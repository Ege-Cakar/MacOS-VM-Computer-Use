#!/usr/bin/env python3
"""
VNC GUI Automation MCP Server

Provides MCP server capabilities for VNC remote control.
"""

import os
import tempfile
from datetime import datetime
import sys
import signal
import asyncio
import traceback
from contextlib import asynccontextmanager
from urllib.parse import urlparse
from typing import Dict, Any, Optional, Tuple, List
from enum import StrEnum

import asyncvnc
import asyncssh
from PIL import Image

# Import the MCP SDK
from mcp.server.fastmcp import FastMCP

# ─── CONSTANTS ───────────────────────────────────────────────
# Create a temporary directory for screenshots
SCREENSHOT_DIR = os.path.join(tempfile.gettempdir(), "vnc_screenshots")
os.makedirs(SCREENSHOT_DIR, exist_ok=True)


# ─── LOGGING UTILITIES ───────────────────────────────────────────
def log(message):
    """Print log messages to stderr instead of stdout to not interfere with MCP protocol"""
    print(message, file=sys.stderr, flush=True)

# ─── SCALING SYSTEM ───────────────────────────────────────────
class ScalingSource(StrEnum):
    """Source of coordinates for scaling"""
    COMPUTER = "computer"  # VM's coordinate system
    API = "api"  # LLM's coordinate system

class CoordinateScaler:
    """Handles scaling between LLM coordinate system and VM coordinate system"""
    def __init__(self):
        # Default to standard resolution for LLM's coordinate system
        self.llm_width = 1280
        self.llm_height = 800
        self.vm_width = 1280  # Will be updated when we get actual VM dimensions
        self.vm_height = 800  # Will be updated when we get actual VM dimensions
        self.scale_enabled = True
    
    def update_vm_dimensions(self, width: int, height: int):
        """Update VM screen dimensions"""
        self.vm_width = width
        self.vm_height = height
        log(f"Updated VM dimensions to {width}x{height}")
    
    def scale_coordinates(self, source: ScalingSource, x: int, y: int) -> Tuple[int, int]:
        """Scale coordinates between LLM coordinate system and VM coordinate system"""
        if not self.scale_enabled:
            return x, y
        
        x_scaling_factor = self.vm_width / self.llm_width
        y_scaling_factor = self.vm_height / self.llm_height
        
        if source == ScalingSource.API:
            # LLM's coordinates -> VM screen coordinates
            return round(x * x_scaling_factor), round(y * y_scaling_factor)
        else:
            # VM screen coordinates -> LLM's coordinate system
            return round(x / x_scaling_factor), round(y / y_scaling_factor)
    
    def scale_image(self, image: Image.Image) -> Image.Image:
        """Scale an image from VM resolution to LLM resolution"""
        if not self.scale_enabled or (self.vm_width == self.llm_width and self.vm_height == self.llm_height):
            return image
        
        return image.resize((self.llm_width, self.llm_height))

# ─── INPUT TIMING CONFIGURATION ───────────────────────────────────
TYPING_DELAY_MS = 12
TYPING_GROUP_SIZE = 50
DEFAULT_ACTION_DELAY = 0.1  # 100ms delay between consecutive actions

def chunks(text: str, chunk_size: int) -> List[str]:
    """Split text into chunks of specified size"""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# ─── KEY MAPPING ───────────────────────────────────────────────
def map_key(key: str) -> str:
    """Map common key names to VNC key names"""
    aliases = {
        "command":   "Super_L",
        "enter":     "Return",
        "return":    "Return",
        "esc":       "Escape",
        "escape":    "Escape",
        "ctrl":      "Control_L",
        "control":   "Control_L",
        "shift":     "Shift_L",
        "alt":       "Alt_L",
        "tab":       "Tab",
        "backspace": "BackSpace",
        "cmd":       "Super_L",
        "super":     "Super_L",
        "space":     "space",
        "up":        "Up",
        "down":      "Down",
        "left":      "Left",
        "right":     "Right",
        "pageup":    "Page_Up",
        "pagedown":  "Page_Down",
        "home":      "Home",
        "end":       "End",
        "delete":    "Delete",
        "insert":    "Insert",
        "f1":        "F1",
        "f2":        "F2",
        "f3":        "F3",
        "f4":        "F4",
        "f5":        "F5",
        "f6":        "F6",
        "f7":        "F7",
        "f8":        "F8",
        "f9":        "F9",
        "f10":       "F10",
        "f11":       "F11",
        "f12":       "F12",
    }
    return aliases.get(key.lower(), key)

# ─── VNC CONNECTION ───────────────────────────────────────────
# ─── VNC CONNECTION ───────────────────────────────────────────
@asynccontextmanager
async def connect_vnc(uri: str, scaler: CoordinateScaler):
    """Connect to a VNC server using the provided URI"""
    u = urlparse(uri)
    if u.scheme.lower() != "vnc":
        raise ValueError(f"Unsupported scheme: {u.scheme!r}, expected vnc://")
    
    # Extract connection parameters explicitly
    host = u.hostname
    port = u.port or 5900
    username = u.username
    password = u.password
    
    log(f"Connecting to VNC: {host}:{port} as {username}")
    
    try:
        async with asyncvnc.connect(
            host,
            port=port,
            username=username,
            password=password,
        ) as client:
            # Get screen dimensions from the client
            # Taking a screenshot first to get dimensions
            pixels = await client.screenshot()
            height, width = pixels.shape[:2]  # Screenshot array has shape (height, width, channels)
            scaler.update_vm_dimensions(width, height)
            log(f"Connected to VNC, screen dimensions: {width}x{height}")
            yield client
    except Exception as e:
        log(f"VNC connection error: {e}")
        raise

# ─── VNC CONNECTION MANAGER ───────────────────────────────────────
class VNCManager:
    def __init__(self):
        self.connections = {}  # Store connection info
        self.active_clients = {}  # Store active VNC clients
        self.active_cm = {}  # Store active context managers
        self.coordinate_scaler = CoordinateScaler()
    
    async def register_connection(self, name: str, uri: str, ssh_user: str = None, ssh_password: str = None) -> bool:
        """Register a new VNC connection with the given name and URI"""
        parsed_uri = urlparse(uri)
        host = parsed_uri.hostname
        self.connections[name] = {
            "uri": uri,
            "active": False,
            "host": host,
            "ssh_user": ssh_user or parsed_uri.username,
            "ssh_password": ssh_password or parsed_uri.password
        }
        log(f"Registered connection: {name} -> {uri}")
        return True
    
    async def connect(self, name: str) -> bool:
        """Connect to a registered VNC server"""
        if name not in self.connections:
            log(f"Connection {name} not registered")
            return False
        
        if self.connections[name]["active"]:
            log(f"Connection {name} is already active")
            return True
        
        try:
            uri = self.connections[name]["uri"]
            log(f"Attempting to connect to {name} at {uri}")
            
            # Use the connect_vnc context manager
            cm = connect_vnc(uri, self.coordinate_scaler)
            client = await cm.__aenter__()
            
            self.active_clients[name] = client
            self.active_cm[name] = cm
            self.connections[name]["active"] = True
            log(f"Connected to: {name}")
            return True
        except Exception as e:
            log(f"Failed to connect to {name}: {e}")
            log(traceback.format_exc())
            return False
    
    async def disconnect(self, name: str) -> bool:
        """Disconnect from a VNC server"""
        if name not in self.active_clients:
            log(f"Connection {name} is not active")
            return False
        
        try:
            cm = self.active_cm[name]
            await cm.__aexit__(None, None, None)
            del self.active_clients[name]
            del self.active_cm[name]
            self.connections[name]["active"] = False
            log(f"Disconnected from: {name}")
            return True
        except Exception as e:
            log(f"Failed to disconnect from {name}: {e}")
            log(traceback.format_exc())
            return False
    
    def get_client(self, name: str):
        """Get the active VNC client for a connection"""
        return self.active_clients.get(name)
    
    def get_scaler(self):
        """Get the coordinate scaler"""
        return self.coordinate_scaler
    
    async def cleanup(self):
        """Clean up all active connections"""
        for name in list(self.active_clients.keys()):
            await self.disconnect(name)

# ─── VNC ACTIONS ───────────────────────────────────────────────
async def click_at(client, scaler, x: int, y: int, button: str = "left"):
    """Click at the specified coordinates with scaling"""
    # Scale coordinates from API to VM
    vm_x, vm_y = scaler.scale_coordinates(ScalingSource.API, x, y)
    log(f"Clicking at {x},{y} (scaled to {vm_x},{vm_y})")
    
    client.mouse.move(vm_x, vm_y)
    await asyncio.sleep(DEFAULT_ACTION_DELAY)  # Brief delay before clicking
    
    if button == "left":
        client.mouse.click()
    elif button == "right":
        client.mouse.right_click()
    else:
        client.mouse.middle_click()
    
    await asyncio.sleep(DEFAULT_ACTION_DELAY)  # Brief delay after clicking

async def send_text(client, text: str, delay: float = 0.0):
    """Type text on the remote system with chunking"""
    # Split text into smaller chunks to prevent overwhelming the system
    text_chunks = chunks(text, TYPING_GROUP_SIZE)
    
    for chunk in text_chunks:
        client.keyboard.write(chunk)
        # Small delay between chunks (convert ms to seconds)
        await asyncio.sleep(len(chunk) * (TYPING_DELAY_MS / 1000))
    
    if delay:
        await asyncio.sleep(delay)

async def press_key(client, key: str, delay: float = 0.0):
    """Press a key on the remote system"""
    ks = map_key(key)
    try:
        client.keyboard.press(ks)
        await asyncio.sleep(DEFAULT_ACTION_DELAY)  # Brief delay after key press
    except KeyError:
        # fallback for single characters
        if len(key) == 1:
            client.keyboard.write(key)
            await asyncio.sleep(TYPING_DELAY_MS / 1000)  # Small delay after typing
        else:
            raise
    if delay:
        await asyncio.sleep(delay)

async def hotkey(client, *keys: str):
    """Press a key combination"""
    mkeys = [map_key(k) for k in keys]
    mods, last = mkeys[:-1], mkeys[-1]

    # press modifiers down
    with client.keyboard.hold(*mods):
        # press & release the "real" key
        client.keyboard.press(last)
    
    await asyncio.sleep(DEFAULT_ACTION_DELAY)  # Brief delay after hotkey

async def take_screenshot(client, scaler, outfile: str = "screenshot.png"):
    """Take a screenshot of the remote system and scale if needed"""
    # Ensure the path is within the temp directory
    if os.path.dirname(outfile) == "":
        outfile = os.path.join(SCREENSHOT_DIR, outfile)
    else:
        # If a specific path was provided, still make sure the directory exists
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
    
    pixels = await client.screenshot()
    img = Image.fromarray(pixels)
    
    # Scale the image if needed
    img = scaler.scale_image(img)
    
    img.save(outfile)
    log(f"Wrote screenshot: {outfile} (dimensions: {img.width}x{img.height})")
    return outfile


async def run_ssh_command(host, user, pwd, cmd):
    """Run a command via SSH on the remote system"""
    log(f"SSH connecting to {host} as {user}")
    async with asyncssh.connect(
        host,
        username=user,
        password=pwd,
        known_hosts=None
    ) as conn:
        log(f"Running SSH command: {cmd}")
        result = await conn.run(cmd, check=True)
        return result.stdout, result.stderr

# ─── INITIALIZATION ───────────────────────────────────────────
async def setup_default_connection(vnc_manager):
    """Set up the default VNC connection"""
    try:
        await vnc_manager.register_connection(
            "default",
            "vnc://claude:1234@192.168.64.3",
            ssh_user="claude",
            ssh_password="1234"
        )
        log("Default connection registered")
    except Exception as e:
        log(f"Error setting up default connection: {e}")
        log(traceback.format_exc())

# ─── MCP SERVER ───────────────────────────────────────────────
def create_mcp_server():
    """Create a FastMCP server with VNC automation tools"""
    # Initialize FastMCP server
    mcp = FastMCP("VNC Automation")
    vnc_manager = VNCManager()
    
    # Register signal handlers for cleanup
    def handle_signal(signum, frame):
        log(f"Received signal {signum}, shutting down...")
        # We need to run cleanup in a new event loop
        loop = asyncio.new_event_loop()
        loop.run_until_complete(vnc_manager.cleanup())
        loop.close()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    
    # Register a default VNC connection
    # We'll run this setup before starting the server
    loop = asyncio.get_event_loop()
    loop.run_until_complete(setup_default_connection(vnc_manager))
    
    # Add a debug tool to test connections directly
    @mcp.tool()
    async def vnc_test_connection(uri: str = "vnc://claude:1234@192.168.64.3") -> dict:
        """
        Test a VNC connection directly using the original method
        
        Args:
            uri: VNC URI to test
            
        Returns:
            Status of the test
        """
        try:
            log(f"Testing direct VNC connection to {uri}")
            test_screenshot_path = os.path.join(SCREENSHOT_DIR, "test_connection.png")
            async with connect_vnc(uri, vnc_manager.get_scaler()) as client:
                # Take a quick screenshot to verify connection
                await take_screenshot(client, vnc_manager.get_scaler(), test_screenshot_path)
                return {"success": True, "message": "Connection successful, screenshot saved", "file": test_screenshot_path}
        except Exception as e:
            log(f"Test connection failed: {e}")
            log(traceback.format_exc())
            return {"success": False, "error": str(e)}
        
    # Define MCP tools
    @mcp.tool()
    async def vnc_register(name: str, uri: str, ssh_user: str = None, ssh_password: str = None) -> bool:
        """
        Register a VNC connection with the given name and URI.
        
        Args:
            name: Name to identify this connection
            uri: VNC URI in the format vnc://user:pass@host:port
            ssh_user: Optional SSH username for remote commands
            ssh_password: Optional SSH password for remote commands
            
        Returns:
            True if registration was successful
        """
        return await vnc_manager.register_connection(name, uri, ssh_user, ssh_password)
    
    @mcp.tool()
    async def vnc_connect(name: str) -> bool:
        """
        Connect to a registered VNC server.
        
        Args:
            name: Name of the connection to connect to
            
        Returns:
            True if connection was successful
        """
        return await vnc_manager.connect(name)
    
    @mcp.tool()
    async def vnc_disconnect(name: str) -> bool:
        """
        Disconnect from a VNC server.
        
        Args:
            name: Name of the connection to disconnect from
            
        Returns:
            True if disconnection was successful
        """
        return await vnc_manager.disconnect(name)
    
    @mcp.tool()
    async def vnc_click(connection: str, x: int, y: int, button: str = "left") -> dict:
        """
        Click at specified coordinates on the remote system.
        
        Args:
            connection: Name of the VNC connection to use
            x: X coordinate to click
            y: Y coordinate to click
            button: Mouse button to use (left, right, middle)
            
        Returns:
            Status of the operation
        """
        client = vnc_manager.get_client(connection)
        if not client:
            success = await vnc_manager.connect(connection)
            if not success:
                return {"success": False, "error": f"Could not connect to {connection}"}
            client = vnc_manager.get_client(connection)
        
        try:
            await click_at(client, vnc_manager.get_scaler(), x, y, button)
            return {
                "success": True,
                "message": f"Clicked at coordinates {x},{y} with {button} button",
                "scaled_coordinates": vnc_manager.get_scaler().scale_coordinates(ScalingSource.API, x, y)
            }
        except Exception as e:
            log(f"Error clicking at {x},{y}: {e}")
            log(traceback.format_exc())
            return {"success": False, "error": str(e)}
    
    @mcp.tool()
    async def vnc_text(connection: str, text: str, delay: float = 0.0) -> dict:
        """
        Type text on the remote system.
        
        Args:
            connection: Name of the VNC connection to use
            text: Text to type
            delay: Optional delay after typing (in seconds)
            
        Returns:
            Status of the operation
        """
        client = vnc_manager.get_client(connection)
        if not client:
            success = await vnc_manager.connect(connection)
            if not success:
                return {"success": False, "error": f"Could not connect to {connection}"}
            client = vnc_manager.get_client(connection)
        
        try:
            await send_text(client, text, delay)
            return {
                "success": True,
                "message": f"Typed text ({len(text)} characters) with chunking",
                "chunk_size": TYPING_GROUP_SIZE,
                "delay_ms": TYPING_DELAY_MS
            }
        except Exception as e:
            log(f"Error sending text: {e}")
            log(traceback.format_exc())
            return {"success": False, "error": str(e)}
    
    @mcp.tool()
    async def vnc_key(connection: str, key: str, delay: float = 0.0) -> dict:
        """
        Press a key on the remote system.
        
        Args:
            connection: Name of the VNC connection to use
            key: Key to press
            delay: Optional delay after pressing (in seconds)
            
        Returns:
            Status of the operation
        """
        client = vnc_manager.get_client(connection)
        if not client:
            success = await vnc_manager.connect(connection)
            if not success:
                return {"success": False, "error": f"Could not connect to {connection}"}
            client = vnc_manager.get_client(connection)
        
        try:
            mapped_key = map_key(key)
            await press_key(client, key, delay)
            return {
                "success": True,
                "message": f"Pressed key: {key} (mapped to: {mapped_key})"
            }
        except Exception as e:
            log(f"Error pressing key {key}: {e}")
            log(traceback.format_exc())
            return {"success": False, "error": str(e)}
    
    @mcp.tool()
    async def vnc_hotkey(connection: str, keys: list[str]) -> dict:
        """
        Press a key combination on the remote system.
        
        Args:
            connection: Name of the VNC connection to use
            keys: List of keys to press (modifiers first, then main key)
            
        Returns:
            Status of the operation
        """
        client = vnc_manager.get_client(connection)
        if not client:
            success = await vnc_manager.connect(connection)
            if not success:
                return {"success": False, "error": f"Could not connect to {connection}"}
            client = vnc_manager.get_client(connection)
        
        try:
            await hotkey(client, *keys)
            mapped_keys = [map_key(k) for k in keys]
            return {
                "success": True,
                "message": f"Pressed hotkey: {'+'.join(keys)} (mapped to: {'+'.join(mapped_keys)})"
            }
        except Exception as e:
            log(f"Error pressing hotkey {keys}: {e}")
            log(traceback.format_exc())
            return {"success": False, "error": str(e)}
    
    @mcp.tool()
    async def vnc_screenshot(connection: str, file: str = None) -> dict:
        """
        Take a screenshot of the remote system.
        
        Args:
            connection: Name of the VNC connection to use
            file: Output file path for the screenshot (defaults to timestamp-based filename in temp dir)
            
        Returns:
            Status of the operation and the path to the screenshot file
        """
        client = vnc_manager.get_client(connection)
        if not client:
            success = await vnc_manager.connect(connection)
            if not success:
                return {"success": False, "error": f"Could not connect to {connection}"}
            client = vnc_manager.get_client(connection)
        
        try:
            # Generate a timestamp-based filename if none provided
            if file is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file = f"vnc_screenshot_{timestamp}.png"
            
            filepath = await take_screenshot(client, vnc_manager.get_scaler(), file)
            dimensions = vnc_manager.get_scaler().llm_width, vnc_manager.get_scaler().llm_height
            return {
                "success": True,
                "file": filepath,
                "dimensions": f"{dimensions[0]}x{dimensions[1]} (scaled from VM resolution)"
            }
        except Exception as e:
            log(f"Error taking screenshot: {e}")
            log(traceback.format_exc())
            return {"success": False, "error": str(e)}
    
    @mcp.tool()
    async def vnc_ssh(connection: str, command: str) -> dict:
        """
        Execute an SSH command on the remote system.
        
        Args:
            connection: Name of the VNC connection to use
            command: SSH command to execute
            
        Returns:
            Status of the operation and the command output
        """
        if connection not in vnc_manager.connections:
            return {"success": False, "error": f"Connection {connection} not registered"}
        
        info = vnc_manager.connections[connection]
        host, user, pw = info["host"], info["ssh_user"], info["ssh_password"]
        if not all([host, user, pw]):
            return {"success": False, "error": "Missing SSH credentials"}
        
        try:
            stdout, stderr = await run_ssh_command(host, user, pw, command)
            return {"success": True, "stdout": stdout, "stderr": stderr}
        except Exception as e:
            log(f"Error executing SSH command: {e}")
            log(traceback.format_exc())
            return {"success": False, "error": str(e)}
    
    return mcp

# ─── MAIN FUNCTION ───────────────────────────────────────────────
def main():
    """Run the VNC MCP server"""
    try:
        # Create and run the MCP server
        mcp = create_mcp_server()
        
        # The run method handles both stdio and other transport modes
        mcp.run()
    except Exception as e:
        log(f"Fatal error: {e}")
        log(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
    