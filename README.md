# UTM-Computer-Use

A tool for remote control and automation of virtual machines through VNC, with a focus on UTM virtual machines.

## Overview

This repository contains two main components:

1. **vnc_mcp.py**: An MCP (Model Context Protocol) server that provides VNC remote control capabilities.
2. **client.py**: A client application that connects to the MCP server and uses Claude AI to interpret user requests and execute them on the virtual machine.

## Prerequisites

- Python 3.8 or higher
- UTM (Universal Type Manager) for macOS
- VNC-enabled virtual machine
- Claude API key

## Installation

### 1. Install UTM

The easiest way to install UTM on macOS is through Homebrew:

```bash
brew install --cask utm
```

Alternatively, you can download UTM directly from the [official website](https://mac.getutm.app/).

### 2. Set up a Virtual Machine with VNC

1. Create a new virtual machine in UTM
2. Enable VNC in your virtual machine:
   - For Linux VMs: Install a VNC server like TigerVNC or x11vnc
   - For Windows VMs: Enable Remote Desktop and use a VNC wrapper
   - For macOS VMs: Enable Screen Sharing in System Preferences

> Note that this has only been tested with macOS VMs, and was the reason it was created. 

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up Environment Variables

Create a `.env` file in the root directory with your Claude API key:

```
ANTHROPIC_API_KEY=your_api_key_here
```

### 5. Configure VNC Connection

The VNC MCP server needs to know how to connect to your virtual machine. You can configure this by creating a `credentials.txt` file:

1. Copy the example file to create your own configuration:
   ```bash
   cp credentials.txt.example credentials.txt
   ```

2. Edit the `credentials.txt` file with your VM's connection details:
   ```
   VNC_URI=vnc://user:password@host:port
   SSH_USER=your_vm_username
   SSH_PASSWORD=your_vm_password
   ```

3. Secure your credentials file (recommended):
   ```bash
   chmod 600 credentials.txt
   ```

If no `credentials.txt` file is found, the server will use default values, which will probably mean it won't work.

## Usage

### Running the VNC MCP Server

The `vnc_mcp.py` file serves as an MCP server that provides VNC remote control capabilities. It can be run directly:

```bash
python vnc_mcp.py
```

However, it's typically used through the client application.

### Running the Client Application

The client application connects to the MCP server and provides an interface for interacting with the virtual machine:

```bash
python client.py vnc
```

This will start the client and connect it to the VNC MCP server.

You can also specify multiple servers:

```bash
python client.py vnc memory
```

#### Client Command-line Options

- `--system-prompt` or `-s`: Provide a custom system prompt for Claude
- `--system-prompt-file` or `-f`: Provide a file containing a custom system prompt
- `--no-system-prompt` or `-n`: Don't use any system prompt

Example:

```bash
python client.py vnc --system-prompt "Your custom prompt here"
```

## Features

The VNC MCP server provides several tools for interacting with a virtual machine:

- **vnc_connect**: Connect to a VNC server
- **vnc_click**: Click at specific coordinates on the remote system
- **vnc_type**: Type text on the remote system
- **vnc_press_key**: Press a specific key on the remote system
- **vnc_hotkey**: Press a combination of keys on the remote system
- **vnc_screenshot**: Take a screenshot of the remote system
- **vnc_ssh**: Execute an SSH command on the remote system (if SSH credentials are provided)

## Troubleshooting

### Connection Issues

If you're having trouble connecting to your VM:

1. Verify that the VNC server is running on your VM
2. Check that the VNC URI is correct (typically `vnc://localhost:5900` or similar)
3. Ensure no firewall is blocking the VNC port

### For best performance

1. Set the resolution of your VM to 1280 x 800 or 1280 x 720 (also improves the performance of Claude).
2. Ensure your host machine has sufficient resources

## License

[MIT License](LICENSE)
