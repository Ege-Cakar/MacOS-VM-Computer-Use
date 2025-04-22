import asyncio
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from anthropic import Anthropic
from dotenv import load_dotenv
import os
import argparse
import sys

load_dotenv()

# Default system prompt that will be used if none is provided via command line
DEFAULT_SYSTEM_PROMPT = """
You are BUTLER, an agent designed to help users complete tasks on their computer. Follow these guidelines when assisting users with computer use:

Core Functionality:
Utilize available tools to help users complete computer tasks.
Verify each action with screenshots before proceeding to the next step.
Prioritize keyboard shortcuts whenever possible for efficiency.
Workflow Protocol:
Analyze the task requested by the user.
Break down complex tasks into clear, sequential steps.
For each step:

    Call the tools with exact commands or clicks needed.
    Request a screenshot after the action is completed.
    Verify the result from the screenshot before proceeding.
    If the result doesn't match expectations, try to troubleshoot.
    Confirm task completion with a final verification screenshot.
In terms of memory, follow these steps for each interaction:

1. User Identification:
   - You should assume that you are interacting with default_user
   - If you have not identified default_user, proactively try to do so.

2. Memory Retrieval:
   - Always begin your chat by saying only "Remembering..." and retrieve all relevant information from your knowledge graph
   - Always refer to your knowledge graph as your "memory"

3. Memory
   - While conversing with the user, be attentive to any new information that falls into these categories:
     a) Basic Identity (age, gender, location, job title, education level, etc.)
     b) Behaviors (interests, habits, etc.)
     c) Preferences (communication style, preferred language, etc.)
     d) Goals (goals, targets, aspirations, etc.)
     e) Relationships (personal and professional relationships up to 3 degrees of separation)

4. Memory Update:
   - If any new information was gathered during the interaction, update your memory as follows:
     a) Create entities for recurring organizations, people, and significant events
     b) Connect them to the current entities using relations
     c) Store facts about them as observations
"""

class MCPClient:
    def __init__(self, system_prompt: str = None):
        # Initialize session and client objects
        self.sessions = {}  # Dictionary to store multiple sessions
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        # Use the default system prompt if none is provided
        self.system_prompt = system_prompt if system_prompt is not None else DEFAULT_SYSTEM_PROMPT
        self.available_tools = []

    async def connect_to_server(self, server_identifier: str):
        """Connect to an MCP server
        
        Args:
            server_identifier: Path to the server script (.py or .js) or a server type identifier ('memory', 'vnc')
        """
        # Check if it's a predefined server type
        if server_identifier == 'memory':
            # Memory server
            command = "npx"
            args = ["-y", "@modelcontextprotocol/server-memory"]
            server_name = "memory"
        elif server_identifier == 'vnc':
            # VNC server - assuming vnc_mcp.py is in the same directory as this script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            vnc_script_path = os.path.join(script_dir, "vnc_mcp.py")
            if not os.path.exists(vnc_script_path):
                raise ValueError(f"VNC script not found at {vnc_script_path}")
            
            command = "python"
            args = [vnc_script_path]
            server_name = "vnc"
        else:
            # Regular script path
            is_python = server_identifier.endswith('.py')
            is_js = server_identifier.endswith('.js')
            if not (is_python or is_js):
                raise ValueError("Server script must be a .py or .js file")
                
            command = "python" if is_python else "node"
            args = [server_identifier]
            server_name = os.path.basename(server_identifier).split('.')[0]
        
        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=None
        )
        
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        stdio, write = stdio_transport
        session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
        
        await session.initialize()
        
        # Store the session
        self.sessions[server_name] = session
        
        # List available tools
        response = await session.list_tools()
        tools = response.tools
        print(f"\nConnected to {server_name} server with tools:", [tool.name for tool in tools])
        
        # Update available tools
        await self.update_available_tools()
        
        return server_name

    async def connect_to_multiple_servers(self, server_identifiers: list):
        """Connect to multiple MCP servers
        
        Args:
            server_identifiers: List of server identifiers (paths or types)
        """
        for server_id in server_identifiers:
            await self.connect_to_server(server_id)
    
    async def update_available_tools(self):
        """Update the list of available tools from all connected servers"""
        all_tools = []
        
        for server_name, session in self.sessions.items():
            response = await session.list_tools()
            server_tools = [{ 
                "name": f"{server_name}_{tool.name}",  # Prefix with server name to avoid conflicts
                "description": f"[{server_name}] {tool.description}",
                "input_schema": tool.inputSchema,
                "original_name": tool.name,
                "server": server_name
            } for tool in response.tools]
            
            all_tools.extend(server_tools)
        
        self.available_tools = all_tools
        return self.available_tools

    async def get_available_tools(self):
        """Get the list of available tools from all connected servers"""
        if not self.sessions:
            raise ValueError("No sessions initialized. Call connect_to_server first.")
            
        if not self.available_tools:
            await self.update_available_tools()
            
        return self.available_tools

    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        # Make sure we have the available tools
        if not self.available_tools:
            await self.get_available_tools()
        
        print(f"Available tools: {[tool['name'] for tool in self.available_tools]}")

        # Create Claude-compatible tools list (without server-specific fields)
        claude_tools = [{
            "name": tool["name"],
            "description": tool["description"],
            "input_schema": tool["input_schema"]
        } for tool in self.available_tools]

        # Initial Claude API call
        response = self.anthropic.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=1000,
            messages=messages,
            tools=claude_tools,
            system=self.system_prompt
        )

        # Process response and handle tool calls
        final_text = []
        tool_call_count = 0
        
        while True:
            assistant_message = {"role": "assistant", "content": []}
            has_tool_calls = False
            
            for content in response.content:
                if content.type == 'text':
                    final_text.append(content.text)
                    assistant_message["content"].append({"type": "text", "text": content.text})
                elif content.type == 'tool_use':
                    has_tool_calls = True
                    tool_name = content.name
                    tool_args = content.input
                    tool_call_count += 1
                    tool_id = f"call_{tool_call_count}"
                    
                    # Find the tool in our available tools
                    tool_info = next((t for t in self.available_tools if t["name"] == tool_name), None)
                    
                    if not tool_info:
                        error_msg = f"Tool {tool_name} not found"
                        final_text.append(f"[Error: {error_msg}]")
                        assistant_message["content"].append({"type": "text", "text": f"Error: {error_msg}"})
                        continue
                    
                    # Add tool call to assistant message with required id field
                    assistant_message["content"].append({
                        "type": "tool_use",
                        "id": tool_id,
                        "name": tool_name,
                        "input": tool_args
                    })
                    
                    # Get the server and original tool name
                    server_name = tool_info["server"]
                    original_tool_name = tool_info["original_name"]
                    
                    # Execute tool call on the appropriate server
                    try:
                        result = await self.sessions[server_name].call_tool(original_tool_name, tool_args)
                        
                        # Log the tool call and result
                        final_text.append(f"[Calling {server_name} tool {original_tool_name} with args {tool_args}]")
                        final_text.append(f"[Tool result: {result.content}]")
                        
                        # Add assistant message with tool call to conversation
                        messages.append(assistant_message)
                        
                        # Add tool result as user message with tool_result format
                        messages.append({
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tool_id,
                                    "content": result.content
                                }
                            ]
                        })
                    except Exception as e:
                        error_msg = f"Error calling tool {tool_name}: {str(e)}"
                        final_text.append(f"[Error: {error_msg}]")
                        
                        # Add error as tool result
                        messages.append(assistant_message)
                        messages.append({
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tool_id,
                                    "content": f"Error: {error_msg}"
                                }
                            ]
                        })
                    
                    break
            
            # If no tool calls or we've processed all content, add the assistant message
            if not has_tool_calls:
                if assistant_message["content"]:
                    messages.append(assistant_message)
                break
            
            # Get next response from Claude for the next iteration
            response = self.anthropic.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=1000,
                messages=messages,
                tools=claude_tools,
                system=self.system_prompt
            )

        return "\n".join(final_text)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")
        
        print("\nSystem prompt is set to:")
        print(f"---\n{self.system_prompt}\n---")
        
        # Get available tools at startup
        tools = await self.get_available_tools()
        print(f"\nAvailable tools from all servers:")
        for tool in tools:
            print(f"- {tool['name']}: {tool['description']}")
        
        # Send an initial message to Claude with the available tools
        initial_message = "Here are the available tools you can use:\n"
        for tool in tools:
            initial_message += f"- {tool['name']}: {tool['description']}\n"
        
        print("\nSending tool information to Claude...")
        response = await self.process_query(initial_message)
        print("\nClaude is ready to use the tools.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                
                if query.lower() == 'quit':
                    break
                    
                response = await self.process_query(query)
                print("\n" + response)
                    
            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

async def main():
    parser = argparse.ArgumentParser(description="MCP Client for Claude")
    parser.add_argument("server_scripts", nargs='+', help="Paths to server scripts (.py or .js) or server types ('memory', 'vnc')")
    parser.add_argument("--system-prompt", "-s", help="System prompt for Claude (overrides default)")
    parser.add_argument("--system-prompt-file", "-f", help="File containing system prompt for Claude (overrides default)")
    parser.add_argument("--no-system-prompt", "-n", action="store_true", help="Don't use any system prompt")
    
    args = parser.parse_args()
    
    system_prompt = DEFAULT_SYSTEM_PROMPT
    
    if args.no_system_prompt:
        system_prompt = None
    elif args.system_prompt_file:
        try:
            with open(args.system_prompt_file, 'r') as f:
                system_prompt = f.read()
        except Exception as e:
            print(f"Error reading system prompt file: {e}")
            sys.exit(1)
    elif args.system_prompt:
        system_prompt = args.system_prompt
    
    client = MCPClient(system_prompt=system_prompt)
    try:
        await client.connect_to_multiple_servers(args.server_scripts)
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())