from fastmcp import FastMCP


mcp = FastMCP(name= 'Calculator')

@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers.

    Args:
        a (float): First number.
        b (float): Second number.

    Returns:
        float: a * b
    """
    return a * b

@mcp.tool()
def divide(a: float, b: float) -> float:
    """Divide two numbers.

    Args:
        a (float): First number.
        b (float): Second number.

    Returns:
        float: a / b
    """
    return a / b

@mcp.tool()
def subtract(a: float, b: float) -> float:
    """Subtract two numbers.

    Args:
        a (float): First number.
        b (float): Second number.

    Returns:
        float: a - b
    """
    return a - b

# can also do it this way
@mcp.tool(
    name= 'add',
    description= 'Adds two numbers',
    tags= {'maths', 'arithmetic', 'addition'}
)
def addition(a: float, b: float) -> float:
    """Adds two numbers.

    Args:
        a (float): First number.
        b (float): Second number.

    Returns:
        float: a + b
    """
    return a + b


if __name__ == '__main__':
    mcp.run()     # STDIO by default

    # run the script then type this command in other terminal: npx @modelcontextprotocol/inspector python 01_fastmcp_calculator.py