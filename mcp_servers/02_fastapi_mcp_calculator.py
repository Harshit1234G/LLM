#HTTP
import uvicorn
from fastapi import FastAPI
from fastapi_mcp import FastApiMCP


app = FastAPI(title="Calculator API")

@app.post("/multiply")
def multiply_numbers(a: float, b: float):
    """
    Multiplies two numbers and returns the result.
    """
    result = a * b
    return {"result": result}


@app.post("/add")
def add_numbers(a: float, b: float):
    """
    Adds two numbers and returns the result.
    """
    result = a + b
    return {"result": result}


@app.post("/subtract")
def subtract_numbers(a: float, b: float):
    """
    Subtracts two numbers and returns the result.
    """
    result = a - b
    return {"result": result}


@app.post("/divide")
def divide_numbers(a: float, b: float):
    """
    Divides two numbers and returns the result.
    """
    if b == 0:
        return {"error": "Division by zero is not allowed."}
    
    result = a / b
    return {"result": result}



# Converting the app to MCP
mcp = FastApiMCP(app, name="Calculator MCP")
mcp.mount_http()


if __name__ == "__main__":
    uvicorn.run(app)
    # run the script then type this command in other terminal: npx @modelcontextprotocol/inspector <host url>