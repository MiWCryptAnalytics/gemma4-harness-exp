"""Checks for the native <|tool_call> parser (no model/GPU needed)."""
from tools import parse_tool_calls

# Exactly what the model emitted in probe_native_toolcall.py.
out = parse_tool_calls('<|tool_call>call:get_weather{location:<|"|>Tokyo<|"|>}<tool_call|><|tool_response>')
assert out == [{"name": "get_weather", "arguments": {"location": "Tokyo"}}], out
print("single string arg: OK")

# Mixed types: bare integer + delimited string (template serialization order).
out = parse_tool_calls('<|tool_call>call:get_weather{days:3,location:<|"|>Tokyo<|"|>}<tool_call|>')
assert out == [{"name": "get_weather", "arguments": {"days": 3, "location": "Tokyo"}}], out
print("int + string args: OK")

# The case that broke the OLD harness: multi-line code with newlines/quotes in a
# string value. Here it's wrapped in the reserved delimiter, so it just works.
multiline = (
    '<|tool_call>call:write_file{'
    'content:<|"|>def f():\n    return "hi, there"\n<|"|>,'
    'path:<|"|>a.py<|"|>}<tool_call|>'
)
out = parse_tool_calls(multiline)
assert out[0]["name"] == "write_file"
assert out[0]["arguments"]["content"] == 'def f():\n    return "hi, there"\n', repr(out[0]["arguments"]["content"])
assert out[0]["arguments"]["path"] == "a.py"
print("multi-line/quotes/commas inside string: OK")
print(out[0]["arguments"]["content"])

# Booleans, nested arrays/objects.
out = parse_tool_calls('<|tool_call>call:cfg{flag:true,nums:[1,2,3],meta:{k:<|"|>v<|"|>}}<tool_call|>')
assert out == [{"name": "cfg", "arguments": {"flag": True, "nums": [1, 2, 3], "meta": {"k": "v"}}}], out
print("bool/array/object: OK")

# A final-answer turn (no tool call) yields nothing.
assert parse_tool_calls("The weather in Tokyo is sunny.") == []
print("no-call final answer: OK")

# Two calls in one turn.
out = parse_tool_calls('<|tool_call>call:a{x:1}<tool_call|> then <|tool_call>call:b{y:2}<tool_call|>')
assert [c["name"] for c in out] == ["a", "b"], out
print("multiple calls: OK")

print("\nall native parser tests passed")
