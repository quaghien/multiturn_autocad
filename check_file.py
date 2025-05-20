from trl import maybe_convert_to_chatml
import inspect

# Get the module path
module_path = inspect.getfile(maybe_convert_to_chatml)
print(f"Module path: {module_path}")    