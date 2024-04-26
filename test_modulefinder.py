import os
import modulefinder
import sys

# Get the current working directory
cwd = os.getcwd()

# Append the parent directory to the Python path
sys.path.append(os.path.dirname(cwd))

# Get the absolute path of the current working directory
project_dir = os.path.abspath(cwd)

# Create a ModuleFinder object
finder = modulefinder.ModuleFinder(path=[project_dir] + sys.path)

# Add the project directory to the list of modules to analyze
finder.run_script(os.path.join(project_dir, '__init__.py'))

# Print the modules found
print("Modules found:")
for module_name, module_obj in finder.modules.items():
    print(f"{module_name}: {module_obj}")

# Print the bad modules
print("\nBad modules:")
for module_name, exception in finder.badmodules.items():
    print(f"{module_name}: {exception}")