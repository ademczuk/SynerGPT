import sys
from pyflakes.api import check
from pyflakes.reporter import Reporter
from io import StringIO

class CustomReporter(Reporter):
    """
    Custom reporter for handling pyflakes output.
    Suppresses "no errors found" messages and prints only non-urgent errors.
    """
    def __init__(self):
        self._captured = StringIO()
        super().__init__(self._captured, self._captured)

    def handle(self):
        """
        Handles the output after analysis.
        Prints the output if it contains non-urgent errors, and suppresses it otherwise.
        """
        content = self._captured.getvalue()
        if "no errors found" not in content.lower():
            print(content)

def analyze_script(file_path):
    """
    Analyzes the specified Python file using pyflakes and uses a custom reporter
    to handle and print messages based on error urgency.
    """
    reporter = CustomReporter()
    # Check the script with pyflakes
    with open(file_path, 'r') as file:
        check(file.read(), file_path, reporter=reporter)
    # Handle the output based on the content
    reporter.handle()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_pyflakes.py <path_to_python_script>")
    else:
        analyze_script(sys.argv[1])
