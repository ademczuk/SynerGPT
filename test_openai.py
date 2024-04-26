# test_openai.py
import openai

def test_openai_import():
    """Tests basic import and access of OpenAI API resources."""
    try:
        # Correctly create a completion request
        response = openai.Completion.create(
            engine="davinci",
            prompt="Hello, world!",
            max_tokens=5
        )
        print("Completion request successful. Response:", response)

        # Correctly list all engines
        engines = openai.Engine.list()
        assert isinstance(engines.data, list)  # Check if a list of engines is returned
        print("Engine listing successful. Engines data:", engines.data)

        print("All tests passed successfully.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
