import click
from langchain_community.llms import Ollama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager

def load_llm():
    return Ollama(
        model="qwen2:0.5b",
        verbose=True,
        temperature=0.5,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )

def generate_summary(text):
    llm = load_llm()
    prompt = f"Please provide a concise summary of the following text:\n\n{text}\n\nSummary:"
    return llm.invoke(prompt)

@click.command()
@click.option('-t', '--text-file', type=click.Path(exists=True), help='Path to the text file to summarize')
@click.argument('text', required=False)
def summarize(text_file, text):
    if text_file:
        try:
            # Try reading with UTF-8 encoding first
            with open(text_file, 'r', encoding='utf-8') as file:
                content = file.read()
        except UnicodeDecodeError:
            # If UTF-8 fails, try with cp1252 and ignore errors
            try:
                with open(text_file, 'r', encoding='cp1252', errors='ignore') as file:
                    content = file.read()
            except Exception as e:
                print(f"Error: Unable to read file {text_file}. {str(e)}")
                return
        
        print(f"Summary of {text_file}:")
        generate_summary(content)
    elif text:
        print("Summary of text pasted:")
        generate_summary(text)
    else:
        print("Error: Please provide either a text file or text input.")

if __name__ == '__main__':
    summarize()