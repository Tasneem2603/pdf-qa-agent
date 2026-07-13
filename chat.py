"""
Interactive PDF chatbot.

Run this, give it a PDF filename once, then ask as many questions as you like.
Type 'quit' or 'exit' to stop.
"""
from agent import PDFAgent

def main():
    print("=== PDF Q&A Chatbot ===")
    pdf_path = input("PDF filename (e.g. demo.pdf): ").strip()

    print(f"\nLoading and reading '{pdf_path}'...")
    try:
        agent = PDFAgent(pdf_path)
    except FileNotFoundError:
        print(f"Could not find '{pdf_path}'. Make sure it's in this folder, or give the full path.")
        return
    print(f"Done — loaded {len(agent.chunks)} sentences from the PDF. Ask away.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if not question:
            continue
        answer = agent.ask(question)
        print(f"Agent: {answer}\n")


if __name__ == "__main__":
    main()
