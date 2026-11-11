import os
from dotenv import load_dotenv
from crew import ProtocrewCrew


def run():
    load_dotenv()

    # Define o Google Gemini como o motor de raciocínio da Crew
    os.environ["MODEL"] = "gemini/gemini-2.5-pro"

    folder_id = os.getenv("DRIVE_FOLDER_ID", "").strip()
    if not folder_id:
        raise ValueError(
            "Variável de ambiente DRIVE_FOLDER_ID não configurada. "
            "Copie .env.example para .env e informe o ID da pasta do Google Drive."
        )

    inputs = {
        "folder_id": folder_id
    }

    print("=" * 50)
    print("Iniciando O PROTOCREW (Pipeline Autônomo)")
    print("=" * 50)

    try:
        ProtocrewCrew().crew().kickoff(inputs=inputs)
        print("\n✅ Execução completa do processo CrewAI!")
    except Exception as e:
        print(f"\n❌ Falha na execução: {e}")


if __name__ == "__main__":
    run()
