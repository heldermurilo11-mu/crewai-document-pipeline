"""CrewAI sequential agent crew orchestration."""

from tools._crew_compat import Agent, Crew, CrewBase, Process, Task, agent, crew, task
from tools.crew_wrappers import (
    baixar_arquivos_drive,
    extrair_dados_cadeia_custodia,
    validar_e_exportar_dados,
)


@CrewBase
class ProtocrewCrew:
    """Sequential CrewAI pipeline orchestrating Drive acquisition, OCR extraction, and CSV export."""

    @agent
    def drive_manager(self) -> Agent:
        return Agent(
            config=self.agents_config["drive_manager"],
            tools=[baixar_arquivos_drive],
            verbose=True,
        )

    @agent
    def ocr_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config["ocr_specialist"],
            tools=[extrair_dados_cadeia_custodia],
            verbose=True,
        )

    @agent
    def data_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["data_analyst"],
            tools=[validar_e_exportar_dados],
            verbose=True,
        )

    @task
    def download_task(self) -> Task:
        return Task(
            config=self.tasks_config["download_files"],
            agent=self.drive_manager(),
        )

    @task
    def extraction_task(self) -> Task:
        return Task(
            config=self.tasks_config["extract_data"],
            agent=self.ocr_specialist(),
        )

    @task
    def validation_and_export_task(self) -> Task:
        return Task(
            config=self.tasks_config["validate_and_export"],
            agent=self.data_analyst(),
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
