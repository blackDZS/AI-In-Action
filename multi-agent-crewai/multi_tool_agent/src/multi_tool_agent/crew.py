import os
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import DirectorySearchTool, PDFSearchTool

# If you want to run a snippet of code before or after the crew starts, 
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class MultiToolAgent():
	"""MultiToolAgent crew"""

	# Learn more about YAML configuration files here:
	# Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
	# Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	pdf_search_tool = PDFSearchTool(
		pdf="/Users/dizhensheng/Library/Mobile Documents/iCloud~md~obsidian/Documents/AI-In-Action/AI-In-Action/UsingAI/大模型/深圳大数据局/数据/大数据局一局一档知识库/南山区城市更新和土地整备局.pdf",
		config=dict(
			llm=dict(
				provider="openai", # Options include ollama, google, anthropic, llama2, and more
				config=dict(
					model=os.environ.get("MODEL"),
					api_key=os.environ.get("OPENAI_API_KEY"),
					base_url=os.environ.get("OPENAI_API_BASE"),
					# Additional configurations here
				),
			),
			embedder=dict(
				provider="openai", # or openai, ollama, ...
				config=dict(
					model="text-embedding-3-large",
					api_key=os.environ.get("OPENAI_API_KEY"),
					api_base=os.environ.get("OPENAI_API_BASE"),
					# task_type="retrieval_document",
					# title="Embeddings",
				),
			),
		)
	)
	# If you would like to add tools to your agents, you can learn more about it here:
	# https://docs.crewai.com/concepts/agents#agent-tools
	@agent
	def researcher(self) -> Agent:
		return Agent(
			config=self.agents_config['researcher'],
			verbose=True,
			allow_delegation=False,
			tools=[self.pdf_search_tool]
		)

	@agent
	def professional_writer(self) -> Agent:
		return Agent(
			config=self.agents_config['professional_writer'],
			verbose=True,
			allow_delegation=False,
		)

	# To learn more about structured task outputs, 
	# task dependencies, and task callbacks, check out the documentation:
	# https://docs.crewai.com/concepts/tasks#overview-of-a-task
	@task
	def research_task(self) -> Task:
		return Task(
			config=self.tasks_config['research_task'],
			agent=self.researcher()
		)

	@task
	def write_email_task(self) -> Task:
		return Task(
			config=self.tasks_config['write_email_task'],
			output_file='report.md',
			agent=self.professional_writer()
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the MultiToolAgent crew"""
		# To learn how to add knowledge sources to your crew, check out the documentation:
		# https://docs.crewai.com/concepts/knowledge#what-is-knowledge

		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			# process=Process.sequential,
			verbose=True,
			# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
		)
