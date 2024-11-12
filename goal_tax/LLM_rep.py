import os
from langchain.llms import OpenAI
from langchain.agents import Tool, initialize_agent, AgentType
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import KMeans
from typing import List

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = "your_openai_api_key"  # Replace with your OpenAI API key

# Define the list of goals
goals = [
    "Lose weight",
    "Learn a new language",
    "Save money for retirement",
    "Improve public speaking skills",
    "Travel to Japan",
    "Start a small business",
    "Reduce carbon footprint",
    "Run a marathon",
    "Build a mobile app",
    "Write a novel",
    "Volunteer at a local shelter",
    "Adopt a healthier diet",
    "Practice meditation daily",
    "Learn to play the guitar",
    "Obtain a master's degree",
    "Buy a house",
    "Get a pilot's license",
    "Organize a community event",
    "Improve time management",
    "Read 50 books in a year",
]

# Load the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Tool to compute embeddings
def compute_embeddings(goals: List[str]) -> str:
    embeddings = embedding_model.encode(goals).tolist()  # Convert to list for JSON serialization
    return str(embeddings)

embedding_tool = Tool(
    name="ComputeEmbeddings",
    func=compute_embeddings,
    description="Use this tool to compute embeddings for a list of goals. Input should be a list of goals. Output is a list of embedding vectors.",
)

# Tool to compute similarity between goals
def compute_similarity(input_str: str) -> str:
    goals = input_str.split("||")
    if len(goals) != 2:
        return "Error: Please provide two goals separated by '||'"
    goal1 = goals[0].strip()
    goal2 = goals[1].strip()
    emb1 = embedding_model.encode(goal1)
    emb2 = embedding_model.encode(goal2)
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return f"Similarity between '{goal1}' and '{goal2}': {similarity:.4f}"

similarity_tool = Tool(
    name="ComputeSimilarity",
    func=compute_similarity,
    description="Use this tool to compute the cosine similarity between two goals. Input should be two goals separated by '||'. Output is the similarity score.",
)

# Tool to cluster goals
def cluster_goals(input_str: str) -> str:
    import ast
    inputs = input_str.split("||")
    if len(inputs) != 2:
        return "Error: Please provide embeddings and number of clusters separated by '||'"
    goal_embeddings_str = inputs[0].strip()
    num_clusters = int(inputs[1].strip())
    # Convert the string representation back to list
    goal_embeddings = np.array(ast.literal_eval(goal_embeddings_str))
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(goal_embeddings)
    labels = kmeans.labels_
    return str(labels.tolist())

cluster_tool = Tool(
    name="ClusterGoals",
    func=cluster_goals,
    description="Use this tool to cluster goals based on their embeddings. Input should be the embeddings as a string and the number of clusters separated by '||'. Output is a list of cluster labels.",
)

# Initialize the LLM
llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Define the list of tools
tools = [embedding_tool, similarity_tool, cluster_tool]

# Initialize the agent
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

# Define the task instruction
task_instruction = """
Please analyze the following list of goals and organize them into a taxonomy, grouping similar goals together and identifying any hierarchical relationships.

You have access to the following tools to help you:

- ComputeEmbeddings: Compute embeddings for a list of goals.
- ComputeSimilarity: Compute the similarity between two goals.
- ClusterGoals: Cluster goals based on their embeddings.

Use these tools as needed to help you create an interpretable taxonomy.

Here is the list of goals:

{goals_list}
"""

# Format the goals list
goals_list_str = "\n".join([f"{i+1}. {goal}" for i, goal in enumerate(goals)])

# Combine the task instruction and goals list
task = task_instruction.format(goals_list=goals_list_str)

# Run the agent with the task
agent_output = agent.run(task)
