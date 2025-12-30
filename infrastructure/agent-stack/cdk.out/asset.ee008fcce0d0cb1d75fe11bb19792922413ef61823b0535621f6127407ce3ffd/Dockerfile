FROM --platform=linux/arm64 python:3.11-slim-bullseye

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install Python, pip, and other dependencies
RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-venv curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install uv (Python package manager)
RUN pip3 install uv

# Set working directory
WORKDIR /app

# Set AWS environment variables
ENV AWS_REGION=us-east-1
ENV AWS_DEFAULT_REGION=us-east-1
ENV DOCKER_CONTAINER=1
ENV PYTHONPATH=/app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt
RUN uv pip install --system aws-opentelemetry-distro>=0.10.1

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 bedrock_agentcore && \
    chown -R bedrock_agentcore:bedrock_agentcore /app
USER bedrock_agentcore

# Expose ports
EXPOSE 8080
EXPOSE 8000

# Add HEALTHCHECK 
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD pgrep -f "python agent.py" || exit 1

# Run the agent
CMD ["opentelemetry-instrument", "python", "agent.py"]