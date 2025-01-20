FROM python:3.10-slim

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends python3-tk && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY project.py .

# Check if report.html is a directory, handle gracefully
CMD ["sh", "-c", "if [ -d report.html ]; then echo '<h1>Error: report.html is a directory</h1>' > /app/error.html && mv /app/error.html report.html; fi && python -m http.server 8082"]
