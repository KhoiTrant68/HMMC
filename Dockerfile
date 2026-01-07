FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel
RUN apt update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN pip install -r requirements.txt && \
	if [ -d /workspace/vmamba ]; then \
		cd /workspace/vmamba && pip install .; \
	elif [ -d vmamba ]; then \
		cd vmamba && pip install .; \
	fi
