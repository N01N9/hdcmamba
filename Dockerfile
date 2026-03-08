# A100(Ampere) 및 최신 Triton 커널에 최적화된 CUDA 12.1 공식 PyTorch 이미지 사용
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel

# 시스템 환경 변수 설정
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="8.0" 

# ⚡ 필수 패키지에 openssh-server 추가 설치
RUN apt-get update && apt-get install -y \
    git \
    curl \
    ninja-build \
    build-essential \
    openssh-server \
    && rm -rf /var/lib/apt/lists/*

# ⚡ SSH 환경 설정 (Root 로그인 허용 및 비밀번호 설정)
RUN mkdir /var/run/sshd && \
    echo 'root:root' | chpasswd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    # SSH 접속 시 환경변수 초기화 방지
    sed -i 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' /etc/pam.d/sshd

# GitHub 리포지토리 클론
WORKDIR /workspace
RUN git clone https://github.com/N01N9/hdcmamba.git

# 작업 디렉토리 이동
WORKDIR /workspace/hdcmamba

# 파이썬 의존성 패키지 설치
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    datasets \
    transformers \
    wandb \
    tqdm \
    triton

# Mamba-2 아키텍처 비교 학습을 위한 공식 패키지 설치
RUN pip install --no-cache-dir packaging wheel && \
    pip install --no-cache-dir causal-conv1d>=1.2.0 && \
    pip install --no-cache-dir mamba-ssm

# ⚡ 외부에서 접속할 SSH 포트(22) 개방
EXPOSE 22

# ⚡ 컨테이너 실행 시 SSH 데몬을 백그라운드가 아닌 포그라운드로 실행하여 컨테이너 유지
CMD ["/usr/sbin/sshd", "-D"]