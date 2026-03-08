# A100(Ampere) 및 최신 Triton 커널에 최적화된 CUDA 12.1 공식 PyTorch 이미지
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel

# 시스템 환경 변수 설정
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="8.0" 

# 필수 패키지 설치 (wget 추가)
RUN apt-get update && apt-get install -y \
    git curl wget ninja-build build-essential openssh-server \
    && rm -rf /var/lib/apt/lists/*

# SSH 환경 설정 (Root 로그인 허용 및 비밀번호 설정)
RUN mkdir /var/run/sshd && \
    echo 'root:root' | chpasswd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' /etc/pam.d/sshd

# 1. 기본 파이썬 패키지 설치
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    datasets transformers wandb tqdm triton packaging wheel

# =========================================================================
# ⚡ 2. WHL 파일 직접 다운로드 및 쾌속 설치 (컴파일 0초)
# =========================================================================
WORKDIR /tmp/wheels
# 공식 GitHub 릴리즈에서 Mamba-2 전용 바이너리(whl)를 직접 당겨옵니다.
RUN wget https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.2.0.post2/causal_conv1d-1.2.0.post2+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl && \
    wget https://github.com/state-spaces/mamba/releases/download/v2.1.0/mamba_ssm-2.1.0+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl && \
    # 다운받은 파일 즉시 설치
    pip install --no-cache-dir ./causal_conv1d-*.whl && \
    pip install --no-cache-dir ./mamba_ssm-*.whl && \
    # 도커 이미지 용량 다이어트를 위해 다운받은 원본 파일 삭제
    rm -rf /tmp/wheels

# =========================================================================
# ⚡ 3. 소스 코드 클론 및 작업 준비
# =========================================================================
WORKDIR /workspace
RUN git clone https://github.com/N01N9/hdcmamba.git

# 작업 디렉토리를 클론한 폴더로 이동
WORKDIR /workspace/hdcmamba

# 외부 SSH 접속 포트 개방
EXPOSE 22

# 컨테이너 실행 시 SSH 데몬 유지 (백그라운드 생존)
CMD ["/usr/sbin/sshd", "-D"]