# Video Subtitle Generator and Translator

A comprehensive Python tool for automatically generating, adding, and translating subtitles for Japanese-language videos, making them accessible to non-Japanese speakers.

## Project Overview

This tool addresses limitations in existing subtitle generation methods by combining OpenAI Whisper's high-accuracy speech recognition with OpenAI's GPT models for translation. This two-step approach delivers better quality subtitles than using Whisper's translation feature alone.

### Key Features

- **High-Accuracy Transcription**: Uses Whisper AI to generate accurate SRT files from Japanese audio
- **Quality Translation**: Employs OpenAI's API to translate subtitles with better context understanding
- **Custom Subtitle Integration**: Embeds subtitles into videos with adjustable formatting
- **Complete Automation**: Streamlines the entire process from speech recognition to subtitle embedding

## System Requirements

- CUDA-capable NVIDIA GPU (recommended for faster processing)
- Docker and NVIDIA Container Toolkit
- OpenAI API key

## Docker Setup

### 1. Install Docker Engine

```bash
# Remove old Docker installations if necessary
for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do 
  sudo apt remove $pkg
done

# Install Docker
sudo apt update
sudo apt install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Test installation
sudo docker run hello-world
```

### 2. Configure Docker for Non-Root Usage

```bash
sudo groupadd docker
sudo usermod -aG docker $USER
wsl --shutdown  # If using WSL
```

### 3. Setup Auto-start for Docker (Optional)

```bash
sudo visudo
# Add to the last line:
docker ALL=(ALL) NOPASSWD: /usr/sbin/service docker start

sudo nano ~/.bashrc
# Add to the last line:
if [[ $(service docker status | awk '{print $4}') = "not" ]]; then
  sudo service docker start > /dev/null
fi

source ~/.bashrc
```

### 4. NVIDIA Docker Setup (for GPU support)

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
&& curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## Creating and Setting Up the Docker Container

### 1. Pull CUDA Docker Image

```bash
docker pull nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
docker run -it --gpus all nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
```

### 2. Install Dependencies

```bash
apt update && apt full-upgrade -y
apt install git wget nano ffmpeg -y
```

### 3. Install Miniconda

```bash
cd ~
mkdir tmp
cd tmp
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Follow prompts: yes, enter, yes

# Clean up temp folder
cd ..
rm -rf tmp

# Exit and restart container
exit
docker container ls -a
docker start <container id>
docker exec -it <container id> /bin/bash
```

### 4. Create Conda Environment

```bash
mkdir subtitle_generator
cd subtitle_generator
nano subtitle-generator.yml
```

Add the following content to subtitle-generator.yml:

```yaml
name: subtitle-generator
channels:
  - conda-forge
  - pytorch
  - nvidia
  - defaults
dependencies:
  - python=3.9
  - pip
  - cudatoolkit=11.8
  - tiktoken
  - pillow
  - tqdm
  - srt
  - moviepy
  - python-dotenv
  - pip:
    - openai-whisper
    - openai
    - torch
    - torchvision
    - torchaudio
```

Create the environment:

```bash
conda env create -f subtitle-generator.yml
conda activate subtitle-generator
```

## Setting Up the Script

1. Copy the Python script to your container:

```bash
nano subtitle_generator.py
# Paste the script content and save
```

2. Create `.env` file with your OpenAI API key:

```bash
nano .env
# Add the following content:
OPENAI_API_KEY=your_openai_api_key
```

## Transferring Files To/From Docker Container

```bash
# Copy from Windows to container
docker cp "/mnt/c/Windows_path/video.mp4" container_name:root/subtitle_generator/

# Copy from container to Windows
docker cp container_name:root/subtitle_generator/output.mp4 "/mnt/c/Windows_path/"
```

## Usage Examples

### 1. Generate Japanese Subtitles (SRT)

```bash
python subtitle_generator.py generate --input video.mp4 --output_srt japanese.srt --model large-v3
```

### 2. Add Japanese Subtitles to Video

```bash
python subtitle_generator.py add --input video.mp4 --output_video japanese_subtitled.mp4 --input_srt japanese.srt
```

### 3. Generate English Subtitles Using Whisper Translation

```bash
python subtitle_generator.py generate --input video.mp4 --output_srt english_whisper.srt --model large-v3 --translate
```

### 4. Translate SRT File Using OpenAI API (Recommended Method)

```bash
python subtitle_generator.py translate --input_srt japanese.srt --output_srt english_gpt.srt --source_lang Japanese --target_lang English --temperature 0.3
```

### 5. Add English Subtitles to Video

```bash
python subtitle_generator.py add --input video.mp4 --output_video english_subtitled.mp4 --input_srt english_gpt.srt
```

## Available Commands

```
usage: subtitle_generator.py [-h] {generate,add,translate} ...

Subtitle Generator and Adder

positional arguments:
  {generate,add,translate}

optional arguments:
  -h, --help            show this help message and exit
```

### Generate Command Options

```
--input INPUT         Input video file path
--output_srt OUTPUT_SRT
                      Output SRT file path
--model MODEL         Whisper model name (default: large-v3)
--language LANGUAGE   Language of the audio (default: Japanese)
--translate           Translate the audio to English (using Whisper)
```

### Add Command Options

```
--input INPUT         Input video file path
--output_video OUTPUT_VIDEO
                      Output video file path
--input_srt INPUT_SRT
                      Input SRT file path
```

### Translate Command Options

```
--input_srt INPUT_SRT
                      Input SRT file path
--output_srt OUTPUT_SRT
                      Output SRT file path
--source_lang SOURCE_LANG
                      Source language of the SRT file (default: Japanese)
--target_lang TARGET_LANG
                      Target language for translation (default: English)
--temperature TEMPERATURE
                      Temperature setting for OpenAI (default: 0.3)
```

## Technical Details

This tool works through three main components:

1. **SRTGenerator**: Uses Whisper AI to extract and transcribe audio from videos, creating SRT files
2. **SubtitleAdder**: Adds subtitle text to videos, intelligently handling formatting and placement
3. **SRTTranslator**: Leverages OpenAI's API to translate subtitle content while preserving timing and formatting

## Why Use This Tool?

- **Better Quality Translations**: The two-step approach (Whisper for transcription + GPT for translation) avoids the omissions and inconsistencies sometimes found in direct Whisper translations
- **Customization**: Fine-tune subtitle appearance and translation quality
- **Efficiency**: Automate a typically time-consuming process
- **GPU Acceleration**: Leverage CUDA for faster processing

## License

[MIT License](LICENSE)

## Acknowledgements

- [OpenAI Whisper](https://github.com/openai/whisper) for speech recognition
- [OpenAI API](https://openai.com/api/) for translation
- [MoviePy](https://zulko.github.io/moviepy/) for video processing
