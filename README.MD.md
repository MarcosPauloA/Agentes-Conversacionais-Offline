# SLM-Edge-Navigation-Assist

Este repositório contém a componente prática desenvolvida para a disciplina PCC557 (UFLA).
O objetivo é avaliar o desempenho de Small Language Models (SLMs) locais atuando como 
agentes conversacionais offline para a navegação assistiva de idosos.

## Pré-requisitos
1. Instale o Ollama em sua máquina (https://ollama.com)
2. Baixe os modelos locais via terminal:
   ollama run qwen:0.5b
   ollama run gemma:2b
   ollama run phi3

## Como Executar os Experimentos
1. Instale as dependências do Python:
   pip install ollama psutil pandas matplotlib

2. Execute o pipeline de IA para coletar as métricas brutas:
   python benchmark_slm.py

3. Gere o dashboard analítico automatizado para o artigo:
   python gerar_dashboard.py