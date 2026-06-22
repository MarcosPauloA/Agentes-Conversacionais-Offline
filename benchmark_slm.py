import os
import time
import json
import psutil
import ollama

# 1. Configuração do Dataset e Modelos
DATASET_FILE = "dataset_idosos.json"
MODELOS_PARA_TESTAR = ["qwen:0.5b", "gemma:2b", "phi3"]

# System prompt restritivo para garantir que a IA se comporte como um middleware de navegação
SYSTEM_PROMPT = (
    "Você é uma IA offline de acessibilidade integrada a um aplicativo mobile para idosos. "
    "Sua única tarefa é extrair a pergunta do usuário e responder EXCLUSIVAMENTE com um JSON válido. "
    "Não adicione nenhuma saudação, explicação ou texto fora do JSON. "
    "O formato do JSON deve ser estritamente: {\"target\": \"NOME_DA_TELA\"}. "
    "As telas válidas do sistema são apenas: "
    "['screen_medications', 'screen_contacts', 'screen_appointments', 'screen_alarm', 'screen_gallery']."
)

def carregar_dataset():
    with open(DATASET_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def medir_memoria():
    # Retorna o uso de memória RAM do processo atual em MB
    processo = psutil.Process(os.getpid())
    return processo.memory_info().rss / (1024 * 1024)

def executar_experimento():
    dataset = carregar_dataset()
    resultados_finais = {}

    print("=== INICIANDO EXPERIMENTO PRÁTICO DE IA ===")
    
    for modelo in MODELOS_PARA_TESTAR:
        print(f"\nAvaliando o modelo local: {modelo}...")
        resultados_finais[modelo] = []
        acertos = 0
        
        for item in dataset:
            id_caso = item["id"]
            prompt_usuario = item["prompt"]
            alvo_esperado = item["expected_target"]
            
            # Coleta métricas antes da inferência
            ram_antes = medir_memoria()
            tempo_inicio = time.time()
            
            # Chamada local ao SLM (Edge AI)
            try:
                resposta = ollama.generate(
                    model=modelo,
                    system=SYSTEM_PROMPT,
                    prompt=prompt_usuario,
                    options={"temperature": 0.0} # Temperatura 0 garante maior determinismo
                )
                
                texto_resposta = resposta['response'].strip()
                print("texto resposta: " + texto_resposta)
                tempo_fim = time.time()
                ram_depois = medir_memoria()
                
                # Parsing da resposta estruturada
                dados_json = json.loads(texto_resposta)
                alvo_predito = dados_json.get("target", "erro")
                
            except Exception as e:
                # Tratamento caso o modelo quebre o formato JSON esperado
                texto_resposta = f"Falha no Parsing: {str(e)}"
                alvo_predito = "erro"
                tempo_fim = time.time()
                ram_depois = medir_memoria()

            # Avaliação de acerto
            sucesso = (alvo_predito == alvo_esperado)
            if sucesso:
                acertos += 1
                
            latencia = tempo_fim - tempo_inicio
            consumo_ram = max(0, ram_depois - ram_antes) # Delta de impacto no processo
            
            resultados_finais[modelo].append({
                "id": id_caso,
                "latencia_segundos": round(latencia, 2),
                "consumo_ram_mb": round(consumo_ram, 2),
                "sucesso": sucesso
            })
            
        total_testes = len(dataset)
        acuracia = (acertos / total_testes) * 100
        print(f"Resultado final de {modelo} -> Acurácia: {acuracia}% | Pronto.")
        
    salvar_resultados(resultados_finais)

def salvar_resultados(resultados):
    with open("resultados_experimento.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=4, ensure_ascii=False)
    print("\n[SUCESSO] Métricas consolidadas em 'resultados_experimento.json'")

if __name__ == "__main__":
    executar_experimento()