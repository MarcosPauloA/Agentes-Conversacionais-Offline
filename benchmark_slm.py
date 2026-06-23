import os
import time
import json
import psutil
import ollama
import re

# 1. Configuração do Dataset e Modelos
DATASET_FILE = "dataset_idosos.json"
MODELOS_PARA_TESTAR = ["qwen:0.5b", "gemma:2b", "phi3"]
LOG_FILE = "experimento_modelos.log"

# System Prompt bem descritivo
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
    processo = psutil.Process(os.getpid())
    return processo.memory_info().rss / (1024 * 1024)

def registrar_log(mensagem, f_log):
    print(mensagem)
    f_log.write(mensagem + "\n")

def extrair_json_defensivo(texto_bruto):
    """
    Localiza e extrai cirurgicamente o primeiro padrão JSON válido na resposta,
    ignorando qualquer conversa fiada antes ou depois.
    """
    texto_limpo = texto_bruto.strip()
    
    # 1. Procura o padrão exato de navegação: {"target": "..."}
    match_estrito = re.search(r'\{\s*"target"\s*:\s*"[^"]+"\s*\}', texto_limpo, re.IGNORECASE)
    if match_estrito:
        return match_estrito.group(0)
    
    # 2. Se não achar, tenta pegar qualquer conteúdo entre chaves { ... }
    match_generico = re.search(r'\{.*?\}', texto_limpo, re.DOTALL)
    if match_generico:
        return match_generico.group(0)
    
    return texto_limpo

def executar_experimento():
    dataset = carregar_dataset()
    resultados_finais = {}

    with open(LOG_FILE, "w", encoding="utf-8") as f_log:
        registrar_log("=== INICIANDO EXPERIMENTO PRÁTICO DE IA CORRIGIDO ===", f_log)
        
        for modelo in MODELOS_PARA_TESTAR:
            registrar_log("\n" + "="*60, f_log)
            registrar_log(f"AVALIANDO O MODELO LOCAL: {modelo}", f_log)
            registrar_log("="*60, f_log)
            resultados_finais[modelo] = []
            acertos = 0
            
            for item in dataset:
                id_caso = item["id"]
                prompt_usuario = item["prompt"]
                alvo_esperado = item["expected_target"]
                
                registrar_log(f"\n[Caso #{id_caso}] Entrada do Idoso: '{prompt_usuario}'", f_log)
                
                ram_antes = medir_memoria()
                tempo_inicio = time.time()
                texto_resposta = ""
                
                try:
                    # Chamada sem format="json" para evitar paralisia de amostragem
                    resposta = ollama.generate(
                        model=modelo,
                        system=SYSTEM_PROMPT,
                        prompt=prompt_usuario,
                        options={
                            "temperature": 0.0,
                            "num_predict": 150 # Token limit expandido para dar margem à IA
                        }
                    )
                    
                    texto_resposta = resposta['response'].strip()
                    registrar_log(f"  --> [RESPOSTA BRUTA DO {modelo}]:\n{texto_resposta}", f_log)
                    
                    # Extração via Regex
                    json_isolado = extrair_json_defensivo(texto_resposta)
                    
                    tempo_fim = time.time()
                    ram_depois = medir_memoria()
                    
                    dados_json = json.loads(json_isolado)
                    alvo_predito = dados_json.get("target", "erro")
                    
                except Exception as e:
                    tempo_fim = time.time()
                    ram_depois = medir_memoria()
                    
                    registrar_log(f"  [!] ERRO DE PROCESSAMENTO NO MODELO {modelo}!", f_log)
                    registrar_log(f"  --> [DETALHE DO ERRO]: {str(e)}", f_log)
                    alvo_predito = "erro"

                sucesso = (alvo_predito == alvo_esperado)
                if sucesso:
                    acertos += 1
                    registrar_log(f"  [✓] SUCESSO: Mapeou corretamente para '{alvo_predito}'", f_log)
                else:
                    registrar_log(f"  [X] FALHA: Predisse '{alvo_predito}' | Esperado: '{alvo_esperado}'", f_log)
                    
                latencia = tempo_fim - tempo_inicio
                consumo_ram = max(0, ram_depois - ram_antes)
                
                resultados_finais[modelo].append({
                    "id": id_caso,
                    "latencia_segundos": round(latencia, 2),
                    "consumo_ram_mb": round(consumo_ram, 2),
                    "sucesso": sucesso
                })
                
            total_testes = len(dataset)
            acuracia = (acertos / total_testes) * 100
            registrar_log(f"\n>>> Resumo de {modelo} -> Acurácia: {acuracia}% | Concluído.", f_log)
            
    salvar_resultados(resultados_finais)

def salvar_resultados(resultados):
    with open("resultados_experimento.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=4, ensure_ascii=False)
    print(f"\n[SUCESSO] Métricas consolidadas com sucesso.")

if __name__ == "__main__":
    executar_experimento()