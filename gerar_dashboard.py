import json
import pandas as pd
import matplotlib.pyplot as plt

def gerar_graficos():
    with open("resultados_experimento.json", "r") as f:
        dados = json.load(f)
        
    linhas = []
    for modelo, testes in dados.items():
        for t in testes:
            linhas.append({
                "Modelo": modelo,
                "Latência (s)": t["latencia_segundos"],
                "Acurácia": 1 if t["sucesso"] else 0
            })
            
    df = pd.DataFrame(linhas)
    
    # Agrupando médias
    df_resumo = df.groupby("Modelo").agg({
        "Latência (s)": "mean",
        "Acurácia": "mean"
    }).reset_index()
    df_resumo["Acurácia (%)"] = df_resumo["Acurácia"] * 100
    
    # Plotando Latência vs Acurácia
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color = '#1f77b4'
    ax1.set_xlabel('Modelos Testados')
    ax1.set_ylabel('Latência Média (segundos)', color=color)
    ax1.bar(df_resumo['Modelo'], df_resumo['Latência (s)'], color=color, alpha=0.6, width=0.4)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = '#2ca02c'
    ax2.set_ylabel('Acurácia Média (%)', color=color)
    ax2.plot(df_resumo['Modelo'], df_resumo['Acurácia (%)'], color=color, marker='o', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Trade-off Técnico: Latência vs Acurácia em Dispositivos Locais')
    fig.tight_layout()
    plt.savefig('dashboard_performance.png')
    print("[DASHBOARD] Gráfico gerado com sucesso e salvo como 'dashboard_performance.png'.")

if __name__ == "__main__":
    gerar_graficos()