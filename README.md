# Sistema de Estimação de Consumo Energético - MLP

## Descrição do Projeto

Este projeto implementa um **Perceptron Multicamadas (MLP)** para estimar o consumo de energia elétrica em edifícios comerciais. O modelo utiliza três variáveis de entrada:

- **x1**: Temperatura ambiente (°C)
- **x2**: Umidade relativa (%)
- **x3**: Nível de ocupação do prédio (pessoas por 100 m²)

E prevê o **consumo de energia (y)** em kWh.

## Arquitetura da Rede Neural

```
Entrada (3 neurônios) → Camada Oculta (10 neurônios) → Saída (1 neurônio)
```

- **Função de ativação**: ReLU (camada oculta)
- **Otimizador**: Adam
- **Função de perda**: MSE (Mean Squared Error)
- **Early Stopping**: Com paciência de 20 épocas

## Estrutura dos Arquivos

```
trabalho 2 iac/
│
├── consumo_energia_full.csv       # Dataset completo
├── consumo_energia_train.csv      # Dataset de treino
├── consumo_energia_test.csv       # Dataset de teste
│
├── mlp_consumo_energia.py         # Script principal
├── requirements.txt               # Dependências do projeto
├── README.md                      # Este arquivo
│
└── Saídas geradas:
    ├── exploracao_dados.png
    ├── curvas_treinamento.png
    ├── avaliacao_modelo.png
    ├── residuos_vs_preditos.png
    └── modelo_mlp_energia.h5
```

## Requisitos

- Python 3.8 ou superior
- Bibliotecas listadas em `requirements.txt`

## Instruções de Instalação

### 1. Criar ambiente virtual (opcional, mas recomendado)

```bash
python -m venv venv
```

#### Ativar o ambiente virtual:

```bash
venv\Scripts\activate
```

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

## Instruções de Execução

### Executar o pipeline completo

```bash
python mlp_consumo_energia.py
```

O script executará automaticamente:

1. **Carregamento dos dados** do arquivo CSV
2. **Exploração de dados** com estatísticas descritivas e gráficos
3. **Pré-processamento** (normalização e divisão treino/teste/validação)
4. **Construção do modelo MLP**
5. **Treinamento** com early stopping (até 300 épocas)
6. **Avaliação** com métricas MSE, MAE e R²
7. **Geração de gráficos** de análise
8. **Salvamento do modelo** treinado

## Pipeline de Execução

### 1. Exploração de Dados
- Estatísticas descritivas de todas as variáveis
- Histogramas de distribuição
- Gráficos de dispersão (relação entrada vs saída)
- Matriz de correlação
- Boxplots para análise de outliers

### 2. Pré-processamento
- Normalização Z-score (StandardScaler)
- Divisão: 80% treino, 20% teste
- Validação: 10% dos dados de treino

### 3. Construção da Rede Neural
- Arquitetura: 3 → 10 → 1
- Total de parâmetros treináveis: 51

### 4. Treinamento
- Épocas máximas: 300
- Batch size: 32
- Early stopping com paciência: 20 épocas
- Redução de learning rate adaptativa

### 5. Avaliação
Métricas calculadas:
- **MSE**: Erro Quadrático Médio
- **RMSE**: Raiz do Erro Quadrático Médio
- **MAE**: Erro Absoluto Médio
- **R²**: Coeficiente de Determinação

Gráficos gerados:
- Valores reais vs preditos
- Distribuição dos resíduos
- Resíduos vs valores preditos

## Arquivos de Saída

Após a execução, os seguintes arquivos serão gerados:

1. **exploracao_dados.png**: Análise exploratória completa com 12 gráficos
2. **curvas_treinamento.png**: Curvas de Loss e MAE (treino e validação)
3. **avaliacao_modelo.png**: Valores reais vs preditos + distribuição de resíduos
4. **residuos_vs_preditos.png**: Análise de homocedasticidade
5. **modelo_mlp_energia.h5**: Modelo treinado salvo em formato Keras

## Resultados Esperados

O modelo deve alcançar:
- **R² > 0.90**: Excelente capacidade de explicação da variabilidade
- **MAE baixo**: Erro médio em kWh reduzido
- **Resíduos centrados em zero**: Indicando predições não enviesadas

## Interpretação dos Resultados

### Correlações esperadas:
- **Temperatura (x1)**: Correlação positiva (maior temperatura → maior uso de ar-condicionado)
- **Umidade (x2)**: Influencia a eficiência dos sistemas de climatização
- **Ocupação (x3)**: Correlação forte positiva (mais pessoas → maior consumo)

## Tecnologias Utilizadas

- **Python**: Linguagem de programação
- **TensorFlow/Keras**: Framework de Deep Learning
- **Scikit-learn**: Pré-processamento e métricas
- **Pandas**: Manipulação de dados
- **Matplotlib/Seaborn**: Visualização de dados
- **NumPy**: Computação numérica
