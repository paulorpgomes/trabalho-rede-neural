import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class MLPConsumEnergia:

    def __init__(self, csv_path='consumo_energia_full.csv'):
        self.csv_path = csv_path
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_val = None
        self.X_val = None
        self.y_train_val = None
        self.y_val = None
        self.scaler = StandardScaler()
        self.model = None
        self.history = None

    def carregar_dados(self):
        print("=" * 80)
        print("1. CARREGAMENTO DOS DADOS")
        print("=" * 80)

        self.data = pd.read_csv(self.csv_path)
        print(f"\nDados carregados com sucesso!")
        print(f"Dimensões do dataset: {self.data.shape}")
        print(f"\nPrimeiras linhas:\n{self.data.head()}")
        print(f"\nInformações do dataset:")
        print(self.data.info())

    def explorar_dados(self):
        print("\n" + "=" * 80)
        print("2. EXPLORAÇÃO DE DADOS")
        print("=" * 80)

        print("\nEstatísticas Descritivas:")
        print(self.data.describe())

        fig = plt.figure(figsize=(16, 12))

        print("\nGerando histogramas das variáveis...")
        for i, col in enumerate(['x1', 'x2', 'x3', 'y'], 1):
            plt.subplot(4, 3, i)
            plt.hist(self.data[col], bins=30, edgecolor='black', alpha=0.7)
            plt.xlabel(col)
            plt.ylabel('Frequência')
            plt.title(f'Histograma de {col}')
            plt.grid(True, alpha=0.3)

        print("Gerando gráficos de dispersão...")
        variaveis = ['x1', 'x2', 'x3']
        nomes = ['Temperatura (°C)', 'Umidade (%)', 'Ocupação (pessoas/100m²)']

        for i, (var, nome) in enumerate(zip(variaveis, nomes), 5):
            plt.subplot(4, 3, i)
            plt.scatter(self.data[var], self.data['y'], alpha=0.5, s=20)
            plt.xlabel(nome)
            plt.ylabel('Consumo de Energia (kWh)')
            plt.title(f'Consumo vs {nome}')
            plt.grid(True, alpha=0.3)

        print("Gerando matriz de correlação...")
        plt.subplot(4, 3, 8)
        correlation_matrix = self.data.corr()
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                    square=True, cbar_kws={'shrink': 0.8})
        plt.title('Matriz de Correlação')

        print("Gerando boxplots...")
        for i, col in enumerate(['x1', 'x2', 'x3', 'y'], 9):
            plt.subplot(4, 3, i)
            plt.boxplot(self.data[col])
            plt.ylabel(col)
            plt.title(f'Boxplot de {col}')
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('exploracao_dados.png', dpi=300, bbox_inches='tight')
        print("\nGráfico salvo como 'exploracao_dados.png'")
        plt.show()

        print("\n" + "-" * 80)
        print("ANÁLISE DAS RELAÇÕES ENTRE VARIÁVEIS:")
        print("-" * 80)
        print("\nCorrelações com o consumo de energia (y):")
        correlacoes = self.data.corr()['y'].sort_values(ascending=False)
        for var, corr in correlacoes.items():
            if var != 'y':
                print(f"  - {var}: {corr:.4f}")

        print("\nInterpretação:")
        print("  • x1 (Temperatura): Correlação positiva indica que maior temperatura")
        print("    aumenta o consumo (uso de ar-condicionado)")
        print("  • x2 (Umidade): Influencia a eficiência dos sistemas de climatização")
        print("  • x3 (Ocupação): Correlação positiva forte - mais ocupantes aumentam")
        print("    significativamente o consumo energético")

    def preprocessar_dados(self):
        print("\n" + "=" * 80)
        print("3. PRÉ-PROCESSAMENTO DOS DADOS")
        print("=" * 80)

        self.X = self.data[['x1', 'x2', 'x3']].values
        self.y = self.data['y'].values

        print(f"\nDimensões das features (X): {self.X.shape}")
        print(f"Dimensões do target (y): {self.y.shape}")

        print("\nDividindo dados em treino (80%) e teste (20%)...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        print(f"  - Treino: {self.X_train.shape[0]} amostras")
        print(f"  - Teste: {self.X_test.shape[0]} amostras")

        print("\nSeparando conjunto de validação (10% do treino)...")
        self.X_train_val, self.X_val, self.y_train_val, self.y_val = train_test_split(
            self.X_train, self.y_train, test_size=0.1, random_state=42
        )

        print(f"  - Treino final: {self.X_train_val.shape[0]} amostras")
        print(f"  - Validação: {self.X_val.shape[0]} amostras")

        print("\nNormalizando os dados (padronização Z-score)...")
        self.X_train_val = self.scaler.fit_transform(self.X_train_val)
        self.X_val = self.scaler.transform(self.X_val)
        self.X_test = self.scaler.transform(self.X_test)

        print("  ✓ Dados normalizados com sucesso!")
        print(f"\nEstatísticas após normalização (conjunto de treino):")
        print(f"  - Média: {self.X_train_val.mean(axis=0)}")
        print(f"  - Desvio padrão: {self.X_train_val.std(axis=0)}")

    def construir_modelo(self, activation='relu'):
        print("\n" + "=" * 80)
        print("4. CONSTRUÇÃO DA REDE NEURAL")
        print("=" * 80)

        print("\nArquitetura do modelo:")
        print("  - Camada de entrada: 3 neurônios (x1, x2, x3)")
        print("  - Camada oculta: 10 neurônios")
        print(f"  - Função de ativação: {activation.upper()}")
        print("  - Camada de saída: 1 neurônio (y)")
        print("  - Otimizador: Adam")
        print("  - Função de perda: MSE (Mean Squared Error)")

        tf.random.set_seed(42)
        np.random.seed(42)

        self.model = keras.Sequential([
            layers.Input(shape=(3,)),
            layers.Dense(10, activation=activation, name='camada_oculta'),
            layers.Dense(1, name='camada_saida')
        ])

        self.model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )

        print("\nResumo do modelo:")
        self.model.summary()

        total_params = self.model.count_params()
        print(f"\nTotal de parâmetros treináveis: {total_params}")

    def treinar_modelo(self, epochs=300, batch_size=32, patience=20):
        print("\n" + "=" * 80)
        print("5. TREINAMENTO DO MODELO")
        print("=" * 80)

        print(f"\nParâmetros de treinamento:")
        print(f"  - Épocas máximas: {epochs}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Early stopping patience: {patience}")

        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )

        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        )

        print("\nIniciando treinamento...\n")

        self.history = self.model.fit(
            self.X_train_val, self.y_train_val,
            validation_data=(self.X_val, self.y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )

        print("\n✓ Treinamento concluído!")

        self.plotar_curvas_treinamento()

    def plotar_curvas_treinamento(self):
        print("\nGerando curvas de aprendizado...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(self.history.history['loss'], label='Treino', linewidth=2)
        ax1.plot(self.history.history['val_loss'], label='Validação', linewidth=2)
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Loss (MSE)')
        ax1.set_title('Curva de Perda durante o Treinamento')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(self.history.history['mae'], label='Treino', linewidth=2)
        ax2.plot(self.history.history['val_mae'], label='Validação', linewidth=2)
        ax2.set_xlabel('Época')
        ax2.set_ylabel('MAE')
        ax2.set_title('Erro Absoluto Médio durante o Treinamento')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('curvas_treinamento.png', dpi=300, bbox_inches='tight')
        print("Gráfico salvo como 'curvas_treinamento.png'")
        plt.show()

    def avaliar_modelo(self):
        print("\n" + "=" * 80)
        print("6. AVALIAÇÃO DO MODELO")
        print("=" * 80)

        y_pred_train = self.model.predict(self.X_train_val, verbose=0).flatten()
        y_pred_test = self.model.predict(self.X_test, verbose=0).flatten()

        print("\nMétricas no conjunto de TREINO:")
        mse_train = mean_squared_error(self.y_train_val, y_pred_train)
        mae_train = mean_absolute_error(self.y_train_val, y_pred_train)
        r2_train = r2_score(self.y_train_val, y_pred_train)
        rmse_train = np.sqrt(mse_train)

        print(f"  • MSE (Erro Quadrático Médio):  {mse_train:.4f}")
        print(f"  • RMSE (Raiz do MSE):            {rmse_train:.4f}")
        print(f"  • MAE (Erro Absoluto Médio):     {mae_train:.4f}")
        print(f"  • R² (Coeficiente de Determinação): {r2_train:.4f}")

        print("\nMétricas no conjunto de TESTE:")
        mse_test = mean_squared_error(self.y_test, y_pred_test)
        mae_test = mean_absolute_error(self.y_test, y_pred_test)
        r2_test = r2_score(self.y_test, y_pred_test)
        rmse_test = np.sqrt(mse_test)

        print(f"  • MSE (Erro Quadrático Médio):  {mse_test:.4f}")
        print(f"  • RMSE (Raiz do MSE):            {rmse_test:.4f}")
        print(f"  • MAE (Erro Absoluto Médio):     {mae_test:.4f}")
        print(f"  • R² (Coeficiente de Determinação): {r2_test:.4f}")

        print("\nInterpretação:")
        print(f"  O modelo explica {r2_test*100:.2f}% da variabilidade do consumo energético.")

        self.plotar_graficos_avaliacao(y_pred_test)

        return {
            'mse_train': mse_train,
            'mae_train': mae_train,
            'r2_train': r2_train,
            'mse_test': mse_test,
            'mae_test': mae_test,
            'r2_test': r2_test
        }

    def plotar_graficos_avaliacao(self, y_pred):
        print("\nGerando gráficos de avaliação...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.scatter(self.y_test, y_pred, alpha=0.6, s=40)

        min_val = min(self.y_test.min(), y_pred.min())
        max_val = max(self.y_test.max(), y_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Predição Perfeita')

        ax1.set_xlabel('Consumo Real (kWh)')
        ax1.set_ylabel('Consumo Predito (kWh)')
        ax1.set_title('Valores Reais vs Preditos')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        r2 = r2_score(self.y_test, y_pred)
        ax1.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax1.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        residuos = y_pred - self.y_test
        ax2.hist(residuos, bins=30, edgecolor='black', alpha=0.7)
        ax2.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero')
        ax2.set_xlabel('Resíduos (Predito - Real)')
        ax2.set_ylabel('Frequência')
        ax2.set_title('Distribuição dos Resíduos')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        ax2.text(0.05, 0.95, f'Média: {residuos.mean():.2f}\nDesvio: {residuos.std():.2f}',
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig('avaliacao_modelo.png', dpi=300, bbox_inches='tight')
        print("Gráfico salvo como 'avaliacao_modelo.png'")
        plt.show()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_pred, residuos, alpha=0.6, s=40)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax.set_xlabel('Valores Preditos (kWh)')
        ax.set_ylabel('Resíduos')
        ax.set_title('Resíduos vs Valores Preditos')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('residuos_vs_preditos.png', dpi=300, bbox_inches='tight')
        print("Gráfico salvo como 'residuos_vs_preditos.png'")
        plt.show()

    def salvar_modelo(self, filename='modelo_mlp_energia.h5'):
        self.model.save(filename)
        print(f"\n✓ Modelo salvo como '{filename}'")

    def executar_pipeline_completo(self):
        print("\n" + "=" * 80)
        print("SISTEMA DE ESTIMAÇÃO DE CONSUMO ENERGÉTICO")
        print("Perceptron Multicamadas (MLP)")
        print("=" * 80)

        self.carregar_dados()
        self.explorar_dados()
        self.preprocessar_dados()
        self.construir_modelo(activation='relu')
        self.treinar_modelo(epochs=300, batch_size=32, patience=20)
        metricas = self.avaliar_modelo()
        self.salvar_modelo()

        print("\n" + "=" * 80)
        print("PIPELINE CONCLUÍDO COM SUCESSO!")
        print("=" * 80)

        return metricas


def main():
    mlp = MLPConsumEnergia(csv_path='consumo_energia_full.csv')
    metricas = mlp.executar_pipeline_completo()

    print("\n" + "=" * 80)
    print("RESUMO FINAL")
    print("=" * 80)
    print("\nArquivos gerados:")
    print("  1. exploracao_dados.png - Análise exploratória completa")
    print("  2. curvas_treinamento.png - Curvas de perda durante treinamento")
    print("  3. avaliacao_modelo.png - Valores reais vs preditos e resíduos")
    print("  4. residuos_vs_preditos.png - Análise de resíduos")
    print("  5. modelo_mlp_energia.h5 - Modelo treinado salvo")

    print("\n✓ Execução finalizada com sucesso!")


if __name__ == "__main__":
    main()
