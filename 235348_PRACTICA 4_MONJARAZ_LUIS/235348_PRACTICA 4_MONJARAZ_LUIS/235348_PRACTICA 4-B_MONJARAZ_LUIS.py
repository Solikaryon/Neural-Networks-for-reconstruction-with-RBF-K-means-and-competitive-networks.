import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

def load_medication_data(file_path):
    """Carga y prepara los datos de medicamentos"""
    data = pd.read_csv(file_path)
    print(f"Datos cargados: {data.shape}")
    print(data.head())
    
    # Extraer características (Peso e Indice PH)
    X = data[['Peso', 'Indice PH']].values
    return X, data

def kmeans_clustering(X, k=3):
    """Implementa algoritmo K-Means"""
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_
    return labels, centers, kmeans

def manual_competitive_clustering(X, n_neurons=3, max_epochs=100):
    """Implementación manual de red competitiva"""
    # Inicializar pesos aleatorios (centros de clusters)
    np.random.seed(42)
    centers = np.random.rand(n_neurons, X.shape[1]) * (X.max(axis=0) - X.min(axis=0)) + X.min(axis=0)
    
    learning_rate = 0.01
    labels_history = []
    centers_history = [centers.copy()]
    
    for epoch in range(max_epochs):
        labels = []
        for i in range(len(X)):
            # Calcular distancias a todos los centros
            distances = np.linalg.norm(X[i] - centers, axis=1)
            # Encontrar la neurona ganadora (más cercana)
            winner = np.argmin(distances)
            labels.append(winner)
            
            # Actualizar el centro ganador
            centers[winner] += learning_rate * (X[i] - centers[winner])
        
        labels_history.append(np.array(labels))
        centers_history.append(centers.copy())
        
        # Verificar convergencia
        if epoch > 0 and np.all(labels_history[-1] == labels_history[-2]):
            break
    
    return np.array(labels), centers, len(centers_history)

def compare_clustering_algorithms(file_path):
    """Compara K-Means vs Red Neuronal Competitiva"""
    
    # Cargar datos
    X, data = load_medication_data(file_path)
    
    # Escalar datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Probar diferentes valores de K/neuronas
    k_values = [2, 3, 4, 5, 7]  # Incluyendo +2 y -2 para comparación
    results = {}
    
    fig, axes = plt.subplots(2, len(k_values), figsize=(20, 8))
    
    for i, k in enumerate(k_values):
        print(f"\n--- Analizando con K/Neuronas = {k} ---")
        
        # K-Means
        kmeans_labels, kmeans_centers, kmeans_model = kmeans_clustering(X_scaled, k)
        kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
        
        # Red Competitiva
        comp_labels, comp_centers, comp_iterations = manual_competitive_clustering(X_scaled, k)
        comp_silhouette = silhouette_score(X_scaled, comp_labels)
        
        results[k] = {
            'kmeans': {
                'labels': kmeans_labels,
                'centers': scaler.inverse_transform(kmeans_centers),
                'silhouette': kmeans_silhouette,
                'iterations': kmeans_model.n_iter_
            },
            'competitive': {
                'labels': comp_labels,
                'centers': scaler.inverse_transform(comp_centers),
                'silhouette': comp_silhouette,
                'iterations': comp_iterations
            }
        }
        
        # Visualización K-Means
        axes[0, i].scatter(data['Peso'], data['Indice PH'], c=kmeans_labels, cmap='viridis', alpha=0.6, s=10)
        axes[0, i].scatter(results[k]['kmeans']['centers'][:, 0], 
                          results[k]['kmeans']['centers'][:, 1], 
                          marker='X', s=200, c='red', label='Centros', edgecolors='black')
        axes[0, i].set_title(f'K-Means (K={k})\nSilhouette: {kmeans_silhouette:.3f}')
        axes[0, i].set_xlabel('Peso')
        axes[0, i].set_ylabel('Indice PH')
        axes[0, i].legend()
        
        # Visualización Red Competitiva
        axes[1, i].scatter(data['Peso'], data['Indice PH'], c=comp_labels, cmap='viridis', alpha=0.6, s=10)
        axes[1, i].scatter(results[k]['competitive']['centers'][:, 0], 
                          results[k]['competitive']['centers'][:, 1], 
                          marker='X', s=200, c='red', label='Centros', edgecolors='black')
        axes[1, i].set_title(f'Red Competitiva (Neuronas={k})\nSilhouette: {comp_silhouette:.3f}')
        axes[1, i].set_xlabel('Peso')
        axes[1, i].set_ylabel('Indice PH')
        axes[1, i].legend()
    
    plt.tight_layout()
    plt.show()
    
    # ANÁLISIS COMPARATIVO MEJORADO
    print("\n" + "="*100)
    print("ANÁLISIS COMPARATIVO: K-MEANS vs RED NEURONAL COMPETITIVA")
    print("="*100)
    
    # 1. ¿Cuántos K o cuantas neuronas fueron necesarias para poder agrupar los datos?
    print("\n1. ¿CUÁNTOS K O CUÁNTAS NEURONAS FUERON NECESARIAS PARA PODER AGRUPAR LOS DATOS?")
    best_k_kmeans = max(results.items(), key=lambda x: x[1]['kmeans']['silhouette'])[0]
    best_k_comp = max(results.items(), key=lambda x: x[1]['competitive']['silhouette'])[0]
    
    print(f"   • K-Means: El mejor resultado se obtuvo con K = {best_k_kmeans} clusters")
    print(f"     (Silhouette Score: {results[best_k_kmeans]['kmeans']['silhouette']:.3f})")
    print(f"   • Red Competitiva: El mejor resultado se obtuvo con {best_k_comp} neuronas")
    print(f"     (Silhouette Score: {results[best_k_comp]['competitive']['silhouette']:.3f})")
    print(f"   • Conclusión: Ambos algoritmos funcionan mejor con {best_k_kmeans} clusters/neuronas")
    
    # 2. ¿Cuál de los dos algoritmos convergen más rápido a una misma solución?
    print("\n2. ¿CUÁL DE LOS DOS ALGORITMOS CONVERGE MÁS RÁPIDO A UNA MISMA SOLUCIÓN?")
    print("   Comparación de iteraciones para convergencia:")
    for k in [3, 5]:
        kmeans_iters = results[k]['kmeans']['iterations']
        comp_iters = results[k]['competitive']['iterations']
        diferencia = kmeans_iters - comp_iters
        mas_rapido = "K-Means" if kmeans_iters < comp_iters else "Red Competitiva"
        
        print(f"   • K={k}:")
        print(f"     - K-Means: {kmeans_iters} iteraciones")
        print(f"     - Red Competitiva: {comp_iters} iteraciones")
        print(f"     - Más rápido: {mas_rapido} (diferencia: {abs(diferencia)} iteraciones)")
    
    # 3. Si aumenta 2 neuronas o 'k' en su algoritmo, ¿qué variación hay en el agrupamiento?
    print("\n3. SI AUMENTA 2 NEURONAS O 'K' EN SU ALGORITMO, ¿QUÉ VARIACIÓN HAY EN EL AGRUPAMIENTO?")
    k_ref = 3
    k_plus_2 = 5
    
    sil_k3_kmeans = results[k_ref]['kmeans']['silhouette']
    sil_k5_kmeans = results[k_plus_2]['kmeans']['silhouette']
    sil_k3_comp = results[k_ref]['competitive']['silhouette']
    sil_k5_comp = results[k_plus_2]['competitive']['silhouette']
    
    print(f"   • Comparación K={k_ref} → K={k_plus_2}:")
    print(f"     K-Means:")
    print(f"       - Silhouette: {sil_k3_kmeans:.3f} → {sil_k5_kmeans:.3f}")
    print(f"       - Variación: {sil_k5_kmeans - sil_k3_kmeans:+.3f} ({(sil_k5_kmeans/sil_k3_kmeans - 1)*100:+.1f}%)")
    print(f"       - Interpretación: {'MEJORA' if sil_k5_kmeans > sil_k3_kmeans else 'EMPEORA'} la calidad del agrupamiento")
    
    print(f"     Red Competitiva:")
    print(f"       - Silhouette: {sil_k3_comp:.3f} → {sil_k5_comp:.3f}")
    print(f"       - Variación: {sil_k5_comp - sil_k3_comp:+.3f} ({(sil_k5_comp/sil_k3_comp - 1)*100:+.1f}%)")
    print(f"       - Interpretación: {'MEJORA' if sil_k5_comp > sil_k3_comp else 'EMPEORA'} la calidad del agrupamiento")
    
    # 4. Si disminuye 2 neuronas o 'k' en su algoritmo, ¿qué variación hay en el agrupamiento?
    print("\n4. SI DISMINUYE 2 NEURONAS O 'K' EN SU ALGORITMO, ¿QUÉ VARIACIÓN HAY EN EL AGRUPAMIENTO?")
    k_minus_2 = 1
    
    if k_minus_2 in results:
        sil_k1_kmeans = results[k_minus_2]['kmeans']['silhouette']
        sil_k1_comp = results[k_minus_2]['competitive']['silhouette']
        
        print(f"   • Comparación K={k_ref} → K={k_minus_2}:")
        print(f"     K-Means:")
        print(f"       - Silhouette: {sil_k3_kmeans:.3f} → {sil_k1_kmeans:.3f}")
        print(f"       - Variación: {sil_k1_kmeans - sil_k3_kmeans:+.3f} ({(sil_k1_kmeans/sil_k3_kmeans - 1)*100:+.1f}%)")
        print(f"       - Interpretación: {'MEJORA' if sil_k1_kmeans > sil_k3_kmeans else 'EMPEORA'} la calidad del agrupamiento")
        
        print(f"     Red Competitiva:")
        print(f"       - Silhouette: {sil_k3_comp:.3f} → {sil_k1_comp:.3f}")
        print(f"       - Variación: {sil_k1_comp - sil_k3_comp:+.3f} ({(sil_k1_comp/sil_k3_comp - 1)*100:+.1f}%)")
        print(f"       - Interpretación: {'MEJORA' if sil_k1_comp > sil_k3_comp else 'EMPEORA'} la calidad del agrupamiento")
    else:
        print(f"   • No se tiene datos para K=1 en esta ejecución")
    
    # 5. Compare cuál de las dos redes ha proporcionado un mejor resultado
    print("\n5. COMPARE CUÁL DE LAS DOS REDES HA PROPORCIONADO UN MEJOR RESULTADO")
    
    # Calcular el mejor silhouette promedio
    avg_sil_kmeans = np.mean([results[k]['kmeans']['silhouette'] for k in k_values])
    avg_sil_comp = np.mean([results[k]['competitive']['silhouette'] for k in k_values])
    
    best_overall_kmeans = max([results[k]['kmeans']['silhouette'] for k in k_values])
    best_overall_comp = max([results[k]['competitive']['silhouette'] for k in k_values])
    
    print(f"   • Métricas de comparación:")
    print(f"     - Mejor Silhouette Score:")
    print(f"       * K-Means: {best_overall_kmeans:.3f} (K={best_k_kmeans})")
    print(f"       * Red Competitiva: {best_overall_comp:.3f} (Neuronas={best_k_comp})")
    print(f"     - Silhouette Score Promedio:")
    print(f"       * K-Means: {avg_sil_kmeans:.3f}")
    print(f"       * Red Competitiva: {avg_sil_comp:.3f}")
    
    if best_overall_kmeans > best_overall_comp:
        print(f"   • CONCLUSIÓN: K-MEANS proporciona MEJORES resultados")
        mejor_algoritmo = "K-Means"
        ventaja = best_overall_kmeans - best_overall_comp
    else:
        print(f"   • CONCLUSIÓN: RED COMPETITIVA proporciona MEJORES resultados")
        mejor_algoritmo = "Red Competitiva"
        ventaja = best_overall_comp - best_overall_kmeans
    
    print(f"     (Ventaja: {ventaja:.3f} en Silhouette Score)")
    
    # 6. Posición final de los centros obtenida por cada algoritmo (para K=3)
    print("\n6. POSICIÓN FINAL DE LOS CENTROS OBTENIDA POR CADA ALGORITMO (K=3)")
    
    print(f"   • K-MEANS - Centros finales (K=3):")
    for i, center in enumerate(results[3]['kmeans']['centers']):
        print(f"     Cluster {i+1}: Peso = {center[0]:.3f}, Índice PH = {center[1]:.3f}")
    
    print(f"   • RED COMPETITIVA - Centros finales (3 Neuronas):")
    for i, center in enumerate(results[3]['competitive']['centers']):
        print(f"     Neurona {i+1}: Peso = {center[0]:.3f}, Índice PH = {center[1]:.3f}")
    
    # Análisis adicional de estabilidad
    print("\n7. ANÁLISIS ADICIONAL - ESTABILIDAD DE LOS ALGORITMOS")
    
    # Calcular desviación estándar de silhouette scores
    std_kmeans = np.std([results[k]['kmeans']['silhouette'] for k in k_values])
    std_comp = np.std([results[k]['competitive']['silhouette'] for k in k_values])
    
    print(f"   • Estabilidad (desviación estándar de Silhouette Scores):")
    print(f"     - K-Means: {std_kmeans:.4f}")
    print(f"     - Red Competitiva: {std_comp:.4f}")
    print(f"     - Más estable: {'K-Means' if std_kmeans < std_comp else 'Red Competitiva'}")
    
    return results, mejor_algoritmo

# Ejecutar análisis
if __name__ == "__main__":
    print("INICIO DEL ANÁLISIS DE AGRUPAMIENTO")
    print("="*50)
    results, mejor_algoritmo = compare_clustering_algorithms('datos_medicamentos2025A.csv')
    print("\n" + "="*50)
    print(f"ANÁLISIS COMPLETADO - ALGORITMO RECOMENDADO: {mejor_algoritmo}")
    print("="*50)