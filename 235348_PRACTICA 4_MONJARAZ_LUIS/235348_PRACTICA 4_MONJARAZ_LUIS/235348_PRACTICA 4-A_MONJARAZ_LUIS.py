import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import cv2
import warnings
warnings.filterwarnings('ignore')

def rbf_image_reconstruction(image_path, damage_percentages=[0.2, 0.6, 0.85]):
    """
    Reconstruye una imagen usando RBF con diferentes porcentajes de daño
    """
    # Cargar imagen y convertir a escala de grises
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: No se pudo cargar la imagen {image_path}")
        return
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = img_gray.astype(np.float64) / 255.0
    
    m, n = img_gray.shape
    print(f"Tamaño de la imagen: {m}x{n} píxeles")
    
    # Si hay múltiples porcentajes, crear subplots
    if len(damage_percentages) > 1:
        fig, axes = plt.subplots(len(damage_percentages), 3, figsize=(15, 5*len(damage_percentages)))
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes = [axes]  # Convertir a lista para consistencia
    
    for idx, damage_pct in enumerate(damage_percentages):
        print(f"Procesando {damage_pct*100}% de daño...")
        
        # Simulación de daño en la imagen
        np.random.seed(42)  # Para resultados reproducibles
        mask = np.random.random((m, n)) > damage_pct
        img_damaged = img_gray.copy()
        img_damaged[~mask] = np.nan
        
        # Extraer datos de entrenamiento (píxeles no dañados)
        X, Y = np.meshgrid(range(n), range(m))
        coordinates = np.column_stack((X.ravel(), Y.ravel()))
        pixel_values = img_gray.ravel()
        
        # Filtrar solo píxeles no dañados
        valid_mask = ~np.isnan(img_damaged.ravel())
        X_train = coordinates[valid_mask]
        Y_train = pixel_values[valid_mask]
        
        print(f"Píxeles válidos para entrenamiento: {len(Y_train)}")
        
        # Reducir número de muestras para evitar problemas de memoria
        num_samples = min(5000, len(Y_train))  # Reducido a 5000 para mayor velocidad
        if num_samples < len(Y_train):
            indices = np.random.choice(len(Y_train), num_samples, replace=False)
            X_train = X_train[indices]
            Y_train = Y_train[indices]
            print(f"Muestras utilizadas para entrenamiento: {num_samples}")
        
        # Escalar datos
        scaler_X = StandardScaler()
        scaler_Y = StandardScaler()
        
        X_train_scaled = scaler_X.fit_transform(X_train)
        Y_train_scaled = scaler_Y.fit_transform(Y_train.reshape(-1, 1)).ravel()
        
        # Entrenar modelo RBF usando Gaussian Process con kernel RBF
        print("Entrenando modelo RBF...")
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, alpha=0.1)
        
        # Entrenar con subconjunto más pequeño si es necesario
        if len(X_train_scaled) > 2000:
            train_subset = np.random.choice(len(X_train_scaled), 2000, replace=False)
            gp.fit(X_train_scaled[train_subset], Y_train_scaled[train_subset])
        else:
            gp.fit(X_train_scaled, Y_train_scaled)
        
        # Predicción de píxeles dañados
        damaged_coords = coordinates[~valid_mask]
        print(f"Píxeles a reconstruir: {len(damaged_coords)}")
        
        if len(damaged_coords) > 0:
            # Procesar en lotes para evitar memoria insuficiente
            batch_size = 1000
            Y_pred = np.zeros(len(damaged_coords))
            
            for i in range(0, len(damaged_coords), batch_size):
                end_idx = min(i + batch_size, len(damaged_coords))
                batch_coords = damaged_coords[i:end_idx]
                
                X_test_scaled = scaler_X.transform(batch_coords)
                Y_pred_scaled = gp.predict(X_test_scaled)
                Y_pred[i:end_idx] = scaler_Y.inverse_transform(Y_pred_scaled.reshape(-1, 1)).ravel()
            
            # Reconstruir imagen
            img_restored = img_damaged.copy()
            img_restored_flat = img_restored.ravel()
            img_restored_flat[~valid_mask] = Y_pred
            img_restored = img_restored_flat.reshape((m, n))
            
            # Asegurar que los valores estén en [0, 1]
            img_restored = np.clip(img_restored, 0, 1)
        else:
            img_restored = img_damaged.copy()
        
        # Visualización
        if len(damage_percentages) > 1:
            ax_orig, ax_damaged, ax_restored = axes[idx, 0], axes[idx, 1], axes[idx, 2]
        else:
            ax_orig, ax_damaged, ax_restored = axes[0], axes[1], axes[2]
        
        ax_orig.imshow(img_gray, cmap='gray')
        ax_orig.set_title(f'Imagen Original')
        ax_orig.axis('off')
        
        # Para mostrar imagen dañada, reemplazar NaN con 0
        img_damaged_display = img_damaged.copy()
        img_damaged_display[np.isnan(img_damaged_display)] = 0
        ax_damaged.imshow(img_damaged_display, cmap='gray')
        ax_damaged.set_title(f'Dañada ({damage_pct*100:.0f}% píxeles eliminados)')
        ax_damaged.axis('off')
        
        ax_restored.imshow(img_restored, cmap='gray')
        ax_restored.set_title(f'Restaurada con RBF')
        ax_restored.axis('off')
        
        print(f"Completado {damage_pct*100}% de daño\n")
    
    plt.tight_layout()
    plt.show()

# Ejecutar la función
rbf_image_reconstruction('Orb.jpg')

