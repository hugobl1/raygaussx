import torch
import numpy as np

def quantify_pos_min_max_np(pos, nbits=10):
    """
    Quantifie les points sur l'intervalle [0, 2^nbits - 1] en utilisant le min et max de chaque dimension.

    Parameters:
        pos   : np.ndarray de forme (N, 3) en float
        nbits : int, nombre de bits pour la quantification (défaut 10)
        
    Returns:
        pos_quant : np.ndarray de forme (N, 3) en np.uint32, points quantifiés
        min_vals  : np.ndarray de forme (3,), minimum de chaque dimension
        max_vals  : np.ndarray de forme (3,), maximum de chaque dimension
    """
    # Calculer le min et le max par dimension
    min_vals = pos.min(axis=0)
    max_vals = pos.max(axis=0)
    
    # Pour éviter une division par zéro en cas d'étendue nulle, on ajoute une petite constante.
    scale = (2**nbits - 1) / (max_vals - min_vals + 1e-9)
    
    # Normaliser et quantifier
    pos_quant = ((pos - min_vals) * scale).astype(np.uint32)
    return pos_quant, min_vals, max_vals

def morton_code_np(pos):
    """
    Calcule le code de Morton (Z-order) pour chaque point de pos.

    Parameters:
      pos : np.ndarray de forme (N, 3)
            Contient les coordonnées entières des points.

    Returns:
      codes : np.ndarray de forme (N,), dtype=np.uint64
              Les codes de Morton pour chaque point.
    """
    N = pos.shape[0]
    # On détermine le nombre de bits à traiter (en supposant des entiers non négatifs)
    max_val = int(pos.max())
    nbits = max_val.bit_length()
    
    codes = np.zeros(N, dtype=np.uint64)
    
    for i in range(nbits):
        # Extraction et intercalage des bits de x, y et z
        codes |= ((pos[:, 0] >> i) & 1).astype(np.uint64) << (3 * i)
        codes |= ((pos[:, 1] >> i) & 1).astype(np.uint64) << (3 * i + 1)
        codes |= ((pos[:, 2] >> i) & 1).astype(np.uint64) << (3 * i + 2)
    
    return codes

def quantify_pos_min_max(pos, nbits=10):
    """
    Quantifie les points sur l'intervalle [0, 2^nbits - 1] en utilisant le min et max de chaque dimension.

    Parameters:
        pos   : torch.Tensor de forme (N, 3) en float
        nbits : int, nombre de bits pour la quantification (défaut 10)
        
    Returns:
        pos_quant : torch.Tensor de forme (N, 3) en np.uint32, points quantifiés
        min_vals  : torch.Tensor de forme (3,), minimum de chaque dimension
        max_vals  : torch.Tensor de forme (3,), maximum de chaque dimension
    """
    # Calculer le min et le max par dimension
    min_vals = torch.min(pos, dim=0).values
    max_vals = torch.max(pos, dim=0).values
    
    # Pour éviter une division par zéro en cas d'étendue nulle, on ajoute une petite constante.
    scale = (2**nbits - 1) / (max_vals - min_vals + 1e-9)
    
    # Normaliser et quantifier
    pos_quant = ((pos - min_vals) * scale).to(torch.int32)
    return pos_quant, min_vals, max_vals

def morton_code(pos):
    """
    Calcule le code de Morton (Z-order) pour chaque point de pos.

    Parameters:
      pos : torch.Tensor de forme (N, 3)
            Contient les coordonnées entières des points.

    Returns:
      codes : torch.Tensor de forme (N,), dtype=torch.uint64
              Les codes de Morton pour chaque point.
    """
    N = pos.shape[0]
    # On détermine le nombre de bits à traiter (en supposant des entiers non négatifs)
    max_val = int(pos.max())
    nbits = max_val.bit_length()
    
    codes = torch.zeros(N, dtype=torch.int64, device=pos.device)
    
    for i in range(nbits):
        # Extraction et intercalage des bits de x, y et z
        codes |= ((pos[:, 0] >> i) & 1).to(torch.int64) << (3 * i)
        codes |= ((pos[:, 1] >> i) & 1).to(torch.int64) << (3 * i + 1)
        codes |= ((pos[:, 2] >> i) & 1).to(torch.int64) << (3 * i + 2)
    
    return codes
