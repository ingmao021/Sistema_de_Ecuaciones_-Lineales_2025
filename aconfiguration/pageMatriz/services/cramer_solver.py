from typing import List, Union, Optional, Dict
from .matrix_operations import MatrixOperations
import numpy as np

class CramerSolver:
    """Clase que implementa el método de Cramer siguiendo el principio de Abierto/Cerrado"""
    
    def __init__(self, matrix_ops: MatrixOperations = None):
        self.matrix_ops = matrix_ops or MatrixOperations()
    
    def solve(self, matrix: List[List[float]], vector: List[float]) -> Optional[Dict[str, Union[List[float], str]]]:
        """
        Resuelve un sistema de ecuaciones usando el método de Cramer
        Retorna un diccionario con la solución o mensaje de error
        """
        try:
            # Validación inicial de dimensiones
            if not matrix or not vector:
                return {"error": "La matriz o el vector están vacíos"}
            
            if not self.matrix_ops.validate_matrix_dimensions(matrix, vector):
                return {"error": "Las dimensiones de la matriz o vector son inválidas"}
            
            # Convertir a numpy arrays para mejor manejo numérico
            A = np.array(matrix, dtype=np.float64)
            b = np.array(vector, dtype=np.float64)
            
            # Calcular el determinante principal con mayor precisión
            main_det = np.linalg.det(A)
            
            if abs(main_det) < 1e-10:  # Consideramos 0 si es muy cercano
                return {"error": "El sistema no tiene solución única (determinante = 0)"}
            
            n = len(matrix)
            solution = []
            cramer_matrices = []
            cramer_determinants = []
            
            # Calcular cada variable usando Cramer
            for i in range(n):
                # Crear copia de la matriz para no modificar la original
                Ai = A.copy()
                # Sustituir la columna i con el vector b
                Ai[:, i] = b
                # Calcular determinante de la nueva matriz
                det_i = np.linalg.det(Ai)
                # Calcular el valor de x_i
                x_i = det_i / main_det
                solution.append(float(x_i))  # Convertir a float Python estándar
                cramer_matrices.append(Ai.tolist())
                cramer_determinants.append(float(det_i))
            
            return {
                "solution": solution,
                "determinant": float(main_det),
                "original_matrix": matrix,
                "cramer_matrices": cramer_matrices,
                "cramer_determinants": cramer_determinants,
                "debug_info": {
                    "matrix_shape": A.shape,
                    "vector_shape": b.shape,
                    "main_det": float(main_det)
                }
            }
            
        except Exception as e:
            import traceback
            return {
                "error": f"Error al resolver el sistema: {str(e)}",
                "debug_info": {
                    "traceback": traceback.format_exc(),
                    "matrix": matrix,
                    "vector": vector
                }
            } 