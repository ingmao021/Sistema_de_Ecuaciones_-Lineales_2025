from typing import List, Union, Optional, Dict
import numpy as np
from .matrix_operations import MatrixOperations

class InverseSolver:
    """Clase que implementa el método de matriz inversa siguiendo el principio de Abierto/Cerrado"""
    
    def __init__(self, matrix_ops: MatrixOperations = None):
        self.matrix_ops = matrix_ops or MatrixOperations()
    
    def solve(self, matrix: List[List[float]], vector: List[float]) -> Optional[Dict[str, Union[List[float], str]]]:
        """
        Resuelve un sistema de ecuaciones usando el método de matriz inversa
        Retorna un diccionario con la solución o mensaje de error
        """
        try:
            if not self.matrix_ops.validate_matrix_dimensions(matrix, vector):
                return {"error": "Las dimensiones de la matriz o vector son inválidas"}
            
            # Convertir a arrays de numpy para operaciones matriciales
            A = np.array(matrix)
            b = np.array(vector)
            
            # Calcular el determinante para verificar si la matriz es invertible
            det = np.linalg.det(A)
            
            if abs(det) < 1e-10:  # Consideramos 0 si es muy cercano
                return {"error": "El sistema no tiene solución única (determinante = 0)"}
            
            # Calcular la matriz inversa
            A_inv = np.linalg.inv(A)
            
            # Calcular la solución: x = A^(-1) * b
            solution = A_inv.dot(b)
            
            # Convertir a lista para la respuesta
            solution_list = solution.tolist()
            
            return {
                "solution": solution_list,
                "determinant": float(det),
                "inverse_matrix": A_inv.tolist()  # Incluimos la matriz inversa en la respuesta
            }
            
        except Exception as e:
            return {"error": f"Error al resolver el sistema: {str(e)}"} 