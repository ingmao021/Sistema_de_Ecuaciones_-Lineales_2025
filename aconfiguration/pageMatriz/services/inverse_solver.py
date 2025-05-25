from typing import List, Union, Optional, Dict
import numpy as np
from .matrix_operations import MatrixOperations

class InverseSolver:
    """Clase que implementa el método de matriz inversa siguiendo el principio de Abierto/Cerrado"""
    
    def __init__(self, matrix_ops: MatrixOperations = None):
        self.matrix_ops = matrix_ops or MatrixOperations()
    
    def _clean_number(self, number: float) -> Union[int, float]:
        """Convierte un número a entero si no tiene parte decimal"""
        if isinstance(number, (int, float)):
            # Convertir a float primero para manejar números numpy
            num_float = float(number)
            # Si es entero, convertir a int
            if num_float.is_integer():
                return int(num_float)
            # Si tiene decimales, mantener como float
            return num_float
        return number

    def _clean_matrix(self, matrix: List[List[float]]) -> List[List[Union[int, float]]]:
        """Limpia todos los números en una matriz"""
        return [[self._clean_number(num) for num in row] for row in matrix]

    def _clean_vector(self, vector: List[float]) -> List[Union[int, float]]:
        """Limpia todos los números en un vector"""
        return [self._clean_number(num) for num in vector]
    
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
            
            # Limpiar y convertir los resultados
            solution_list = self._clean_vector(solution.tolist())
            inverse_matrix = self._clean_matrix(A_inv.tolist())
            clean_det = self._clean_number(det)
            
            return {
                "solution": solution_list,
                "determinant": clean_det,
                "inverse_matrix": inverse_matrix
            }
            
        except Exception as e:
            return {"error": f"Error al resolver el sistema: {str(e)}"} 