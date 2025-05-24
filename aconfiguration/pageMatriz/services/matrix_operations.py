from typing import List, Union
import numpy as np

class MatrixOperations:
    """Clase base para operaciones matriciales siguiendo el principio de Responsabilidad Única"""
    
    @staticmethod
    def calculate_determinant(matrix: List[List[float]]) -> float:
        """Calcula el determinante de una matriz"""
        return float(np.linalg.det(np.array(matrix)))
    
    @staticmethod
    def create_substituted_matrix(matrix: List[List[float]], vector: List[float], column: int) -> List[List[float]]:
        """Crea una nueva matriz sustituyendo una columna con un vector"""
        result = [row[:] for row in matrix]  # Copia profunda de la matriz
        for i in range(len(matrix)):
            result[i][column] = vector[i]
        return result
    
    @staticmethod
    def validate_matrix_dimensions(matrix: List[List[float]], vector: List[float]) -> bool:
        """Valida que las dimensiones de la matriz y el vector sean correctas"""
        if not matrix or not vector:
            return False
        n = len(matrix)
        return all(len(row) == n for row in matrix) and len(vector) == n 