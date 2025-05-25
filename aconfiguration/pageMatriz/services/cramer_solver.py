from typing import List, Union, Optional, Dict
from .matrix_operations import MatrixOperations
import numpy as np

class CramerSolver:
    """Clase que implementa el método de Cramer siguiendo el principio de Abierto/Cerrado"""
    
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
                solution.append(self._clean_number(x_i))
                cramer_matrices.append(self._clean_matrix(Ai.tolist()))
                cramer_determinants.append(self._clean_number(det_i))
            
            return {
                "solution": solution,
                "determinant": self._clean_number(main_det),
                "original_matrix": self._clean_matrix(matrix),
                "cramer_matrices": cramer_matrices,
                "cramer_determinants": cramer_determinants,
                "debug_info": {
                    "matrix_shape": A.shape,
                    "vector_shape": b.shape,
                    "main_det": self._clean_number(main_det)
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