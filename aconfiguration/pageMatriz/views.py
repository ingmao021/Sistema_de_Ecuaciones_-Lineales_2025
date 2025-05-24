from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_http_methods
from .services.cramer_solver import CramerSolver
from .services.inverse_solver import InverseSolver
import json
import logging

logger = logging.getLogger(__name__)

# Create your views here.

@ensure_csrf_cookie
def index(request):
    """
    Vista para renderizar la página principal del sistema de ecuaciones
    """
    return render(request, 'index.html')

@require_http_methods(["POST"])
def solve_system(request):
    """Vista para resolver el sistema de ecuaciones"""
    try:
        data = json.loads(request.body)
        matrix = data.get('matrix', [])
        vector = data.get('vector', [])
        method = data.get('method', 'cramer')

        # Log de datos recibidos
        logger.debug(f"Datos recibidos - Matriz: {matrix}, Vector: {vector}, Método: {method}")
        
        # Validar que la matriz y el vector no estén vacíos
        if not matrix or not vector:
            return JsonResponse({
                "error": "La matriz o el vector están vacíos",
                "debug_info": {"matrix": matrix, "vector": vector}
            }, status=400)
        
        # Convertir strings a float
        try:
            matrix = [[float(val) for val in row] for row in matrix]
            vector = [float(val) for val in vector]
            logger.debug(f"Matriz convertida: {matrix}")
            logger.debug(f"Vector convertido: {vector}")
        except (ValueError, TypeError) as e:
            return JsonResponse({
                "error": "Valores inválidos en la matriz o vector",
                "debug_info": {
                    "error_detail": str(e),
                    "matrix": matrix,
                    "vector": vector
                }
            }, status=400)
        
        # Seleccionar el método de solución
        try:
            if method == 'inverse':
                solver = InverseSolver()
            else:
                solver = CramerSolver()
                
            result = solver.solve(matrix, vector)
            logger.debug(f"Resultado del solver: {result}")
            
            if "error" in result:
                return JsonResponse({
                    "error": result["error"],
                    "debug_info": result.get("debug_info", {})
                }, status=400)
                
            return JsonResponse(result)
            
        except Exception as e:
            logger.error(f"Error al resolver el sistema: {str(e)}", exc_info=True)
            return JsonResponse({
                "error": "Error al resolver el sistema",
                "debug_info": {
                    "error_detail": str(e),
                    "matrix": matrix,
                    "vector": vector,
                    "method": method
                }
            }, status=500)
            
    except json.JSONDecodeError as e:
        return JsonResponse({
            "error": "Formato JSON inválido",
            "debug_info": {"error_detail": str(e)}
        }, status=400)
    except Exception as e:
        logger.error(f"Error interno del servidor: {str(e)}", exc_info=True)
        return JsonResponse({
            "error": f"Error interno del servidor: {str(e)}",
            "debug_info": {"error_detail": str(e)}
        }, status=500)
