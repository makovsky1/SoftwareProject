#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctype.h>
#include "spkmeans.c"


static PyObject* spkFit(PyObject *self, PyObject *args){
    double **mat, **weighted_matrix, **diagonal_matrix, **lnorm_matrix, **eigan_vectors_matrix, *eigan_values;
    int N, dim, K;

    PyObject* mat_py;
    PyObject* res_py;
    
    if(!PyArg_ParseTuple(args, "Oiii",&mat_py, &N, &dim, &K)){
        return Py_BuildValue("");
    }  
      
    mat = InitMat(mat_py, N, dim); // Need to check if InitMat returns NULL
    if (mat == NULL)
        return Py_BuildValue("");

    weighted_matrix = AllocateMat(N, N);
    diagonal_matrix = AllocateMat(N, N);
    lnorm_matrix = AllocateMat(N, N);
    eigan_vectors_matrix = AllocateMat(N, N);
    eigan_values = (double*)calloc(N, sizeof(double));

    if (weighted_matrix == NULL || diagonal_matrix == NULL || lnorm_matrix == NULL || eigan_vectors_matrix == NULL || eigan_values == NULL){
        return Py_BuildValue("");
    }
    if (WeightedAdjancencyMatrix(&mat, &weighted_matrix, N, dim) == 0){
        return Py_BuildValue("");
    }
    if (DiagonalDegreeMatirx(&weighted_matrix, &diagonal_matrix, N, dim) == 0){
        return Py_BuildValue("");
    }
    if (NormalizedGraphLaplasian(&diagonal_matrix, &weighted_matrix, &lnorm_matrix, N) == 0){
        return Py_BuildValue("");
    }
    if (Jacobi(&mat, &eigan_vectors_matrix, &eigan_values, N) == 0){
        return Py_BuildValue("");
    }
    if (K == 0){
        K = Eigengap(&eigan_values, N);
        if (K == -1){
        return Py_BuildValue("");
        }   
    }
}

static PyObject* weightedFit(PyObject *self, PyObject *args){
    double **mat, **weighted_matrix;
    int N,dim;


    PyObject* mat_py;
    PyObject* res_py;
    
    if(!PyArg_ParseTuple(args, "Oii",&mat_py, &N, &dim)){
        return Py_BuildValue("");
    }  
      
    mat = InitMat(mat_py, N, dim); 
    if (mat == NULL)
        return Py_BuildValue("");

    weighted_matrix = AllocateMat(N, N);
    if (weighted_matrix == NULL){
        return Py_BuildValue("");
    }

    if (WeightedAdjancencyMatrix(&mat, &weighted_matrix, N, dim) == 0){
        return Py_BuildValue("");
    }

    res_py = InitPyObject(&mat, N, N);
    FreeMat(&mat, N);
    FreeMat(&weighted_matrix, N);
    return Py_BuildValue("O", res_py); /* return data to python file */
}

static PyObject* diagonalFit(PyObject *self, PyObject *args){
    double **mat, **weighted_matrix, **diagonal_matrix;
    int N,dim;

    PyObject* mat_py;
    PyObject* res_py;
    
    if(!PyArg_ParseTuple(args, "Oii",&mat_py, &N, &dim)){
        return Py_BuildValue("");
    }  
      
    mat = InitMat(mat_py, N, dim); 

    if (mat == NULL)
        return Py_BuildValue("");

    weighted_matrix = AllocateMat(N, N);
    diagonal_matrix = AllocateMat(N, N);
    if (weighted_matrix == NULL || diagonal_matrix == NULL){
        return Py_BuildValue("");
    }

    if (WeightedAdjancencyMatrix(&mat, &weighted_matrix, N, dim) == 0){
        return Py_BuildValue("");
    }
    if (DiagonalDegreeMatirx(&weighted_matrix, &diagonal_matrix, N, dim) == 0){
        return Py_BuildValue("");
    }

    res_py = InitPyObject(&diagonal_matrix, N, N);
    FreeMat(&mat, N);
    FreeMat(&weighted_matrix, N);
    FreeMat(&diagonal_matrix, N);
    return Py_BuildValue("O", res_py); /* return data to python file */
}

static PyObject* lnormFit(PyObject *self, PyObject *args){
    double **mat, **weighted_matrix, **diagonal_matrix, **lnorm_matrix;
    int N,dim;

    PyObject* mat_py;
    PyObject* res_py;
    
    if(!PyArg_ParseTuple(args, "Oii",&mat_py, &N, &dim)){
        return Py_BuildValue("");
    }  
      
    mat = InitMat(mat_py, N, dim); 

    if (mat == NULL)
        return Py_BuildValue("");

    weighted_matrix = AllocateMat(N, N);
    diagonal_matrix = AllocateMat(N, N);
    lnorm_matrix = AllocateMat(N, N);
    if (weighted_matrix == NULL || diagonal_matrix == NULL || lnorm_matrix == NULL){
        return Py_BuildValue("");
    }

    if (WeightedAdjancencyMatrix(&mat, &weighted_matrix, N, dim) == 0){
        return Py_BuildValue("");
    }
    if (DiagonalDegreeMatirx(&weighted_matrix, &diagonal_matrix, N, dim) == 0){
        return Py_BuildValue("");
    }
    if (NormalizedGraphLaplasian(&diagonal_matrix, &weighted_matrix, &lnorm_matrix, N) == 0){
        return Py_BuildValue("");
    }

    res_py = InitPyObject(&lnorm_matrix, N, N);
    FreeMat(&mat, N);
    FreeMat(&weighted_matrix, N);
    FreeMat(&diagonal_matrix, N);
    FreeMat(&lnorm_matrix, N);
    return Py_BuildValue("O", res_py); /* return data to python file */
}

static PyObject* jacobiFit(PyObject *self, PyObject *args){
    double **mat, **eigan_vectors_matrix, *eigan_values;
    int N, i;

    PyObject* mat_py;
    PyObject* eigan_vectors_py;
    PyObject* eigan_values_py;
    
    if(!PyArg_ParseTuple(args, "Oii",&mat_py, &N)){
        return Py_BuildValue("");
    }  
      
    mat = InitMat(mat_py, N, N); 

    if (mat == NULL)
        return Py_BuildValue("");

    eigan_vectors_matrix = AllocateMat(N, N);

    eigan_values = (double*)calloc(N, sizeof(double));
    if (eigan_vectors_matrix == NULL || eigan_values == NULL){
        return Py_BuildValue("");
    }

    if (Jacobi(&mat, &eigan_vectors_matrix, &eigan_values, N) == 0){
        return Py_BuildValue("");
    }

    eigan_vectors_py = InitPyObject(&eigan_vectors_matrix, N, N);
    eigan_values_py = PyList_New(N);
    for (i = 0; i < N; i++){
        PyList_SetItem(eigan_values_py,i,Py_BuildValue("d", eigan_values[i]));
    }

    FreeMat(&mat, N);
    FreeMat(&eigan_vectors_matrix, N);
    free(eigan_values);

    return Py_BuildValue("(OO)", eigan_vectors_py, eigan_values_py); 
}

static PyObject* kmeansFit(PyObject *self, PyObject *args){
    double **mat, **centroids;
    int N, dim, K, i, j, MAX_ITER;

    PyObject* mat_py;
    PyObject* centroids_py;
    
    if(!PyArg_ParseTuple(args, "Oiiii",&mat_py, &N, &dim, &K, &MAX_ITER)){
        return Py_BuildValue("");
    }  
      
    mat = InitMat(mat_py, N, dim); 
    if (mat == NULL)
        return Py_BuildValue("");
    
    centroids = AllocateMat(N, dim);
    if (centroids == NULL){
        return Py_BuildValue("");
    }
    if (Kmeans(&mat, &centroids, N, dim, K, MAX_ITER) == 0){
        return Py_BuildValue("");
    }
    centroids_py = InitPyObject(&centroids, N, dim);
    FreeMat(&mat, N);
    FreeMat(&centroids, N);

    return Py_BuildValue("O", centroids_py); 

}

double ** InitMat(PyObject * mat_py,int N, int dim){
    double ** mat;
    int i, j;
    PyObject * curr_row;
    PyObject * curr_cord;


    mat = (double **)calloc(N, sizeof(double *));
    if (mat == NULL){
        return NULL;
    }
    for (i = 0; i < N; i++){
        mat[i] = (double*)calloc(dim, sizeof(double));
        if (mat[i] == NULL){
            return NULL;
        }
    }

    for(i = 0; i < N ;i++){ /* fill new array with python vectors */
        curr_row = PyList_GetItem(mat_py,i);

        for(j = 0; j < dim ;j++){
            curr_cord = PyList_GetItem(curr_row,j);
            mat[i][j] = PyFloat_AsDouble(curr_cord);
        }
    } 

    return mat;
}

PyObject* InitPyObject (double *** mat, int N, int dim){
    PyObject * res_py, *weighted_Row;
    int i,j;

    for (i = 0; i < N; i++){
        weighted_Row = PyList_New(dim);

        for (j = 0; j < dim; j++){
            PyList_SetItem(weighted_Row,j,Py_BuildValue("d", (*mat)[i][j]));   
        }
        PyList_SetItem(res_py,i,weighted_Row);
    }
    return res_py;
}