#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctype.h>
#include "spkmeans.h"

/*static PyObject* spkFit(PyObject *, PyObject *);
static PyObject* weightedFit(PyObject *, PyObject *);
static PyObject* diagonalFit(PyObject *, PyObject *);
static PyObject* lnormFit(PyObject *, PyObject *);
static PyObject* jacobiFit(PyObject *, PyObject *);
static PyObject* kmeansFit(PyObject *, PyObject *);*/
double ** InitMat(PyObject *, int, int);
PyObject* InitPyObject (double ***, int, int);



static PyObject* spkFit(PyObject *self, PyObject *args){
    double **mat, **weighted_matrix, **diagonal_matrix, **lnorm_matrix, **eigan_vectors_matrix, *eigan_values, **u_matrix, **t_matrix;
    int N, dim, K;

    PyObject* mat_py;
    PyObject* res_py;
    
    if(!PyArg_ParseTuple(args, "Oiii",&mat_py, &N, &dim, &K)){
        return Py_BuildValue("");
    }  
      
    mat = (double**)InitMat(mat_py, N, dim); 
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
    if (DiagonalDegreeMatirx(&weighted_matrix, &diagonal_matrix, N) == 0){
        return Py_BuildValue("");
    }
    if (NormalizedGraphLaplasian(&diagonal_matrix, &weighted_matrix, &lnorm_matrix, N) == 0){
        return Py_BuildValue("");
    }
    if (Jacobi(&lnorm_matrix, &eigan_vectors_matrix, &eigan_values, N) == 0){
        return Py_BuildValue("");
    }

    if (K == 0){
        K = Eigengap(&eigan_values, N);
        if (K == -1){
            return Py_BuildValue("");
        }   
    }

    if (SortVectors(&eigan_vectors_matrix, &eigan_values, N) == 0){
        return Py_BuildValue("");
    }
    u_matrix = AllocateMat(N, K);
    if (UMatrix(&eigan_vectors_matrix, &u_matrix, N, K) == 0){
        return Py_BuildValue("");
    }

    t_matrix = AllocateMat(N, K);

    if (TMatrix(&u_matrix, &t_matrix, N, K) == 0){
        return Py_BuildValue("");
    }

    FreeMat(&mat, N);
    FreeMat(&weighted_matrix, N);
    FreeMat(&diagonal_matrix, N);
    FreeMat(&lnorm_matrix, N);
    FreeMat(&eigan_vectors_matrix, N);
    FreeMat(&u_matrix, N);
    free(eigan_values);
    
    res_py = InitPyObject(&t_matrix, N, K);
    FreeMat(&t_matrix, N);
    return Py_BuildValue("O", res_py); 
}

static PyObject* weightedFit(PyObject *self, PyObject *args){
    double **mat, **weighted_matrix;
    int N,dim;

    PyObject* mat_py;
    PyObject* res_py;
    
    if(!PyArg_ParseTuple(args, "Oii",&mat_py, &N, &dim)){
        return Py_BuildValue("");
    }

    mat = (double**)InitMat(mat_py, N, dim); 
    if (mat == NULL)
        return Py_BuildValue("");
    

    weighted_matrix = AllocateMat(N, N);
    if (weighted_matrix == NULL){
        return Py_BuildValue("");
    }


    if (WeightedAdjancencyMatrix(&mat, &weighted_matrix, N, dim) == 0){
        return Py_BuildValue("");
    }
    
    
    res_py = InitPyObject(&weighted_matrix, N, N);
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
      
    mat = (double**)InitMat(mat_py, N, dim); 

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

    if (DiagonalDegreeMatirx(&weighted_matrix, &diagonal_matrix, N) == 0){
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
      
    mat = (double**)InitMat(mat_py, N, dim); 

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
    if (DiagonalDegreeMatirx(&weighted_matrix, &diagonal_matrix, N) == 0){
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
    if(!PyArg_ParseTuple(args, "Oi",&mat_py, &N)){
        return Py_BuildValue("");
    }  

    mat = (double**)InitMat(mat_py, N, N); 

    if (mat == NULL){
        return Py_BuildValue("");
    }

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
    int N, dim, K, MAX_ITER;

    PyObject* mat_py;
    PyObject* centroids_py;
    
    if(!PyArg_ParseTuple(args, "Oiiii",&mat_py, &N, &dim, &K, &MAX_ITER)){
        return Py_BuildValue("");
    }  
      
    mat = (double**)InitMat(mat_py, N, dim); 
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

    for (i = 0; i < N ; i++){ /* fill new array with python vectors */
        curr_row = PyList_GetItem(mat_py, i);

        for (j = 0; j < dim ; j++){
            curr_cord = PyList_GetItem(curr_row, j);
            mat[i][j] = PyFloat_AsDouble(curr_cord);
        }
    } 

    return mat;
}

PyObject* InitPyObject (double *** mat, int N, int dim){
    PyObject * res_py, * list_row;
    int i,j;

    res_py = PyList_New(N);
    for (i = 0; i < N; i++){
        list_row = PyList_New(dim);

        for (j = 0; j < dim; j++){
            PyList_SetItem(list_row,j,Py_BuildValue("d", (*mat)[i][j]));   
        }
        PyList_SetItem(res_py, i, list_row);
    }
    return res_py;
}


static PyMethodDef capiMethods[] = {
    {"spkFit", (PyCFunction) spkFit, METH_VARARGS, PyDoc_STR("spk")},
    {"weightedFit", (PyCFunction) weightedFit, METH_VARARGS, PyDoc_STR("WeightedAdjancencyMatrix")},
    {"diagonalFit", (PyCFunction) diagonalFit, METH_VARARGS, PyDoc_STR("DiagonalMatrix")},
    {"lnormFit", (PyCFunction) lnormFit, METH_VARARGS, PyDoc_STR("lnormFit")},
    {"jacobiFit", (PyCFunction) jacobiFit, METH_VARARGS, PyDoc_STR("jacobiFit")},
    {"kmeansFit", (PyCFunction) kmeansFit, METH_VARARGS, PyDoc_STR("kmeansFit")},
    {NULL,NULL,0,NULL}
};


static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "mykmeanssp",
    NULL,
    -1,
    capiMethods
};


PyMODINIT_FUNC
PyInit_mykmeanssp(void)
{
    PyObject *m;
    m = PyModule_Create(&moduledef);
    if (!m){
        return NULL;
    }
    return m;
}