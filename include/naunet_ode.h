#ifndef __NAUNET_ODE_H__
#define __NAUNET_ODE_H__

#include <sundials/sundials_matrix.h>
#include <sundials/sundials_nvector.h>
#include <sundials/sundials_types.h>  // defs. of realtype, sunindextype

#include "naunet_data.h"
#include "naunet_macros.h"

/* */
__device__ int EvalRates(realtype *k, realtype *y, NaunetData *user_data);
#if NHEATPROCS
__device__ int EvalHeatingRates(realtype *kc, realtype *y,
                                NaunetData *user_data);
#endif
#if NCOOLPROCS
__device__ int EvalCoolingRates(realtype *kc, realtype *y,
                                NaunetData *user_data);
#endif
/* */
int Fex(realtype t, N_Vector u, N_Vector udot, void *user_data);
int Jac(realtype t, N_Vector u, N_Vector fu, SUNMatrix jmatrix, void *user_data,
        N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);

/* */
int InitJac(SUNMatrix jmatrix);
/* */

#endif