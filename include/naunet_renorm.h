#ifndef __NAUNET_RENORM_H__
#define __NAUNET_RENORM_H__

#include <sundials/sundials_matrix.h>

// clang-format off
__host__ int InitRenorm(realtype *ab, SUNMatrix A);
__host__ int RenormAbundance(realtype *rptr, realtype *ab);

#endif