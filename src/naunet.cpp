#include <cvode/cvode.h>  // prototypes for CVODE fcts., consts.
/* */
#include <nvector/nvector_cuda.h>
#include <nvector/nvector_serial.h>  // access to serial N_Vector
#include <sunlinsol/sunlinsol_cusolversp_batchqr.h>
#include <sunlinsol/sunlinsol_dense.h>  // access to dense SUNLinearSolver
#include <sunmatrix/sunmatrix_cusparse.h>
/* */
#include "naunet.h"
#include "naunet_ode.h"
#include "naunet_physics.h"
#include "naunet_renorm.h"
/* */

Naunet::Naunet(){};

Naunet::~Naunet(){};

// Adaptedfrom the cvDiurnals_ky.c example from the CVODE package.
// Check function return value...
//   opt == 0 means SUNDIALS function allocates memory so check if
//            returned NULL pointer
//   opt == 1 means SUNDIALS function returns a flag so check if
//            flag >= 0
//   opt == 2 means function allocates memory so check if returned
//            NULL pointer
int Naunet::CheckFlag(void *flagvalue, const char *funcname, int opt,
                      FILE *errf) {
    int *errflag;

    /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
    if (opt == 0 && flagvalue == NULL) {
        fprintf(errf,
                "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n",
                funcname);
        return NAUNET_FAIL;
    }

    /* Check if flag < 0 */
    else if (opt == 1) {
        errflag = (int *)flagvalue;
        if (*errflag < 0) {
            fprintf(errf, "\nSUNDIALS_ERROR: %s() failed with flag = %d\n\n",
                    funcname, *errflag);
            return NAUNET_FAIL;
        }
    }

    /* Check if function returned NULL pointer - no memory allocated */
    else if (opt == 2 && flagvalue == NULL) {
        fprintf(errf, "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n",
                funcname);
        return NAUNET_FAIL;
    }

    return NAUNET_SUCCESS;
};

int Naunet::Finalize() {
    /* */
    for (int i = 0; i < n_stream_in_use_; i++) {
        // N_VDestroy(cv_y_[i]);
        N_VFreeEmpty(cv_y_[i]);
        SUNMatDestroy(cv_a_[i]);
        // CVodeFree(&cv_mem_[i]);
        SUNLinSolFree(cv_ls_[i]);
        SUNContext_Free(&cv_sunctx_[i]);

        cusparseDestroy(cusp_handle_[i]);
        cusolverSpDestroy(cusol_handle_[i]);
        cudaStreamDestroy(custream_[i]);
    }

    cudaFreeHost(h_ab);
    cudaFreeHost(h_data);

    /*  */

    fclose(errfp_);

    return NAUNET_SUCCESS;
};

int Naunet::GetCVStates(void *cv_mem, long int &nst, long int &nfe,
                        long int &nsetups, long int &nje, long int &netf,
                        long int &nge, long int &nni, long int &ncfn) {
    int flag;

    flag = CVodeGetNumSteps(cv_mem, &nst);
    if (CheckFlag(&flag, "CVodeGetNumSteps", 1, errfp_) == NAUNET_FAIL) {
        return NAUNET_FAIL;
    }

    flag = CVodeGetNumRhsEvals(cv_mem, &nfe);
    if (CheckFlag(&flag, "CVodeGetNumRhsEvals", 1, errfp_) == NAUNET_FAIL) {
        return NAUNET_FAIL;
    }

    flag = CVodeGetNumLinSolvSetups(cv_mem, &nsetups);
    if (CheckFlag(&flag, "CVodeGetNumLinSolvSetups", 1, errfp_) ==
        NAUNET_FAIL) {
        return NAUNET_FAIL;
    }

    flag = CVodeGetNumErrTestFails(cv_mem, &netf);
    if (CheckFlag(&flag, "CVodeGetNumErrTestFails", 1, errfp_) == NAUNET_FAIL) {
        return NAUNET_FAIL;
    }

    flag = CVodeGetNumNonlinSolvIters(cv_mem, &nni);
    if (CheckFlag(&flag, "CVodeGetNumNonlinSolvIters", 1, errfp_) ==
        NAUNET_FAIL) {
        return NAUNET_FAIL;
    }

    flag = CVodeGetNumNonlinSolvConvFails(cv_mem, &ncfn);
    if (CheckFlag(&flag, "CVodeGetNumNonlinSolvConvFails", 1, errfp_) ==
        NAUNET_FAIL) {
        return NAUNET_FAIL;
    }

    flag = CVodeGetNumJacEvals(cv_mem, &nje);
    if (CheckFlag(&flag, "CVodeGetNumJacEvals", 1, errfp_) == NAUNET_FAIL) {
        return NAUNET_FAIL;
    }

    flag = CVodeGetNumGEvals(cv_mem, &nge);
    if (CheckFlag(&flag, "CVodeGetNumGEvals", 1, errfp_) == NAUNET_FAIL) {
        return NAUNET_FAIL;
    }

    return NAUNET_SUCCESS;
};

int Naunet::HandleError(int cvflag, realtype *ab, realtype dt, realtype t0) {
    if (cvflag >= 0) {
        return NAUNET_SUCCESS;
    }

    fprintf(errfp_, "CVode failed in Naunet! Flag = %d\n", cvflag);
    fprintf(errfp_, "Calling HandleError to fix the problem\n");

    /* */

    // TODO: No error handling for cusparse solver yet

    /* */

    return NAUNET_FAIL;
}

int Naunet::Init(int nsystem, double atol, double rtol, int mxsteps) {
    n_system_ = nsystem;
    mxsteps_  = mxsteps;
    atol_     = atol;
    rtol_     = rtol;
    errfp_    = fopen("naunet_error_record.txt", "a");

    int flag;

    /* */

    // if (nsystem < NSTREAMS ||  nsystem % NSTREAMS != 0) {
    //     printf("Invalid size of system!");
    //     return NAUNET_FAIL;
    // }

    cudaMallocHost((void **)&h_ab, sizeof(realtype) * n_system_ * NEQUATIONS);
    cudaMallocHost((void **)&h_data, sizeof(NaunetData) * n_system_);

    n_stream_in_use_        = nsystem / NSTREAMS >= 32 ? NSTREAMS : 1;
    int n_system_per_stream = nsystem / n_stream_in_use_;
    int n_thread_per_stream = std::min(BLOCKSIZE, n_system_per_stream);

    cudaError_t cuerr;

    for (int i = 0; i < n_stream_in_use_; i++) {
        cuerr                  = cudaStreamCreate(&custream_[i]);
        // SUNCudaThreadDirectExecPolicy stream_exec_policy(n_thread_per_stream,
        // custream_[i]); SUNCudaBlockReduceExecPolicy
        // reduce_exec_policy(n_thread_per_stream, 0, custream_[i]);
        stream_exec_policy_[i] = new SUNCudaThreadDirectExecPolicy(
            n_thread_per_stream, custream_[i]);
        reduce_exec_policy_[i] = new SUNCudaBlockReduceExecPolicy(
            n_thread_per_stream, 0, custream_[i]);

        cusparseCreate(&cusp_handle_[i]);
        cusparseSetStream(cusp_handle_[i], custream_[i]);
        cusolverSpCreate(&cusol_handle_[i]);
        cusolverSpSetStream(cusol_handle_[i], custream_[i]);
        flag = SUNContext_Create(NULL, &cv_sunctx_[i]);
        if (CheckFlag(&flag, "SUNContext_Create", 1, errfp_) == NAUNET_FAIL) {
            return NAUNET_FAIL;
        }
        cv_y_[i] = N_VNew_Cuda(NEQUATIONS * n_system_per_stream, cv_sunctx_[i]);
        // cv_y_[i]  = N_VNewEmpty_Cuda(cv_sunctx_[i]);
        flag     = N_VSetKernelExecPolicy_Cuda(cv_y_[i], stream_exec_policy_[i],
                                               reduce_exec_policy_[i]);
        if (CheckFlag(&flag, "N_VSetKernelExecPolicy_Cuda", 1, errfp_) ==
            NAUNET_FAIL) {
            return NAUNET_FAIL;
        }
        cv_a_[i] = SUNMatrix_cuSparse_NewBlockCSR(
            n_system_per_stream, NEQUATIONS, NEQUATIONS, NNZ, cusp_handle_[i], cv_sunctx_[i]);
        cv_ls_[i] =
            SUNLinSol_cuSolverSp_batchQR(cv_y_[i], cv_a_[i], cusol_handle_[i], cv_sunctx_[i]);
        // abstol = N_VNew_Cuda(neq, cv_sunctx_[i]);
        SUNMatrix_cuSparse_SetFixedPattern(cv_a_[i], 1);
        InitJac(cv_a_[i]);
    }

    /*  */

    // reset the n_vector to empty, maybe not necessary
    /* */

    // N_VDestroy(cv_y_);
    // cv_y_ = N_VNewEmpty_Cuda(cv_sunctx_);

    /* */

    /* */

    return NAUNET_SUCCESS;
};

int Naunet::PrintDebugInfo() {
    long int nst, nfe, nsetups, nje, netf, nge, nni, ncfn;
    int flag;

    /*  */

    size_t cuSpInternalSize, cuSpWorkSize;

    for (int i = 0; i < n_stream_in_use_; i++) {
        if (GetCVStates(cv_mem_[i], nst, nfe, nsetups, nje, netf, nge, nni,
                        ncfn) == NAUNET_FAIL) {
            return NAUNET_FAIL;
        }

        SUNLinSol_cuSolverSp_batchQR_GetDeviceSpace(
            cv_ls_[i], &cuSpInternalSize, &cuSpWorkSize);

        printf("\nFinal Statistics of %d stream:\n", i);
        printf("nst = %-6ld nfe  = %-6ld nsetups = %-6ld nje = %ld\n", nst, nfe,
               nsetups, nje);
        printf("nni = %-6ld ncfn = %-6ld netf = %-6ld    nge = %ld\n \n", nni,
               ncfn, netf, nge);
        printf(
            "cuSolverSp numerical factorization workspace size (in bytes) = "
            "%ld\n",
            cuSpWorkSize);
        printf("cuSolverSp internal Q, R buffer size (in bytes) = %ld\n",
               cuSpInternalSize);
    }
    /*  */

    return NAUNET_SUCCESS;
};

#ifdef IDX_ELEM_H
int Naunet::Renorm(realtype *ab) {

    SUNContext sunctx;
    int flag;
    flag = SUNContext_Create(NULL, &sunctx);
    if (CheckFlag(&flag, "SUNContext_Create", 1, errfp_) == NAUNET_FAIL) {
        return NAUNET_FAIL;
    }

    N_Vector b  = N_VMake_Serial(NELEMENTS, ab_ref_, sunctx);
    N_Vector r  = N_VNew_Serial(NELEMENTS, sunctx);
    SUNMatrix A = SUNDenseMatrix(NELEMENTS, NELEMENTS, sunctx);

    N_VConst(0.0, r);

    InitRenorm(ab, A);

    SUNLinearSolver LS = SUNLinSol_Dense(r, A, sunctx);

    flag = SUNLinSolSetup(LS, A);
    if (CheckFlag(&flag, "SUNLinSolSetup", 1, errfp_) == NAUNET_FAIL) {
        return NAUNET_FAIL;
    }
    flag = SUNLinSolSolve(LS, A, r, b, 0.0);
    if (CheckFlag(&flag, "SUNLinSolSolve", 1, errfp_) == NAUNET_FAIL) {
        return NAUNET_FAIL;
    }

    realtype *rptr = N_VGetArrayPointer(r);

    RenormAbundance(rptr, ab);

    N_VDestroy(b);
    N_VDestroy(r);
    SUNMatDestroy(A);
    SUNLinSolFree(LS);
    SUNContext_Free(&sunctx);

    return NAUNET_SUCCESS;
};
#endif

// To reset the size of cusparse solver
int Naunet::Reset(int nsystem, double atol, double rtol, int mxsteps) {
    n_system_ = nsystem;
    mxsteps_  = mxsteps;
    atol_     = atol;
    rtol_     = rtol;

    /*  */

    // if (nsystem < NSTREAMS ||  nsystem % NSTREAMS != 0) {
    //     printf("Invalid size of system!");
    //     return NAUNET_FAIL;
    // }

    n_stream_in_use_ = nsystem / NSTREAMS >= 32 ? NSTREAMS : 1;
    int n_system_per_stream = nsystem / n_stream_in_use_;
    int n_thread_per_stream = std::min(BLOCKSIZE, n_system_per_stream);

    cudaFreeHost(h_ab);
    cudaFreeHost(h_data);

    cudaMallocHost((void **)&h_ab, sizeof(realtype) * n_system_ * NEQUATIONS);
    cudaMallocHost((void **)&h_data, sizeof(NaunetData) * n_system_);

    int flag;

    for (int i = 0; i < n_stream_in_use_; i++) {
        N_VDestroy(cv_y_[i]);
        SUNMatDestroy(cv_a_[i]);
        SUNLinSolFree(cv_ls_[i]);

        delete stream_exec_policy_[i];
        delete reduce_exec_policy_[i];

        // SUNCudaThreadDirectExecPolicy stream_exec_policy(n_thread_per_stream,
        // custream_[i]); SUNCudaBlockReduceExecPolicy
        // reduce_exec_policy(n_thread_per_stream, 0, custream_[i]);

        stream_exec_policy_[i] = new SUNCudaThreadDirectExecPolicy(
            n_thread_per_stream, custream_[i]);
        reduce_exec_policy_[i] = new SUNCudaBlockReduceExecPolicy(
            n_thread_per_stream, 0, custream_[i]);

        cv_y_[i] = N_VNew_Cuda(NEQUATIONS * n_system_per_stream, cv_sunctx_[i]);
        // cv_y_[i] = N_VNewEmpty_Cuda(cv_sunctx_[i]);
        flag     = N_VSetKernelExecPolicy_Cuda(cv_y_[i], stream_exec_policy_[i],
                                               reduce_exec_policy_[i]);
        if (CheckFlag(&flag, "N_VSetKernelExecPolicy_Cuda", 1, errfp_) ==
            NAUNET_FAIL) {
            return NAUNET_FAIL;
        }
        cv_a_[i] = SUNMatrix_cuSparse_NewBlockCSR(
            n_system_per_stream, NEQUATIONS, NEQUATIONS, NNZ, cusp_handle_[i], cv_sunctx_[i]);
        cv_ls_[i] =
            SUNLinSol_cuSolverSp_batchQR(cv_y_[i], cv_a_[i], cusol_handle_[i], cv_sunctx_[i]);
        SUNMatrix_cuSparse_SetFixedPattern(cv_a_[i], 1);
        InitJac(cv_a_[i]);

        // reset the n_vector to empty, maybe not necessary
        // N_VDestroy(cv_y_);
        // cv_y_ = N_VNewEmpty_Cuda(cv_sunctx_[i]);
    }

    /*  */

    return NAUNET_SUCCESS;
};

#ifdef IDX_ELEM_H
int Naunet::SetReferenceAbund(realtype *ref, int opt) {
    if (opt == 0) {
        for (int i = 0; i < NELEMENTS; i++) {
            ab_ref_[i] = ref[i] / ref[IDX_ELEM_H];
        }
    } else if (opt == 1) {
        double Hnuclei = GetHNuclei(ref);
        for (int i = 0; i < NELEMENTS; i++) {
            ab_ref_[i] = GetElementAbund(ref, i) / Hnuclei;
        }
    }

    return NAUNET_SUCCESS;
}
#endif

int Naunet::Solve(realtype *ab, realtype dt, NaunetData *data) {
    /* */

    for (int i = 0; i < n_system_; i++) {
        h_data[i] = data[i];
        for (int j = 0; j < NEQUATIONS; j++) {
            int idx   = i * NEQUATIONS + j;
            h_ab[idx] = ab[idx];
        }
    }

    for (int i = 0; i < n_stream_in_use_; i++) {
        int cvflag;
        realtype t0 = 0.0;

        // ! Bug: I don't know why n_vector does not save the stream_exec_policy
        // and reduce_exec_policy
        N_VSetKernelExecPolicy_Cuda(cv_y_[i], stream_exec_policy_[i],
                                    reduce_exec_policy_[i]);

        // This way is too slow
        // realtype *ydata = N_VGetArrayPointer(cv_y_[i]);
        // for (int i = 0; i < NEQUATIONS; i++)
        // {
        //     ydata[i] = ab[i];
        // }
        N_VSetHostArrayPointer_Cuda(
            h_ab + i * n_system_ * NEQUATIONS / n_stream_in_use_, cv_y_[i]);
        N_VCopyToDevice_Cuda(cv_y_[i]);

#ifdef NAUNET_DEBUG
        // sunindextype lrw, liw;
        // N_VSpace_Cuda(cv_y_[i], &lrw, &liw);
        // printf("NVector space: real-%d, int-%d\n", lrw, liw);
#endif

        cv_mem_[i] = CVodeCreate(CV_BDF, cv_sunctx_[i]);

        cvflag     = CVodeSetErrFile(cv_mem_[i], errfp_);
        if (CheckFlag(&cvflag, "CVodeSetErrFile", 1, errfp_) == NAUNET_FAIL) {
            return NAUNET_FAIL;
        }

        cvflag = CVodeSetMaxNumSteps(cv_mem_[i], mxsteps_);
        if (CheckFlag(&cvflag, "CVodeSetMaxNumSteps", 1, errfp_) ==
            NAUNET_FAIL) {
            return NAUNET_FAIL;
        }

        cvflag = CVodeInit(cv_mem_[i], Fex, t0, cv_y_[i]);
        if (CheckFlag(&cvflag, "CVodeInit", 1, errfp_) == NAUNET_FAIL) {
            return NAUNET_FAIL;
        }

        cvflag = CVodeSStolerances(cv_mem_[i], rtol_, atol_);
        if (CheckFlag(&cvflag, "CVodeSStolerances", 1, errfp_) == NAUNET_FAIL) {
            return NAUNET_FAIL;
        }

        cvflag = CVodeSetLinearSolver(cv_mem_[i], cv_ls_[i], cv_a_[i]);
        if (CheckFlag(&cvflag, "CVodeSetLinearSolver", 1, errfp_) ==
            NAUNET_FAIL) {
            return NAUNET_FAIL;
        }

        cvflag = CVodeSetJacFn(cv_mem_[i], Jac);
        if (CheckFlag(&cvflag, "CVodeSetJacFn", 1, errfp_) == NAUNET_FAIL) {
            return NAUNET_FAIL;
        }

        cvflag = CVodeSetUserData(cv_mem_[i],
                                  h_data + i * n_system_ / n_stream_in_use_);
        if (CheckFlag(&cvflag, "CVodeSetUserData", 1, errfp_) == NAUNET_FAIL) {
            return NAUNET_FAIL;
        }

        cvflag = CVode(cv_mem_[i], dt, cv_y_[i], &t0, CV_NORMAL);

        N_VCopyFromDevice_Cuda(cv_y_[i]);
        realtype *local_ab = N_VGetHostArrayPointer_Cuda(cv_y_[i]);
        for (int idx = 0; idx < n_system_ * NEQUATIONS / n_stream_in_use_;
             idx++) {
            ab[idx + i * n_system_ * NEQUATIONS / n_stream_in_use_] =
                local_ab[idx];
        }

        CVodeFree(&cv_mem_[i]);
    }

    // TODO: error handling

    // cudaDeviceSynchronize();

    return NAUNET_SUCCESS;

    /* */
};

#ifdef PYMODULE
#ifdef IDX_ELEM_H
py::array_t<realtype> Naunet::PyWrapRenorm(py::array_t<realtype> arr) {
    py::buffer_info info = arr.request();
    realtype *ab         = static_cast<realtype *>(info.ptr);

    int flag             = Renorm(ab);
    if (flag == NAUNET_FAIL) {
        throw std::runtime_error("Fail to renormalization");
    }

    return py::array_t<realtype>(info.shape, ab);
}
py::array_t<realtype> Naunet::PyWrapSetReferenceAbund(py::array_t<realtype> arr,
                                                      int opt) {
    py::buffer_info info = arr.request();
    realtype *ab         = static_cast<realtype *>(info.ptr);

    int flag             = SetReferenceAbund(ab, opt);
    if (flag == NAUNET_FAIL) {
        throw std::runtime_error("Fail to set reference abundance");
    }

    return py::array_t<realtype>(info.shape, ab);
}
#endif
py::array_t<realtype> Naunet::PyWrapSolve(py::array_t<realtype> arr,
                                          realtype dt, NaunetData *data) {
    py::buffer_info info = arr.request();
    realtype *ab         = static_cast<realtype *>(info.ptr);

    int flag             = Solve(ab, dt, data);
    if (flag == NAUNET_FAIL) {
        throw std::runtime_error("Something unrecoverable occurred");
    }

    return py::array_t<realtype>(info.shape, ab);
}
#endif