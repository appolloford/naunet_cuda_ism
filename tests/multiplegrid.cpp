// 
#include <stdio.h>

#include <stdexcept>
#include <vector>

#include "naunet.h"
#include "naunet_data.h"
#include "naunet_macros.h"
#include "naunet_ode.h"
#include "naunet_timer.h"

int main() {
    int nsystem      = 2048;
    double spy       = 86400.0 * 365.0;

    NaunetData *data = new NaunetData[nsystem];
    //
    double nH        = 2e4;
    double zeta_cr   = 1.3e-17;
    double zeta_xr   = 0.0;
    double Tgas      = 10.0;
    double Tdust     = 10.0;
    double Av        = 10.0;
    double G0        = 4.0;
    double rG        = 1e-5;
    double omega     = 0.5;
    double barr      = 1.5e-8;
    double sites     = 1.5e15;
    double hop       = 0.3;
    double nMono     = 2.0;
    double duty      = 3.16e-19;
    double Tcr       = 70.0;
    double branch    = 1e-2;

    for (int isys = 0; isys < nsystem; isys++) {
        data[isys].nH      = nH;
        data[isys].zeta_cr = zeta_cr;
        data[isys].zeta_xr = zeta_xr;
        data[isys].Tgas    = Tgas;
        data[isys].Tdust   = Tdust;
        data[isys].Av      = Av;
        data[isys].G0      = G0;
        data[isys].rG      = rG;
        data[isys].omega   = omega;
        data[isys].barr    = barr;
        data[isys].sites   = sites;
        data[isys].hop     = hop;
        data[isys].nMono   = nMono;
        data[isys].duty    = duty;
        data[isys].Tcr     = Tcr;
        data[isys].branch  = branch;
    }


    Naunet naunet;
    if (naunet.Init() == NAUNET_FAIL) {
        printf("Initialize Fail\n");
        return 1;
    }

    if (naunet.Reset(nsystem) == NAUNET_FAIL) {
        throw std::runtime_error("Fail to reset the number of systems");
    }

    //
    double *y = new double[nsystem * NEQUATIONS];
    for (int isys = 0; isys < nsystem; isys++) {
        for (int i = 0; i < NEQUATIONS; i++) {
            y[isys * NEQUATIONS + i] = 1.e-40;
        }
        y[isys * NEQUATIONS + IDX_H2I]     = 0.5 * nH;
        y[isys * NEQUATIONS + IDX_HI]      = 5.0e-5 * nH;
        y[isys * NEQUATIONS + IDX_HeI]     = 9.75e-2 * nH;
        y[isys * NEQUATIONS + IDX_NI]      = 7.5e-5 * nH;
        y[isys * NEQUATIONS + IDX_OI]      = 3.2e-4 * nH;
        y[isys * NEQUATIONS + IDX_CI]      = 1.4e-4 * nH;
        y[isys * NEQUATIONS + IDX_SI]      = 8.0e-8 * nH;
        y[isys * NEQUATIONS + IDX_SiI]     = 8.0e-9 * nH;
        y[isys * NEQUATIONS + IDX_NaI]     = 2.0e-9 * nH;
        y[isys * NEQUATIONS + IDX_MgI]     = 7.0e-9 * nH;
        y[isys * NEQUATIONS + IDX_FeI]     = 3.0e-9 * nH;
        y[isys * NEQUATIONS + IDX_ClI]     = 4.0e-9 * nH;
        y[isys * NEQUATIONS + IDX_FI]      = 2.0e-8 * nH;
        y[isys * NEQUATIONS + IDX_GRAIN0I] = 1.3e-12 * nH;
    }


    FILE *fbin = fopen("evolution_multiplegrid.bin", "w");
    FILE *ftxt = fopen("evolution_multiplegrid.txt", "w");
    FILE *ttxt = fopen("time_parallel.txt", "w");

#ifdef NAUNET_DEBUG
    printf("Initialization is done. Start to evolve.\n");
    // FILE *rtxt = fopen("reactionrates.txt", "w");
    // double rates[NREACTIONS];
#endif

    //
    std::vector<double> timesteps;
    double logtstart = 2.0, logtend = 4.0, logtstep = 0.1;
    double time = 0.0;
    for (double logtime = logtstart; logtime < logtend + 0.1 * logtstep;
         logtime += logtstep) {
        double dtyr = pow(10.0, logtime) - time;
        timesteps.push_back(dtyr);
        time += dtyr;
    }
    //

    double dtyr = 0.0, curtime = 0.0;

    // write the initial abundances
    for (int isys = 0; isys < nsystem; isys++) {
        fwrite((double *)&isys, sizeof(double), 1, fbin);
        fwrite(&curtime, sizeof(double), 1, fbin);
        fwrite(&y[isys * NEQUATIONS], sizeof(double), NEQUATIONS, fbin);

        fprintf(ftxt, "%13.7e ", (double)isys);
        fprintf(ftxt, "%13.7e ", curtime);
        for (int j = 0; j < NEQUATIONS; j++) {
            fprintf(ftxt, "%13.7e ", y[isys * NEQUATIONS + j]);
        }
        fprintf(ftxt, "\n");
    }
    for (auto step = timesteps.begin(); step != timesteps.end(); step++) {
#ifdef NAUNET_DEBUG
        // EvalRates only receive one system as input, disabled in parallel test
        // EvalRates(rates, y, data);
        // for (int j = 0; j < NREACTIONS; j++) {
        //     fprintf(rtxt, "%13.7e ", rates[j]);
        // }
        // fprintf(rtxt, "\n");
#endif

        //
        //

        dtyr = *step;

        Timer timer;
        timer.start();
        naunet.Solve(y, dtyr * spy, data);
        timer.stop();

        curtime += dtyr;

        // write the abundances after each step
        for (int isys = 0; isys < nsystem; isys++) {
            fwrite((double *)&isys, sizeof(double), 1, fbin);
            fwrite(&curtime, sizeof(double), 1, fbin);
            fwrite(&y[isys * NEQUATIONS], sizeof(double), NEQUATIONS, fbin);

            fprintf(ftxt, "%13.7e ", (double)isys);
            fprintf(ftxt, "%13.7e ", curtime);
            for (int j = 0; j < NEQUATIONS; j++) {
                fprintf(ftxt, "%13.7e ", y[isys * NEQUATIONS + j]);
            }
            fprintf(ftxt, "\n");
        }

        // float duration = (float)timer.elapsed() / 1e6;
        double duration = timer.elapsed();
        fprintf(ttxt, "%8.5e \n", duration);
        printf("Time = %13.7e yr, elapsed: %8.5e sec\n", curtime, duration);
    }

    fclose(fbin);
    fclose(ftxt);
    fclose(ttxt);

#ifdef NAUNET_DEBUG
    // fclose(rtxt);
#endif

    if (naunet.Finalize() == NAUNET_FAIL) {
        printf("Finalize Fail\n");
        return 1;
    }

    delete[] data;
    delete[] y;

    return 0;
}