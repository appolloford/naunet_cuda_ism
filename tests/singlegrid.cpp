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
    double spy = 86400.0 * 365.0;

    NaunetData data;
    //
    double nH      = 2e4;
    double zeta_cr = 1.3e-17;
    double zeta_xr = 0.0;
    double Tgas    = 10.0;
    double Tdust   = 10.0;
    double Av      = 10.0;
    double G0      = 4.0;
    double rG      = 1e-5;
    double omega   = 0.5;
    double barr    = 1.5e-8;
    double sites   = 1.5e15;
    double hop     = 0.3;
    double nMono   = 2.0;
    double duty    = 3.16e-19;
    double Tcr     = 70.0;
    double branch  = 1e-2;

    data.nH      = nH;
    data.zeta_cr = zeta_cr;
    data.zeta_xr = zeta_xr;
    data.Tgas    = Tgas;
    data.Tdust   = Tdust;
    data.Av      = Av;
    data.G0      = G0;
    data.rG      = rG;
    data.omega   = omega;
    data.barr    = barr;
    data.sites   = sites;
    data.hop     = hop;
    data.nMono   = nMono;
    data.duty    = duty;
    data.Tcr     = Tcr;
    data.branch  = branch;


    Naunet naunet;
    if (naunet.Init() == NAUNET_FAIL) {
        printf("Initialize Fail\n");
        return 1;
    }

#ifdef USE_CUDA
    if (naunet.Reset(1) == NAUNET_FAIL) {
        throw std::runtime_error("Fail to reset the number of systems");
    }
#endif

    //
    double y[NEQUATIONS] = {0.0};
    // for (int i = 0; i < NEQUATIONS; i++)
    // {
    //     y[i] = 1.e-40;
    // }
    y[IDX_H2I]           = 0.5 * nH;
    y[IDX_HI]            = 5.0e-5 * nH;
    y[IDX_HeI]           = 9.75e-2 * nH;
    y[IDX_NI]            = 7.5e-5 * nH;
    y[IDX_OI]            = 3.2e-4 * nH;
    y[IDX_CI]            = 1.4e-4 * nH;
    y[IDX_SI]            = 8.0e-8 * nH;
    y[IDX_SiI]           = 8.0e-9 * nH;
    y[IDX_NaI]           = 2.0e-9 * nH;
    y[IDX_MgI]           = 7.0e-9 * nH;
    y[IDX_FeI]           = 3.0e-9 * nH;
    y[IDX_ClI]           = 4.0e-9 * nH;
    y[IDX_FI]            = 2.0e-8 * nH;
    y[IDX_GRAIN0I]       = 1.3e-12 * nH;


    FILE *fbin = fopen("evolution_singlegrid.bin", "w");
    FILE *ftxt = fopen("evolution_singlegrid.txt", "w");
    FILE *ttxt = fopen("time_singlegrid.txt", "w");
#ifdef NAUNET_DEBUG
    printf("Initialization is done. Start to evolve.\n");
    FILE *rtxt               = fopen("reactionrates.txt", "w");
    double rates[NREACTIONS] = {0.0};
#endif

    //
    std::vector<double> timesteps;
    double logtstart = 3.0, logtend = 7.0, logtstep = 0.1;
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
    fwrite(&curtime, sizeof(double), 1, fbin);
    fwrite(y, sizeof(double), NEQUATIONS, fbin);

    fprintf(ftxt, "%13.7e ", curtime);
    for (int j = 0; j < NEQUATIONS; j++) {
        fprintf(ftxt, "%13.7e ", y[j]);
    }
    fprintf(ftxt, "\n");

    for (auto step = timesteps.begin(); step != timesteps.end(); step++) {
#ifdef NAUNET_DEBUG
        EvalRates(rates, y, &data);
        for (int j = 0; j < NREACTIONS; j++) {
            fprintf(rtxt, "%13.7e ", rates[j]);
        }
        fprintf(rtxt, "\n");
#endif

        //
        //

        dtyr = *step;

        Timer timer;
        timer.start();
        naunet.Solve(y, dtyr * spy, &data);
        timer.stop();

        curtime += dtyr;

        // write the abundances after each step
        fwrite(&curtime, sizeof(double), 1, fbin);
        fwrite(y, sizeof(double), NEQUATIONS, fbin);

        fprintf(ftxt, "%13.7e ", curtime);
        for (int j = 0; j < NEQUATIONS; j++) {
            fprintf(ftxt, "%13.7e ", y[j]);
        }
        fprintf(ftxt, "\n");

        // float duration = (float)timer.elapsed() / 1e6;
        double duration = timer.elapsed();
        fprintf(ttxt, "%8.5e \n", duration);
        printf("Time = %13.7e yr, elapsed: %8.5e sec\n", curtime, duration);
    }

    fclose(fbin);
    fclose(ftxt);
    fclose(ttxt);
#ifdef NAUNET_DEBUG
    fclose(rtxt);
#endif

    if (naunet.Finalize() == NAUNET_FAIL) {
        printf("Finalize Fail\n");
        return 1;
    }

    return 0;
}