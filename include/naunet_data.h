#ifndef __NAUNET_DATA_H__
#define __NAUNET_DATA_H__

// Struct for holding the nessesary additional variables for the problem.
struct NaunetData {
    // clang-format off
    double rG;
    double barr;
    double sites;
    double hop;
    double nMono;
    double duty;
    double Tcr;
    double branch;
    double nH;
    double zeta_cr;
    double zeta_xr;
    double Tgas;
    double Tdust;
    double Av;
    double G0;
    double omega;
    
    // clang-format on
};

#endif