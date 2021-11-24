#ifndef __NAUNET_CONSTANTS_H__
#define __NAUNET_CONSTANTS_H__
// clang-format off

extern __constant__ double pi;

// atomic mass unit (g)
extern __constant__ double amu;

// mass of electron (g)
extern __constant__ double me;

// mass or electron (u)
extern __constant__ double meu;

// mass of proton (g)
extern __constant__ double mp;

// mass of neutron (g)
extern __constant__ double mn;

// mass of hydrogen (g)
extern __constant__ double mh;

// electron charge (esu)
extern __constant__ double echarge;

// Boltzmann constant (erg/K)
extern __constant__ double kerg;

extern __constant__ double gism           ;
extern __constant__ double habing         ;
extern __constant__ double zism           ;
extern __constant__ double crphot         ;
extern __constant__ double hbar           ;
extern __constant__ double eb_GCI         ;
extern __constant__ double eb_GC10I       ;
extern __constant__ double eb_GC10HI      ;
extern __constant__ double eb_GC10H2I     ;
extern __constant__ double eb_GC11I       ;
extern __constant__ double eb_GC2I        ;
extern __constant__ double eb_GC2HI       ;
extern __constant__ double eb_GC2H2I      ;
extern __constant__ double eb_GC2H3I      ;
extern __constant__ double eb_GC2H4I      ;
extern __constant__ double eb_GC2H4CNI    ;
extern __constant__ double eb_GC2H5I      ;
extern __constant__ double eb_GC2H5CNI    ;
extern __constant__ double eb_GC2H5OHI    ;
extern __constant__ double eb_GC2H6I      ;
extern __constant__ double eb_GC2NI       ;
extern __constant__ double eb_GC2OI       ;
extern __constant__ double eb_GC2SI       ;
extern __constant__ double eb_GC3I        ;
extern __constant__ double eb_GC3HI       ;
extern __constant__ double eb_GC3H2I      ;
extern __constant__ double eb_GC3NI       ;
extern __constant__ double eb_GC3OI       ;
extern __constant__ double eb_GC3PI       ;
extern __constant__ double eb_GC3SI       ;
extern __constant__ double eb_GC4I        ;
extern __constant__ double eb_GC4HI       ;
extern __constant__ double eb_GC4H2I      ;
extern __constant__ double eb_GC4H3I      ;
extern __constant__ double eb_GC4H6I      ;
extern __constant__ double eb_GC4NI       ;
extern __constant__ double eb_GC4PI       ;
extern __constant__ double eb_GC4SI       ;
extern __constant__ double eb_GC5I        ;
extern __constant__ double eb_GC5HI       ;
extern __constant__ double eb_GC5H2I      ;
extern __constant__ double eb_GC5NI       ;
extern __constant__ double eb_GC6I        ;
extern __constant__ double eb_GC6HI       ;
extern __constant__ double eb_GC6H2I      ;
extern __constant__ double eb_GC6H6I      ;
extern __constant__ double eb_GC7I        ;
extern __constant__ double eb_GC7HI       ;
extern __constant__ double eb_GC7H2I      ;
extern __constant__ double eb_GC7NI       ;
extern __constant__ double eb_GC8I        ;
extern __constant__ double eb_GC8HI       ;
extern __constant__ double eb_GC8H2I      ;
extern __constant__ double eb_GC9I        ;
extern __constant__ double eb_GC9HI       ;
extern __constant__ double eb_GC9H2I      ;
extern __constant__ double eb_GC9NI       ;
extern __constant__ double eb_GCCPI       ;
extern __constant__ double eb_GCClI       ;
extern __constant__ double eb_GCHI        ;
extern __constant__ double eb_GCH2I       ;
extern __constant__ double eb_GCH2CCHI    ;
extern __constant__ double eb_GCH2CCH2I   ;
extern __constant__ double eb_GCH2CHCCHI  ;
extern __constant__ double eb_GCH2CHCNI   ;
extern __constant__ double eb_GCH2CNI     ;
extern __constant__ double eb_GCH2COI     ;
extern __constant__ double eb_GCH2NHI     ;
extern __constant__ double eb_GCH2OHI     ;
extern __constant__ double eb_GCH2OHCHOI  ;
extern __constant__ double eb_GCH2OHCOI   ;
extern __constant__ double eb_GCH2PHI     ;
extern __constant__ double eb_GCH3I       ;
extern __constant__ double eb_GCH3C3NI    ;
extern __constant__ double eb_GCH3C4HI    ;
extern __constant__ double eb_GCH3C5NI    ;
extern __constant__ double eb_GCH3C6HI    ;
extern __constant__ double eb_GCH3C7NI    ;
extern __constant__ double eb_GCH3CCHI    ;
extern __constant__ double eb_GCH3CHCH2I  ;
extern __constant__ double eb_GCH3CHOI    ;
extern __constant__ double eb_GCH3CNI     ;
extern __constant__ double eb_GCH3COI     ;
extern __constant__ double eb_GCH3COCH3I  ;
extern __constant__ double eb_GCH3COOHI   ;
extern __constant__ double eb_GCH3OI      ;
extern __constant__ double eb_GCH3OCH3I   ;
extern __constant__ double eb_GCH3OHI     ;
extern __constant__ double eb_GCH4I       ;
extern __constant__ double eb_GCNI        ;
extern __constant__ double eb_GCNOI       ;
extern __constant__ double eb_GCOI        ;
extern __constant__ double eb_GCO2I       ;
extern __constant__ double eb_GCOOCH3I    ;
extern __constant__ double eb_GCOOHI      ;
extern __constant__ double eb_GCPI        ;
extern __constant__ double eb_GCSI        ;
extern __constant__ double eb_GClI        ;
extern __constant__ double eb_GClOI       ;
extern __constant__ double eb_GFI         ;
extern __constant__ double eb_GFeI        ;
extern __constant__ double eb_GHI         ;
extern __constant__ double eb_GH2I        ;
extern __constant__ double eb_GH2CCCI     ;
extern __constant__ double eb_GH2CNI      ;
extern __constant__ double eb_GH2COI      ;
extern __constant__ double eb_GH2CSI      ;
extern __constant__ double eb_GH2OI       ;
extern __constant__ double eb_GH2O2I      ;
extern __constant__ double eb_GH2SI       ;
extern __constant__ double eb_GH2S2I      ;
extern __constant__ double eb_GH2SiOI     ;
extern __constant__ double eb_GHC2OI      ;
extern __constant__ double eb_GHC2PI      ;
extern __constant__ double eb_GHC3NI      ;
extern __constant__ double eb_GHC5NI      ;
extern __constant__ double eb_GHC7NI      ;
extern __constant__ double eb_GHC9NI      ;
extern __constant__ double eb_GHCCNI      ;
extern __constant__ double eb_GHCNI       ;
extern __constant__ double eb_GHCNOI      ;
extern __constant__ double eb_GHCOI       ;
extern __constant__ double eb_GHCOOCH3I   ;
extern __constant__ double eb_GHCOOHI     ;
extern __constant__ double eb_GHCPI       ;
extern __constant__ double eb_GHCSI       ;
extern __constant__ double eb_GHCSiI      ;
extern __constant__ double eb_GHClI       ;
extern __constant__ double eb_GHFI        ;
extern __constant__ double eb_GHNCI       ;
extern __constant__ double eb_GHNC3I      ;
extern __constant__ double eb_GHNCOI      ;
extern __constant__ double eb_GHNOI       ;
extern __constant__ double eb_GHNSiI      ;
extern __constant__ double eb_GHOCNI      ;
extern __constant__ double eb_GHONCI      ;
extern __constant__ double eb_GHPOI       ;
extern __constant__ double eb_GHSI        ;
extern __constant__ double eb_GHS2I       ;
extern __constant__ double eb_GHeI        ;
extern __constant__ double eb_GMgI        ;
extern __constant__ double eb_GNI         ;
extern __constant__ double eb_GN2I        ;
extern __constant__ double eb_GN2OI       ;
extern __constant__ double eb_GNCCNI      ;
extern __constant__ double eb_GNHI        ;
extern __constant__ double eb_GNH2I       ;
extern __constant__ double eb_GNH2CNI     ;
extern __constant__ double eb_GNH3I       ;
extern __constant__ double eb_GNOI        ;
extern __constant__ double eb_GNO2I       ;
extern __constant__ double eb_GNSI        ;
extern __constant__ double eb_GNaI        ;
extern __constant__ double eb_GOI         ;
extern __constant__ double eb_GO2I        ;
extern __constant__ double eb_GO2HI       ;
extern __constant__ double eb_GOCNI       ;
extern __constant__ double eb_GOCSI       ;
extern __constant__ double eb_GOHI        ;
extern __constant__ double eb_GPI         ;
extern __constant__ double eb_GPHI        ;
extern __constant__ double eb_GPH2I       ;
extern __constant__ double eb_GPNI        ;
extern __constant__ double eb_GPOI        ;
extern __constant__ double eb_GSI         ;
extern __constant__ double eb_GS2I        ;
extern __constant__ double eb_GSOI        ;
extern __constant__ double eb_GSO2I       ;
extern __constant__ double eb_GSiI        ;
extern __constant__ double eb_GSiCI       ;
extern __constant__ double eb_GSiC2I      ;
extern __constant__ double eb_GSiC2HI     ;
extern __constant__ double eb_GSiC2H2I    ;
extern __constant__ double eb_GSiC3I      ;
extern __constant__ double eb_GSiC3HI     ;
extern __constant__ double eb_GSiC4I      ;
extern __constant__ double eb_GSiCH2I     ;
extern __constant__ double eb_GSiCH3I     ;
extern __constant__ double eb_GSiHI       ;
extern __constant__ double eb_GSiH2I      ;
extern __constant__ double eb_GSiH3I      ;
extern __constant__ double eb_GSiH4I      ;
extern __constant__ double eb_GSiNI       ;
extern __constant__ double eb_GSiNCI      ;
extern __constant__ double eb_GSiOI       ;
extern __constant__ double eb_GSiO2I      ;
extern __constant__ double eb_GSiSI       ;

// H2 column density
extern __device__ double H2ShieldingTableX[105];
// H2 shielding factor
extern __device__ double H2ShieldingTable[105];

// Excitation temperature
extern __device__ double COShieldingTableX[5];
// H2 column density
extern __device__ double COShieldingTableY[41];
// CO column density
extern __device__ double COShieldingTableZ[46];
// CO shielding factor
extern __device__ double COShieldingTable[5][41][46];

// Excitation temperature
extern __device__ double N2ShieldingTableX[5];
// H2 column density
extern __device__ double N2ShieldingTableY[46];
// N2 column density
extern __device__ double N2ShieldingTableZ[46];
// N2 shielding factor
extern __device__ double N2ShieldingTable[5][46][46];


#endif