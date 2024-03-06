// 
#include <math.h>
#include <stdio.h>

#include <algorithm>

#include "naunet_constants.h"
#include "naunet_macros.h"
#include "naunet_physics.h"
#include "naunet_utilities.h"

/*  */
// clang-format off
__device__ __host__ double GetElementAbund(double *y, int elemidx) {
    if (elemidx == IDX_ELEM_F) {
        return 1.0*y[IDX_GHFI] + 1.0*y[IDX_GFI] + 1.0*y[IDX_CFII] + 1.0*y[IDX_HFII] + 
               1.0*y[IDX_SiFII] + 1.0*y[IDX_H2FII] + 1.0*y[IDX_FII] + 1.0*y[IDX_FI] + 
               1.0*y[IDX_HFI] + 0.0;
    }
    if (elemidx == IDX_ELEM_Cl) {
        return 1.0*y[IDX_GCClI] + 1.0*y[IDX_GClOI] + 1.0*y[IDX_GHClI] + 1.0*y[IDX_ClOII] + 
               1.0*y[IDX_GClI] + 1.0*y[IDX_ClOI] + 1.0*y[IDX_H2CClII] + 1.0*y[IDX_HClII] + 
               1.0*y[IDX_CClII] + 1.0*y[IDX_CClI] + 1.0*y[IDX_ClII] + 1.0*y[IDX_H2ClII] + 
               1.0*y[IDX_ClI] + 1.0*y[IDX_HClI] + 0.0;
    }
    if (elemidx == IDX_ELEM_P) {
        return 1.0*y[IDX_GHC2PI] + 1.0*y[IDX_GHPOI] + 1.0*y[IDX_GPNI] + 1.0*y[IDX_GC4PI] + 
               1.0*y[IDX_GCH2PHI] + 1.0*y[IDX_GHCPI] + 1.0*y[IDX_GPH2I] + 1.0*y[IDX_GPOI] + 
               1.0*y[IDX_GC3PI] + 1.0*y[IDX_GPHI] + 1.0*y[IDX_GCCPI] + 1.0*y[IDX_C4PII] + 
               1.0*y[IDX_GCPI] + 1.0*y[IDX_PC2H4II] + 1.0*y[IDX_PC2H3II] + 1.0*y[IDX_PNII] + 
               1.0*y[IDX_PNH3II] + 1.0*y[IDX_GPI] + 1.0*y[IDX_PCH3II] + 1.0*y[IDX_PNH2II] + 
               1.0*y[IDX_HPNII] + 1.0*y[IDX_PC3HII] + 1.0*y[IDX_PCH4II] + 1.0*y[IDX_CCPII] + 
               1.0*y[IDX_H2POII] + 1.0*y[IDX_PC2H2II] + 1.0*y[IDX_PC4HII] + 1.0*y[IDX_PH3II] + 
               1.0*y[IDX_CPII] + 1.0*y[IDX_HC2PII] + 1.0*y[IDX_PCH2II] + 1.0*y[IDX_HCPII] + 
               1.0*y[IDX_HPOII] + 1.0*y[IDX_CH2PHI] + 1.0*y[IDX_HPOI] + 1.0*y[IDX_POII] + 
               1.0*y[IDX_PH2I] + 1.0*y[IDX_PNI] + 1.0*y[IDX_C4PI] + 1.0*y[IDX_HCPI] + 
               1.0*y[IDX_C3PI] + 1.0*y[IDX_PH2II] + 1.0*y[IDX_POI] + 1.0*y[IDX_HC2PI] + 
               1.0*y[IDX_CCPI] + 1.0*y[IDX_CPI] + 1.0*y[IDX_PHI] + 1.0*y[IDX_PHII] + 
               1.0*y[IDX_PII] + 1.0*y[IDX_PI] + 0.0;
    }
    if (elemidx == IDX_ELEM_Fe) {
        return 1.0*y[IDX_GFeI] + 1.0*y[IDX_FeII] + 1.0*y[IDX_FeI] + 0.0;
    }
    if (elemidx == IDX_ELEM_Mg) {
        return 1.0*y[IDX_GMgI] + 1.0*y[IDX_MgII] + 1.0*y[IDX_MgI] + 0.0;
    }
    if (elemidx == IDX_ELEM_Na) {
        return 1.0*y[IDX_GNaI] + 1.0*y[IDX_NaII] + 1.0*y[IDX_NaI] + 0.0;
    }
    if (elemidx == IDX_ELEM_Si) {
        return 1.0*y[IDX_GHNSiI] + 1.0*y[IDX_GSiC3HI] + 1.0*y[IDX_GSiCH3I] + 1.0*y[IDX_GSiO2I] + 
               1.0*y[IDX_GSiSI] + 1.0*y[IDX_GH2SiOI] + 1.0*y[IDX_GSiC2HI] + 1.0*y[IDX_GSiC2H2I] + 
               1.0*y[IDX_GSiC4I] + 1.0*y[IDX_GSiNCI] + 1.0*y[IDX_GHCSiI] + 1.0*y[IDX_GSiH4I] + 
               1.0*y[IDX_GSiC3I] + 1.0*y[IDX_GSiH3I] + 1.0*y[IDX_GSiNI] + 1.0*y[IDX_GSiOI] + 
               1.0*y[IDX_GSiCH2I] + 1.0*y[IDX_GSiHI] + 1.0*y[IDX_GSiH2I] + 1.0*y[IDX_SiFII] + 
               1.0*y[IDX_GSiCI] + 1.0*y[IDX_GSiC2I] + 1.0*y[IDX_HSiO2II] + 1.0*y[IDX_H2SiOII] + 
               1.0*y[IDX_H3SiOII] + 1.0*y[IDX_SiH4II] + 1.0*y[IDX_SiO2I] + 1.0*y[IDX_SiC3H2II] + 
               1.0*y[IDX_SiC4II] + 1.0*y[IDX_SiC4HII] + 1.0*y[IDX_SiH5II] + 1.0*y[IDX_SiNCHII] + 
               1.0*y[IDX_HNSiII] + 1.0*y[IDX_SiCH4II] + 1.0*y[IDX_SiC2H3II] + 1.0*y[IDX_SiC3HI] + 
               1.0*y[IDX_SiCH3II] + 1.0*y[IDX_SiNH2II] + 1.0*y[IDX_HSiSII] + 1.0*y[IDX_SiNII] + 
               1.0*y[IDX_H2SiOI] + 1.0*y[IDX_SiC4I] + 1.0*y[IDX_SiNCI] + 1.0*y[IDX_SiNCII] + 
               1.0*y[IDX_HNSiI] + 1.0*y[IDX_SiC2H2I] + 1.0*y[IDX_SiC3II] + 1.0*y[IDX_GSiI] + 
               1.0*y[IDX_SiC3HII] + 1.0*y[IDX_SiCH3I] + 1.0*y[IDX_SiC2II] + 1.0*y[IDX_SiC2H2II] + 
               1.0*y[IDX_SiC2HI] + 1.0*y[IDX_SiC3I] + 1.0*y[IDX_SiCII] + 1.0*y[IDX_HCSiI] + 
               1.0*y[IDX_SiCH2I] + 1.0*y[IDX_HCSiII] + 1.0*y[IDX_SiC2I] + 1.0*y[IDX_SiNI] + 
               1.0*y[IDX_SiC2HII] + 1.0*y[IDX_SiH2I] + 1.0*y[IDX_SiH2II] + 1.0*y[IDX_SiH3II] + 
               1.0*y[IDX_SiCI] + 1.0*y[IDX_SiH3I] + 1.0*y[IDX_SiCH2II] + 1.0*y[IDX_SiHII] + 
               1.0*y[IDX_SiHI] + 1.0*y[IDX_SiH4I] + 1.0*y[IDX_SiSII] + 1.0*y[IDX_SiSI] + 
               1.0*y[IDX_SiOII] + 1.0*y[IDX_SiOHII] + 1.0*y[IDX_SiOI] + 1.0*y[IDX_SiII] + 
               1.0*y[IDX_SiI] + 0.0;
    }
    if (elemidx == IDX_ELEM_S) {
        return 2.0*y[IDX_GH2S2I] + 1.0*y[IDX_GC4SI] + 1.0*y[IDX_GSiSI] + 2.0*y[IDX_GS2I] + 
               1.0*y[IDX_GC3SI] + 1.0*y[IDX_GH2SI] + 2.0*y[IDX_GHS2I] + 1.0*y[IDX_GSO2I] + 
               1.0*y[IDX_GC2SI] + 1.0*y[IDX_GH2CSI] + 1.0*y[IDX_GHCSI] + 1.0*y[IDX_GOCSI] + 
               1.0*y[IDX_GSOI] + 1.0*y[IDX_GNSI] + 1.0*y[IDX_HNSII] + 1.0*y[IDX_HSOII] + 
               1.0*y[IDX_CH3CSII] + 2.0*y[IDX_H3S2II] + 1.0*y[IDX_C2SII] + 1.0*y[IDX_C3SII] + 
               1.0*y[IDX_HSO2II] + 1.0*y[IDX_GCSI] + 1.0*y[IDX_GHSI] + 2.0*y[IDX_H2S2I] + 
               1.0*y[IDX_HOCSII] + 1.0*y[IDX_NSII] + 1.0*y[IDX_HC4SII] + 2.0*y[IDX_H2S2II] + 
               1.0*y[IDX_HSiSII] + 1.0*y[IDX_SO2II] + 1.0*y[IDX_H3CSII] + 1.0*y[IDX_HC3SII] + 
               2.0*y[IDX_HS2II] + 2.0*y[IDX_HS2I] + 1.0*y[IDX_H2CSII] + 1.0*y[IDX_H2CSI] + 
               2.0*y[IDX_S2I] + 2.0*y[IDX_S2II] + 1.0*y[IDX_C3SI] + 1.0*y[IDX_OCSII] + 
               1.0*y[IDX_GSI] + 1.0*y[IDX_HCSI] + 1.0*y[IDX_NSI] + 1.0*y[IDX_SO2I] + 
               1.0*y[IDX_HCSII] + 1.0*y[IDX_CSII] + 1.0*y[IDX_H3SII] + 1.0*y[IDX_C4SII] + 
               1.0*y[IDX_HSII] + 1.0*y[IDX_OCSI] + 1.0*y[IDX_SM] + 1.0*y[IDX_SiSII] + 
               1.0*y[IDX_HC2SII] + 1.0*y[IDX_C4SI] + 1.0*y[IDX_HSI] + 1.0*y[IDX_CSI] + 
               1.0*y[IDX_SiSI] + 1.0*y[IDX_C2SI] + 1.0*y[IDX_SOI] + 1.0*y[IDX_SOII] + 
               1.0*y[IDX_H2SII] + 1.0*y[IDX_H2SI] + 1.0*y[IDX_SII] + 1.0*y[IDX_SI] + 0.0;
    }
    if (elemidx == IDX_ELEM_N) {
        return 1.0*y[IDX_GCH3C3NI] + 1.0*y[IDX_GCH3C5NI] + 1.0*y[IDX_GCH3C7NI] + 1.0*y[IDX_GHNSiI] + 
               2.0*y[IDX_GNH2CNI] + 1.0*y[IDX_GNO2I] + 1.0*y[IDX_GPNI] + 2.0*y[IDX_GNCCNI] + 
               1.0*y[IDX_C2H4CNI] + 1.0*y[IDX_GC4NI] + 1.0*y[IDX_GHCNOI] + 1.0*y[IDX_GHNC3I] + 
               1.0*y[IDX_GHNCOI] + 1.0*y[IDX_GHOCNI] + 1.0*y[IDX_GHONCI] + 2.0*y[IDX_GN2OI] + 
               1.0*y[IDX_GSiNCI] + 2.0*y[IDX_NCCNCH3II] + 1.0*y[IDX_GC2H4CNI] + 1.0*y[IDX_GCH2CNI] + 
               1.0*y[IDX_GCH3CNI] + 1.0*y[IDX_GH2CNI] + 1.0*y[IDX_GHC5NI] + 1.0*y[IDX_GHC7NI] + 
               1.0*y[IDX_GHC9NI] + 1.0*y[IDX_GHCCNI] + 1.0*y[IDX_GSiNI] + 1.0*y[IDX_HCCNI] + 
               1.0*y[IDX_CH3C3NII] + 1.0*y[IDX_GCH2CHCNI] + 1.0*y[IDX_GHC3NI] + 1.0*y[IDX_C5NII] + 
               1.0*y[IDX_CH2CHCNII] + 1.0*y[IDX_GC2H5CNI] + 1.0*y[IDX_GC9NI] + 1.0*y[IDX_GOCNI] + 
               1.0*y[IDX_H2CNOII] + 1.0*y[IDX_H2NCOII] + 1.0*y[IDX_H2OCNII] + 1.0*y[IDX_C7NII] + 
               1.0*y[IDX_C9NII] + 1.0*y[IDX_CH3NHII] + 1.0*y[IDX_GC2NI] + 1.0*y[IDX_GCH2NHI] + 
               1.0*y[IDX_GCNOI] + 1.0*y[IDX_GHNCI] + 1.0*y[IDX_GNSI] + 1.0*y[IDX_H2C4NII] + 
               1.0*y[IDX_H2NOII] + 1.0*y[IDX_HNSII] + 2.0*y[IDX_NH2CNHII] + 1.0*y[IDX_GHNOI] + 
               2.0*y[IDX_GN2I] + 1.0*y[IDX_HC4NII] + 1.0*y[IDX_HCNOHII] + 1.0*y[IDX_HNCOHII] + 
               1.0*y[IDX_HOCNII] + 1.0*y[IDX_PNII] + 1.0*y[IDX_PNH3II] + 1.0*y[IDX_GC5NI] + 
               1.0*y[IDX_H2C7NII] + 1.0*y[IDX_H2C9NII] + 1.0*y[IDX_HONCII] + 1.0*y[IDX_PNH2II] + 
               1.0*y[IDX_C2H5CNHII] + 1.0*y[IDX_CH2NH2II] + 1.0*y[IDX_CH3C5NHII] + 1.0*y[IDX_CH3C7NHII] + 
               1.0*y[IDX_GC3NI] + 1.0*y[IDX_GC7NI] + 1.0*y[IDX_GHCNI] + 1.0*y[IDX_H2CNI] + 
               1.0*y[IDX_H3C5NII] + 1.0*y[IDX_HCNOII] + 1.0*y[IDX_GNH3I] + 1.0*y[IDX_H3C9NII] + 
               1.0*y[IDX_HPNII] + 1.0*y[IDX_C4NI] + 1.0*y[IDX_H3C7NII] + 2.0*y[IDX_NH2CNI] + 
               1.0*y[IDX_CH3C3NHII] + 1.0*y[IDX_HC7NII] + 1.0*y[IDX_HC9NII] + 1.0*y[IDX_HNCOII] + 
               1.0*y[IDX_NO2II] + 1.0*y[IDX_C2H5CNI] + 1.0*y[IDX_CH2CHCNHII] + 2.0*y[IDX_HN2OII] + 
               2.0*y[IDX_N2OII] + 1.0*y[IDX_SiNCHII] + 1.0*y[IDX_HNSiII] + 1.0*y[IDX_NSII] + 
               1.0*y[IDX_OCNII] + 2.0*y[IDX_C2N2II] + 1.0*y[IDX_CH3CNII] + 1.0*y[IDX_HC5NHII] + 
               1.0*y[IDX_SiNH2II] + 1.0*y[IDX_C3NII] + 1.0*y[IDX_CH2CNII] + 1.0*y[IDX_HC5NII] + 
               1.0*y[IDX_HONCI] + 1.0*y[IDX_SiNII] + 1.0*y[IDX_CH3C5NI] + 1.0*y[IDX_CH3C7NI] + 
               1.0*y[IDX_H2NCII] + 1.0*y[IDX_SiNCI] + 1.0*y[IDX_SiNCII] + 1.0*y[IDX_CNOI] + 
               1.0*y[IDX_HNSiI] + 1.0*y[IDX_C2NHII] + 1.0*y[IDX_HCNOI] + 1.0*y[IDX_C4NII] + 
               1.0*y[IDX_CH3C3NI] + 1.0*y[IDX_HOCNI] + 1.0*y[IDX_GNOI] + 1.0*y[IDX_HC9NI] + 
               2.0*y[IDX_NCCNHII] + 1.0*y[IDX_NO2I] + 1.0*y[IDX_PNI] + 1.0*y[IDX_CH2CNI] + 
               1.0*y[IDX_HNCOI] + 1.0*y[IDX_C9NI] + 1.0*y[IDX_GNH2I] + 1.0*y[IDX_CH2NHI] + 
               1.0*y[IDX_SiNI] + 1.0*y[IDX_GNHI] + 1.0*y[IDX_NSI] + 1.0*y[IDX_CH2CHCNI] + 
               1.0*y[IDX_CH3CNHII] + 1.0*y[IDX_HC7NI] + 1.0*y[IDX_C7NI] + 1.0*y[IDX_HNOI] + 
               2.0*y[IDX_N2OI] + 1.0*y[IDX_C2NII] + 1.0*y[IDX_HNC3I] + 1.0*y[IDX_OCNI] + 
               1.0*y[IDX_HC3NII] + 1.0*y[IDX_HC5NI] + 1.0*y[IDX_HC3NHII] + 1.0*y[IDX_CH3CNI] + 
               2.0*y[IDX_NCCNI] + 1.0*y[IDX_HNOII] + 1.0*y[IDX_GCNI] + 1.0*y[IDX_CNII] + 
               1.0*y[IDX_GNI] + 2.0*y[IDX_N2II] + 1.0*y[IDX_HCNII] + 1.0*y[IDX_CNCII] + 
               1.0*y[IDX_NHII] + 1.0*y[IDX_NH2II] + 1.0*y[IDX_C5NM] + 1.0*y[IDX_HC3NI] + 
               1.0*y[IDX_C2NI] + 1.0*y[IDX_NH2I] + 1.0*y[IDX_C5NI] + 1.0*y[IDX_C3NM] + 
               1.0*y[IDX_C3NI] + 1.0*y[IDX_NHI] + 1.0*y[IDX_CNM] + 2.0*y[IDX_N2HII] + 
               1.0*y[IDX_NII] + 1.0*y[IDX_NOII] + 1.0*y[IDX_HCNHII] + 1.0*y[IDX_HNCI] + 
               1.0*y[IDX_NH3II] + 1.0*y[IDX_NH4II] + 2.0*y[IDX_N2I] + 1.0*y[IDX_NOI] + 
               1.0*y[IDX_HCNI] + 1.0*y[IDX_NH3I] + 1.0*y[IDX_CNI] + 1.0*y[IDX_NI] + 0.0;
    }
    if (elemidx == IDX_ELEM_O) {
        return 1.0*y[IDX_GClOI] + 1.0*y[IDX_GHPOI] + 2.0*y[IDX_GNO2I] + 2.0*y[IDX_GSiO2I] + 
               1.0*y[IDX_GH2SiOI] + 2.0*y[IDX_GCH3COOHI] + 1.0*y[IDX_GHCNOI] + 1.0*y[IDX_GHNCOI] + 
               1.0*y[IDX_GHOCNI] + 1.0*y[IDX_GHONCI] + 1.0*y[IDX_GN2OI] + 1.0*y[IDX_GPOI] + 
               1.0*y[IDX_HC2OI] + 2.0*y[IDX_GH2O2I] + 2.0*y[IDX_GSO2I] + 1.0*y[IDX_GC3OI] + 
               1.0*y[IDX_GHC2OI] + 1.0*y[IDX_GSiOI] + 1.0*y[IDX_C3H2OII] + 1.0*y[IDX_ClOII] + 
               1.0*y[IDX_GCH3COCH3I] + 1.0*y[IDX_GCH3OCH3I] + 1.0*y[IDX_H3C3OII] + 1.0*y[IDX_GOCSI] + 
               2.0*y[IDX_COOCH3II] + 1.0*y[IDX_GC2H5OHI] + 1.0*y[IDX_GC2OI] + 2.0*y[IDX_GCO2I] + 
               2.0*y[IDX_GO2HI] + 1.0*y[IDX_GOCNI] + 1.0*y[IDX_GSOI] + 1.0*y[IDX_H2CNOII] + 
               1.0*y[IDX_H2NCOII] + 1.0*y[IDX_H2OCNII] + 1.0*y[IDX_ClOI] + 1.0*y[IDX_GCNOI] + 
               1.0*y[IDX_H2NOII] + 1.0*y[IDX_HOCII] + 1.0*y[IDX_HSOII] + 2.0*y[IDX_HSiO2II] + 
               1.0*y[IDX_C2H5OHII] + 1.0*y[IDX_C2OII] + 2.0*y[IDX_CH2OHCOII] + 1.0*y[IDX_GCH2COI] + 
               1.0*y[IDX_GHNOI] + 1.0*y[IDX_HCNOHII] + 1.0*y[IDX_HNCOHII] + 1.0*y[IDX_HOCNII] + 
               2.0*y[IDX_H2O2I] + 1.0*y[IDX_HONCII] + 2.0*y[IDX_CH2OHCH2OII] + 2.0*y[IDX_GCOOCH3I] + 
               1.0*y[IDX_H2SiOII] + 1.0*y[IDX_HCNOII] + 2.0*y[IDX_CH3COOHII] + 2.0*y[IDX_H5C2O2II] + 
               2.0*y[IDX_HCOOHII] + 2.0*y[IDX_CH3COOH2II] + 1.0*y[IDX_CH3OCH3II] + 2.0*y[IDX_GCH2OHCOI] + 
               2.0*y[IDX_GCOOHI] + 2.0*y[IDX_GO2I] + 1.0*y[IDX_H3SiOII] + 2.0*y[IDX_HSO2II] + 
               1.0*y[IDX_CH3OCH4II] + 1.0*y[IDX_HNCOII] + 2.0*y[IDX_NO2II] + 2.0*y[IDX_SiO2I] + 
               1.0*y[IDX_C2H5OH2II] + 2.0*y[IDX_COOCH3I] + 1.0*y[IDX_H2POII] + 2.0*y[IDX_HCOOCH3II] + 
               1.0*y[IDX_HN2OII] + 1.0*y[IDX_N2OII] + 1.0*y[IDX_C3OII] + 2.0*y[IDX_CH2OHCHOII] + 
               1.0*y[IDX_CH3CHOII] + 2.0*y[IDX_GCH2OHCHOI] + 2.0*y[IDX_GHCOOHI] + 1.0*y[IDX_HOCSII] + 
               1.0*y[IDX_OCNII] + 1.0*y[IDX_GCH3COI] + 2.0*y[IDX_GHCOOCH3I] + 1.0*y[IDX_CH3COCH3II] + 
               1.0*y[IDX_GCH2OHI] + 1.0*y[IDX_GCH3OHI] + 1.0*y[IDX_HONCI] + 2.0*y[IDX_SO2II] + 
               2.0*y[IDX_CH2OHCOI] + 1.0*y[IDX_H2SiOI] + 1.0*y[IDX_HC2OII] + 1.0*y[IDX_CNOI] + 
               1.0*y[IDX_GCH3OI] + 1.0*y[IDX_GH2OI] + 2.0*y[IDX_HCOOH2II] + 1.0*y[IDX_HPOII] + 
               2.0*y[IDX_CH2OHCHOI] + 1.0*y[IDX_HC3OII] + 1.0*y[IDX_HCNOI] + 1.0*y[IDX_CH2COII] + 
               1.0*y[IDX_GCH3CHOI] + 1.0*y[IDX_GH2COI] + 1.0*y[IDX_HOCNI] + 1.0*y[IDX_HPOI] + 
               1.0*y[IDX_CH3COI] + 1.0*y[IDX_CH3COCH4II] + 1.0*y[IDX_CH3OHII] + 2.0*y[IDX_COOHI] + 
               1.0*y[IDX_GNOI] + 1.0*y[IDX_POII] + 2.0*y[IDX_HCOOCH3I] + 2.0*y[IDX_NO2I] + 
               1.0*y[IDX_HNCOI] + 2.0*y[IDX_CH3COOHI] + 1.0*y[IDX_CH3OCH3I] + 1.0*y[IDX_POI] + 
               1.0*y[IDX_CH3OI] + 1.0*y[IDX_GHCOI] + 1.0*y[IDX_OCSII] + 1.0*y[IDX_C2OI] + 
               1.0*y[IDX_CH2OHI] + 1.0*y[IDX_C3OI] + 1.0*y[IDX_CH3CHOHII] + 1.0*y[IDX_CH3COII] + 
               2.0*y[IDX_HCOOHI] + 2.0*y[IDX_O2HI] + 2.0*y[IDX_SO2I] + 1.0*y[IDX_C2H5OHI] + 
               1.0*y[IDX_HNOI] + 1.0*y[IDX_N2OI] + 1.0*y[IDX_CH3OH2II] + 1.0*y[IDX_CH2COI] + 
               1.0*y[IDX_OCNI] + 1.0*y[IDX_CH3COCH3I] + 2.0*y[IDX_CO2II] + 1.0*y[IDX_GOHI] + 
               1.0*y[IDX_GCOI] + 2.0*y[IDX_HCO2II] + 1.0*y[IDX_CH3CHOI] + 2.0*y[IDX_O2HII] + 
               1.0*y[IDX_HNOII] + 1.0*y[IDX_GOI] + 2.0*y[IDX_O2M] + 1.0*y[IDX_OCSI] + 
               1.0*y[IDX_COII] + 1.0*y[IDX_OHM] + 1.0*y[IDX_OM] + 1.0*y[IDX_CH3OHI] + 
               1.0*y[IDX_H3COII] + 1.0*y[IDX_SiOII] + 1.0*y[IDX_SiOHII] + 1.0*y[IDX_OHII] + 
               1.0*y[IDX_H2OII] + 2.0*y[IDX_CO2I] + 2.0*y[IDX_O2II] + 1.0*y[IDX_SiOI] + 
               1.0*y[IDX_SOI] + 1.0*y[IDX_SOII] + 1.0*y[IDX_H2COII] + 1.0*y[IDX_NOII] + 
               1.0*y[IDX_OII] + 2.0*y[IDX_O2I] + 1.0*y[IDX_NOI] + 1.0*y[IDX_HCOI] + 
               1.0*y[IDX_H2COI] + 1.0*y[IDX_OHI] + 1.0*y[IDX_H3OII] + 1.0*y[IDX_OI] + 
               1.0*y[IDX_H2OI] + 1.0*y[IDX_HCOII] + 1.0*y[IDX_COI] + 0.0;
    }
    if (elemidx == IDX_ELEM_He) {
        return 1.0*y[IDX_GHeI] + 1.0*y[IDX_HeHII] + 1.0*y[IDX_HeII] + 1.0*y[IDX_HeI] + 0.0;
    }
    if (elemidx == IDX_ELEM_C) {
        return 4.0*y[IDX_GC4H6I] + 4.0*y[IDX_GC4SI] + 1.0*y[IDX_GCClI] + 4.0*y[IDX_GCH3C3NI] + 
               5.0*y[IDX_GCH3C4HI] + 6.0*y[IDX_GCH3C5NI] + 7.0*y[IDX_GCH3C6HI] + 8.0*y[IDX_GCH3C7NI] + 
               2.0*y[IDX_GHC2PI] + 1.0*y[IDX_GNH2CNI] + 3.0*y[IDX_GSiC3HI] + 1.0*y[IDX_GSiCH3I] + 
               2.0*y[IDX_GNCCNI] + 2.0*y[IDX_GSiC2HI] + 3.0*y[IDX_C2H4CNI] + 3.0*y[IDX_GC3SI] + 
               4.0*y[IDX_GC4NI] + 4.0*y[IDX_GC4PI] + 1.0*y[IDX_GCH2PHI] + 3.0*y[IDX_GCH3CHCH2I] + 
               2.0*y[IDX_GCH3COOHI] + 1.0*y[IDX_GHCNOI] + 1.0*y[IDX_GHCPI] + 3.0*y[IDX_GHNC3I] + 
               1.0*y[IDX_GHNCOI] + 1.0*y[IDX_GHOCNI] + 1.0*y[IDX_GHONCI] + 2.0*y[IDX_GSiC2H2I] + 
               4.0*y[IDX_GSiC4I] + 1.0*y[IDX_GSiNCI] + 2.0*y[IDX_HC2OI] + 3.0*y[IDX_GC3PI] + 
               6.0*y[IDX_GC6H6I] + 1.0*y[IDX_GHCSiI] + 3.0*y[IDX_NCCNCH3II] + 3.0*y[IDX_GC2H4CNI] + 
               2.0*y[IDX_GC2SI] + 3.0*y[IDX_GC3OI] + 4.0*y[IDX_GC4H3I] + 7.0*y[IDX_GC7H2I] + 
               8.0*y[IDX_GC8H2I] + 9.0*y[IDX_GC9H2I] + 2.0*y[IDX_GCH2CNI] + 2.0*y[IDX_GCH3CNI] + 
               1.0*y[IDX_GH2CNI] + 1.0*y[IDX_GH2CSI] + 2.0*y[IDX_GHC2OI] + 5.0*y[IDX_GHC5NI] + 
               7.0*y[IDX_GHC7NI] + 9.0*y[IDX_GHC9NI] + 2.0*y[IDX_GHCCNI] + 1.0*y[IDX_GHCSI] + 
               3.0*y[IDX_GSiC3I] + 2.0*y[IDX_HCCNI] + 3.0*y[IDX_C3H2OII] + 1.0*y[IDX_CFII] + 
               2.0*y[IDX_GC2H6I] + 3.0*y[IDX_GC3H2I] + 5.0*y[IDX_GC5H2I] + 2.0*y[IDX_GCCPI] + 
               4.0*y[IDX_GCH2CHCCHI] + 3.0*y[IDX_GCH3COCH3I] + 2.0*y[IDX_GCH3OCH3I] + 1.0*y[IDX_GSiCH2I] + 
               3.0*y[IDX_H3C3OII] + 8.0*y[IDX_C8H5II] + 9.0*y[IDX_C9H5II] + 4.0*y[IDX_CH3C3NII] + 
               10.0*y[IDX_GC10H2I] + 4.0*y[IDX_GC4H2I] + 3.0*y[IDX_GCH2CHCNI] + 3.0*y[IDX_GH2CCCI] + 
               3.0*y[IDX_GHC3NI] + 1.0*y[IDX_GOCSI] + 4.0*y[IDX_C4H6I] + 4.0*y[IDX_C4PII] + 
               5.0*y[IDX_C5NII] + 6.0*y[IDX_C6H6II] + 3.0*y[IDX_CH2CHCNII] + 2.0*y[IDX_COOCH3II] + 
               3.0*y[IDX_GC2H5CNI] + 2.0*y[IDX_GC2H5OHI] + 2.0*y[IDX_GC2OI] + 6.0*y[IDX_GC6H2I] + 
               9.0*y[IDX_GC9NI] + 1.0*y[IDX_GCO2I] + 1.0*y[IDX_GCPI] + 1.0*y[IDX_GOCNI] + 
               1.0*y[IDX_GSiCI] + 1.0*y[IDX_H2CNOII] + 1.0*y[IDX_H2NCOII] + 1.0*y[IDX_H2OCNII] + 
               2.0*y[IDX_PC2H4II] + 7.0*y[IDX_C7NII] + 9.0*y[IDX_C9NII] + 1.0*y[IDX_CH3NHII] + 
               11.0*y[IDX_GC11I] + 2.0*y[IDX_GC2NI] + 3.0*y[IDX_GCH2CCH2I] + 1.0*y[IDX_GCH2NHI] + 
               3.0*y[IDX_GCH3CCHI] + 1.0*y[IDX_GCNOI] + 1.0*y[IDX_GHNCI] + 2.0*y[IDX_GSiC2I] + 
               4.0*y[IDX_H2C4NII] + 1.0*y[IDX_H2CClII] + 1.0*y[IDX_HOCII] + 1.0*y[IDX_NH2CNHII] + 
               2.0*y[IDX_C2H5OHII] + 2.0*y[IDX_C2OII] + 4.0*y[IDX_CH2CHCCHI] + 2.0*y[IDX_CH2OHCOII] + 
               3.0*y[IDX_GCH2CCHI] + 2.0*y[IDX_GCH2COI] + 4.0*y[IDX_HC4NII] + 1.0*y[IDX_HCNOHII] + 
               1.0*y[IDX_HNCOHII] + 1.0*y[IDX_HOCNII] + 2.0*y[IDX_PC2H3II] + 2.0*y[IDX_CH3CSII] + 
               5.0*y[IDX_GC5NI] + 7.0*y[IDX_H2C7NII] + 9.0*y[IDX_H2C9NII] + 1.0*y[IDX_HONCII] + 
               1.0*y[IDX_PCH3II] + 10.0*y[IDX_C10H3II] + 3.0*y[IDX_C2H5CNHII] + 1.0*y[IDX_CClII] + 
               1.0*y[IDX_CH2NH2II] + 2.0*y[IDX_CH2OHCH2OII] + 6.0*y[IDX_CH3C5NHII] + 8.0*y[IDX_CH3C7NHII] + 
               2.0*y[IDX_GC2H5I] + 3.0*y[IDX_GC3NI] + 7.0*y[IDX_GC7NI] + 2.0*y[IDX_GCOOCH3I] + 
               1.0*y[IDX_GHCNI] + 1.0*y[IDX_H2CNI] + 5.0*y[IDX_H3C5NII] + 1.0*y[IDX_HCNOII] + 
               1.0*y[IDX_CClI] + 2.0*y[IDX_CH3COOHII] + 9.0*y[IDX_H3C9NII] + 2.0*y[IDX_H5C2O2II] + 
               1.0*y[IDX_HCOOHII] + 2.0*y[IDX_C2SII] + 3.0*y[IDX_C3SII] + 4.0*y[IDX_C4NI] + 
               2.0*y[IDX_CH3COOH2II] + 2.0*y[IDX_CH3OCH3II] + 10.0*y[IDX_GC10I] + 10.0*y[IDX_GC10HI] + 
               8.0*y[IDX_GC8HI] + 2.0*y[IDX_GCH2OHCOI] + 1.0*y[IDX_GCOOHI] + 7.0*y[IDX_H3C7NII] + 
               1.0*y[IDX_NH2CNI] + 4.0*y[IDX_CH3C3NHII] + 2.0*y[IDX_CH3OCH4II] + 2.0*y[IDX_GC2H4I] + 
               9.0*y[IDX_GC9I] + 9.0*y[IDX_GC9HI] + 1.0*y[IDX_GCSI] + 7.0*y[IDX_HC7NII] + 
               9.0*y[IDX_HC9NII] + 1.0*y[IDX_HNCOII] + 3.0*y[IDX_PC3HII] + 1.0*y[IDX_PCH4II] + 
               3.0*y[IDX_C2H5CNI] + 2.0*y[IDX_C2H5OH2II] + 2.0*y[IDX_C2H7II] + 2.0*y[IDX_CCPII] + 
               3.0*y[IDX_CH2CHCNHII] + 2.0*y[IDX_COOCH3I] + 2.0*y[IDX_HCOOCH3II] + 2.0*y[IDX_PC2H2II] + 
               3.0*y[IDX_SiC3H2II] + 4.0*y[IDX_SiC4II] + 4.0*y[IDX_SiC4HII] + 1.0*y[IDX_SiNCHII] + 
               11.0*y[IDX_C11I] + 3.0*y[IDX_C3OII] + 4.0*y[IDX_C4H5II] + 4.0*y[IDX_C4H7II] + 
               2.0*y[IDX_CH2OHCHOII] + 2.0*y[IDX_CH3CHOII] + 2.0*y[IDX_GCH2OHCHOI] + 1.0*y[IDX_GHCOOHI] + 
               1.0*y[IDX_HOCSII] + 1.0*y[IDX_OCNII] + 4.0*y[IDX_PC4HII] + 1.0*y[IDX_SiCH4II] + 
               10.0*y[IDX_C10II] + 10.0*y[IDX_C10H2II] + 2.0*y[IDX_C2N2II] + 2.0*y[IDX_CH3CNII] + 
               2.0*y[IDX_GC2H2I] + 5.0*y[IDX_GC5I] + 7.0*y[IDX_GC7HI] + 2.0*y[IDX_GCH3COI] + 
               2.0*y[IDX_GHCOOCH3I] + 4.0*y[IDX_HC4SII] + 5.0*y[IDX_HC5NHII] + 2.0*y[IDX_SiC2H3II] + 
               3.0*y[IDX_SiC3HI] + 1.0*y[IDX_SiCH3II] + 11.0*y[IDX_C11II] + 3.0*y[IDX_C3NII] + 
               4.0*y[IDX_C4H4II] + 2.0*y[IDX_CH2CNII] + 3.0*y[IDX_CH3COCH3II] + 1.0*y[IDX_CPII] + 
               6.0*y[IDX_GC6I] + 8.0*y[IDX_GC8I] + 1.0*y[IDX_GCH2OHI] + 1.0*y[IDX_GCH3OHI] + 
               2.0*y[IDX_HC2PII] + 5.0*y[IDX_HC5NII] + 1.0*y[IDX_HONCI] + 1.0*y[IDX_PCH2II] + 
               9.0*y[IDX_C9II] + 2.0*y[IDX_CH2OHCOI] + 6.0*y[IDX_CH3C5NI] + 8.0*y[IDX_CH3C7NI] + 
               3.0*y[IDX_GC3HI] + 4.0*y[IDX_GC4HI] + 5.0*y[IDX_GC5HI] + 6.0*y[IDX_GC6HI] + 
               1.0*y[IDX_H2NCII] + 1.0*y[IDX_H3CSII] + 2.0*y[IDX_HC2OII] + 1.0*y[IDX_HCPII] + 
               4.0*y[IDX_SiC4I] + 1.0*y[IDX_SiNCI] + 1.0*y[IDX_SiNCII] + 1.0*y[IDX_CNOI] + 
               2.0*y[IDX_GC2H3I] + 1.0*y[IDX_GCH3OI] + 3.0*y[IDX_HC3SII] + 1.0*y[IDX_HCOOH2II] + 
               2.0*y[IDX_SiC2H2I] + 3.0*y[IDX_SiC3II] + 2.0*y[IDX_C2NHII] + 3.0*y[IDX_C3H6II] + 
               6.0*y[IDX_C6H7II] + 8.0*y[IDX_C8H4II] + 2.0*y[IDX_CH2OHCHOI] + 7.0*y[IDX_GC7I] + 
               3.0*y[IDX_HC3OII] + 1.0*y[IDX_HCNOI] + 3.0*y[IDX_SiC3HII] + 1.0*y[IDX_SiCH3I] + 
               3.0*y[IDX_C3H7II] + 4.0*y[IDX_C4NII] + 8.0*y[IDX_C8II] + 9.0*y[IDX_C9H4II] + 
               2.0*y[IDX_CH2COII] + 1.0*y[IDX_CH2PHI] + 4.0*y[IDX_CH3C3NI] + 4.0*y[IDX_GC4I] + 
               2.0*y[IDX_GCH3CHOI] + 1.0*y[IDX_GH2COI] + 1.0*y[IDX_HOCNI] + 2.0*y[IDX_SiC2II] + 
               2.0*y[IDX_SiC2H2II] + 6.0*y[IDX_C6H4II] + 2.0*y[IDX_CH3COI] + 3.0*y[IDX_CH3COCH4II] + 
               1.0*y[IDX_CH3OHII] + 1.0*y[IDX_COOHI] + 7.0*y[IDX_C7H4II] + 7.0*y[IDX_C7H5II] + 
               7.0*y[IDX_CH3C6HI] + 1.0*y[IDX_H2CSII] + 9.0*y[IDX_HC9NI] + 2.0*y[IDX_HCOOCH3I] + 
               2.0*y[IDX_NCCNHII] + 2.0*y[IDX_SiC2HI] + 3.0*y[IDX_SiC3I] + 4.0*y[IDX_C4PI] + 
               7.0*y[IDX_C7II] + 2.0*y[IDX_CH2CNI] + 5.0*y[IDX_CH3C4HII] + 1.0*y[IDX_GCH4I] + 
               1.0*y[IDX_HCPI] + 1.0*y[IDX_HNCOI] + 3.0*y[IDX_C3PI] + 9.0*y[IDX_C9NI] + 
               5.0*y[IDX_CH3C4HI] + 2.0*y[IDX_CH3COOHI] + 2.0*y[IDX_CH3OCH3I] + 2.0*y[IDX_GC2HI] + 
               1.0*y[IDX_H2CSI] + 1.0*y[IDX_SiCII] + 3.0*y[IDX_C3SI] + 6.0*y[IDX_C6H5II] + 
               1.0*y[IDX_CH2NHI] + 1.0*y[IDX_CH3OI] + 1.0*y[IDX_GHCOI] + 2.0*y[IDX_HC2PI] + 
               1.0*y[IDX_HCSiI] + 1.0*y[IDX_SiCH2I] + 10.0*y[IDX_C10HII] + 1.0*y[IDX_HCSiII] + 
               1.0*y[IDX_OCSII] + 2.0*y[IDX_SiC2I] + 2.0*y[IDX_C2H6II] + 2.0*y[IDX_C2OI] + 
               8.0*y[IDX_C8HII] + 1.0*y[IDX_CH2OHI] + 5.0*y[IDX_C5H5II] + 1.0*y[IDX_HCSI] + 
               2.0*y[IDX_SiC2HII] + 3.0*y[IDX_C3OI] + 6.0*y[IDX_C6HII] + 9.0*y[IDX_C9H2I] + 
               3.0*y[IDX_CH2CHCNI] + 2.0*y[IDX_CH3CHOHII] + 2.0*y[IDX_CH3CNHII] + 2.0*y[IDX_CH3COII] + 
               7.0*y[IDX_HC7NI] + 6.0*y[IDX_C6II] + 6.0*y[IDX_C6H6I] + 7.0*y[IDX_C7NI] + 
               1.0*y[IDX_HCOOHI] + 10.0*y[IDX_C10H2I] + 4.0*y[IDX_C4II] + 2.0*y[IDX_CCPI] + 
               1.0*y[IDX_CPI] + 3.0*y[IDX_GC3I] + 1.0*y[IDX_SiCI] + 2.0*y[IDX_C2H5OHI] + 
               1.0*y[IDX_CH3OH2II] + 2.0*y[IDX_C2NII] + 5.0*y[IDX_C5II] + 6.0*y[IDX_C6H3II] + 
               8.0*y[IDX_C8H3II] + 9.0*y[IDX_C9HII] + 9.0*y[IDX_C9H3II] + 2.0*y[IDX_CH2COI] + 
               3.0*y[IDX_HNC3I] + 1.0*y[IDX_OCNI] + 1.0*y[IDX_SiCH2II] + 3.0*y[IDX_CH3COCH3I] + 
               3.0*y[IDX_HC3NII] + 8.0*y[IDX_C8H2I] + 1.0*y[IDX_CO2II] + 5.0*y[IDX_HC5NI] + 
               7.0*y[IDX_C7HII] + 9.0*y[IDX_C9H2II] + 7.0*y[IDX_C7H2I] + 1.0*y[IDX_GCOI] + 
               1.0*y[IDX_HCSII] + 3.0*y[IDX_C3H4II] + 8.0*y[IDX_C8H2II] + 1.0*y[IDX_GCH2I] + 
               1.0*y[IDX_HCO2II] + 5.0*y[IDX_C5H3II] + 3.0*y[IDX_HC3NHII] + 5.0*y[IDX_C5HII] + 
               2.0*y[IDX_CH3CHOI] + 1.0*y[IDX_CSII] + 3.0*y[IDX_C3H5II] + 7.0*y[IDX_C7H3II] + 
               2.0*y[IDX_CH3CNI] + 2.0*y[IDX_NCCNI] + 7.0*y[IDX_C7H2II] + 2.0*y[IDX_GC2I] + 
               3.0*y[IDX_C3II] + 6.0*y[IDX_C6H2II] + 1.0*y[IDX_CH4II] + 3.0*y[IDX_H2CCCI] + 
               6.0*y[IDX_C6H2I] + 1.0*y[IDX_GCNI] + 2.0*y[IDX_C2H5II] + 5.0*y[IDX_C5H2I] + 
               5.0*y[IDX_C5H2II] + 3.0*y[IDX_CH2CCH2I] + 1.0*y[IDX_CNII] + 2.0*y[IDX_C2H6I] + 
               3.0*y[IDX_C3H2I] + 2.0*y[IDX_C2H5I] + 4.0*y[IDX_C4SII] + 1.0*y[IDX_HCNII] + 
               3.0*y[IDX_CH3CHCH2I] + 2.0*y[IDX_C2II] + 4.0*y[IDX_C4HII] + 2.0*y[IDX_CNCII] + 
               1.0*y[IDX_CHM] + 1.0*y[IDX_OCSI] + 1.0*y[IDX_GCH3I] + 2.0*y[IDX_C2HM] + 
               4.0*y[IDX_C4HM] + 1.0*y[IDX_COII] + 3.0*y[IDX_CH3CCHI] + 1.0*y[IDX_CH5II] + 
               2.0*y[IDX_HC2SII] + 10.0*y[IDX_C10HM] + 3.0*y[IDX_C3HM] + 4.0*y[IDX_C4SI] + 
               6.0*y[IDX_C6HM] + 8.0*y[IDX_C8HM] + 7.0*y[IDX_C7HM] + 9.0*y[IDX_C9HM] + 
               10.0*y[IDX_C10M] + 5.0*y[IDX_C5HM] + 5.0*y[IDX_C5NM] + 1.0*y[IDX_CM] + 
               1.0*y[IDX_CSI] + 9.0*y[IDX_C9M] + 1.0*y[IDX_CH3OHI] + 3.0*y[IDX_HC3NI] + 
               3.0*y[IDX_C3HII] + 3.0*y[IDX_C3H3II] + 8.0*y[IDX_C8M] + 2.0*y[IDX_C2M] + 
               2.0*y[IDX_C2SI] + 3.0*y[IDX_C3M] + 4.0*y[IDX_C4M] + 6.0*y[IDX_C6M] + 
               1.0*y[IDX_H3COII] + 5.0*y[IDX_C5M] + 7.0*y[IDX_C7M] + 2.0*y[IDX_C2NI] + 
               1.0*y[IDX_GCHI] + 4.0*y[IDX_C4H3I] + 2.0*y[IDX_C2HII] + 3.0*y[IDX_C3H2II] + 
               5.0*y[IDX_C5NI] + 1.0*y[IDX_CH2II] + 3.0*y[IDX_C3NM] + 3.0*y[IDX_C3NI] + 
               1.0*y[IDX_CHII] + 1.0*y[IDX_CO2I] + 10.0*y[IDX_C10HI] + 9.0*y[IDX_C9HI] + 
               10.0*y[IDX_C10I] + 8.0*y[IDX_C8HI] + 7.0*y[IDX_C7I] + 9.0*y[IDX_C9I] + 
               8.0*y[IDX_C8I] + 1.0*y[IDX_GCI] + 5.0*y[IDX_C5I] + 6.0*y[IDX_C6I] + 
               1.0*y[IDX_CNM] + 4.0*y[IDX_C4I] + 7.0*y[IDX_C7HI] + 2.0*y[IDX_C2H4II] + 
               4.0*y[IDX_C4H3II] + 3.0*y[IDX_CH2CCHI] + 3.0*y[IDX_CH2CCHII] + 6.0*y[IDX_C6HI] + 
               5.0*y[IDX_C5HI] + 3.0*y[IDX_C3HI] + 1.0*y[IDX_H2COII] + 3.0*y[IDX_C3I] + 
               4.0*y[IDX_C4HI] + 4.0*y[IDX_C4H2I] + 4.0*y[IDX_C4H2II] + 1.0*y[IDX_HCNHII] + 
               1.0*y[IDX_CH2I] + 1.0*y[IDX_HNCI] + 2.0*y[IDX_C2H4I] + 2.0*y[IDX_C2H3I] + 
               2.0*y[IDX_C2H3II] + 1.0*y[IDX_HCOI] + 1.0*y[IDX_CH4I] + 2.0*y[IDX_C2HI] + 
               1.0*y[IDX_H2COI] + 1.0*y[IDX_HCNI] + 1.0*y[IDX_CHI] + 2.0*y[IDX_C2I] + 
               2.0*y[IDX_C2H2II] + 1.0*y[IDX_CH3II] + 2.0*y[IDX_C2H2I] + 1.0*y[IDX_CNI] + 
               1.0*y[IDX_CH3I] + 1.0*y[IDX_CII] + 1.0*y[IDX_CI] + 1.0*y[IDX_HCOII] + 
               1.0*y[IDX_COI] + 0.0;
    }
    if (elemidx == IDX_ELEM_GRAIN) {
        return 1.0*y[IDX_GRAINM] + 1.0*y[IDX_GRAIN0I] + 0.0;
    }
    if (elemidx == IDX_ELEM_H) {
        return 2.0*y[IDX_GH2S2I] + 6.0*y[IDX_GC4H6I] + 3.0*y[IDX_GCH3C3NI] + 4.0*y[IDX_GCH3C4HI] + 
               3.0*y[IDX_GCH3C5NI] + 4.0*y[IDX_GCH3C6HI] + 3.0*y[IDX_GCH3C7NI] + 1.0*y[IDX_GHC2PI] + 
               1.0*y[IDX_GHClI] + 1.0*y[IDX_GHFI] + 1.0*y[IDX_GHNSiI] + 1.0*y[IDX_GHPOI] + 
               2.0*y[IDX_GNH2CNI] + 1.0*y[IDX_GSiC3HI] + 3.0*y[IDX_GSiCH3I] + 2.0*y[IDX_GH2SiOI] + 
               1.0*y[IDX_GSiC2HI] + 4.0*y[IDX_C2H4CNI] + 3.0*y[IDX_GCH2PHI] + 6.0*y[IDX_GCH3CHCH2I] + 
               4.0*y[IDX_GCH3COOHI] + 2.0*y[IDX_GH2SI] + 1.0*y[IDX_GHCNOI] + 1.0*y[IDX_GHCPI] + 
               1.0*y[IDX_GHNC3I] + 1.0*y[IDX_GHNCOI] + 1.0*y[IDX_GHOCNI] + 1.0*y[IDX_GHONCI] + 
               1.0*y[IDX_GHS2I] + 2.0*y[IDX_GPH2I] + 2.0*y[IDX_GSiC2H2I] + 1.0*y[IDX_HC2OI] + 
               6.0*y[IDX_GC6H6I] + 2.0*y[IDX_GH2O2I] + 1.0*y[IDX_GHCSiI] + 1.0*y[IDX_GPHI] + 
               4.0*y[IDX_GSiH4I] + 3.0*y[IDX_NCCNCH3II] + 4.0*y[IDX_GC2H4CNI] + 3.0*y[IDX_GC4H3I] + 
               2.0*y[IDX_GC7H2I] + 2.0*y[IDX_GC8H2I] + 2.0*y[IDX_GC9H2I] + 2.0*y[IDX_GCH2CNI] + 
               3.0*y[IDX_GCH3CNI] + 2.0*y[IDX_GH2CNI] + 2.0*y[IDX_GH2CSI] + 1.0*y[IDX_GHC2OI] + 
               1.0*y[IDX_GHC5NI] + 1.0*y[IDX_GHC7NI] + 1.0*y[IDX_GHC9NI] + 1.0*y[IDX_GHCCNI] + 
               1.0*y[IDX_GHCSI] + 3.0*y[IDX_GSiH3I] + 1.0*y[IDX_HCCNI] + 2.0*y[IDX_C3H2OII] + 
               6.0*y[IDX_GC2H6I] + 2.0*y[IDX_GC3H2I] + 2.0*y[IDX_GC5H2I] + 4.0*y[IDX_GCH2CHCCHI] + 
               6.0*y[IDX_GCH3COCH3I] + 6.0*y[IDX_GCH3OCH3I] + 2.0*y[IDX_GSiCH2I] + 1.0*y[IDX_GSiHI] + 
               2.0*y[IDX_GSiH2I] + 3.0*y[IDX_H3C3OII] + 1.0*y[IDX_HFII] + 5.0*y[IDX_C8H5II] + 
               5.0*y[IDX_C9H5II] + 3.0*y[IDX_CH3C3NII] + 2.0*y[IDX_GC10H2I] + 2.0*y[IDX_GC4H2I] + 
               3.0*y[IDX_GCH2CHCNI] + 2.0*y[IDX_GH2CCCI] + 1.0*y[IDX_GHC3NI] + 2.0*y[IDX_H2FII] + 
               6.0*y[IDX_C4H6I] + 6.0*y[IDX_C6H6II] + 3.0*y[IDX_CH2CHCNII] + 3.0*y[IDX_COOCH3II] + 
               5.0*y[IDX_GC2H5CNI] + 6.0*y[IDX_GC2H5OHI] + 2.0*y[IDX_GC6H2I] + 1.0*y[IDX_GO2HI] + 
               2.0*y[IDX_H2CNOII] + 2.0*y[IDX_H2NCOII] + 2.0*y[IDX_H2OCNII] + 4.0*y[IDX_PC2H4II] + 
               4.0*y[IDX_CH3NHII] + 4.0*y[IDX_GCH2CCH2I] + 3.0*y[IDX_GCH2NHI] + 4.0*y[IDX_GCH3CCHI] + 
               1.0*y[IDX_GHNCI] + 2.0*y[IDX_H2C4NII] + 2.0*y[IDX_H2CClII] + 2.0*y[IDX_H2NOII] + 
               1.0*y[IDX_HClII] + 1.0*y[IDX_HNSII] + 1.0*y[IDX_HOCII] + 1.0*y[IDX_HSOII] + 
               1.0*y[IDX_HSiO2II] + 3.0*y[IDX_NH2CNHII] + 6.0*y[IDX_C2H5OHII] + 4.0*y[IDX_CH2CHCCHI] + 
               3.0*y[IDX_CH2OHCOII] + 3.0*y[IDX_GCH2CCHI] + 2.0*y[IDX_GCH2COI] + 1.0*y[IDX_GHNOI] + 
               1.0*y[IDX_HC4NII] + 2.0*y[IDX_HCNOHII] + 2.0*y[IDX_HNCOHII] + 1.0*y[IDX_HOCNII] + 
               1.0*y[IDX_HeHII] + 3.0*y[IDX_PC2H3II] + 3.0*y[IDX_PNH3II] + 3.0*y[IDX_CH3CSII] + 
               2.0*y[IDX_H2C7NII] + 2.0*y[IDX_H2C9NII] + 2.0*y[IDX_H2O2I] + 3.0*y[IDX_H3S2II] + 
               1.0*y[IDX_HONCII] + 3.0*y[IDX_PCH3II] + 2.0*y[IDX_PNH2II] + 3.0*y[IDX_C10H3II] + 
               6.0*y[IDX_C2H5CNHII] + 4.0*y[IDX_CH2NH2II] + 5.0*y[IDX_CH2OHCH2OII] + 4.0*y[IDX_CH3C5NHII] + 
               4.0*y[IDX_CH3C7NHII] + 5.0*y[IDX_GC2H5I] + 3.0*y[IDX_GCOOCH3I] + 1.0*y[IDX_GHCNI] + 
               2.0*y[IDX_H2CNI] + 2.0*y[IDX_H2SiOII] + 3.0*y[IDX_H3C5NII] + 1.0*y[IDX_HCNOII] + 
               4.0*y[IDX_CH3COOHII] + 3.0*y[IDX_GNH3I] + 2.0*y[IDX_H2ClII] + 3.0*y[IDX_H3C9NII] + 
               5.0*y[IDX_H5C2O2II] + 2.0*y[IDX_HCOOHII] + 1.0*y[IDX_HPNII] + 5.0*y[IDX_CH3COOH2II] + 
               6.0*y[IDX_CH3OCH3II] + 1.0*y[IDX_GC10HI] + 1.0*y[IDX_GC8HI] + 3.0*y[IDX_GCH2OHCOI] + 
               1.0*y[IDX_GCOOHI] + 3.0*y[IDX_H3C7NII] + 3.0*y[IDX_H3SiOII] + 1.0*y[IDX_HSO2II] + 
               2.0*y[IDX_NH2CNI] + 4.0*y[IDX_CH3C3NHII] + 7.0*y[IDX_CH3OCH4II] + 4.0*y[IDX_GC2H4I] + 
               1.0*y[IDX_GC9HI] + 1.0*y[IDX_HC7NII] + 1.0*y[IDX_HC9NII] + 1.0*y[IDX_HNCOII] + 
               1.0*y[IDX_PC3HII] + 4.0*y[IDX_PCH4II] + 4.0*y[IDX_SiH4II] + 5.0*y[IDX_C2H5CNI] + 
               7.0*y[IDX_C2H5OH2II] + 7.0*y[IDX_C2H7II] + 4.0*y[IDX_CH2CHCNHII] + 3.0*y[IDX_COOCH3I] + 
               1.0*y[IDX_GHSI] + 2.0*y[IDX_H2POII] + 4.0*y[IDX_HCOOCH3II] + 1.0*y[IDX_HN2OII] + 
               2.0*y[IDX_PC2H2II] + 2.0*y[IDX_SiC3H2II] + 1.0*y[IDX_SiC4HII] + 5.0*y[IDX_SiH5II] + 
               1.0*y[IDX_SiNCHII] + 5.0*y[IDX_C4H5II] + 7.0*y[IDX_C4H7II] + 4.0*y[IDX_CH2OHCHOII] + 
               4.0*y[IDX_CH3CHOII] + 4.0*y[IDX_GCH2OHCHOI] + 2.0*y[IDX_GHCOOHI] + 2.0*y[IDX_H2S2I] + 
               1.0*y[IDX_HNSiII] + 1.0*y[IDX_HOCSII] + 1.0*y[IDX_PC4HII] + 3.0*y[IDX_PH3II] + 
               4.0*y[IDX_SiCH4II] + 2.0*y[IDX_C10H2II] + 3.0*y[IDX_CH3CNII] + 2.0*y[IDX_GC2H2I] + 
               1.0*y[IDX_GC7HI] + 3.0*y[IDX_GCH3COI] + 4.0*y[IDX_GHCOOCH3I] + 1.0*y[IDX_HC4SII] + 
               2.0*y[IDX_HC5NHII] + 3.0*y[IDX_SiC2H3II] + 1.0*y[IDX_SiC3HI] + 3.0*y[IDX_SiCH3II] + 
               2.0*y[IDX_SiNH2II] + 4.0*y[IDX_C4H4II] + 2.0*y[IDX_CH2CNII] + 6.0*y[IDX_CH3COCH3II] + 
               3.0*y[IDX_GCH2OHI] + 4.0*y[IDX_GCH3OHI] + 2.0*y[IDX_H2S2II] + 1.0*y[IDX_HC2PII] + 
               1.0*y[IDX_HC5NII] + 1.0*y[IDX_HFI] + 1.0*y[IDX_HONCI] + 1.0*y[IDX_HSiSII] + 
               2.0*y[IDX_PCH2II] + 3.0*y[IDX_CH2OHCOI] + 3.0*y[IDX_CH3C5NI] + 3.0*y[IDX_CH3C7NI] + 
               1.0*y[IDX_GC3HI] + 1.0*y[IDX_GC4HI] + 1.0*y[IDX_GC5HI] + 1.0*y[IDX_GC6HI] + 
               2.0*y[IDX_H2NCII] + 2.0*y[IDX_H2SiOI] + 3.0*y[IDX_H3CSII] + 1.0*y[IDX_HC2OII] + 
               1.0*y[IDX_HCPII] + 3.0*y[IDX_GC2H3I] + 3.0*y[IDX_GCH3OI] + 2.0*y[IDX_GH2OI] + 
               1.0*y[IDX_HC3SII] + 3.0*y[IDX_HCOOH2II] + 1.0*y[IDX_HNSiI] + 1.0*y[IDX_HPOII] + 
               1.0*y[IDX_HS2II] + 2.0*y[IDX_SiC2H2I] + 1.0*y[IDX_C2NHII] + 6.0*y[IDX_C3H6II] + 
               7.0*y[IDX_C6H7II] + 4.0*y[IDX_C8H4II] + 4.0*y[IDX_CH2OHCHOI] + 1.0*y[IDX_HC3OII] + 
               1.0*y[IDX_HCNOI] + 1.0*y[IDX_SiC3HII] + 3.0*y[IDX_SiCH3I] + 7.0*y[IDX_C3H7II] + 
               4.0*y[IDX_C9H4II] + 2.0*y[IDX_CH2COII] + 3.0*y[IDX_CH2PHI] + 3.0*y[IDX_CH3C3NI] + 
               4.0*y[IDX_GCH3CHOI] + 2.0*y[IDX_GH2COI] + 1.0*y[IDX_HOCNI] + 1.0*y[IDX_HPOI] + 
               1.0*y[IDX_HS2I] + 2.0*y[IDX_SiC2H2II] + 4.0*y[IDX_C6H4II] + 3.0*y[IDX_CH3COI] + 
               7.0*y[IDX_CH3COCH4II] + 4.0*y[IDX_CH3OHII] + 1.0*y[IDX_COOHI] + 4.0*y[IDX_C7H4II] + 
               5.0*y[IDX_C7H5II] + 4.0*y[IDX_CH3C6HI] + 2.0*y[IDX_H2CSII] + 1.0*y[IDX_HC9NI] + 
               4.0*y[IDX_HCOOCH3I] + 1.0*y[IDX_NCCNHII] + 2.0*y[IDX_PH2I] + 1.0*y[IDX_SiC2HI] + 
               2.0*y[IDX_CH2CNI] + 4.0*y[IDX_CH3C4HII] + 4.0*y[IDX_GCH4I] + 1.0*y[IDX_HCPI] + 
               1.0*y[IDX_HClI] + 1.0*y[IDX_HNCOI] + 4.0*y[IDX_CH3C4HI] + 4.0*y[IDX_CH3COOHI] + 
               6.0*y[IDX_CH3OCH3I] + 1.0*y[IDX_GC2HI] + 2.0*y[IDX_GNH2I] + 2.0*y[IDX_H2CSI] + 
               2.0*y[IDX_PH2II] + 5.0*y[IDX_C6H5II] + 3.0*y[IDX_CH2NHI] + 3.0*y[IDX_CH3OI] + 
               1.0*y[IDX_GHCOI] + 1.0*y[IDX_HC2PI] + 1.0*y[IDX_HCSiI] + 2.0*y[IDX_SiCH2I] + 
               1.0*y[IDX_C10HII] + 1.0*y[IDX_HCSiII] + 6.0*y[IDX_C2H6II] + 1.0*y[IDX_C8HII] + 
               3.0*y[IDX_CH2OHI] + 5.0*y[IDX_C5H5II] + 1.0*y[IDX_GNHI] + 1.0*y[IDX_HCSI] + 
               1.0*y[IDX_SiC2HII] + 2.0*y[IDX_SiH2I] + 1.0*y[IDX_C6HII] + 2.0*y[IDX_C9H2I] + 
               3.0*y[IDX_CH2CHCNI] + 5.0*y[IDX_CH3CHOHII] + 4.0*y[IDX_CH3CNHII] + 3.0*y[IDX_CH3COII] + 
               1.0*y[IDX_HC7NI] + 6.0*y[IDX_C6H6I] + 2.0*y[IDX_HCOOHI] + 2.0*y[IDX_SiH2II] + 
               3.0*y[IDX_SiH3II] + 2.0*y[IDX_C10H2I] + 1.0*y[IDX_O2HI] + 6.0*y[IDX_C2H5OHI] + 
               1.0*y[IDX_HNOI] + 3.0*y[IDX_SiH3I] + 5.0*y[IDX_CH3OH2II] + 3.0*y[IDX_C6H3II] + 
               3.0*y[IDX_C8H3II] + 1.0*y[IDX_C9HII] + 3.0*y[IDX_C9H3II] + 2.0*y[IDX_CH2COI] + 
               1.0*y[IDX_HNC3I] + 2.0*y[IDX_SiCH2II] + 1.0*y[IDX_SiHII] + 6.0*y[IDX_CH3COCH3I] + 
               1.0*y[IDX_HC3NII] + 2.0*y[IDX_C8H2I] + 1.0*y[IDX_GOHI] + 1.0*y[IDX_HC5NI] + 
               1.0*y[IDX_SiHI] + 4.0*y[IDX_SiH4I] + 1.0*y[IDX_C7HII] + 2.0*y[IDX_C9H2II] + 
               1.0*y[IDX_PHI] + 2.0*y[IDX_C7H2I] + 1.0*y[IDX_HCSII] + 1.0*y[IDX_PHII] + 
               4.0*y[IDX_C3H4II] + 2.0*y[IDX_C8H2II] + 2.0*y[IDX_GCH2I] + 1.0*y[IDX_HCO2II] + 
               3.0*y[IDX_C5H3II] + 2.0*y[IDX_HC3NHII] + 1.0*y[IDX_C5HII] + 4.0*y[IDX_CH3CHOI] + 
               5.0*y[IDX_C3H5II] + 3.0*y[IDX_C7H3II] + 3.0*y[IDX_CH3CNI] + 2.0*y[IDX_C7H2II] + 
               3.0*y[IDX_H3SII] + 1.0*y[IDX_O2HII] + 2.0*y[IDX_C6H2II] + 4.0*y[IDX_CH4II] + 
               1.0*y[IDX_HNOII] + 2.0*y[IDX_H2CCCI] + 2.0*y[IDX_C6H2I] + 5.0*y[IDX_C2H5II] + 
               2.0*y[IDX_C5H2I] + 2.0*y[IDX_C5H2II] + 4.0*y[IDX_CH2CCH2I] + 6.0*y[IDX_C2H6I] + 
               2.0*y[IDX_C3H2I] + 5.0*y[IDX_C2H5I] + 1.0*y[IDX_HCNII] + 6.0*y[IDX_CH3CHCH2I] + 
               1.0*y[IDX_C4HII] + 1.0*y[IDX_HSII] + 1.0*y[IDX_NHII] + 1.0*y[IDX_CHM] + 
               2.0*y[IDX_H2II] + 3.0*y[IDX_GCH3I] + 2.0*y[IDX_NH2II] + 1.0*y[IDX_C2HM] + 
               1.0*y[IDX_C4HM] + 4.0*y[IDX_CH3CCHI] + 5.0*y[IDX_CH5II] + 1.0*y[IDX_HC2SII] + 
               1.0*y[IDX_OHM] + 1.0*y[IDX_C10HM] + 1.0*y[IDX_C3HM] + 1.0*y[IDX_C6HM] + 
               1.0*y[IDX_C8HM] + 1.0*y[IDX_C7HM] + 1.0*y[IDX_C9HM] + 1.0*y[IDX_C5HM] + 
               1.0*y[IDX_HSI] + 4.0*y[IDX_CH3OHI] + 1.0*y[IDX_HM] + 1.0*y[IDX_HC3NI] + 
               1.0*y[IDX_C3HII] + 3.0*y[IDX_C3H3II] + 3.0*y[IDX_H3COII] + 1.0*y[IDX_GCHI] + 
               1.0*y[IDX_SiOHII] + 2.0*y[IDX_NH2I] + 3.0*y[IDX_C4H3I] + 1.0*y[IDX_C2HII] + 
               2.0*y[IDX_C3H2II] + 2.0*y[IDX_CH2II] + 1.0*y[IDX_OHII] + 2.0*y[IDX_H2OII] + 
               2.0*y[IDX_GH2I] + 1.0*y[IDX_CHII] + 1.0*y[IDX_C10HI] + 1.0*y[IDX_C9HI] + 
               1.0*y[IDX_C8HI] + 1.0*y[IDX_NHI] + 1.0*y[IDX_C7HI] + 4.0*y[IDX_C2H4II] + 
               3.0*y[IDX_C4H3II] + 1.0*y[IDX_N2HII] + 3.0*y[IDX_CH2CCHI] + 3.0*y[IDX_CH2CCHII] + 
               1.0*y[IDX_C6HI] + 1.0*y[IDX_C5HI] + 2.0*y[IDX_H2SII] + 1.0*y[IDX_C3HI] + 
               2.0*y[IDX_H2COII] + 1.0*y[IDX_C4HI] + 2.0*y[IDX_C4H2I] + 2.0*y[IDX_C4H2II] + 
               2.0*y[IDX_HCNHII] + 2.0*y[IDX_CH2I] + 1.0*y[IDX_HNCI] + 3.0*y[IDX_NH3II] + 
               4.0*y[IDX_C2H4I] + 3.0*y[IDX_C2H3I] + 4.0*y[IDX_NH4II] + 2.0*y[IDX_H2SI] + 
               3.0*y[IDX_C2H3II] + 1.0*y[IDX_HCOI] + 4.0*y[IDX_CH4I] + 1.0*y[IDX_C2HI] + 
               2.0*y[IDX_H2COI] + 1.0*y[IDX_HCNI] + 1.0*y[IDX_CHI] + 1.0*y[IDX_OHI] + 
               3.0*y[IDX_NH3I] + 2.0*y[IDX_C2H2II] + 3.0*y[IDX_CH3II] + 2.0*y[IDX_C2H2I] + 
               1.0*y[IDX_GHI] + 3.0*y[IDX_CH3I] + 3.0*y[IDX_H3OII] + 2.0*y[IDX_H2OI] + 
               1.0*y[IDX_HII] + 1.0*y[IDX_HCOII] + 3.0*y[IDX_H3II] + 2.0*y[IDX_H2I] + 
               1.0*y[IDX_HI] + 0.0;
    }
    
}

__device__ __host__ double GetMantleDens(double *y) {
    return y[IDX_GFeI] + y[IDX_GHeI] + y[IDX_GMgI] + y[IDX_GNaI] + y[IDX_GH2S2I] +
        y[IDX_GC4H6I] + y[IDX_GC4SI] + y[IDX_GCClI] + y[IDX_GCH3C3NI] +
        y[IDX_GCH3C4HI] + y[IDX_GCH3C5NI] + y[IDX_GCH3C6HI] + y[IDX_GCH3C7NI] +
        y[IDX_GClOI] + y[IDX_GHC2PI] + y[IDX_GHClI] + y[IDX_GHFI] +
        y[IDX_GHNSiI] + y[IDX_GHPOI] + y[IDX_GNH2CNI] + y[IDX_GNO2I] +
        y[IDX_GPNI] + y[IDX_GSiC3HI] + y[IDX_GSiCH3I] + y[IDX_GSiO2I] +
        y[IDX_GSiSI] + y[IDX_GFI] + y[IDX_GH2SiOI] + y[IDX_GNCCNI] + y[IDX_GS2I]
        + y[IDX_GSiC2HI] + y[IDX_GC3SI] + y[IDX_GC4NI] + y[IDX_GC4PI] +
        y[IDX_GCH2PHI] + y[IDX_GCH3CHCH2I] + y[IDX_GCH3COOHI] + y[IDX_GH2SI] +
        y[IDX_GHCNOI] + y[IDX_GHCPI] + y[IDX_GHNC3I] + y[IDX_GHNCOI] +
        y[IDX_GHOCNI] + y[IDX_GHONCI] + y[IDX_GHS2I] + y[IDX_GN2OI] +
        y[IDX_GPH2I] + y[IDX_GPOI] + y[IDX_GSiC2H2I] + y[IDX_GSiC4I] +
        y[IDX_GSiNCI] + y[IDX_GC3PI] + y[IDX_GC6H6I] + y[IDX_GH2O2I] +
        y[IDX_GHCSiI] + y[IDX_GPHI] + y[IDX_GSO2I] + y[IDX_GSiH4I] +
        y[IDX_GC2H4CNI] + y[IDX_GC2SI] + y[IDX_GC3OI] + y[IDX_GC4H3I] +
        y[IDX_GC7H2I] + y[IDX_GC8H2I] + y[IDX_GC9H2I] + y[IDX_GCH2CNI] +
        y[IDX_GCH3CNI] + y[IDX_GH2CNI] + y[IDX_GH2CSI] + y[IDX_GHC2OI] +
        y[IDX_GHC5NI] + y[IDX_GHC7NI] + y[IDX_GHC9NI] + y[IDX_GHCCNI] +
        y[IDX_GHCSI] + y[IDX_GSiC3I] + y[IDX_GSiH3I] + y[IDX_GSiNI] +
        y[IDX_GSiOI] + y[IDX_GC2H6I] + y[IDX_GC3H2I] + y[IDX_GC5H2I] +
        y[IDX_GCCPI] + y[IDX_GCH2CHCCHI] + y[IDX_GCH3COCH3I] + y[IDX_GCH3OCH3I]
        + y[IDX_GClI] + y[IDX_GSiCH2I] + y[IDX_GSiHI] + y[IDX_GSiH2I] +
        y[IDX_GC10H2I] + y[IDX_GC4H2I] + y[IDX_GCH2CHCNI] + y[IDX_GH2CCCI] +
        y[IDX_GHC3NI] + y[IDX_GOCSI] + y[IDX_GC2H5CNI] + y[IDX_GC2H5OHI] +
        y[IDX_GC2OI] + y[IDX_GC6H2I] + y[IDX_GC9NI] + y[IDX_GCO2I] + y[IDX_GCPI]
        + y[IDX_GO2HI] + y[IDX_GOCNI] + y[IDX_GSOI] + y[IDX_GSiCI] +
        y[IDX_GC11I] + y[IDX_GC2NI] + y[IDX_GCH2CCH2I] + y[IDX_GCH2NHI] +
        y[IDX_GCH3CCHI] + y[IDX_GCNOI] + y[IDX_GHNCI] + y[IDX_GNSI] +
        y[IDX_GSiC2I] + y[IDX_GCH2CCHI] + y[IDX_GCH2COI] + y[IDX_GHNOI] +
        y[IDX_GN2I] + y[IDX_GC5NI] + y[IDX_GPI] + y[IDX_GC2H5I] + y[IDX_GC3NI] +
        y[IDX_GC7NI] + y[IDX_GCOOCH3I] + y[IDX_GHCNI] + y[IDX_GNH3I] +
        y[IDX_GC10I] + y[IDX_GC10HI] + y[IDX_GC8HI] + y[IDX_GCH2OHCOI] +
        y[IDX_GCOOHI] + y[IDX_GO2I] + y[IDX_GC2H4I] + y[IDX_GC9I] + y[IDX_GC9HI]
        + y[IDX_GCSI] + y[IDX_GHSI] + y[IDX_GCH2OHCHOI] + y[IDX_GHCOOHI] +
        y[IDX_GC2H2I] + y[IDX_GC5I] + y[IDX_GC7HI] + y[IDX_GCH3COI] +
        y[IDX_GHCOOCH3I] + y[IDX_GC6I] + y[IDX_GC8I] + y[IDX_GCH2OHI] +
        y[IDX_GCH3OHI] + y[IDX_GC3HI] + y[IDX_GC4HI] + y[IDX_GC5HI] +
        y[IDX_GC6HI] + y[IDX_GC2H3I] + y[IDX_GCH3OI] + y[IDX_GH2OI] +
        y[IDX_GC7I] + y[IDX_GSiI] + y[IDX_GC4I] + y[IDX_GCH3CHOI] +
        y[IDX_GH2COI] + y[IDX_GNOI] + y[IDX_GCH4I] + y[IDX_GC2HI] + y[IDX_GNH2I]
        + y[IDX_GHCOI] + y[IDX_GNHI] + y[IDX_GSI] + y[IDX_GC3I] + y[IDX_GOHI] +
        y[IDX_GCOI] + y[IDX_GCH2I] + y[IDX_GC2I] + y[IDX_GCNI] + y[IDX_GNI] +
        y[IDX_GOI] + y[IDX_GCH3I] + y[IDX_GCHI] + y[IDX_GH2I] + y[IDX_GCI] +
        y[IDX_GHI] + 0.0;
}

__device__ __host__ double GetHNuclei(double *y) {
#ifdef IDX_ELEM_H
    return GetElementAbund(y, IDX_ELEM_H);
#else
    return 0.0;
#endif
}

__device__ __host__ double GetMu(double *y) {
    // TODO: exclude electron, grain?
    double mass = 56.0*y[IDX_GFeI] + 4.0*y[IDX_GHeI] + 24.0*y[IDX_GMgI] + 23.0*y[IDX_GNaI] + 
                  66.0*y[IDX_GH2S2I] + 54.0*y[IDX_GC4H6I] + 80.0*y[IDX_GC4SI] + 47.0*y[IDX_GCClI] + 
                  65.0*y[IDX_GCH3C3NI] + 64.0*y[IDX_GCH3C4HI] + 89.0*y[IDX_GCH3C5NI] + 88.0*y[IDX_GCH3C6HI] + 
                  113.0*y[IDX_GCH3C7NI] + 51.0*y[IDX_GClOI] + 56.0*y[IDX_GHC2PI] + 36.0*y[IDX_GHClI] + 
                  20.0*y[IDX_GHFI] + 43.0*y[IDX_GHNSiI] + 48.0*y[IDX_GHPOI] + 42.0*y[IDX_GNH2CNI] + 
                  46.0*y[IDX_GNO2I] + 45.0*y[IDX_GPNI] + 65.0*y[IDX_GSiC3HI] + 43.0*y[IDX_GSiCH3I] + 
                  60.0*y[IDX_GSiO2I] + 60.0*y[IDX_GSiSI] + 19.0*y[IDX_GFI] + 46.0*y[IDX_GH2SiOI] + 
                  52.0*y[IDX_GNCCNI] + 64.0*y[IDX_GS2I] + 53.0*y[IDX_GSiC2HI] + 54.0*y[IDX_C2H4CNI] + 
                  68.0*y[IDX_GC3SI] + 62.0*y[IDX_GC4NI] + 79.0*y[IDX_GC4PI] + 46.0*y[IDX_GCH2PHI] + 
                  42.0*y[IDX_GCH3CHCH2I] + 60.0*y[IDX_GCH3COOHI] + 34.0*y[IDX_GH2SI] + 43.0*y[IDX_GHCNOI] + 
                  44.0*y[IDX_GHCPI] + 51.0*y[IDX_GHNC3I] + 43.0*y[IDX_GHNCOI] + 43.0*y[IDX_GHOCNI] + 
                  43.0*y[IDX_GHONCI] + 65.0*y[IDX_GHS2I] + 44.0*y[IDX_GN2OI] + 33.0*y[IDX_GPH2I] + 
                  47.0*y[IDX_GPOI] + 54.0*y[IDX_GSiC2H2I] + 76.0*y[IDX_GSiC4I] + 54.0*y[IDX_GSiNCI] + 
                  41.0*y[IDX_HC2OI] + 67.0*y[IDX_GC3PI] + 78.0*y[IDX_GC6H6I] + 34.0*y[IDX_GH2O2I] + 
                  41.0*y[IDX_GHCSiI] + 32.0*y[IDX_GPHI] + 64.0*y[IDX_GSO2I] + 32.0*y[IDX_GSiH4I] + 
                  67.0*y[IDX_NCCNCH3II] + 54.0*y[IDX_GC2H4CNI] + 56.0*y[IDX_GC2SI] + 52.0*y[IDX_GC3OI] + 
                  51.0*y[IDX_GC4H3I] + 86.0*y[IDX_GC7H2I] + 98.0*y[IDX_GC8H2I] + 110.0*y[IDX_GC9H2I] + 
                  40.0*y[IDX_GCH2CNI] + 41.0*y[IDX_GCH3CNI] + 28.0*y[IDX_GH2CNI] + 46.0*y[IDX_GH2CSI] + 
                  41.0*y[IDX_GHC2OI] + 75.0*y[IDX_GHC5NI] + 99.0*y[IDX_GHC7NI] + 123.0*y[IDX_GHC9NI] + 
                  39.0*y[IDX_GHCCNI] + 45.0*y[IDX_GHCSI] + 64.0*y[IDX_GSiC3I] + 31.0*y[IDX_GSiH3I] + 
                  42.0*y[IDX_GSiNI] + 44.0*y[IDX_GSiOI] + 39.0*y[IDX_HCCNI] + 54.0*y[IDX_C3H2OII] + 
                  31.0*y[IDX_CFII] + 51.0*y[IDX_ClOII] + 30.0*y[IDX_GC2H6I] + 38.0*y[IDX_GC3H2I] + 
                  62.0*y[IDX_GC5H2I] + 55.0*y[IDX_GCCPI] + 52.0*y[IDX_GCH2CHCCHI] + 58.0*y[IDX_GCH3COCH3I] + 
                  46.0*y[IDX_GCH3OCH3I] + 35.0*y[IDX_GClI] + 42.0*y[IDX_GSiCH2I] + 29.0*y[IDX_GSiHI] + 
                  30.0*y[IDX_GSiH2I] + 55.0*y[IDX_H3C3OII] + 20.0*y[IDX_HFII] + 47.0*y[IDX_SiFII] + 
                  101.0*y[IDX_C8H5II] + 113.0*y[IDX_C9H5II] + 65.0*y[IDX_CH3C3NII] + 122.0*y[IDX_GC10H2I] + 
                  50.0*y[IDX_GC4H2I] + 53.0*y[IDX_GCH2CHCNI] + 38.0*y[IDX_GH2CCCI] + 51.0*y[IDX_GHC3NI] + 
                  60.0*y[IDX_GOCSI] + 21.0*y[IDX_H2FII] + 54.0*y[IDX_C4H6I] + 79.0*y[IDX_C4PII] + 
                  74.0*y[IDX_C5NII] + 78.0*y[IDX_C6H6II] + 53.0*y[IDX_CH2CHCNII] + 59.0*y[IDX_COOCH3II] + 
                  19.0*y[IDX_FII] + 55.0*y[IDX_GC2H5CNI] + 46.0*y[IDX_GC2H5OHI] + 40.0*y[IDX_GC2OI] + 
                  74.0*y[IDX_GC6H2I] + 122.0*y[IDX_GC9NI] + 44.0*y[IDX_GCO2I] + 43.0*y[IDX_GCPI] + 
                  33.0*y[IDX_GO2HI] + 42.0*y[IDX_GOCNI] + 48.0*y[IDX_GSOI] + 40.0*y[IDX_GSiCI] + 
                  44.0*y[IDX_H2CNOII] + 44.0*y[IDX_H2NCOII] + 44.0*y[IDX_H2OCNII] + 59.0*y[IDX_PC2H4II] + 
                  98.0*y[IDX_C7NII] + 122.0*y[IDX_C9NII] + 30.0*y[IDX_CH3NHII] + 51.0*y[IDX_ClOI] + 
                  132.0*y[IDX_GC11I] + 38.0*y[IDX_GC2NI] + 40.0*y[IDX_GCH2CCH2I] + 29.0*y[IDX_GCH2NHI] + 
                  40.0*y[IDX_GCH3CCHI] + 42.0*y[IDX_GCNOI] + 27.0*y[IDX_GHNCI] + 46.0*y[IDX_GNSI] + 
                  52.0*y[IDX_GSiC2I] + 64.0*y[IDX_H2C4NII] + 49.0*y[IDX_H2CClII] + 32.0*y[IDX_H2NOII] + 
                  36.0*y[IDX_HClII] + 47.0*y[IDX_HNSII] + 29.0*y[IDX_HOCII] + 49.0*y[IDX_HSOII] + 
                  61.0*y[IDX_HSiO2II] + 43.0*y[IDX_NH2CNHII] + 46.0*y[IDX_C2H5OHII] + 40.0*y[IDX_C2OII] + 
                  52.0*y[IDX_CH2CHCCHI] + 59.0*y[IDX_CH2OHCOII] + 39.0*y[IDX_GCH2CCHI] + 42.0*y[IDX_GCH2COI] + 
                  31.0*y[IDX_GHNOI] + 28.0*y[IDX_GN2I] + 63.0*y[IDX_HC4NII] + 44.0*y[IDX_HCNOHII] + 
                  44.0*y[IDX_HNCOHII] + 43.0*y[IDX_HOCNII] + 5.0*y[IDX_HeHII] + 58.0*y[IDX_PC2H3II] + 
                  45.0*y[IDX_PNII] + 48.0*y[IDX_PNH3II] + 59.0*y[IDX_CH3CSII] + 74.0*y[IDX_GC5NI] + 
                  31.0*y[IDX_GPI] + 100.0*y[IDX_H2C7NII] + 124.0*y[IDX_H2C9NII] + 34.0*y[IDX_H2O2I] + 
                  67.0*y[IDX_H3S2II] + 43.0*y[IDX_HONCII] + 46.0*y[IDX_PCH3II] + 47.0*y[IDX_PNH2II] + 
                  123.0*y[IDX_C10H3II] + 56.0*y[IDX_C2H5CNHII] + 47.0*y[IDX_CClII] + 30.0*y[IDX_CH2NH2II] + 
                  61.0*y[IDX_CH2OHCH2OII] + 90.0*y[IDX_CH3C5NHII] + 114.0*y[IDX_CH3C7NHII] + 29.0*y[IDX_GC2H5I] + 
                  50.0*y[IDX_GC3NI] + 98.0*y[IDX_GC7NI] + 59.0*y[IDX_GCOOCH3I] + 27.0*y[IDX_GHCNI] + 
                  28.0*y[IDX_H2CNI] + 46.0*y[IDX_H2SiOII] + 77.0*y[IDX_H3C5NII] + 43.0*y[IDX_HCNOII] + 
                  47.0*y[IDX_CClI] + 60.0*y[IDX_CH3COOHII] + 35.0*y[IDX_ClII] + 17.0*y[IDX_GNH3I] + 
                  37.0*y[IDX_H2ClII] + 125.0*y[IDX_H3C9NII] + 61.0*y[IDX_H5C2O2II] + 46.0*y[IDX_HCOOHII] + 
                  46.0*y[IDX_HPNII] + 56.0*y[IDX_C2SII] + 68.0*y[IDX_C3SII] + 62.0*y[IDX_C4NI] + 
                  61.0*y[IDX_CH3COOH2II] + 46.0*y[IDX_CH3OCH3II] + 120.0*y[IDX_GC10I] + 121.0*y[IDX_GC10HI] + 
                  97.0*y[IDX_GC8HI] + 59.0*y[IDX_GCH2OHCOI] + 45.0*y[IDX_GCOOHI] + 32.0*y[IDX_GO2I] + 
                  101.0*y[IDX_H3C7NII] + 47.0*y[IDX_H3SiOII] + 65.0*y[IDX_HSO2II] + 42.0*y[IDX_NH2CNI] + 
                  66.0*y[IDX_CH3C3NHII] + 47.0*y[IDX_CH3OCH4II] + 28.0*y[IDX_GC2H4I] + 108.0*y[IDX_GC9I] + 
                  109.0*y[IDX_GC9HI] + 44.0*y[IDX_GCSI] + 99.0*y[IDX_HC7NII] + 123.0*y[IDX_HC9NII] + 
                  43.0*y[IDX_HNCOII] + 46.0*y[IDX_NO2II] + 68.0*y[IDX_PC3HII] + 47.0*y[IDX_PCH4II] + 
                  32.0*y[IDX_SiH4II] + 60.0*y[IDX_SiO2I] + 55.0*y[IDX_C2H5CNI] + 47.0*y[IDX_C2H5OH2II] + 
                  31.0*y[IDX_C2H7II] + 55.0*y[IDX_CCPII] + 54.0*y[IDX_CH2CHCNHII] + 59.0*y[IDX_COOCH3I] + 
                  19.0*y[IDX_FI] + 33.0*y[IDX_GHSI] + 49.0*y[IDX_H2POII] + 60.0*y[IDX_HCOOCH3II] + 
                  45.0*y[IDX_HN2OII] + 44.0*y[IDX_N2OII] + 57.0*y[IDX_PC2H2II] + 66.0*y[IDX_SiC3H2II] + 
                  76.0*y[IDX_SiC4II] + 77.0*y[IDX_SiC4HII] + 33.0*y[IDX_SiH5II] + 55.0*y[IDX_SiNCHII] + 
                  132.0*y[IDX_C11I] + 52.0*y[IDX_C3OII] + 53.0*y[IDX_C4H5II] + 55.0*y[IDX_C4H7II] + 
                  60.0*y[IDX_CH2OHCHOII] + 44.0*y[IDX_CH3CHOII] + 60.0*y[IDX_GCH2OHCHOI] + 46.0*y[IDX_GHCOOHI] + 
                  66.0*y[IDX_H2S2I] + 43.0*y[IDX_HNSiII] + 61.0*y[IDX_HOCSII] + 46.0*y[IDX_NSII] + 
                  42.0*y[IDX_OCNII] + 80.0*y[IDX_PC4HII] + 34.0*y[IDX_PH3II] + 44.0*y[IDX_SiCH4II] + 
                  120.0*y[IDX_C10II] + 122.0*y[IDX_C10H2II] + 52.0*y[IDX_C2N2II] + 41.0*y[IDX_CH3CNII] + 
                  26.0*y[IDX_GC2H2I] + 60.0*y[IDX_GC5I] + 85.0*y[IDX_GC7HI] + 43.0*y[IDX_GCH3COI] + 
                  60.0*y[IDX_GHCOOCH3I] + 81.0*y[IDX_HC4SII] + 76.0*y[IDX_HC5NHII] + 55.0*y[IDX_SiC2H3II] + 
                  65.0*y[IDX_SiC3HI] + 43.0*y[IDX_SiCH3II] + 44.0*y[IDX_SiNH2II] + 132.0*y[IDX_C11II] + 
                  50.0*y[IDX_C3NII] + 52.0*y[IDX_C4H4II] + 40.0*y[IDX_CH2CNII] + 58.0*y[IDX_CH3COCH3II] + 
                  43.0*y[IDX_CPII] + 72.0*y[IDX_GC6I] + 96.0*y[IDX_GC8I] + 31.0*y[IDX_GCH2OHI] + 
                  32.0*y[IDX_GCH3OHI] + 66.0*y[IDX_H2S2II] + 56.0*y[IDX_HC2PII] + 75.0*y[IDX_HC5NII] + 
                  20.0*y[IDX_HFI] + 43.0*y[IDX_HONCI] + 61.0*y[IDX_HSiSII] + 45.0*y[IDX_PCH2II] + 
                  64.0*y[IDX_SO2II] + 42.0*y[IDX_SiNII] + 108.0*y[IDX_C9II] + 59.0*y[IDX_CH2OHCOI] + 
                  89.0*y[IDX_CH3C5NI] + 113.0*y[IDX_CH3C7NI] + 37.0*y[IDX_GC3HI] + 49.0*y[IDX_GC4HI] + 
                  61.0*y[IDX_GC5HI] + 73.0*y[IDX_GC6HI] + 28.0*y[IDX_H2NCII] + 46.0*y[IDX_H2SiOI] + 
                  47.0*y[IDX_H3CSII] + 41.0*y[IDX_HC2OII] + 44.0*y[IDX_HCPII] + 76.0*y[IDX_SiC4I] + 
                  54.0*y[IDX_SiNCI] + 54.0*y[IDX_SiNCII] + 42.0*y[IDX_CNOI] + 27.0*y[IDX_GC2H3I] + 
                  31.0*y[IDX_GCH3OI] + 18.0*y[IDX_GH2OI] + 69.0*y[IDX_HC3SII] + 47.0*y[IDX_HCOOH2II] + 
                  43.0*y[IDX_HNSiI] + 48.0*y[IDX_HPOII] + 65.0*y[IDX_HS2II] + 54.0*y[IDX_SiC2H2I] + 
                  64.0*y[IDX_SiC3II] + 39.0*y[IDX_C2NHII] + 42.0*y[IDX_C3H6II] + 79.0*y[IDX_C6H7II] + 
                  100.0*y[IDX_C8H4II] + 60.0*y[IDX_CH2OHCHOI] + 35.0*y[IDX_ClI] + 84.0*y[IDX_GC7I] + 
                  28.0*y[IDX_GSiI] + 53.0*y[IDX_HC3OII] + 43.0*y[IDX_HCNOI] + 65.0*y[IDX_SiC3HII] + 
                  43.0*y[IDX_SiCH3I] + 43.0*y[IDX_C3H7II] + 62.0*y[IDX_C4NII] + 96.0*y[IDX_C8II] + 
                  112.0*y[IDX_C9H4II] + 42.0*y[IDX_CH2COII] + 46.0*y[IDX_CH2PHI] + 65.0*y[IDX_CH3C3NI] + 
                  48.0*y[IDX_GC4I] + 44.0*y[IDX_GCH3CHOI] + 30.0*y[IDX_GH2COI] + 43.0*y[IDX_HOCNI] + 
                  48.0*y[IDX_HPOI] + 65.0*y[IDX_HS2I] + 52.0*y[IDX_SiC2II] + 54.0*y[IDX_SiC2H2II] + 
                  76.0*y[IDX_C6H4II] + 43.0*y[IDX_CH3COI] + 59.0*y[IDX_CH3COCH4II] + 32.0*y[IDX_CH3OHII] + 
                  45.0*y[IDX_COOHI] + 30.0*y[IDX_GNOI] + 47.0*y[IDX_POII] + 88.0*y[IDX_C7H4II] + 
                  89.0*y[IDX_C7H5II] + 88.0*y[IDX_CH3C6HI] + 46.0*y[IDX_H2CSII] + 123.0*y[IDX_HC9NI] + 
                  60.0*y[IDX_HCOOCH3I] + 53.0*y[IDX_NCCNHII] + 46.0*y[IDX_NO2I] + 33.0*y[IDX_PH2I] + 
                  45.0*y[IDX_PNI] + 53.0*y[IDX_SiC2HI] + 64.0*y[IDX_SiC3I] + 79.0*y[IDX_C4PI] + 
                  84.0*y[IDX_C7II] + 40.0*y[IDX_CH2CNI] + 64.0*y[IDX_CH3C4HII] + 16.0*y[IDX_GCH4I] + 
                  44.0*y[IDX_HCPI] + 36.0*y[IDX_HClI] + 43.0*y[IDX_HNCOI] + 67.0*y[IDX_C3PI] + 
                  122.0*y[IDX_C9NI] + 64.0*y[IDX_CH3C4HI] + 60.0*y[IDX_CH3COOHI] + 46.0*y[IDX_CH3OCH3I] + 
                  25.0*y[IDX_GC2HI] + 16.0*y[IDX_GNH2I] + 46.0*y[IDX_H2CSI] + 33.0*y[IDX_PH2II] + 
                  47.0*y[IDX_POI] + 64.0*y[IDX_S2I] + 64.0*y[IDX_S2II] + 40.0*y[IDX_SiCII] + 
                  68.0*y[IDX_C3SI] + 77.0*y[IDX_C6H5II] + 29.0*y[IDX_CH2NHI] + 31.0*y[IDX_CH3OI] + 
                  29.0*y[IDX_GHCOI] + 56.0*y[IDX_HC2PI] + 41.0*y[IDX_HCSiI] + 42.0*y[IDX_SiCH2I] + 
                  121.0*y[IDX_C10HII] + 41.0*y[IDX_HCSiII] + 60.0*y[IDX_OCSII] + 52.0*y[IDX_SiC2I] + 
                  30.0*y[IDX_C2H6II] + 40.0*y[IDX_C2OI] + 97.0*y[IDX_C8HII] + 31.0*y[IDX_CH2OHI] + 
                  42.0*y[IDX_SiNI] + 65.0*y[IDX_C5H5II] + 15.0*y[IDX_GNHI] + 32.0*y[IDX_GSI] + 
                  45.0*y[IDX_HCSI] + 46.0*y[IDX_NSI] + 53.0*y[IDX_SiC2HII] + 30.0*y[IDX_SiH2I] + 
                  52.0*y[IDX_C3OI] + 73.0*y[IDX_C6HII] + 110.0*y[IDX_C9H2I] + 53.0*y[IDX_CH2CHCNI] + 
                  45.0*y[IDX_CH3CHOHII] + 42.0*y[IDX_CH3CNHII] + 43.0*y[IDX_CH3COII] + 99.0*y[IDX_HC7NI] + 
                  72.0*y[IDX_C6II] + 78.0*y[IDX_C6H6I] + 98.0*y[IDX_C7NI] + 46.0*y[IDX_HCOOHI] + 
                  30.0*y[IDX_SiH2II] + 31.0*y[IDX_SiH3II] + 122.0*y[IDX_C10H2I] + 48.0*y[IDX_C4II] + 
                  55.0*y[IDX_CCPI] + 43.0*y[IDX_CPI] + 36.0*y[IDX_GC3I] + 33.0*y[IDX_O2HI] + 
                  64.0*y[IDX_SO2I] + 40.0*y[IDX_SiCI] + 46.0*y[IDX_C2H5OHI] + 31.0*y[IDX_HNOI] + 
                  44.0*y[IDX_N2OI] + 31.0*y[IDX_SiH3I] + 33.0*y[IDX_CH3OH2II] + 38.0*y[IDX_C2NII] + 
                  60.0*y[IDX_C5II] + 75.0*y[IDX_C6H3II] + 99.0*y[IDX_C8H3II] + 109.0*y[IDX_C9HII] + 
                  111.0*y[IDX_C9H3II] + 42.0*y[IDX_CH2COI] + 51.0*y[IDX_HNC3I] + 42.0*y[IDX_OCNI] + 
                  42.0*y[IDX_SiCH2II] + 29.0*y[IDX_SiHII] + 58.0*y[IDX_CH3COCH3I] + 51.0*y[IDX_HC3NII] + 
                  98.0*y[IDX_C8H2I] + 44.0*y[IDX_CO2II] + 17.0*y[IDX_GOHI] + 75.0*y[IDX_HC5NI] + 
                  29.0*y[IDX_SiHI] + 32.0*y[IDX_SiH4I] + 85.0*y[IDX_C7HII] + 110.0*y[IDX_C9H2II] + 
                  32.0*y[IDX_PHI] + 86.0*y[IDX_C7H2I] + 28.0*y[IDX_GCOI] + 45.0*y[IDX_HCSII] + 
                  32.0*y[IDX_PHII] + 40.0*y[IDX_C3H4II] + 98.0*y[IDX_C8H2II] + 14.0*y[IDX_GCH2I] + 
                  45.0*y[IDX_HCO2II] + 63.0*y[IDX_C5H3II] + 52.0*y[IDX_HC3NHII] + 61.0*y[IDX_C5HII] + 
                  44.0*y[IDX_CH3CHOI] + 44.0*y[IDX_CSII] + 41.0*y[IDX_C3H5II] + 87.0*y[IDX_C7H3II] + 
                  31.0*y[IDX_PII] + 41.0*y[IDX_CH3CNI] + 52.0*y[IDX_NCCNI] + 86.0*y[IDX_C7H2II] + 
                  24.0*y[IDX_GC2I] + 36.0*y[IDX_C3II] + 35.0*y[IDX_H3SII] + 33.0*y[IDX_O2HII] + 
                  74.0*y[IDX_C6H2II] + 16.0*y[IDX_CH4II] + 31.0*y[IDX_HNOII] + 38.0*y[IDX_H2CCCI] + 
                  74.0*y[IDX_C6H2I] + 26.0*y[IDX_GCNI] + 29.0*y[IDX_C2H5II] + 62.0*y[IDX_C5H2I] + 
                  62.0*y[IDX_C5H2II] + 40.0*y[IDX_CH2CCH2I] + 26.0*y[IDX_CNII] + 30.0*y[IDX_C2H6I] + 
                  14.0*y[IDX_GNI] + 38.0*y[IDX_C3H2I] + 29.0*y[IDX_C2H5I] + 16.0*y[IDX_GOI] + 
                  28.0*y[IDX_N2II] + 80.0*y[IDX_C4SII] + 27.0*y[IDX_HCNII] + 42.0*y[IDX_CH3CHCH2I] + 
                  24.0*y[IDX_C2II] + 49.0*y[IDX_C4HII] + 38.0*y[IDX_CNCII] + 33.0*y[IDX_HSII] + 
                  15.0*y[IDX_NHII] + 32.0*y[IDX_O2M] + 13.0*y[IDX_CHM] + 2.0*y[IDX_H2II] + 
                  60.0*y[IDX_OCSI] + 15.0*y[IDX_GCH3I] + 16.0*y[IDX_NH2II] + 31.0*y[IDX_PI] + 
                  32.0*y[IDX_SM] + 60.0*y[IDX_SiSII] + 25.0*y[IDX_C2HM] + 49.0*y[IDX_C4HM] + 
                  28.0*y[IDX_COII] + 40.0*y[IDX_CH3CCHI] + 17.0*y[IDX_CH5II] + 57.0*y[IDX_HC2SII] + 
                  17.0*y[IDX_OHM] + 121.0*y[IDX_C10HM] + 37.0*y[IDX_C3HM] + 80.0*y[IDX_C4SI] + 
                  73.0*y[IDX_C6HM] + 97.0*y[IDX_C8HM] + 85.0*y[IDX_C7HM] + 109.0*y[IDX_C9HM] + 
                  120.0*y[IDX_C10M] + 61.0*y[IDX_C5HM] + 74.0*y[IDX_C5NM] + 33.0*y[IDX_HSI] + 
                  12.0*y[IDX_CM] + 44.0*y[IDX_CSI] + 16.0*y[IDX_OM] + 108.0*y[IDX_C9M] + 
                  32.0*y[IDX_CH3OHI] + 1.0*y[IDX_HM] + 51.0*y[IDX_HC3NI] + 60.0*y[IDX_SiSI] + 
                  37.0*y[IDX_C3HII] + 39.0*y[IDX_C3H3II] + 96.0*y[IDX_C8M] + 24.0*y[IDX_C2M] + 
                  56.0*y[IDX_C2SI] + 36.0*y[IDX_C3M] + 48.0*y[IDX_C4M] + 72.0*y[IDX_C6M] + 
                  31.0*y[IDX_H3COII] + 60.0*y[IDX_C5M] + 84.0*y[IDX_C7M] + 44.0*y[IDX_SiOII] + 
                  38.0*y[IDX_C2NI] + 13.0*y[IDX_GCHI] + 45.0*y[IDX_SiOHII] + 16.0*y[IDX_NH2I] + 
                  51.0*y[IDX_C4H3I] + 25.0*y[IDX_C2HII] + 38.0*y[IDX_C3H2II] + 74.0*y[IDX_C5NI] + 
                  14.0*y[IDX_CH2II] + 17.0*y[IDX_OHII] + 18.0*y[IDX_H2OII] + 50.0*y[IDX_C3NM] + 
                  2.0*y[IDX_GH2I] + 50.0*y[IDX_C3NI] + 13.0*y[IDX_CHII] + 44.0*y[IDX_CO2I] + 
                  32.0*y[IDX_O2II] + 121.0*y[IDX_C10HI] + 109.0*y[IDX_C9HI] + 44.0*y[IDX_SiOI] + 
                  56.0*y[IDX_FeII] + 56.0*y[IDX_FeI] + 24.0*y[IDX_MgII] + 23.0*y[IDX_NaII] + 
                  24.0*y[IDX_MgI] + 23.0*y[IDX_NaI] + 120.0*y[IDX_C10I] + 97.0*y[IDX_C8HI] + 
                  84.0*y[IDX_C7I] + 108.0*y[IDX_C9I] + 96.0*y[IDX_C8I] + 12.0*y[IDX_GCI] + 
                  15.0*y[IDX_NHI] + 60.0*y[IDX_C5I] + 72.0*y[IDX_C6I] + 26.0*y[IDX_CNM] + 
                  48.0*y[IDX_C4I] + 48.0*y[IDX_SOI] + 85.0*y[IDX_C7HI] + 28.0*y[IDX_C2H4II] + 
                  51.0*y[IDX_C4H3II] + 29.0*y[IDX_N2HII] + 39.0*y[IDX_CH2CCHI] + 48.0*y[IDX_SOII] + 
                  39.0*y[IDX_CH2CCHII] + 73.0*y[IDX_C6HI] + 61.0*y[IDX_C5HI] + 34.0*y[IDX_H2SII] + 
                  14.0*y[IDX_NII] + 37.0*y[IDX_C3HI] + 30.0*y[IDX_H2COII] + 36.0*y[IDX_C3I] + 
                  49.0*y[IDX_C4HI] + 30.0*y[IDX_NOII] + 16.0*y[IDX_OII] + 50.0*y[IDX_C4H2I] + 
                  50.0*y[IDX_C4H2II] + 28.0*y[IDX_HCNHII] + 14.0*y[IDX_CH2I] + 27.0*y[IDX_HNCI] + 
                  17.0*y[IDX_NH3II] + 28.0*y[IDX_C2H4I] + 27.0*y[IDX_C2H3I] + 18.0*y[IDX_NH4II] + 
                  28.0*y[IDX_N2I] + 32.0*y[IDX_O2I] + 28.0*y[IDX_SiII] + 34.0*y[IDX_H2SI] + 
                  32.0*y[IDX_SII] + 28.0*y[IDX_SiI] + 27.0*y[IDX_C2H3II] + 30.0*y[IDX_NOI] + 
                  29.0*y[IDX_HCOI] + 16.0*y[IDX_CH4I] + 25.0*y[IDX_C2HI] + 30.0*y[IDX_H2COI] + 
                  27.0*y[IDX_HCNI] + 13.0*y[IDX_CHI] + 32.0*y[IDX_SI] + 24.0*y[IDX_C2I] + 
                  17.0*y[IDX_OHI] + 17.0*y[IDX_NH3I] + 26.0*y[IDX_C2H2II] + 15.0*y[IDX_CH3II] + 
                  26.0*y[IDX_C2H2I] + 26.0*y[IDX_CNI] + 1.0*y[IDX_GHI] + 15.0*y[IDX_CH3I] + 
                  14.0*y[IDX_NI] + 19.0*y[IDX_H3OII] + 16.0*y[IDX_OI] + 4.0*y[IDX_HeII] + 
                  4.0*y[IDX_HeI] + 12.0*y[IDX_CII] + 18.0*y[IDX_H2OI] + 1.0*y[IDX_HII] + 
                  12.0*y[IDX_CI] + 29.0*y[IDX_HCOII] + 3.0*y[IDX_H3II] + 28.0*y[IDX_COI] + 
                  0.0*y[IDX_GRAINM] + 0.0*y[IDX_GRAIN0I] + 2.0*y[IDX_H2I] + 0.0*y[IDX_eM] + 
                  1.0*y[IDX_HI] + 0.0;
    double num = y[IDX_GFeI] + y[IDX_GHeI] + y[IDX_GMgI] + y[IDX_GNaI] +
                 y[IDX_GH2S2I] + y[IDX_GC4H6I] + y[IDX_GC4SI] + y[IDX_GCClI] +
                 y[IDX_GCH3C3NI] + y[IDX_GCH3C4HI] + y[IDX_GCH3C5NI] +
                 y[IDX_GCH3C6HI] + y[IDX_GCH3C7NI] + y[IDX_GClOI] +
                 y[IDX_GHC2PI] + y[IDX_GHClI] + y[IDX_GHFI] + y[IDX_GHNSiI] +
                 y[IDX_GHPOI] + y[IDX_GNH2CNI] + y[IDX_GNO2I] + y[IDX_GPNI] +
                 y[IDX_GSiC3HI] + y[IDX_GSiCH3I] + y[IDX_GSiO2I] + y[IDX_GSiSI]
                 + y[IDX_GFI] + y[IDX_GH2SiOI] + y[IDX_GNCCNI] + y[IDX_GS2I] +
                 y[IDX_GSiC2HI] + y[IDX_C2H4CNI] + y[IDX_GC3SI] + y[IDX_GC4NI] +
                 y[IDX_GC4PI] + y[IDX_GCH2PHI] + y[IDX_GCH3CHCH2I] +
                 y[IDX_GCH3COOHI] + y[IDX_GH2SI] + y[IDX_GHCNOI] + y[IDX_GHCPI]
                 + y[IDX_GHNC3I] + y[IDX_GHNCOI] + y[IDX_GHOCNI] + y[IDX_GHONCI]
                 + y[IDX_GHS2I] + y[IDX_GN2OI] + y[IDX_GPH2I] + y[IDX_GPOI] +
                 y[IDX_GSiC2H2I] + y[IDX_GSiC4I] + y[IDX_GSiNCI] + y[IDX_HC2OI]
                 + y[IDX_GC3PI] + y[IDX_GC6H6I] + y[IDX_GH2O2I] + y[IDX_GHCSiI]
                 + y[IDX_GPHI] + y[IDX_GSO2I] + y[IDX_GSiH4I] + y[IDX_NCCNCH3II]
                 + y[IDX_GC2H4CNI] + y[IDX_GC2SI] + y[IDX_GC3OI] + y[IDX_GC4H3I]
                 + y[IDX_GC7H2I] + y[IDX_GC8H2I] + y[IDX_GC9H2I] +
                 y[IDX_GCH2CNI] + y[IDX_GCH3CNI] + y[IDX_GH2CNI] + y[IDX_GH2CSI]
                 + y[IDX_GHC2OI] + y[IDX_GHC5NI] + y[IDX_GHC7NI] + y[IDX_GHC9NI]
                 + y[IDX_GHCCNI] + y[IDX_GHCSI] + y[IDX_GSiC3I] + y[IDX_GSiH3I]
                 + y[IDX_GSiNI] + y[IDX_GSiOI] + y[IDX_HCCNI] + y[IDX_C3H2OII] +
                 y[IDX_CFII] + y[IDX_ClOII] + y[IDX_GC2H6I] + y[IDX_GC3H2I] +
                 y[IDX_GC5H2I] + y[IDX_GCCPI] + y[IDX_GCH2CHCCHI] +
                 y[IDX_GCH3COCH3I] + y[IDX_GCH3OCH3I] + y[IDX_GClI] +
                 y[IDX_GSiCH2I] + y[IDX_GSiHI] + y[IDX_GSiH2I] + y[IDX_H3C3OII]
                 + y[IDX_HFII] + y[IDX_SiFII] + y[IDX_C8H5II] + y[IDX_C9H5II] +
                 y[IDX_CH3C3NII] + y[IDX_GC10H2I] + y[IDX_GC4H2I] +
                 y[IDX_GCH2CHCNI] + y[IDX_GH2CCCI] + y[IDX_GHC3NI] +
                 y[IDX_GOCSI] + y[IDX_H2FII] + y[IDX_C4H6I] + y[IDX_C4PII] +
                 y[IDX_C5NII] + y[IDX_C6H6II] + y[IDX_CH2CHCNII] +
                 y[IDX_COOCH3II] + y[IDX_FII] + y[IDX_GC2H5CNI] +
                 y[IDX_GC2H5OHI] + y[IDX_GC2OI] + y[IDX_GC6H2I] + y[IDX_GC9NI] +
                 y[IDX_GCO2I] + y[IDX_GCPI] + y[IDX_GO2HI] + y[IDX_GOCNI] +
                 y[IDX_GSOI] + y[IDX_GSiCI] + y[IDX_H2CNOII] + y[IDX_H2NCOII] +
                 y[IDX_H2OCNII] + y[IDX_PC2H4II] + y[IDX_C7NII] + y[IDX_C9NII] +
                 y[IDX_CH3NHII] + y[IDX_ClOI] + y[IDX_GC11I] + y[IDX_GC2NI] +
                 y[IDX_GCH2CCH2I] + y[IDX_GCH2NHI] + y[IDX_GCH3CCHI] +
                 y[IDX_GCNOI] + y[IDX_GHNCI] + y[IDX_GNSI] + y[IDX_GSiC2I] +
                 y[IDX_H2C4NII] + y[IDX_H2CClII] + y[IDX_H2NOII] + y[IDX_HClII]
                 + y[IDX_HNSII] + y[IDX_HOCII] + y[IDX_HSOII] + y[IDX_HSiO2II] +
                 y[IDX_NH2CNHII] + y[IDX_C2H5OHII] + y[IDX_C2OII] +
                 y[IDX_CH2CHCCHI] + y[IDX_CH2OHCOII] + y[IDX_GCH2CCHI] +
                 y[IDX_GCH2COI] + y[IDX_GHNOI] + y[IDX_GN2I] + y[IDX_HC4NII] +
                 y[IDX_HCNOHII] + y[IDX_HNCOHII] + y[IDX_HOCNII] + y[IDX_HeHII]
                 + y[IDX_PC2H3II] + y[IDX_PNII] + y[IDX_PNH3II] + y[IDX_CH3CSII]
                 + y[IDX_GC5NI] + y[IDX_GPI] + y[IDX_H2C7NII] + y[IDX_H2C9NII] +
                 y[IDX_H2O2I] + y[IDX_H3S2II] + y[IDX_HONCII] + y[IDX_PCH3II] +
                 y[IDX_PNH2II] + y[IDX_C10H3II] + y[IDX_C2H5CNHII] +
                 y[IDX_CClII] + y[IDX_CH2NH2II] + y[IDX_CH2OHCH2OII] +
                 y[IDX_CH3C5NHII] + y[IDX_CH3C7NHII] + y[IDX_GC2H5I] +
                 y[IDX_GC3NI] + y[IDX_GC7NI] + y[IDX_GCOOCH3I] + y[IDX_GHCNI] +
                 y[IDX_H2CNI] + y[IDX_H2SiOII] + y[IDX_H3C5NII] + y[IDX_HCNOII]
                 + y[IDX_CClI] + y[IDX_CH3COOHII] + y[IDX_ClII] + y[IDX_GNH3I] +
                 y[IDX_H2ClII] + y[IDX_H3C9NII] + y[IDX_H5C2O2II] +
                 y[IDX_HCOOHII] + y[IDX_HPNII] + y[IDX_C2SII] + y[IDX_C3SII] +
                 y[IDX_C4NI] + y[IDX_CH3COOH2II] + y[IDX_CH3OCH3II] +
                 y[IDX_GC10I] + y[IDX_GC10HI] + y[IDX_GC8HI] + y[IDX_GCH2OHCOI]
                 + y[IDX_GCOOHI] + y[IDX_GO2I] + y[IDX_H3C7NII] + y[IDX_H3SiOII]
                 + y[IDX_HSO2II] + y[IDX_NH2CNI] + y[IDX_CH3C3NHII] +
                 y[IDX_CH3OCH4II] + y[IDX_GC2H4I] + y[IDX_GC9I] + y[IDX_GC9HI] +
                 y[IDX_GCSI] + y[IDX_HC7NII] + y[IDX_HC9NII] + y[IDX_HNCOII] +
                 y[IDX_NO2II] + y[IDX_PC3HII] + y[IDX_PCH4II] + y[IDX_SiH4II] +
                 y[IDX_SiO2I] + y[IDX_C2H5CNI] + y[IDX_C2H5OH2II] +
                 y[IDX_C2H7II] + y[IDX_CCPII] + y[IDX_CH2CHCNHII] +
                 y[IDX_COOCH3I] + y[IDX_FI] + y[IDX_GHSI] + y[IDX_H2POII] +
                 y[IDX_HCOOCH3II] + y[IDX_HN2OII] + y[IDX_N2OII] +
                 y[IDX_PC2H2II] + y[IDX_SiC3H2II] + y[IDX_SiC4II] +
                 y[IDX_SiC4HII] + y[IDX_SiH5II] + y[IDX_SiNCHII] + y[IDX_C11I] +
                 y[IDX_C3OII] + y[IDX_C4H5II] + y[IDX_C4H7II] +
                 y[IDX_CH2OHCHOII] + y[IDX_CH3CHOII] + y[IDX_GCH2OHCHOI] +
                 y[IDX_GHCOOHI] + y[IDX_H2S2I] + y[IDX_HNSiII] + y[IDX_HOCSII] +
                 y[IDX_NSII] + y[IDX_OCNII] + y[IDX_PC4HII] + y[IDX_PH3II] +
                 y[IDX_SiCH4II] + y[IDX_C10II] + y[IDX_C10H2II] + y[IDX_C2N2II]
                 + y[IDX_CH3CNII] + y[IDX_GC2H2I] + y[IDX_GC5I] + y[IDX_GC7HI] +
                 y[IDX_GCH3COI] + y[IDX_GHCOOCH3I] + y[IDX_HC4SII] +
                 y[IDX_HC5NHII] + y[IDX_SiC2H3II] + y[IDX_SiC3HI] +
                 y[IDX_SiCH3II] + y[IDX_SiNH2II] + y[IDX_C11II] + y[IDX_C3NII] +
                 y[IDX_C4H4II] + y[IDX_CH2CNII] + y[IDX_CH3COCH3II] +
                 y[IDX_CPII] + y[IDX_GC6I] + y[IDX_GC8I] + y[IDX_GCH2OHI] +
                 y[IDX_GCH3OHI] + y[IDX_H2S2II] + y[IDX_HC2PII] + y[IDX_HC5NII]
                 + y[IDX_HFI] + y[IDX_HONCI] + y[IDX_HSiSII] + y[IDX_PCH2II] +
                 y[IDX_SO2II] + y[IDX_SiNII] + y[IDX_C9II] + y[IDX_CH2OHCOI] +
                 y[IDX_CH3C5NI] + y[IDX_CH3C7NI] + y[IDX_GC3HI] + y[IDX_GC4HI] +
                 y[IDX_GC5HI] + y[IDX_GC6HI] + y[IDX_H2NCII] + y[IDX_H2SiOI] +
                 y[IDX_H3CSII] + y[IDX_HC2OII] + y[IDX_HCPII] + y[IDX_SiC4I] +
                 y[IDX_SiNCI] + y[IDX_SiNCII] + y[IDX_CNOI] + y[IDX_GC2H3I] +
                 y[IDX_GCH3OI] + y[IDX_GH2OI] + y[IDX_HC3SII] + y[IDX_HCOOH2II]
                 + y[IDX_HNSiI] + y[IDX_HPOII] + y[IDX_HS2II] + y[IDX_SiC2H2I] +
                 y[IDX_SiC3II] + y[IDX_C2NHII] + y[IDX_C3H6II] + y[IDX_C6H7II] +
                 y[IDX_C8H4II] + y[IDX_CH2OHCHOI] + y[IDX_ClI] + y[IDX_GC7I] +
                 y[IDX_GSiI] + y[IDX_HC3OII] + y[IDX_HCNOI] + y[IDX_SiC3HII] +
                 y[IDX_SiCH3I] + y[IDX_C3H7II] + y[IDX_C4NII] + y[IDX_C8II] +
                 y[IDX_C9H4II] + y[IDX_CH2COII] + y[IDX_CH2PHI] + y[IDX_CH3C3NI]
                 + y[IDX_GC4I] + y[IDX_GCH3CHOI] + y[IDX_GH2COI] + y[IDX_HOCNI]
                 + y[IDX_HPOI] + y[IDX_HS2I] + y[IDX_SiC2II] + y[IDX_SiC2H2II] +
                 y[IDX_C6H4II] + y[IDX_CH3COI] + y[IDX_CH3COCH4II] +
                 y[IDX_CH3OHII] + y[IDX_COOHI] + y[IDX_GNOI] + y[IDX_POII] +
                 y[IDX_C7H4II] + y[IDX_C7H5II] + y[IDX_CH3C6HI] + y[IDX_H2CSII]
                 + y[IDX_HC9NI] + y[IDX_HCOOCH3I] + y[IDX_NCCNHII] + y[IDX_NO2I]
                 + y[IDX_PH2I] + y[IDX_PNI] + y[IDX_SiC2HI] + y[IDX_SiC3I] +
                 y[IDX_C4PI] + y[IDX_C7II] + y[IDX_CH2CNI] + y[IDX_CH3C4HII] +
                 y[IDX_GCH4I] + y[IDX_HCPI] + y[IDX_HClI] + y[IDX_HNCOI] +
                 y[IDX_C3PI] + y[IDX_C9NI] + y[IDX_CH3C4HI] + y[IDX_CH3COOHI] +
                 y[IDX_CH3OCH3I] + y[IDX_GC2HI] + y[IDX_GNH2I] + y[IDX_H2CSI] +
                 y[IDX_PH2II] + y[IDX_POI] + y[IDX_S2I] + y[IDX_S2II] +
                 y[IDX_SiCII] + y[IDX_C3SI] + y[IDX_C6H5II] + y[IDX_CH2NHI] +
                 y[IDX_CH3OI] + y[IDX_GHCOI] + y[IDX_HC2PI] + y[IDX_HCSiI] +
                 y[IDX_SiCH2I] + y[IDX_C10HII] + y[IDX_HCSiII] + y[IDX_OCSII] +
                 y[IDX_SiC2I] + y[IDX_C2H6II] + y[IDX_C2OI] + y[IDX_C8HII] +
                 y[IDX_CH2OHI] + y[IDX_SiNI] + y[IDX_C5H5II] + y[IDX_GNHI] +
                 y[IDX_GSI] + y[IDX_HCSI] + y[IDX_NSI] + y[IDX_SiC2HII] +
                 y[IDX_SiH2I] + y[IDX_C3OI] + y[IDX_C6HII] + y[IDX_C9H2I] +
                 y[IDX_CH2CHCNI] + y[IDX_CH3CHOHII] + y[IDX_CH3CNHII] +
                 y[IDX_CH3COII] + y[IDX_HC7NI] + y[IDX_C6II] + y[IDX_C6H6I] +
                 y[IDX_C7NI] + y[IDX_HCOOHI] + y[IDX_SiH2II] + y[IDX_SiH3II] +
                 y[IDX_C10H2I] + y[IDX_C4II] + y[IDX_CCPI] + y[IDX_CPI] +
                 y[IDX_GC3I] + y[IDX_O2HI] + y[IDX_SO2I] + y[IDX_SiCI] +
                 y[IDX_C2H5OHI] + y[IDX_HNOI] + y[IDX_N2OI] + y[IDX_SiH3I] +
                 y[IDX_CH3OH2II] + y[IDX_C2NII] + y[IDX_C5II] + y[IDX_C6H3II] +
                 y[IDX_C8H3II] + y[IDX_C9HII] + y[IDX_C9H3II] + y[IDX_CH2COI] +
                 y[IDX_HNC3I] + y[IDX_OCNI] + y[IDX_SiCH2II] + y[IDX_SiHII] +
                 y[IDX_CH3COCH3I] + y[IDX_HC3NII] + y[IDX_C8H2I] + y[IDX_CO2II]
                 + y[IDX_GOHI] + y[IDX_HC5NI] + y[IDX_SiHI] + y[IDX_SiH4I] +
                 y[IDX_C7HII] + y[IDX_C9H2II] + y[IDX_PHI] + y[IDX_C7H2I] +
                 y[IDX_GCOI] + y[IDX_HCSII] + y[IDX_PHII] + y[IDX_C3H4II] +
                 y[IDX_C8H2II] + y[IDX_GCH2I] + y[IDX_HCO2II] + y[IDX_C5H3II] +
                 y[IDX_HC3NHII] + y[IDX_C5HII] + y[IDX_CH3CHOI] + y[IDX_CSII] +
                 y[IDX_C3H5II] + y[IDX_C7H3II] + y[IDX_PII] + y[IDX_CH3CNI] +
                 y[IDX_NCCNI] + y[IDX_C7H2II] + y[IDX_GC2I] + y[IDX_C3II] +
                 y[IDX_H3SII] + y[IDX_O2HII] + y[IDX_C6H2II] + y[IDX_CH4II] +
                 y[IDX_HNOII] + y[IDX_H2CCCI] + y[IDX_C6H2I] + y[IDX_GCNI] +
                 y[IDX_C2H5II] + y[IDX_C5H2I] + y[IDX_C5H2II] + y[IDX_CH2CCH2I]
                 + y[IDX_CNII] + y[IDX_C2H6I] + y[IDX_GNI] + y[IDX_C3H2I] +
                 y[IDX_C2H5I] + y[IDX_GOI] + y[IDX_N2II] + y[IDX_C4SII] +
                 y[IDX_HCNII] + y[IDX_CH3CHCH2I] + y[IDX_C2II] + y[IDX_C4HII] +
                 y[IDX_CNCII] + y[IDX_HSII] + y[IDX_NHII] + y[IDX_O2M] +
                 y[IDX_CHM] + y[IDX_H2II] + y[IDX_OCSI] + y[IDX_GCH3I] +
                 y[IDX_NH2II] + y[IDX_PI] + y[IDX_SM] + y[IDX_SiSII] +
                 y[IDX_C2HM] + y[IDX_C4HM] + y[IDX_COII] + y[IDX_CH3CCHI] +
                 y[IDX_CH5II] + y[IDX_HC2SII] + y[IDX_OHM] + y[IDX_C10HM] +
                 y[IDX_C3HM] + y[IDX_C4SI] + y[IDX_C6HM] + y[IDX_C8HM] +
                 y[IDX_C7HM] + y[IDX_C9HM] + y[IDX_C10M] + y[IDX_C5HM] +
                 y[IDX_C5NM] + y[IDX_HSI] + y[IDX_CM] + y[IDX_CSI] + y[IDX_OM] +
                 y[IDX_C9M] + y[IDX_CH3OHI] + y[IDX_HM] + y[IDX_HC3NI] +
                 y[IDX_SiSI] + y[IDX_C3HII] + y[IDX_C3H3II] + y[IDX_C8M] +
                 y[IDX_C2M] + y[IDX_C2SI] + y[IDX_C3M] + y[IDX_C4M] + y[IDX_C6M]
                 + y[IDX_H3COII] + y[IDX_C5M] + y[IDX_C7M] + y[IDX_SiOII] +
                 y[IDX_C2NI] + y[IDX_GCHI] + y[IDX_SiOHII] + y[IDX_NH2I] +
                 y[IDX_C4H3I] + y[IDX_C2HII] + y[IDX_C3H2II] + y[IDX_C5NI] +
                 y[IDX_CH2II] + y[IDX_OHII] + y[IDX_H2OII] + y[IDX_C3NM] +
                 y[IDX_GH2I] + y[IDX_C3NI] + y[IDX_CHII] + y[IDX_CO2I] +
                 y[IDX_O2II] + y[IDX_C10HI] + y[IDX_C9HI] + y[IDX_SiOI] +
                 y[IDX_FeII] + y[IDX_FeI] + y[IDX_MgII] + y[IDX_NaII] +
                 y[IDX_MgI] + y[IDX_NaI] + y[IDX_C10I] + y[IDX_C8HI] +
                 y[IDX_C7I] + y[IDX_C9I] + y[IDX_C8I] + y[IDX_GCI] + y[IDX_NHI]
                 + y[IDX_C5I] + y[IDX_C6I] + y[IDX_CNM] + y[IDX_C4I] +
                 y[IDX_SOI] + y[IDX_C7HI] + y[IDX_C2H4II] + y[IDX_C4H3II] +
                 y[IDX_N2HII] + y[IDX_CH2CCHI] + y[IDX_SOII] + y[IDX_CH2CCHII] +
                 y[IDX_C6HI] + y[IDX_C5HI] + y[IDX_H2SII] + y[IDX_NII] +
                 y[IDX_C3HI] + y[IDX_H2COII] + y[IDX_C3I] + y[IDX_C4HI] +
                 y[IDX_NOII] + y[IDX_OII] + y[IDX_C4H2I] + y[IDX_C4H2II] +
                 y[IDX_HCNHII] + y[IDX_CH2I] + y[IDX_HNCI] + y[IDX_NH3II] +
                 y[IDX_C2H4I] + y[IDX_C2H3I] + y[IDX_NH4II] + y[IDX_N2I] +
                 y[IDX_O2I] + y[IDX_SiII] + y[IDX_H2SI] + y[IDX_SII] +
                 y[IDX_SiI] + y[IDX_C2H3II] + y[IDX_NOI] + y[IDX_HCOI] +
                 y[IDX_CH4I] + y[IDX_C2HI] + y[IDX_H2COI] + y[IDX_HCNI] +
                 y[IDX_CHI] + y[IDX_SI] + y[IDX_C2I] + y[IDX_OHI] + y[IDX_NH3I]
                 + y[IDX_C2H2II] + y[IDX_CH3II] + y[IDX_C2H2I] + y[IDX_CNI] +
                 y[IDX_GHI] + y[IDX_CH3I] + y[IDX_NI] + y[IDX_H3OII] + y[IDX_OI]
                 + y[IDX_HeII] + y[IDX_HeI] + y[IDX_CII] + y[IDX_H2OI] +
                 y[IDX_HII] + y[IDX_CI] + y[IDX_HCOII] + y[IDX_H3II] +
                 y[IDX_COI] + y[IDX_GRAINM] + y[IDX_GRAIN0I] + y[IDX_H2I] +
                 y[IDX_eM] + y[IDX_HI] + 0.0;

    return mass / num;
}

__device__ __host__ double GetGamma(double *y) {
    // TODO: different ways to get adiabatic index
    return 5.0 / 3.0;
}

__device__ __host__ double GetNumDens(double *y) {
    double numdens = 0.0;

    for (int i = 0; i < NSPECIES; i++) numdens += y[i];
    return numdens;
}
// clang-format on

/*  */
// clang-format off
__device__ 
double GetShieldingFactor(int specidx, double h2coldens, double spcoldens,
                          double tgas, int method) {
    // clang-format on
    double factor;
#ifdef IDX_H2I
    if (specidx == IDX_H2I) {
        factor = GetH2shielding(h2coldens, method);
    }
#endif
#ifdef IDX_COI
    if (specidx == IDX_COI) {
        factor = GetCOshielding(tgas, h2coldens, spcoldens, method);
    }
#endif
#ifdef IDX_N2I
    if (specidx == IDX_N2I) {
        factor = GetN2shielding(tgas, h2coldens, spcoldens, method);
    }
#endif

    return factor;
}

// clang-format off
__device__ 
double GetH2shielding(double coldens, int method) {
    // clang-format on
    double shielding = -1.0;
    switch (method) {
        case 0:
            shielding = GetH2shieldingInt(coldens);
            break;
        case 1:
            shielding = GetH2shieldingFGK(coldens);
            break;
        default:
            break;
    }
    return shielding;
}

// clang-format off
__device__ double GetCOshielding(double tgas, double h2col, double coldens, int method) {
    // clang-format on
    double shielding = -1.0;
    switch (method) {
        case 0:
            shielding = GetCOshieldingInt(tgas, h2col, coldens);
            break;
        case 1:
            shielding = GetCOshieldingInt1(h2col, coldens);
            break;
        default:
            break;
    }
    return shielding;
}

// clang-format off
__device__ double GetN2shielding(double tgas, double h2col, double coldens, int method) {
    // clang-format on
    double shielding = -1.0;
    switch (method) {
        case 0:
            shielding = GetN2shieldingInt(tgas, h2col, coldens);
            break;
        default:
            break;
    }
    return shielding;
}

// Interpolate/Extropolate from table (must be rendered in naunet constants)
// clang-format off
__device__ double GetH2shieldingInt(double coldens) {
    // clang-format on

    double shielding = -1.0;

    /* */
    int i;
    for (i = 0; i < 103; i++) {
        if (coldens < H2ShieldingTableX[i + 1]) {
            break;
        }
    }
    double x1 = H2ShieldingTableX[i];
    double x2 = H2ShieldingTableX[i + 1];
    double y1 = H2ShieldingTable[i];
    double y2 = H2ShieldingTable[i + 1];
    shielding =
        log10(y1) + log10(y2 / y1) * log10(coldens / x1) / log10(x2 / x1);
    shielding = pow(10.0, shielding);
    return shielding;

    /* */

    return shielding;
}

// Calculates the line self shielding function
// Ref: Federman et al. apj vol.227 p.466.
// Originally implemented in UCLCHEM
// clang-format off
__device__ double GetH2shieldingFGK(double coldens) {
    // clang-format on

    const double dopplerwidth       = 3.0e10;
    const double radiativewidth     = 8.0e7;
    const double oscillatorstrength = 1.0e-2;

    double shielding                = -1.0;

    double taud = 0.5 * coldens * 1.5e-2 * oscillatorstrength / dopplerwidth;

    // Calculate wing contribution of self shielding function sr
    if (taud < 0.0) taud = 0.0;

    double sr = 0.0;
    if (radiativewidth != 0.0) {
        double r = radiativewidth / (1.7724539 * dopplerwidth);
        double t = 3.02 * pow(1000.0 * r, -0.064);
        double u = pow(taud * r, 0.5) / t;
        sr       = pow((u * u + 0.78539816), -0.5) * r / t;
    }

    // Calculate doppler contribution of self shielding function sj
    double sj = 0.0;
    if (taud == 0.0) {
        sj = 1.0;
    } else if (taud < 2.0) {
        sj = exp(-0.6666667 * taud);
    } else if (taud < 10.0) {
        sj = 0.638 * pow(taud, -1.25);
    } else if (taud < 100.0) {
        sj = 0.505 * pow(taud, -1.15);
    } else {
        sj = 0.344 * pow(taud, -1.0667);
    }

    shielding = sj + sr;

    return shielding;
}

// Interpolate/Extropolate from table (must be rendered in naunet constants)
// clang-format off
__device__ double GetCOshieldingInt(double tgas, double h2col, double coldens) {
    // clang-format on
    double shielding = -1.0;

    /* */
    double x1, x2, y1, y2, z1, z2;
    int i1, i2, j1, j2, k1, k2;
    for (i1 = 0; i1 < 3; i1++) {
        if (tgas < COShieldingTableX[i1 + 1]) {
            break;
        }
    }
    i2 = i1 + 1;
    x1 = COShieldingTableX[i1];
    x2 = COShieldingTableX[i2];

    for (j1 = 0; j1 < 39; j1++) {
        if (h2col < COShieldingTableY[j1 + 1]) {
            break;
        }
    }
    j2 = j1 + 1;
    y1 = COShieldingTableY[j1];
    y2 = COShieldingTableY[j2];

    for (k1 = 0; k1 < 44; k1++) {
        if (coldens < COShieldingTableZ[k1 + 1]) {
            break;
        }
    }
    k2        = k1 + 1;
    z1        = COShieldingTableZ[k1];
    z2        = COShieldingTableZ[k2];

    double mx = log10(tgas / x1) / log10(x2 / x1);
    double my = log10(h2col / y1) / log10(y2 / y1);
    double mz = log10(coldens / z1) / log10(z2 / z1);

    double f1 = log10(COShieldingTable[i1][j1][k1]) * (1 - mx) +
                log10(COShieldingTable[i2][j1][k1]) * mx;
    double f2 = log10(COShieldingTable[i1][j2][k1]) * (1 - mx) +
                log10(COShieldingTable[i2][j2][k1]) * mx;
    double f3 = log10(COShieldingTable[i1][j1][k2]) * (1 - mx) +
                log10(COShieldingTable[i2][j1][k2]) * mx;
    double f4 = log10(COShieldingTable[i1][j2][k2]) * (1 - mx) +
                log10(COShieldingTable[i2][j2][k2]) * mx;

    shielding =
        (f1 * (1 - my) + f2 * my) * (1 - mz) + (f3 * (1 - my) + f4 * my) * mz;
    shielding = pow(10.0, shielding);

    /* */

    return shielding;
}

// clang-format off
__device__ double GetCOshieldingInt1(double h2col, double coldens) {
    // clang-format on
    double shielding = -1.0;

    /* */

    printf("WARNING!! Not Implemented! Return CO shielding = -1.\n");

    /* */

    return shielding;
}

// Interpolate/Extropolate from table (must be rendered in naunet constants)
// clang-format off
__device__ double GetN2shieldingInt(double tgas, double h2col, double coldens) {
    // clang-format on

    double shielding = -1.0;

    /* */
    double x1, x2, y1, y2, z1, z2;
    int i1, i2, j1, j2, k1, k2;
    // find the index where tags falls in the range]
    for (i1 = 0; i1 < 3; i1++) {
        if (tgas < N2ShieldingTableX[i1 + 1]) {
            break;
        }
    }
    i2 = i1 + 1;
    x1 = N2ShieldingTableX[i1];
    x2 = N2ShieldingTableX[i2];

    for (j1 = 0; j1 < 44; j1++) {
        if (h2col < N2ShieldingTableY[j1 + 1]) {
            break;
        }
    }
    j2 = j1 + 1;
    y1 = N2ShieldingTableY[j1];
    y2 = N2ShieldingTableY[j2];

    for (k1 = 0; k1 < 44; k1++) {
        if (coldens < N2ShieldingTableZ[k1 + 1]) {
            break;
        }
    }
    k2        = k1 + 1;
    z1        = N2ShieldingTableZ[k1];
    z2        = N2ShieldingTableZ[k2];

    double mx = log10(tgas / x1) / log10(x2 / x1);
    double my = log10(h2col / y1) / log10(y2 / y1);
    double mz = log10(coldens / z1) / log10(z2 / z1);

    double f1 = log10(N2ShieldingTable[i1][j1][k1]) * (1 - mx) +
                log10(N2ShieldingTable[i2][j1][k1]) * mx;
    double f2 = log10(N2ShieldingTable[i1][j2][k1]) * (1 - mx) +
                log10(N2ShieldingTable[i2][j2][k1]) * mx;
    double f3 = log10(N2ShieldingTable[i1][j1][k2]) * (1 - mx) +
                log10(N2ShieldingTable[i2][j1][k2]) * mx;
    double f4 = log10(N2ShieldingTable[i1][j2][k2]) * (1 - mx) +
                log10(N2ShieldingTable[i2][j2][k2]) * mx;

    shielding =
        (f1 * (1 - my) + f2 * my) * (1 - mz) + (f3 * (1 - my) + f4 * my) * mz;
    shielding = pow(10.0, shielding);

    /* */

    return shielding;
}

// Calculate xlamda := tau(lambda) / tau(visual)
// tau(lambda) is the opt. depth for dust extinction at
// wavelength x (cf. b.d.savage and j.s.mathis, annual
// review of astronomy and astrophysics vol.17(1979),p.84)
// clang-format off
__device__ double xlamda(double wavelength) {
    // clang-format on
    double x[29] = {910.0,  950.0,  1000.0,  1050.0,  1110.0, 1180.0,
                    1250.0, 1390.0, 1490.0,  1600.0,  1700.0, 1800.0,
                    1900.0, 2000.0, 2100.0,  2190.0,  2300.0, 2400.0,
                    2500.0, 2740.0, 3440.0,  4000.0,  4400.0, 5500.0,
                    7000.0, 9000.0, 12500.0, 22000.0, 34000.0};

    double y[29] = {5.76, 5.18, 4.65, 4.16, 3.73, 3.4,  3.11, 2.74, 2.63, 2.62,
                    2.54, 2.5,  2.58, 2.78, 3.01, 3.12, 2.86, 2.58, 2.35, 2.0,
                    1.58, 1.42, 1.32, 1.0,  0.75, 0.48, 0.28, 0.12, 0.05};

    if (wavelength < x[0]) {
        return 5.76;
    }

    else if (wavelength >= x[28]) {
        return 0.05 - 5.16e-11 * (wavelength - x[28]);
    }

    for (int i = 0; i < 28; i++) {
        if (wavelength >= x[i] && wavelength < x[i + 1]) {
            return y[i] +
                   (y[i + 1] - y[i]) * (wavelength - x[i]) / (x[i + 1] - x[i]);
        }
    }

    return 0.0;
}

// Calculate the influence of dust extinction (g=0.8, omega=0.3)
// Ref: Wagenblast & Hartquist, mnras237, 1019 (1989)
// Adapted from UCLCHEM
// clang-format off
__device__ double GetGrainScattering(double av, double wavelength) {
    // clang-format on
    double c[6] = {1.0e0, 2.006e0, -1.438e0, 7.364e-1, -5.076e-1, -5.920e-2};
    double k[6] = {7.514e-1, 8.490e-1, 1.013e0, 1.282e0, 2.005e0, 5.832e0};

    double tv   = av / 1.086;
    double tl   = tv * xlamda(wavelength);

    double scat = 0.0;
    double expo;
    if (tl < 1.0) {
        expo = k[0] * tl;
        if (expo < 35.0) {
            scat = c[0] * exp(-expo);
        }
    } else {
        for (int i = 1; i < 6; i++) {
            expo = k[i] * tl;
            if (expo < 35.0) {
                scat = scat + c[i] * exp(-expo);
            }
        }
    }

    return scat;
}

// Calculate lambda bar (in a) according to equ. 4 of van dishoeck
// and black, apj 334, p771 (1988)
// Adapted from UCLCHEM
// clang-format off
__device__ double GetCharactWavelength(double h2col, double cocol) {
    // clang-format on
    double logco = log10(abs(cocol) + 1.0);
    double logh2 = log10(abs(h2col) + 1.0);

    double lbar  = (5675.0 - 200.6 * logh2) - (571.6 - 24.09 * logh2) * logco +
                  (18.22 - 0.7664 * logh2) * pow(logco, 2.0);

    // lbar represents the mean of the wavelengths of the 33
    // dissociating bands weighted by their fractional contribution
    // to the total rate of each depth. lbar cannot be larger than
    // the wavelength of band 33 (1076.1a) and not be smaller than
    // the wavelength of band 1 (913.6a).

    /* */
    lbar = min(1076.0, max(913.0, lbar));
    /* */
    return lbar;
}