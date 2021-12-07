#include <math.h>

#include "naunet_constants.h"
#include "naunet_macros.h"
#include "naunet_physics.h"

// clang-format off
__device__ __host__ double GetMantleDens(double *y) {
    return y[IDX_GCI] + y[IDX_GC10I] + y[IDX_GC10HI] + y[IDX_GC10H2I] +
        y[IDX_GC11I] + y[IDX_GC2I] + y[IDX_GC2HI] + y[IDX_GC2H2I] +
        y[IDX_GC2H3I] + y[IDX_GC2H4I] + y[IDX_GC2H4CNI] + y[IDX_GC2H5I] +
        y[IDX_GC2H5CNI] + y[IDX_GC2H5OHI] + y[IDX_GC2H6I] + y[IDX_GC2NI] +
        y[IDX_GC2OI] + y[IDX_GC2SI] + y[IDX_GC3I] + y[IDX_GC3HI] + y[IDX_GC3H2I]
        + y[IDX_GC3NI] + y[IDX_GC3OI] + y[IDX_GC3PI] + y[IDX_GC3SI] +
        y[IDX_GC4I] + y[IDX_GC4HI] + y[IDX_GC4H2I] + y[IDX_GC4H3I] +
        y[IDX_GC4H6I] + y[IDX_GC4NI] + y[IDX_GC4PI] + y[IDX_GC4SI] + y[IDX_GC5I]
        + y[IDX_GC5HI] + y[IDX_GC5H2I] + y[IDX_GC5NI] + y[IDX_GC6I] +
        y[IDX_GC6HI] + y[IDX_GC6H2I] + y[IDX_GC6H6I] + y[IDX_GC7I] +
        y[IDX_GC7HI] + y[IDX_GC7H2I] + y[IDX_GC7NI] + y[IDX_GC8I] + y[IDX_GC8HI]
        + y[IDX_GC8H2I] + y[IDX_GC9I] + y[IDX_GC9HI] + y[IDX_GC9H2I] +
        y[IDX_GC9NI] + y[IDX_GCCPI] + y[IDX_GCClI] + y[IDX_GCHI] + y[IDX_GCH2I]
        + y[IDX_GCH2CCHI] + y[IDX_GCH2CCH2I] + y[IDX_GCH2CHCCHI] +
        y[IDX_GCH2CHCNI] + y[IDX_GCH2CNI] + y[IDX_GCH2COI] + y[IDX_GCH2NHI] +
        y[IDX_GCH2OHI] + y[IDX_GCH2OHCHOI] + y[IDX_GCH2OHCOI] + y[IDX_GCH2PHI] +
        y[IDX_GCH3I] + y[IDX_GCH3C3NI] + y[IDX_GCH3C4HI] + y[IDX_GCH3C5NI] +
        y[IDX_GCH3C6HI] + y[IDX_GCH3C7NI] + y[IDX_GCH3CCHI] + y[IDX_GCH3CHCH2I]
        + y[IDX_GCH3CHOI] + y[IDX_GCH3CNI] + y[IDX_GCH3COI] + y[IDX_GCH3COCH3I]
        + y[IDX_GCH3COOHI] + y[IDX_GCH3OI] + y[IDX_GCH3OCH3I] + y[IDX_GCH3OHI] +
        y[IDX_GCH4I] + y[IDX_GCNI] + y[IDX_GCNOI] + y[IDX_GCOI] + y[IDX_GCO2I] +
        y[IDX_GCOOCH3I] + y[IDX_GCOOHI] + y[IDX_GCPI] + y[IDX_GCSI] +
        y[IDX_GClI] + y[IDX_GClOI] + y[IDX_GFI] + y[IDX_GFeI] + y[IDX_GHI] +
        y[IDX_GH2I] + y[IDX_GH2CCCI] + y[IDX_GH2CNI] + y[IDX_GH2COI] +
        y[IDX_GH2CSI] + y[IDX_GH2OI] + y[IDX_GH2O2I] + y[IDX_GH2SI] +
        y[IDX_GH2S2I] + y[IDX_GH2SiOI] + y[IDX_GHC2OI] + y[IDX_GHC2PI] +
        y[IDX_GHC3NI] + y[IDX_GHC5NI] + y[IDX_GHC7NI] + y[IDX_GHC9NI] +
        y[IDX_GHCCNI] + y[IDX_GHCNI] + y[IDX_GHCNOI] + y[IDX_GHCOI] +
        y[IDX_GHCOOCH3I] + y[IDX_GHCOOHI] + y[IDX_GHCPI] + y[IDX_GHCSI] +
        y[IDX_GHCSiI] + y[IDX_GHClI] + y[IDX_GHFI] + y[IDX_GHNCI] +
        y[IDX_GHNC3I] + y[IDX_GHNCOI] + y[IDX_GHNOI] + y[IDX_GHNSiI] +
        y[IDX_GHOCNI] + y[IDX_GHONCI] + y[IDX_GHPOI] + y[IDX_GHSI] +
        y[IDX_GHS2I] + y[IDX_GHeI] + y[IDX_GMgI] + y[IDX_GNI] + y[IDX_GN2I] +
        y[IDX_GN2OI] + y[IDX_GNCCNI] + y[IDX_GNHI] + y[IDX_GNH2I] +
        y[IDX_GNH2CNI] + y[IDX_GNH3I] + y[IDX_GNOI] + y[IDX_GNO2I] + y[IDX_GNSI]
        + y[IDX_GNaI] + y[IDX_GOI] + y[IDX_GO2I] + y[IDX_GO2HI] + y[IDX_GOCNI] +
        y[IDX_GOCSI] + y[IDX_GOHI] + y[IDX_GPI] + y[IDX_GPHI] + y[IDX_GPH2I] +
        y[IDX_GPNI] + y[IDX_GPOI] + y[IDX_GSI] + y[IDX_GS2I] + y[IDX_GSOI] +
        y[IDX_GSO2I] + y[IDX_GSiI] + y[IDX_GSiCI] + y[IDX_GSiC2I] +
        y[IDX_GSiC2HI] + y[IDX_GSiC2H2I] + y[IDX_GSiC3I] + y[IDX_GSiC3HI] +
        y[IDX_GSiC4I] + y[IDX_GSiCH2I] + y[IDX_GSiCH3I] + y[IDX_GSiHI] +
        y[IDX_GSiH2I] + y[IDX_GSiH3I] + y[IDX_GSiH4I] + y[IDX_GSiNI] +
        y[IDX_GSiNCI] + y[IDX_GSiOI] + y[IDX_GSiO2I] + y[IDX_GSiSI];
}

__device__ __host__ double GetMu(double *y) {
    return (y[IDX_CI]*12.0 + y[IDX_CII]*12.0 + y[IDX_CM]*12.0 + y[IDX_C10I]*120.0 +
        y[IDX_C10II]*120.0 + y[IDX_C10M]*120.0 + y[IDX_C10HI]*121.0 +
        y[IDX_C10HII]*121.0 + y[IDX_C10HM]*121.0 + y[IDX_C10H2I]*122.0 +
        y[IDX_C10H2II]*122.0 + y[IDX_C10H3II]*123.0 + y[IDX_C11I]*132.0 +
        y[IDX_C11II]*132.0 + y[IDX_C2I]*24.0 + y[IDX_C2II]*24.0 +
        y[IDX_C2M]*24.0 + y[IDX_C2HI]*25.0 + y[IDX_C2HII]*25.0 +
        y[IDX_C2HM]*25.0 + y[IDX_C2H2I]*26.0 + y[IDX_C2H2II]*26.0 +
        y[IDX_C2H3I]*27.0 + y[IDX_C2H3II]*27.0 + y[IDX_C2H4I]*28.0 +
        y[IDX_C2H4II]*28.0 + y[IDX_C2H4CNI]*54.0 + y[IDX_C2H5I]*29.0 +
        y[IDX_C2H5II]*29.0 + y[IDX_C2H5CNI]*55.0 + y[IDX_C2H5CNHII]*56.0 +
        y[IDX_C2H5OHI]*46.0 + y[IDX_C2H5OHII]*46.0 + y[IDX_C2H5OH2II]*47.0 +
        y[IDX_C2H6I]*30.0 + y[IDX_C2H6II]*30.0 + y[IDX_C2H7II]*31.0 +
        y[IDX_C2NI]*38.0 + y[IDX_C2NII]*38.0 + y[IDX_C2N2II]*52.0 +
        y[IDX_C2NHII]*39.0 + y[IDX_C2OI]*40.0 + y[IDX_C2OII]*40.0 +
        y[IDX_C2SI]*56.0 + y[IDX_C2SII]*56.0 + y[IDX_C3I]*36.0 +
        y[IDX_C3II]*36.0 + y[IDX_C3M]*36.0 + y[IDX_C3HI]*37.0 +
        y[IDX_C3HII]*37.0 + y[IDX_C3HM]*37.0 + y[IDX_C3H2I]*38.0 +
        y[IDX_C3H2II]*38.0 + y[IDX_C3H2OII]*54.0 + y[IDX_C3H3II]*39.0 +
        y[IDX_C3H4II]*40.0 + y[IDX_C3H5II]*41.0 + y[IDX_C3H6II]*42.0 +
        y[IDX_C3H7II]*43.0 + y[IDX_C3NI]*50.0 + y[IDX_C3NII]*50.0 +
        y[IDX_C3NM]*50.0 + y[IDX_C3OI]*52.0 + y[IDX_C3OII]*52.0 +
        y[IDX_C3PI]*67.0 + y[IDX_C3SI]*68.0 + y[IDX_C3SII]*68.0 +
        y[IDX_C4I]*48.0 + y[IDX_C4II]*48.0 + y[IDX_C4M]*48.0 + y[IDX_C4HI]*49.0
        + y[IDX_C4HII]*49.0 + y[IDX_C4HM]*49.0 + y[IDX_C4H2I]*50.0 +
        y[IDX_C4H2II]*50.0 + y[IDX_C4H3I]*51.0 + y[IDX_C4H3II]*51.0 +
        y[IDX_C4H4II]*52.0 + y[IDX_C4H5II]*53.0 + y[IDX_C4H6I]*54.0 +
        y[IDX_C4H7II]*55.0 + y[IDX_C4NI]*62.0 + y[IDX_C4NII]*62.0 +
        y[IDX_C4PI]*79.0 + y[IDX_C4PII]*79.0 + y[IDX_C4SI]*80.0 +
        y[IDX_C4SII]*80.0 + y[IDX_C5I]*60.0 + y[IDX_C5II]*60.0 + y[IDX_C5M]*60.0
        + y[IDX_C5HI]*61.0 + y[IDX_C5HII]*61.0 + y[IDX_C5HM]*61.0 +
        y[IDX_C5H2I]*62.0 + y[IDX_C5H2II]*62.0 + y[IDX_C5H3II]*63.0 +
        y[IDX_C5H5II]*65.0 + y[IDX_C5NI]*74.0 + y[IDX_C5NII]*74.0 +
        y[IDX_C5NM]*74.0 + y[IDX_C6I]*72.0 + y[IDX_C6II]*72.0 + y[IDX_C6M]*72.0
        + y[IDX_C6HI]*73.0 + y[IDX_C6HII]*73.0 + y[IDX_C6HM]*73.0 +
        y[IDX_C6H2I]*74.0 + y[IDX_C6H2II]*74.0 + y[IDX_C6H3II]*75.0 +
        y[IDX_C6H4II]*76.0 + y[IDX_C6H5II]*77.0 + y[IDX_C6H6I]*78.0 +
        y[IDX_C6H6II]*78.0 + y[IDX_C6H7II]*79.0 + y[IDX_C7I]*84.0 +
        y[IDX_C7II]*84.0 + y[IDX_C7M]*84.0 + y[IDX_C7HI]*85.0 +
        y[IDX_C7HII]*85.0 + y[IDX_C7HM]*85.0 + y[IDX_C7H2I]*86.0 +
        y[IDX_C7H2II]*86.0 + y[IDX_C7H3II]*87.0 + y[IDX_C7H4II]*88.0 +
        y[IDX_C7H5II]*89.0 + y[IDX_C7NI]*98.0 + y[IDX_C7NII]*98.0 +
        y[IDX_C8I]*96.0 + y[IDX_C8II]*96.0 + y[IDX_C8M]*96.0 + y[IDX_C8HI]*97.0
        + y[IDX_C8HII]*97.0 + y[IDX_C8HM]*97.0 + y[IDX_C8H2I]*98.0 +
        y[IDX_C8H2II]*98.0 + y[IDX_C8H3II]*99.0 + y[IDX_C8H4II]*100.0 +
        y[IDX_C8H5II]*101.0 + y[IDX_C9I]*108.0 + y[IDX_C9II]*108.0 +
        y[IDX_C9M]*108.0 + y[IDX_C9HI]*109.0 + y[IDX_C9HII]*109.0 +
        y[IDX_C9HM]*109.0 + y[IDX_C9H2I]*110.0 + y[IDX_C9H2II]*110.0 +
        y[IDX_C9H3II]*111.0 + y[IDX_C9H4II]*112.0 + y[IDX_C9H5II]*113.0 +
        y[IDX_C9NI]*122.0 + y[IDX_C9NII]*122.0 + y[IDX_CCPI]*55.0 +
        y[IDX_CCPII]*55.0 + y[IDX_CClI]*47.0 + y[IDX_CClII]*47.0 +
        y[IDX_CFII]*31.0 + y[IDX_CHI]*13.0 + y[IDX_CHII]*13.0 + y[IDX_CHM]*13.0
        + y[IDX_CH2I]*14.0 + y[IDX_CH2II]*14.0 + y[IDX_CH2CCHI]*39.0 +
        y[IDX_CH2CCHII]*39.0 + y[IDX_CH2CCH2I]*40.0 + y[IDX_CH2CHCCHI]*52.0 +
        y[IDX_CH2CHCNI]*53.0 + y[IDX_CH2CHCNII]*53.0 + y[IDX_CH2CHCNHII]*54.0 +
        y[IDX_CH2CNI]*40.0 + y[IDX_CH2CNII]*40.0 + y[IDX_CH2COI]*42.0 +
        y[IDX_CH2COII]*42.0 + y[IDX_CH2NHI]*29.0 + y[IDX_CH2NH2II]*30.0 +
        y[IDX_CH2OHI]*31.0 + y[IDX_CH2OHCH2OII]*61.0 + y[IDX_CH2OHCHOI]*60.0 +
        y[IDX_CH2OHCHOII]*60.0 + y[IDX_CH2OHCOI]*59.0 + y[IDX_CH2OHCOII]*59.0 +
        y[IDX_CH2PHI]*46.0 + y[IDX_CH3I]*15.0 + y[IDX_CH3II]*15.0 +
        y[IDX_CH3C3NI]*65.0 + y[IDX_CH3C3NII]*65.0 + y[IDX_CH3C3NHII]*66.0 +
        y[IDX_CH3C4HI]*64.0 + y[IDX_CH3C4HII]*64.0 + y[IDX_CH3C5NI]*89.0 +
        y[IDX_CH3C5NHII]*90.0 + y[IDX_CH3C6HI]*88.0 + y[IDX_CH3C7NI]*113.0 +
        y[IDX_CH3C7NHII]*114.0 + y[IDX_CH3CCHI]*40.0 + y[IDX_CH3CHCH2I]*42.0 +
        y[IDX_CH3CHOI]*44.0 + y[IDX_CH3CHOII]*44.0 + y[IDX_CH3CHOHII]*45.0 +
        y[IDX_CH3CNI]*41.0 + y[IDX_CH3CNII]*41.0 + y[IDX_CH3CNHII]*42.0 +
        y[IDX_CH3COI]*43.0 + y[IDX_CH3COII]*43.0 + y[IDX_CH3COCH3I]*58.0 +
        y[IDX_CH3COCH3II]*58.0 + y[IDX_CH3COCH4II]*59.0 + y[IDX_CH3COOHI]*60.0 +
        y[IDX_CH3COOHII]*60.0 + y[IDX_CH3COOH2II]*61.0 + y[IDX_CH3CSII]*59.0 +
        y[IDX_CH3NHII]*30.0 + y[IDX_CH3OI]*31.0 + y[IDX_CH3OCH3I]*46.0 +
        y[IDX_CH3OCH3II]*46.0 + y[IDX_CH3OCH4II]*47.0 + y[IDX_CH3OHI]*32.0 +
        y[IDX_CH3OHII]*32.0 + y[IDX_CH3OH2II]*33.0 + y[IDX_CH4I]*16.0 +
        y[IDX_CH4II]*16.0 + y[IDX_CH5II]*17.0 + y[IDX_CNI]*26.0 +
        y[IDX_CNII]*26.0 + y[IDX_CNM]*26.0 + y[IDX_CNCII]*38.0 +
        y[IDX_CNOI]*42.0 + y[IDX_COI]*28.0 + y[IDX_COII]*28.0 + y[IDX_CO2I]*44.0
        + y[IDX_CO2II]*44.0 + y[IDX_COOCH3I]*59.0 + y[IDX_COOCH3II]*59.0 +
        y[IDX_COOHI]*45.0 + y[IDX_CPI]*43.0 + y[IDX_CPII]*43.0 + y[IDX_CSI]*44.0
        + y[IDX_CSII]*44.0 + y[IDX_ClI]*35.0 + y[IDX_ClII]*35.0 +
        y[IDX_ClOI]*51.0 + y[IDX_ClOII]*51.0 + y[IDX_FI]*19.0 + y[IDX_FII]*19.0
        + y[IDX_FeI]*56.0 + y[IDX_FeII]*56.0 + y[IDX_GCI]*12.0 +
        y[IDX_GC10I]*120.0 + y[IDX_GC10HI]*121.0 + y[IDX_GC10H2I]*122.0 +
        y[IDX_GC11I]*132.0 + y[IDX_GC2I]*24.0 + y[IDX_GC2HI]*25.0 +
        y[IDX_GC2H2I]*26.0 + y[IDX_GC2H3I]*27.0 + y[IDX_GC2H4I]*28.0 +
        y[IDX_GC2H4CNI]*54.0 + y[IDX_GC2H5I]*29.0 + y[IDX_GC2H5CNI]*55.0 +
        y[IDX_GC2H5OHI]*46.0 + y[IDX_GC2H6I]*30.0 + y[IDX_GC2NI]*38.0 +
        y[IDX_GC2OI]*40.0 + y[IDX_GC2SI]*56.0 + y[IDX_GC3I]*36.0 +
        y[IDX_GC3HI]*37.0 + y[IDX_GC3H2I]*38.0 + y[IDX_GC3NI]*50.0 +
        y[IDX_GC3OI]*52.0 + y[IDX_GC3PI]*67.0 + y[IDX_GC3SI]*68.0 +
        y[IDX_GC4I]*48.0 + y[IDX_GC4HI]*49.0 + y[IDX_GC4H2I]*50.0 +
        y[IDX_GC4H3I]*51.0 + y[IDX_GC4H6I]*54.0 + y[IDX_GC4NI]*62.0 +
        y[IDX_GC4PI]*79.0 + y[IDX_GC4SI]*80.0 + y[IDX_GC5I]*60.0 +
        y[IDX_GC5HI]*61.0 + y[IDX_GC5H2I]*62.0 + y[IDX_GC5NI]*74.0 +
        y[IDX_GC6I]*72.0 + y[IDX_GC6HI]*73.0 + y[IDX_GC6H2I]*74.0 +
        y[IDX_GC6H6I]*78.0 + y[IDX_GC7I]*84.0 + y[IDX_GC7HI]*85.0 +
        y[IDX_GC7H2I]*86.0 + y[IDX_GC7NI]*98.0 + y[IDX_GC8I]*96.0 +
        y[IDX_GC8HI]*97.0 + y[IDX_GC8H2I]*98.0 + y[IDX_GC9I]*108.0 +
        y[IDX_GC9HI]*109.0 + y[IDX_GC9H2I]*110.0 + y[IDX_GC9NI]*122.0 +
        y[IDX_GCCPI]*55.0 + y[IDX_GCClI]*47.0 + y[IDX_GCHI]*13.0 +
        y[IDX_GCH2I]*14.0 + y[IDX_GCH2CCHI]*39.0 + y[IDX_GCH2CCH2I]*40.0 +
        y[IDX_GCH2CHCCHI]*52.0 + y[IDX_GCH2CHCNI]*53.0 + y[IDX_GCH2CNI]*40.0 +
        y[IDX_GCH2COI]*42.0 + y[IDX_GCH2NHI]*29.0 + y[IDX_GCH2OHI]*31.0 +
        y[IDX_GCH2OHCHOI]*60.0 + y[IDX_GCH2OHCOI]*59.0 + y[IDX_GCH2PHI]*46.0 +
        y[IDX_GCH3I]*15.0 + y[IDX_GCH3C3NI]*65.0 + y[IDX_GCH3C4HI]*64.0 +
        y[IDX_GCH3C5NI]*89.0 + y[IDX_GCH3C6HI]*88.0 + y[IDX_GCH3C7NI]*113.0 +
        y[IDX_GCH3CCHI]*40.0 + y[IDX_GCH3CHCH2I]*42.0 + y[IDX_GCH3CHOI]*44.0 +
        y[IDX_GCH3CNI]*41.0 + y[IDX_GCH3COI]*43.0 + y[IDX_GCH3COCH3I]*58.0 +
        y[IDX_GCH3COOHI]*60.0 + y[IDX_GCH3OI]*31.0 + y[IDX_GCH3OCH3I]*46.0 +
        y[IDX_GCH3OHI]*32.0 + y[IDX_GCH4I]*16.0 + y[IDX_GCNI]*26.0 +
        y[IDX_GCNOI]*42.0 + y[IDX_GCOI]*28.0 + y[IDX_GCO2I]*44.0 +
        y[IDX_GCOOCH3I]*59.0 + y[IDX_GCOOHI]*45.0 + y[IDX_GCPI]*43.0 +
        y[IDX_GCSI]*44.0 + y[IDX_GClI]*35.0 + y[IDX_GClOI]*51.0 +
        y[IDX_GFI]*19.0 + y[IDX_GFeI]*56.0 + y[IDX_GHI]*1.0 + y[IDX_GH2I]*2.0 +
        y[IDX_GH2CCCI]*38.0 + y[IDX_GH2CNI]*28.0 + y[IDX_GH2COI]*30.0 +
        y[IDX_GH2CSI]*46.0 + y[IDX_GH2OI]*18.0 + y[IDX_GH2O2I]*34.0 +
        y[IDX_GH2SI]*34.0 + y[IDX_GH2S2I]*66.0 + y[IDX_GH2SiOI]*46.0 +
        y[IDX_GHC2OI]*41.0 + y[IDX_GHC2PI]*56.0 + y[IDX_GHC3NI]*51.0 +
        y[IDX_GHC5NI]*75.0 + y[IDX_GHC7NI]*99.0 + y[IDX_GHC9NI]*123.0 +
        y[IDX_GHCCNI]*39.0 + y[IDX_GHCNI]*27.0 + y[IDX_GHCNOI]*43.0 +
        y[IDX_GHCOI]*29.0 + y[IDX_GHCOOCH3I]*60.0 + y[IDX_GHCOOHI]*46.0 +
        y[IDX_GHCPI]*44.0 + y[IDX_GHCSI]*45.0 + y[IDX_GHCSiI]*41.0 +
        y[IDX_GHClI]*36.0 + y[IDX_GHFI]*20.0 + y[IDX_GHNCI]*27.0 +
        y[IDX_GHNC3I]*51.0 + y[IDX_GHNCOI]*43.0 + y[IDX_GHNOI]*31.0 +
        y[IDX_GHNSiI]*43.0 + y[IDX_GHOCNI]*43.0 + y[IDX_GHONCI]*43.0 +
        y[IDX_GHPOI]*48.0 + y[IDX_GHSI]*33.0 + y[IDX_GHS2I]*65.0 +
        y[IDX_GHeI]*4.0 + y[IDX_GMgI]*24.0 + y[IDX_GNI]*14.0 + y[IDX_GN2I]*28.0
        + y[IDX_GN2OI]*44.0 + y[IDX_GNCCNI]*52.0 + y[IDX_GNHI]*15.0 +
        y[IDX_GNH2I]*16.0 + y[IDX_GNH2CNI]*42.0 + y[IDX_GNH3I]*17.0 +
        y[IDX_GNOI]*30.0 + y[IDX_GNO2I]*46.0 + y[IDX_GNSI]*46.0 +
        y[IDX_GNaI]*23.0 + y[IDX_GOI]*16.0 + y[IDX_GO2I]*32.0 +
        y[IDX_GO2HI]*33.0 + y[IDX_GOCNI]*42.0 + y[IDX_GOCSI]*60.0 +
        y[IDX_GOHI]*17.0 + y[IDX_GPI]*31.0 + y[IDX_GPHI]*32.0 +
        y[IDX_GPH2I]*33.0 + y[IDX_GPNI]*45.0 + y[IDX_GPOI]*47.0 +
        y[IDX_GRAINM]*0.0 + y[IDX_GRAIN0I]*0.0 + y[IDX_GSI]*32.0 +
        y[IDX_GS2I]*64.0 + y[IDX_GSOI]*48.0 + y[IDX_GSO2I]*64.0 +
        y[IDX_GSiI]*28.0 + y[IDX_GSiCI]*40.0 + y[IDX_GSiC2I]*52.0 +
        y[IDX_GSiC2HI]*53.0 + y[IDX_GSiC2H2I]*54.0 + y[IDX_GSiC3I]*64.0 +
        y[IDX_GSiC3HI]*65.0 + y[IDX_GSiC4I]*76.0 + y[IDX_GSiCH2I]*42.0 +
        y[IDX_GSiCH3I]*43.0 + y[IDX_GSiHI]*29.0 + y[IDX_GSiH2I]*30.0 +
        y[IDX_GSiH3I]*31.0 + y[IDX_GSiH4I]*32.0 + y[IDX_GSiNI]*42.0 +
        y[IDX_GSiNCI]*54.0 + y[IDX_GSiOI]*44.0 + y[IDX_GSiO2I]*60.0 +
        y[IDX_GSiSI]*60.0 + y[IDX_HI]*1.0 + y[IDX_HII]*1.0 + y[IDX_HM]*1.0 +
        y[IDX_H2I]*2.0 + y[IDX_H2II]*2.0 + y[IDX_H2C4NII]*64.0 +
        y[IDX_H2C7NII]*100.0 + y[IDX_H2C9NII]*124.0 + y[IDX_H2CCCI]*38.0 +
        y[IDX_H2CClII]*49.0 + y[IDX_H2CNI]*28.0 + y[IDX_H2CNOII]*44.0 +
        y[IDX_H2COI]*30.0 + y[IDX_H2COII]*30.0 + y[IDX_H2CSI]*46.0 +
        y[IDX_H2CSII]*46.0 + y[IDX_H2ClII]*37.0 + y[IDX_H2FII]*21.0 +
        y[IDX_H2NCII]*28.0 + y[IDX_H2NCOII]*44.0 + y[IDX_H2NOII]*32.0 +
        y[IDX_H2OI]*18.0 + y[IDX_H2OII]*18.0 + y[IDX_H2O2I]*34.0 +
        y[IDX_H2OCNII]*44.0 + y[IDX_H2POII]*49.0 + y[IDX_H2SI]*34.0 +
        y[IDX_H2SII]*34.0 + y[IDX_H2S2I]*66.0 + y[IDX_H2S2II]*66.0 +
        y[IDX_H2SiOI]*46.0 + y[IDX_H2SiOII]*46.0 + y[IDX_H3II]*3.0 +
        y[IDX_H3C3OII]*55.0 + y[IDX_H3C5NII]*77.0 + y[IDX_H3C7NII]*101.0 +
        y[IDX_H3C9NII]*125.0 + y[IDX_H3COII]*31.0 + y[IDX_H3CSII]*47.0 +
        y[IDX_H3OII]*19.0 + y[IDX_H3SII]*35.0 + y[IDX_H3S2II]*67.0 +
        y[IDX_H3SiOII]*47.0 + y[IDX_H5C2O2II]*61.0 + y[IDX_HC2OI]*41.0 +
        y[IDX_HC2OII]*41.0 + y[IDX_HC2PI]*56.0 + y[IDX_HC2PII]*56.0 +
        y[IDX_HC2SII]*57.0 + y[IDX_HC3NI]*51.0 + y[IDX_HC3NII]*51.0 +
        y[IDX_HC3NHII]*52.0 + y[IDX_HC3OII]*53.0 + y[IDX_HC3SII]*69.0 +
        y[IDX_HC4NII]*63.0 + y[IDX_HC4SII]*81.0 + y[IDX_HC5NI]*75.0 +
        y[IDX_HC5NII]*75.0 + y[IDX_HC5NHII]*76.0 + y[IDX_HC7NI]*99.0 +
        y[IDX_HC7NII]*99.0 + y[IDX_HC9NI]*123.0 + y[IDX_HC9NII]*123.0 +
        y[IDX_HCCNI]*39.0 + y[IDX_HCNI]*27.0 + y[IDX_HCNII]*27.0 +
        y[IDX_HCNHII]*28.0 + y[IDX_HCNOI]*43.0 + y[IDX_HCNOII]*43.0 +
        y[IDX_HCNOHII]*44.0 + y[IDX_HCOI]*29.0 + y[IDX_HCOII]*29.0 +
        y[IDX_HCO2II]*45.0 + y[IDX_HCOOCH3I]*60.0 + y[IDX_HCOOCH3II]*60.0 +
        y[IDX_HCOOHI]*46.0 + y[IDX_HCOOHII]*46.0 + y[IDX_HCOOH2II]*47.0 +
        y[IDX_HCPI]*44.0 + y[IDX_HCPII]*44.0 + y[IDX_HCSI]*45.0 +
        y[IDX_HCSII]*45.0 + y[IDX_HCSiI]*41.0 + y[IDX_HCSiII]*41.0 +
        y[IDX_HClI]*36.0 + y[IDX_HClII]*36.0 + y[IDX_HFI]*20.0 +
        y[IDX_HFII]*20.0 + y[IDX_HN2OII]*45.0 + y[IDX_HNCI]*27.0 +
        y[IDX_HNC3I]*51.0 + y[IDX_HNCOI]*43.0 + y[IDX_HNCOII]*43.0 +
        y[IDX_HNCOHII]*44.0 + y[IDX_HNOI]*31.0 + y[IDX_HNOII]*31.0 +
        y[IDX_HNSII]*47.0 + y[IDX_HNSiI]*43.0 + y[IDX_HNSiII]*43.0 +
        y[IDX_HOCII]*29.0 + y[IDX_HOCNI]*43.0 + y[IDX_HOCNII]*43.0 +
        y[IDX_HOCSII]*61.0 + y[IDX_HONCI]*43.0 + y[IDX_HONCII]*43.0 +
        y[IDX_HPNII]*46.0 + y[IDX_HPOI]*48.0 + y[IDX_HPOII]*48.0 +
        y[IDX_HSI]*33.0 + y[IDX_HSII]*33.0 + y[IDX_HS2I]*65.0 +
        y[IDX_HS2II]*65.0 + y[IDX_HSOII]*49.0 + y[IDX_HSO2II]*65.0 +
        y[IDX_HSiO2II]*61.0 + y[IDX_HSiSII]*61.0 + y[IDX_HeI]*4.0 +
        y[IDX_HeII]*4.0 + y[IDX_HeHII]*5.0 + y[IDX_MgI]*24.0 + y[IDX_MgII]*24.0
        + y[IDX_NI]*14.0 + y[IDX_NII]*14.0 + y[IDX_N2I]*28.0 + y[IDX_N2II]*28.0
        + y[IDX_N2HII]*29.0 + y[IDX_N2OI]*44.0 + y[IDX_N2OII]*44.0 +
        y[IDX_NCCNI]*52.0 + y[IDX_NCCNCH3II]*67.0 + y[IDX_NCCNHII]*53.0 +
        y[IDX_NHI]*15.0 + y[IDX_NHII]*15.0 + y[IDX_NH2I]*16.0 +
        y[IDX_NH2II]*16.0 + y[IDX_NH2CNI]*42.0 + y[IDX_NH2CNHII]*43.0 +
        y[IDX_NH3I]*17.0 + y[IDX_NH3II]*17.0 + y[IDX_NH4II]*18.0 +
        y[IDX_NOI]*30.0 + y[IDX_NOII]*30.0 + y[IDX_NO2I]*46.0 +
        y[IDX_NO2II]*46.0 + y[IDX_NSI]*46.0 + y[IDX_NSII]*46.0 + y[IDX_NaI]*23.0
        + y[IDX_NaII]*23.0 + y[IDX_OI]*16.0 + y[IDX_OII]*16.0 + y[IDX_OM]*16.0 +
        y[IDX_O2I]*32.0 + y[IDX_O2II]*32.0 + y[IDX_O2M]*32.0 + y[IDX_O2HI]*33.0
        + y[IDX_O2HII]*33.0 + y[IDX_OCNI]*42.0 + y[IDX_OCNII]*42.0 +
        y[IDX_OCSI]*60.0 + y[IDX_OCSII]*60.0 + y[IDX_OHI]*17.0 +
        y[IDX_OHII]*17.0 + y[IDX_OHM]*17.0 + y[IDX_PI]*31.0 + y[IDX_PII]*31.0 +
        y[IDX_PC2H2II]*57.0 + y[IDX_PC2H3II]*58.0 + y[IDX_PC2H4II]*59.0 +
        y[IDX_PC3HII]*68.0 + y[IDX_PC4HII]*80.0 + y[IDX_PCH2II]*45.0 +
        y[IDX_PCH3II]*46.0 + y[IDX_PCH4II]*47.0 + y[IDX_PHI]*32.0 +
        y[IDX_PHII]*32.0 + y[IDX_PH2I]*33.0 + y[IDX_PH2II]*33.0 +
        y[IDX_PH3II]*34.0 + y[IDX_PNI]*45.0 + y[IDX_PNII]*45.0 +
        y[IDX_PNH2II]*47.0 + y[IDX_PNH3II]*48.0 + y[IDX_POI]*47.0 +
        y[IDX_POII]*47.0 + y[IDX_SI]*32.0 + y[IDX_SII]*32.0 + y[IDX_SM]*32.0 +
        y[IDX_S2I]*64.0 + y[IDX_S2II]*64.0 + y[IDX_SOI]*48.0 + y[IDX_SOII]*48.0
        + y[IDX_SO2I]*64.0 + y[IDX_SO2II]*64.0 + y[IDX_SiI]*28.0 +
        y[IDX_SiII]*28.0 + y[IDX_SiCI]*40.0 + y[IDX_SiCII]*40.0 +
        y[IDX_SiC2I]*52.0 + y[IDX_SiC2II]*52.0 + y[IDX_SiC2HI]*53.0 +
        y[IDX_SiC2HII]*53.0 + y[IDX_SiC2H2I]*54.0 + y[IDX_SiC2H2II]*54.0 +
        y[IDX_SiC2H3II]*55.0 + y[IDX_SiC3I]*64.0 + y[IDX_SiC3II]*64.0 +
        y[IDX_SiC3HI]*65.0 + y[IDX_SiC3HII]*65.0 + y[IDX_SiC3H2II]*66.0 +
        y[IDX_SiC4I]*76.0 + y[IDX_SiC4II]*76.0 + y[IDX_SiC4HII]*77.0 +
        y[IDX_SiCH2I]*42.0 + y[IDX_SiCH2II]*42.0 + y[IDX_SiCH3I]*43.0 +
        y[IDX_SiCH3II]*43.0 + y[IDX_SiCH4II]*44.0 + y[IDX_SiFII]*47.0 +
        y[IDX_SiHI]*29.0 + y[IDX_SiHII]*29.0 + y[IDX_SiH2I]*30.0 +
        y[IDX_SiH2II]*30.0 + y[IDX_SiH3I]*31.0 + y[IDX_SiH3II]*31.0 +
        y[IDX_SiH4I]*32.0 + y[IDX_SiH4II]*32.0 + y[IDX_SiH5II]*33.0 +
        y[IDX_SiNI]*42.0 + y[IDX_SiNII]*42.0 + y[IDX_SiNCI]*54.0 +
        y[IDX_SiNCII]*54.0 + y[IDX_SiNCHII]*55.0 + y[IDX_SiNH2II]*44.0 +
        y[IDX_SiOI]*44.0 + y[IDX_SiOII]*44.0 + y[IDX_SiO2I]*60.0 +
        y[IDX_SiOHII]*45.0 + y[IDX_SiSI]*60.0 + y[IDX_SiSII]*60.0 +
        y[IDX_eM]*0.0) / (y[IDX_CI] + y[IDX_CII] + y[IDX_CM] + y[IDX_C10I] +
        y[IDX_C10II] + y[IDX_C10M] + y[IDX_C10HI] + y[IDX_C10HII] + y[IDX_C10HM]
        + y[IDX_C10H2I] + y[IDX_C10H2II] + y[IDX_C10H3II] + y[IDX_C11I] +
        y[IDX_C11II] + y[IDX_C2I] + y[IDX_C2II] + y[IDX_C2M] + y[IDX_C2HI] +
        y[IDX_C2HII] + y[IDX_C2HM] + y[IDX_C2H2I] + y[IDX_C2H2II] + y[IDX_C2H3I]
        + y[IDX_C2H3II] + y[IDX_C2H4I] + y[IDX_C2H4II] + y[IDX_C2H4CNI] +
        y[IDX_C2H5I] + y[IDX_C2H5II] + y[IDX_C2H5CNI] + y[IDX_C2H5CNHII] +
        y[IDX_C2H5OHI] + y[IDX_C2H5OHII] + y[IDX_C2H5OH2II] + y[IDX_C2H6I] +
        y[IDX_C2H6II] + y[IDX_C2H7II] + y[IDX_C2NI] + y[IDX_C2NII] +
        y[IDX_C2N2II] + y[IDX_C2NHII] + y[IDX_C2OI] + y[IDX_C2OII] + y[IDX_C2SI]
        + y[IDX_C2SII] + y[IDX_C3I] + y[IDX_C3II] + y[IDX_C3M] + y[IDX_C3HI] +
        y[IDX_C3HII] + y[IDX_C3HM] + y[IDX_C3H2I] + y[IDX_C3H2II] +
        y[IDX_C3H2OII] + y[IDX_C3H3II] + y[IDX_C3H4II] + y[IDX_C3H5II] +
        y[IDX_C3H6II] + y[IDX_C3H7II] + y[IDX_C3NI] + y[IDX_C3NII] + y[IDX_C3NM]
        + y[IDX_C3OI] + y[IDX_C3OII] + y[IDX_C3PI] + y[IDX_C3SI] + y[IDX_C3SII]
        + y[IDX_C4I] + y[IDX_C4II] + y[IDX_C4M] + y[IDX_C4HI] + y[IDX_C4HII] +
        y[IDX_C4HM] + y[IDX_C4H2I] + y[IDX_C4H2II] + y[IDX_C4H3I] +
        y[IDX_C4H3II] + y[IDX_C4H4II] + y[IDX_C4H5II] + y[IDX_C4H6I] +
        y[IDX_C4H7II] + y[IDX_C4NI] + y[IDX_C4NII] + y[IDX_C4PI] + y[IDX_C4PII]
        + y[IDX_C4SI] + y[IDX_C4SII] + y[IDX_C5I] + y[IDX_C5II] + y[IDX_C5M] +
        y[IDX_C5HI] + y[IDX_C5HII] + y[IDX_C5HM] + y[IDX_C5H2I] + y[IDX_C5H2II]
        + y[IDX_C5H3II] + y[IDX_C5H5II] + y[IDX_C5NI] + y[IDX_C5NII] +
        y[IDX_C5NM] + y[IDX_C6I] + y[IDX_C6II] + y[IDX_C6M] + y[IDX_C6HI] +
        y[IDX_C6HII] + y[IDX_C6HM] + y[IDX_C6H2I] + y[IDX_C6H2II] +
        y[IDX_C6H3II] + y[IDX_C6H4II] + y[IDX_C6H5II] + y[IDX_C6H6I] +
        y[IDX_C6H6II] + y[IDX_C6H7II] + y[IDX_C7I] + y[IDX_C7II] + y[IDX_C7M] +
        y[IDX_C7HI] + y[IDX_C7HII] + y[IDX_C7HM] + y[IDX_C7H2I] + y[IDX_C7H2II]
        + y[IDX_C7H3II] + y[IDX_C7H4II] + y[IDX_C7H5II] + y[IDX_C7NI] +
        y[IDX_C7NII] + y[IDX_C8I] + y[IDX_C8II] + y[IDX_C8M] + y[IDX_C8HI] +
        y[IDX_C8HII] + y[IDX_C8HM] + y[IDX_C8H2I] + y[IDX_C8H2II] +
        y[IDX_C8H3II] + y[IDX_C8H4II] + y[IDX_C8H5II] + y[IDX_C9I] + y[IDX_C9II]
        + y[IDX_C9M] + y[IDX_C9HI] + y[IDX_C9HII] + y[IDX_C9HM] + y[IDX_C9H2I] +
        y[IDX_C9H2II] + y[IDX_C9H3II] + y[IDX_C9H4II] + y[IDX_C9H5II] +
        y[IDX_C9NI] + y[IDX_C9NII] + y[IDX_CCPI] + y[IDX_CCPII] + y[IDX_CClI] +
        y[IDX_CClII] + y[IDX_CFII] + y[IDX_CHI] + y[IDX_CHII] + y[IDX_CHM] +
        y[IDX_CH2I] + y[IDX_CH2II] + y[IDX_CH2CCHI] + y[IDX_CH2CCHII] +
        y[IDX_CH2CCH2I] + y[IDX_CH2CHCCHI] + y[IDX_CH2CHCNI] + y[IDX_CH2CHCNII]
        + y[IDX_CH2CHCNHII] + y[IDX_CH2CNI] + y[IDX_CH2CNII] + y[IDX_CH2COI] +
        y[IDX_CH2COII] + y[IDX_CH2NHI] + y[IDX_CH2NH2II] + y[IDX_CH2OHI] +
        y[IDX_CH2OHCH2OII] + y[IDX_CH2OHCHOI] + y[IDX_CH2OHCHOII] +
        y[IDX_CH2OHCOI] + y[IDX_CH2OHCOII] + y[IDX_CH2PHI] + y[IDX_CH3I] +
        y[IDX_CH3II] + y[IDX_CH3C3NI] + y[IDX_CH3C3NII] + y[IDX_CH3C3NHII] +
        y[IDX_CH3C4HI] + y[IDX_CH3C4HII] + y[IDX_CH3C5NI] + y[IDX_CH3C5NHII] +
        y[IDX_CH3C6HI] + y[IDX_CH3C7NI] + y[IDX_CH3C7NHII] + y[IDX_CH3CCHI] +
        y[IDX_CH3CHCH2I] + y[IDX_CH3CHOI] + y[IDX_CH3CHOII] + y[IDX_CH3CHOHII] +
        y[IDX_CH3CNI] + y[IDX_CH3CNII] + y[IDX_CH3CNHII] + y[IDX_CH3COI] +
        y[IDX_CH3COII] + y[IDX_CH3COCH3I] + y[IDX_CH3COCH3II] +
        y[IDX_CH3COCH4II] + y[IDX_CH3COOHI] + y[IDX_CH3COOHII] +
        y[IDX_CH3COOH2II] + y[IDX_CH3CSII] + y[IDX_CH3NHII] + y[IDX_CH3OI] +
        y[IDX_CH3OCH3I] + y[IDX_CH3OCH3II] + y[IDX_CH3OCH4II] + y[IDX_CH3OHI] +
        y[IDX_CH3OHII] + y[IDX_CH3OH2II] + y[IDX_CH4I] + y[IDX_CH4II] +
        y[IDX_CH5II] + y[IDX_CNI] + y[IDX_CNII] + y[IDX_CNM] + y[IDX_CNCII] +
        y[IDX_CNOI] + y[IDX_COI] + y[IDX_COII] + y[IDX_CO2I] + y[IDX_CO2II] +
        y[IDX_COOCH3I] + y[IDX_COOCH3II] + y[IDX_COOHI] + y[IDX_CPI] +
        y[IDX_CPII] + y[IDX_CSI] + y[IDX_CSII] + y[IDX_ClI] + y[IDX_ClII] +
        y[IDX_ClOI] + y[IDX_ClOII] + y[IDX_FI] + y[IDX_FII] + y[IDX_FeI] +
        y[IDX_FeII] + y[IDX_GCI] + y[IDX_GC10I] + y[IDX_GC10HI] + y[IDX_GC10H2I]
        + y[IDX_GC11I] + y[IDX_GC2I] + y[IDX_GC2HI] + y[IDX_GC2H2I] +
        y[IDX_GC2H3I] + y[IDX_GC2H4I] + y[IDX_GC2H4CNI] + y[IDX_GC2H5I] +
        y[IDX_GC2H5CNI] + y[IDX_GC2H5OHI] + y[IDX_GC2H6I] + y[IDX_GC2NI] +
        y[IDX_GC2OI] + y[IDX_GC2SI] + y[IDX_GC3I] + y[IDX_GC3HI] + y[IDX_GC3H2I]
        + y[IDX_GC3NI] + y[IDX_GC3OI] + y[IDX_GC3PI] + y[IDX_GC3SI] +
        y[IDX_GC4I] + y[IDX_GC4HI] + y[IDX_GC4H2I] + y[IDX_GC4H3I] +
        y[IDX_GC4H6I] + y[IDX_GC4NI] + y[IDX_GC4PI] + y[IDX_GC4SI] + y[IDX_GC5I]
        + y[IDX_GC5HI] + y[IDX_GC5H2I] + y[IDX_GC5NI] + y[IDX_GC6I] +
        y[IDX_GC6HI] + y[IDX_GC6H2I] + y[IDX_GC6H6I] + y[IDX_GC7I] +
        y[IDX_GC7HI] + y[IDX_GC7H2I] + y[IDX_GC7NI] + y[IDX_GC8I] + y[IDX_GC8HI]
        + y[IDX_GC8H2I] + y[IDX_GC9I] + y[IDX_GC9HI] + y[IDX_GC9H2I] +
        y[IDX_GC9NI] + y[IDX_GCCPI] + y[IDX_GCClI] + y[IDX_GCHI] + y[IDX_GCH2I]
        + y[IDX_GCH2CCHI] + y[IDX_GCH2CCH2I] + y[IDX_GCH2CHCCHI] +
        y[IDX_GCH2CHCNI] + y[IDX_GCH2CNI] + y[IDX_GCH2COI] + y[IDX_GCH2NHI] +
        y[IDX_GCH2OHI] + y[IDX_GCH2OHCHOI] + y[IDX_GCH2OHCOI] + y[IDX_GCH2PHI] +
        y[IDX_GCH3I] + y[IDX_GCH3C3NI] + y[IDX_GCH3C4HI] + y[IDX_GCH3C5NI] +
        y[IDX_GCH3C6HI] + y[IDX_GCH3C7NI] + y[IDX_GCH3CCHI] + y[IDX_GCH3CHCH2I]
        + y[IDX_GCH3CHOI] + y[IDX_GCH3CNI] + y[IDX_GCH3COI] + y[IDX_GCH3COCH3I]
        + y[IDX_GCH3COOHI] + y[IDX_GCH3OI] + y[IDX_GCH3OCH3I] + y[IDX_GCH3OHI] +
        y[IDX_GCH4I] + y[IDX_GCNI] + y[IDX_GCNOI] + y[IDX_GCOI] + y[IDX_GCO2I] +
        y[IDX_GCOOCH3I] + y[IDX_GCOOHI] + y[IDX_GCPI] + y[IDX_GCSI] +
        y[IDX_GClI] + y[IDX_GClOI] + y[IDX_GFI] + y[IDX_GFeI] + y[IDX_GHI] +
        y[IDX_GH2I] + y[IDX_GH2CCCI] + y[IDX_GH2CNI] + y[IDX_GH2COI] +
        y[IDX_GH2CSI] + y[IDX_GH2OI] + y[IDX_GH2O2I] + y[IDX_GH2SI] +
        y[IDX_GH2S2I] + y[IDX_GH2SiOI] + y[IDX_GHC2OI] + y[IDX_GHC2PI] +
        y[IDX_GHC3NI] + y[IDX_GHC5NI] + y[IDX_GHC7NI] + y[IDX_GHC9NI] +
        y[IDX_GHCCNI] + y[IDX_GHCNI] + y[IDX_GHCNOI] + y[IDX_GHCOI] +
        y[IDX_GHCOOCH3I] + y[IDX_GHCOOHI] + y[IDX_GHCPI] + y[IDX_GHCSI] +
        y[IDX_GHCSiI] + y[IDX_GHClI] + y[IDX_GHFI] + y[IDX_GHNCI] +
        y[IDX_GHNC3I] + y[IDX_GHNCOI] + y[IDX_GHNOI] + y[IDX_GHNSiI] +
        y[IDX_GHOCNI] + y[IDX_GHONCI] + y[IDX_GHPOI] + y[IDX_GHSI] +
        y[IDX_GHS2I] + y[IDX_GHeI] + y[IDX_GMgI] + y[IDX_GNI] + y[IDX_GN2I] +
        y[IDX_GN2OI] + y[IDX_GNCCNI] + y[IDX_GNHI] + y[IDX_GNH2I] +
        y[IDX_GNH2CNI] + y[IDX_GNH3I] + y[IDX_GNOI] + y[IDX_GNO2I] + y[IDX_GNSI]
        + y[IDX_GNaI] + y[IDX_GOI] + y[IDX_GO2I] + y[IDX_GO2HI] + y[IDX_GOCNI] +
        y[IDX_GOCSI] + y[IDX_GOHI] + y[IDX_GPI] + y[IDX_GPHI] + y[IDX_GPH2I] +
        y[IDX_GPNI] + y[IDX_GPOI] + y[IDX_GRAINM] + y[IDX_GRAIN0I] + y[IDX_GSI]
        + y[IDX_GS2I] + y[IDX_GSOI] + y[IDX_GSO2I] + y[IDX_GSiI] + y[IDX_GSiCI]
        + y[IDX_GSiC2I] + y[IDX_GSiC2HI] + y[IDX_GSiC2H2I] + y[IDX_GSiC3I] +
        y[IDX_GSiC3HI] + y[IDX_GSiC4I] + y[IDX_GSiCH2I] + y[IDX_GSiCH3I] +
        y[IDX_GSiHI] + y[IDX_GSiH2I] + y[IDX_GSiH3I] + y[IDX_GSiH4I] +
        y[IDX_GSiNI] + y[IDX_GSiNCI] + y[IDX_GSiOI] + y[IDX_GSiO2I] +
        y[IDX_GSiSI] + y[IDX_HI] + y[IDX_HII] + y[IDX_HM] + y[IDX_H2I] +
        y[IDX_H2II] + y[IDX_H2C4NII] + y[IDX_H2C7NII] + y[IDX_H2C9NII] +
        y[IDX_H2CCCI] + y[IDX_H2CClII] + y[IDX_H2CNI] + y[IDX_H2CNOII] +
        y[IDX_H2COI] + y[IDX_H2COII] + y[IDX_H2CSI] + y[IDX_H2CSII] +
        y[IDX_H2ClII] + y[IDX_H2FII] + y[IDX_H2NCII] + y[IDX_H2NCOII] +
        y[IDX_H2NOII] + y[IDX_H2OI] + y[IDX_H2OII] + y[IDX_H2O2I] +
        y[IDX_H2OCNII] + y[IDX_H2POII] + y[IDX_H2SI] + y[IDX_H2SII] +
        y[IDX_H2S2I] + y[IDX_H2S2II] + y[IDX_H2SiOI] + y[IDX_H2SiOII] +
        y[IDX_H3II] + y[IDX_H3C3OII] + y[IDX_H3C5NII] + y[IDX_H3C7NII] +
        y[IDX_H3C9NII] + y[IDX_H3COII] + y[IDX_H3CSII] + y[IDX_H3OII] +
        y[IDX_H3SII] + y[IDX_H3S2II] + y[IDX_H3SiOII] + y[IDX_H5C2O2II] +
        y[IDX_HC2OI] + y[IDX_HC2OII] + y[IDX_HC2PI] + y[IDX_HC2PII] +
        y[IDX_HC2SII] + y[IDX_HC3NI] + y[IDX_HC3NII] + y[IDX_HC3NHII] +
        y[IDX_HC3OII] + y[IDX_HC3SII] + y[IDX_HC4NII] + y[IDX_HC4SII] +
        y[IDX_HC5NI] + y[IDX_HC5NII] + y[IDX_HC5NHII] + y[IDX_HC7NI] +
        y[IDX_HC7NII] + y[IDX_HC9NI] + y[IDX_HC9NII] + y[IDX_HCCNI] +
        y[IDX_HCNI] + y[IDX_HCNII] + y[IDX_HCNHII] + y[IDX_HCNOI] +
        y[IDX_HCNOII] + y[IDX_HCNOHII] + y[IDX_HCOI] + y[IDX_HCOII] +
        y[IDX_HCO2II] + y[IDX_HCOOCH3I] + y[IDX_HCOOCH3II] + y[IDX_HCOOHI] +
        y[IDX_HCOOHII] + y[IDX_HCOOH2II] + y[IDX_HCPI] + y[IDX_HCPII] +
        y[IDX_HCSI] + y[IDX_HCSII] + y[IDX_HCSiI] + y[IDX_HCSiII] + y[IDX_HClI]
        + y[IDX_HClII] + y[IDX_HFI] + y[IDX_HFII] + y[IDX_HN2OII] + y[IDX_HNCI]
        + y[IDX_HNC3I] + y[IDX_HNCOI] + y[IDX_HNCOII] + y[IDX_HNCOHII] +
        y[IDX_HNOI] + y[IDX_HNOII] + y[IDX_HNSII] + y[IDX_HNSiI] + y[IDX_HNSiII]
        + y[IDX_HOCII] + y[IDX_HOCNI] + y[IDX_HOCNII] + y[IDX_HOCSII] +
        y[IDX_HONCI] + y[IDX_HONCII] + y[IDX_HPNII] + y[IDX_HPOI] + y[IDX_HPOII]
        + y[IDX_HSI] + y[IDX_HSII] + y[IDX_HS2I] + y[IDX_HS2II] + y[IDX_HSOII] +
        y[IDX_HSO2II] + y[IDX_HSiO2II] + y[IDX_HSiSII] + y[IDX_HeI] +
        y[IDX_HeII] + y[IDX_HeHII] + y[IDX_MgI] + y[IDX_MgII] + y[IDX_NI] +
        y[IDX_NII] + y[IDX_N2I] + y[IDX_N2II] + y[IDX_N2HII] + y[IDX_N2OI] +
        y[IDX_N2OII] + y[IDX_NCCNI] + y[IDX_NCCNCH3II] + y[IDX_NCCNHII] +
        y[IDX_NHI] + y[IDX_NHII] + y[IDX_NH2I] + y[IDX_NH2II] + y[IDX_NH2CNI] +
        y[IDX_NH2CNHII] + y[IDX_NH3I] + y[IDX_NH3II] + y[IDX_NH4II] + y[IDX_NOI]
        + y[IDX_NOII] + y[IDX_NO2I] + y[IDX_NO2II] + y[IDX_NSI] + y[IDX_NSII] +
        y[IDX_NaI] + y[IDX_NaII] + y[IDX_OI] + y[IDX_OII] + y[IDX_OM] +
        y[IDX_O2I] + y[IDX_O2II] + y[IDX_O2M] + y[IDX_O2HI] + y[IDX_O2HII] +
        y[IDX_OCNI] + y[IDX_OCNII] + y[IDX_OCSI] + y[IDX_OCSII] + y[IDX_OHI] +
        y[IDX_OHII] + y[IDX_OHM] + y[IDX_PI] + y[IDX_PII] + y[IDX_PC2H2II] +
        y[IDX_PC2H3II] + y[IDX_PC2H4II] + y[IDX_PC3HII] + y[IDX_PC4HII] +
        y[IDX_PCH2II] + y[IDX_PCH3II] + y[IDX_PCH4II] + y[IDX_PHI] + y[IDX_PHII]
        + y[IDX_PH2I] + y[IDX_PH2II] + y[IDX_PH3II] + y[IDX_PNI] + y[IDX_PNII] +
        y[IDX_PNH2II] + y[IDX_PNH3II] + y[IDX_POI] + y[IDX_POII] + y[IDX_SI] +
        y[IDX_SII] + y[IDX_SM] + y[IDX_S2I] + y[IDX_S2II] + y[IDX_SOI] +
        y[IDX_SOII] + y[IDX_SO2I] + y[IDX_SO2II] + y[IDX_SiI] + y[IDX_SiII] +
        y[IDX_SiCI] + y[IDX_SiCII] + y[IDX_SiC2I] + y[IDX_SiC2II] +
        y[IDX_SiC2HI] + y[IDX_SiC2HII] + y[IDX_SiC2H2I] + y[IDX_SiC2H2II] +
        y[IDX_SiC2H3II] + y[IDX_SiC3I] + y[IDX_SiC3II] + y[IDX_SiC3HI] +
        y[IDX_SiC3HII] + y[IDX_SiC3H2II] + y[IDX_SiC4I] + y[IDX_SiC4II] +
        y[IDX_SiC4HII] + y[IDX_SiCH2I] + y[IDX_SiCH2II] + y[IDX_SiCH3I] +
        y[IDX_SiCH3II] + y[IDX_SiCH4II] + y[IDX_SiFII] + y[IDX_SiHI] +
        y[IDX_SiHII] + y[IDX_SiH2I] + y[IDX_SiH2II] + y[IDX_SiH3I] +
        y[IDX_SiH3II] + y[IDX_SiH4I] + y[IDX_SiH4II] + y[IDX_SiH5II] +
        y[IDX_SiNI] + y[IDX_SiNII] + y[IDX_SiNCI] + y[IDX_SiNCII] +
        y[IDX_SiNCHII] + y[IDX_SiNH2II] + y[IDX_SiOI] + y[IDX_SiOII] +
        y[IDX_SiO2I] + y[IDX_SiOHII] + y[IDX_SiSI] + y[IDX_SiSII] + y[IDX_eM]);
}

__device__ __host__ double GetGamma(double *y) {
    return 5.0 / 3.0;
}

__device__ __host__ double GetNumDens(double *y) {
    double numdens = 0.0;

    for (int i = 0; i < NSPECIES; i++) numdens += y[i];
    return numdens;
}
// clang-format on

// clang-format off
__device__ double GetShieldingFactor(int specidx, double h2coldens, double spcoldens,
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
__device__ double GetH2shielding(double coldens, int method) {
    // clang-format on
    double shielding = -1.0;
    switch (method) {
        case 0:
            shielding = GetH2shieldingInt(coldens);
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
    if (coldens >= H2ShieldingTableX[104]) {
        double x1 = H2ShieldingTableX[103];
        double x2 = H2ShieldingTableX[104];
        double y1 = H2ShieldingTable[103];
        double y2 = H2ShieldingTable[104];
        shielding =
            log10(y1) + log10(y2 / y1) * log10(coldens / x1) / log10(x2 / x1);
        shielding = pow(10.0, shielding);
        return shielding;
    }

    for (int i = 0; i < 104; i++) {
        if (coldens >= H2ShieldingTableX[i] &&
            coldens < H2ShieldingTableX[i + 1]) {
            double x1 = H2ShieldingTableX[i];
            double x2 = H2ShieldingTableX[i + 1];
            double y1 = H2ShieldingTable[i];
            double y2 = H2ShieldingTable[i + 1];
            shielding = log10(y1) +
                        log10(y2 / y1) * log10(coldens / x1) / log10(x2 / x1);
            shielding = pow(10.0, shielding);
            return shielding;
        }
    }
    /* */

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
    for (int i = 0; i < 4; i++) {
        if (tgas >= COShieldingTableX[i] && tgas < COShieldingTableX[i + 1]) {
            x1 = COShieldingTableX[i];
            x2 = COShieldingTableX[i + 1];
            i1 = i;
            i2 = i + 1;
        }
    }

    if (tgas >= COShieldingTableX[4]) {
        x1 = COShieldingTableX[3];
        x2 = COShieldingTableX[4];
        i1 = 3;
        i2 = 4;
    }

    for (int i = 0; i < 40; i++) {
        if (h2col >= COShieldingTableY[i] && h2col < COShieldingTableY[i + 1]) {
            y1 = COShieldingTableY[i];
            y2 = COShieldingTableY[i + 1];
            j1 = i;
            j2 = i + 1;
        }
    }

    if (h2col >= COShieldingTableY[40]) {
        y1 = COShieldingTableY[39];
        y2 = COShieldingTableY[40];
        j1 = 39;
        j2 = 40;
    }

    for (int i = 0; i < 45; i++) {
        if (coldens >= COShieldingTableZ[i] &&
            coldens < COShieldingTableZ[i + 1]) {
            z1 = COShieldingTableZ[i];
            z2 = COShieldingTableZ[i + 1];
            k1 = i;
            k2 = i + 1;
        }
    }

    if (coldens >= COShieldingTableZ[45]) {
        z1 = COShieldingTableZ[44];
        z2 = COShieldingTableZ[45];
        k1 = 44;
        k2 = 45;
    }

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

// Interpolate/Extropolate from table (must be rendered in naunet constants)
// clang-format off
__device__ double GetN2shieldingInt(double tgas, double h2col, double coldens) {
    // clang-format on

    double shielding = -1.0;

    /* */
    double x1, x2, y1, y2, z1, z2;
    int i1, i2, j1, j2, k1, k2;
    for (int i = 0; i < 4; i++) {
        if (tgas >= N2ShieldingTableX[i] && tgas < N2ShieldingTableX[i + 1]) {
            x1 = N2ShieldingTableX[i];
            x2 = N2ShieldingTableX[i + 1];
            i1 = i;
            i2 = i + 1;
        }
    }

    if (tgas >= N2ShieldingTableX[4]) {
        x1 = N2ShieldingTableX[3];
        x2 = N2ShieldingTableX[4];
        i1 = 3;
        i2 = 4;
    }

    for (int i = 0; i < 45; i++) {
        if (h2col >= N2ShieldingTableY[i] && h2col < N2ShieldingTableY[i + 1]) {
            y1 = N2ShieldingTableY[i];
            y2 = N2ShieldingTableY[i + 1];
            j1 = i;
            j2 = i + 1;
        }
    }

    if (h2col >= N2ShieldingTableY[45]) {
        y1 = N2ShieldingTableY[44];
        y2 = N2ShieldingTableY[45];
        j1 = 44;
        j2 = 45;
    }

    for (int i = 0; i < 45; i++) {
        if (coldens >= N2ShieldingTableZ[i] &&
            coldens < N2ShieldingTableZ[i + 1]) {
            z1 = N2ShieldingTableZ[i];
            z2 = N2ShieldingTableZ[i + 1];
            k1 = i;
            k2 = i + 1;
        }
    }

    if (coldens >= N2ShieldingTableZ[45]) {
        z1 = N2ShieldingTableZ[44];
        z2 = N2ShieldingTableZ[45];
        k1 = 44;
        k2 = 45;
    }

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