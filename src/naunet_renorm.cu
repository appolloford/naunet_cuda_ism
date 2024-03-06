#include "naunet_macros.h"
#include "naunet_physics.h"
#include "naunet_renorm.h"

// clang-format off
__host__ int InitRenorm(realtype *ab, SUNMatrix A) {
    // clang-format on
    realtype Hnuclei = GetHNuclei(ab);

    // clang-format off
            
    IJth(A, IDX_ELEM_F, IDX_ELEM_F) = 0.0 + 19.0 * ab[IDX_GHFI] / 20.0 / Hnuclei +
                                    19.0 * ab[IDX_GFI] / 19.0 / Hnuclei + 19.0 *
                                    ab[IDX_CFII] / 31.0 / Hnuclei + 19.0 *
                                    ab[IDX_HFII] / 20.0 / Hnuclei + 19.0 *
                                    ab[IDX_SiFII] / 47.0 / Hnuclei + 19.0 *
                                    ab[IDX_H2FII] / 21.0 / Hnuclei + 19.0 *
                                    ab[IDX_FII] / 19.0 / Hnuclei + 19.0 *
                                    ab[IDX_FI] / 19.0 / Hnuclei + 19.0 *
                                    ab[IDX_HFI] / 20.0 / Hnuclei;
    IJth(A, IDX_ELEM_F, IDX_ELEM_Cl) = 0.0;
    IJth(A, IDX_ELEM_F, IDX_ELEM_P) = 0.0;
    IJth(A, IDX_ELEM_F, IDX_ELEM_Fe) = 0.0;
    IJth(A, IDX_ELEM_F, IDX_ELEM_Mg) = 0.0;
    IJth(A, IDX_ELEM_F, IDX_ELEM_Na) = 0.0;
    IJth(A, IDX_ELEM_F, IDX_ELEM_Si) = 0.0 + 28.0 * ab[IDX_SiFII] / 47.0 / Hnuclei;
    IJth(A, IDX_ELEM_F, IDX_ELEM_S) = 0.0;
    IJth(A, IDX_ELEM_F, IDX_ELEM_N) = 0.0;
    IJth(A, IDX_ELEM_F, IDX_ELEM_O) = 0.0;
    IJth(A, IDX_ELEM_F, IDX_ELEM_He) = 0.0;
    IJth(A, IDX_ELEM_F, IDX_ELEM_C) = 0.0 + 12.0 * ab[IDX_CFII] / 31.0 / Hnuclei;
    IJth(A, IDX_ELEM_F, IDX_ELEM_GRAIN) = 0.0;
    IJth(A, IDX_ELEM_F, IDX_ELEM_H) = 0.0 + 1.0 * ab[IDX_GHFI] / 20.0 / Hnuclei +
                                    1.0 * ab[IDX_HFII] / 20.0 / Hnuclei + 2.0 *
                                    ab[IDX_H2FII] / 21.0 / Hnuclei + 1.0 *
                                    ab[IDX_HFI] / 20.0 / Hnuclei;
    IJth(A, IDX_ELEM_Cl, IDX_ELEM_F) = 0.0;
    IJth(A, IDX_ELEM_Cl, IDX_ELEM_Cl) = 0.0 + 35.0 * ab[IDX_GCClI] / 47.0 / Hnuclei
                                    + 35.0 * ab[IDX_GClOI] / 51.0 / Hnuclei +
                                    35.0 * ab[IDX_GHClI] / 36.0 / Hnuclei + 35.0
                                    * ab[IDX_ClOII] / 51.0 / Hnuclei + 35.0 *
                                    ab[IDX_GClI] / 35.0 / Hnuclei + 35.0 *
                                    ab[IDX_ClOI] / 51.0 / Hnuclei + 35.0 *
                                    ab[IDX_H2CClII] / 49.0 / Hnuclei + 35.0 *
                                    ab[IDX_HClII] / 36.0 / Hnuclei + 35.0 *
                                    ab[IDX_CClII] / 47.0 / Hnuclei + 35.0 *
                                    ab[IDX_CClI] / 47.0 / Hnuclei + 35.0 *
                                    ab[IDX_ClII] / 35.0 / Hnuclei + 35.0 *
                                    ab[IDX_H2ClII] / 37.0 / Hnuclei + 35.0 *
                                    ab[IDX_ClI] / 35.0 / Hnuclei + 35.0 *
                                    ab[IDX_HClI] / 36.0 / Hnuclei;
    IJth(A, IDX_ELEM_Cl, IDX_ELEM_P) = 0.0;
    IJth(A, IDX_ELEM_Cl, IDX_ELEM_Fe) = 0.0;
    IJth(A, IDX_ELEM_Cl, IDX_ELEM_Mg) = 0.0;
    IJth(A, IDX_ELEM_Cl, IDX_ELEM_Na) = 0.0;
    IJth(A, IDX_ELEM_Cl, IDX_ELEM_Si) = 0.0;
    IJth(A, IDX_ELEM_Cl, IDX_ELEM_S) = 0.0;
    IJth(A, IDX_ELEM_Cl, IDX_ELEM_N) = 0.0;
    IJth(A, IDX_ELEM_Cl, IDX_ELEM_O) = 0.0 + 16.0 * ab[IDX_GClOI] / 51.0 / Hnuclei
                                    + 16.0 * ab[IDX_ClOII] / 51.0 / Hnuclei +
                                    16.0 * ab[IDX_ClOI] / 51.0 / Hnuclei;
    IJth(A, IDX_ELEM_Cl, IDX_ELEM_He) = 0.0;
    IJth(A, IDX_ELEM_Cl, IDX_ELEM_C) = 0.0 + 12.0 * ab[IDX_GCClI] / 47.0 / Hnuclei
                                    + 12.0 * ab[IDX_H2CClII] / 49.0 / Hnuclei +
                                    12.0 * ab[IDX_CClII] / 47.0 / Hnuclei + 12.0
                                    * ab[IDX_CClI] / 47.0 / Hnuclei;
    IJth(A, IDX_ELEM_Cl, IDX_ELEM_GRAIN) = 0.0;
    IJth(A, IDX_ELEM_Cl, IDX_ELEM_H) = 0.0 + 1.0 * ab[IDX_GHClI] / 36.0 / Hnuclei +
                                    2.0 * ab[IDX_H2CClII] / 49.0 / Hnuclei + 1.0
                                    * ab[IDX_HClII] / 36.0 / Hnuclei + 2.0 *
                                    ab[IDX_H2ClII] / 37.0 / Hnuclei + 1.0 *
                                    ab[IDX_HClI] / 36.0 / Hnuclei;
    IJth(A, IDX_ELEM_P, IDX_ELEM_F) = 0.0;
    IJth(A, IDX_ELEM_P, IDX_ELEM_Cl) = 0.0;
    IJth(A, IDX_ELEM_P, IDX_ELEM_P) = 0.0 + 31.0 * ab[IDX_GHC2PI] / 56.0 / Hnuclei
                                    + 31.0 * ab[IDX_GHPOI] / 48.0 / Hnuclei +
                                    31.0 * ab[IDX_GPNI] / 45.0 / Hnuclei + 31.0
                                    * ab[IDX_GC4PI] / 79.0 / Hnuclei + 31.0 *
                                    ab[IDX_GCH2PHI] / 46.0 / Hnuclei + 31.0 *
                                    ab[IDX_GHCPI] / 44.0 / Hnuclei + 31.0 *
                                    ab[IDX_GPH2I] / 33.0 / Hnuclei + 31.0 *
                                    ab[IDX_GPOI] / 47.0 / Hnuclei + 31.0 *
                                    ab[IDX_GC3PI] / 67.0 / Hnuclei + 31.0 *
                                    ab[IDX_GPHI] / 32.0 / Hnuclei + 31.0 *
                                    ab[IDX_GCCPI] / 55.0 / Hnuclei + 31.0 *
                                    ab[IDX_C4PII] / 79.0 / Hnuclei + 31.0 *
                                    ab[IDX_GCPI] / 43.0 / Hnuclei + 31.0 *
                                    ab[IDX_PC2H4II] / 59.0 / Hnuclei + 31.0 *
                                    ab[IDX_PC2H3II] / 58.0 / Hnuclei + 31.0 *
                                    ab[IDX_PNII] / 45.0 / Hnuclei + 31.0 *
                                    ab[IDX_PNH3II] / 48.0 / Hnuclei + 31.0 *
                                    ab[IDX_GPI] / 31.0 / Hnuclei + 31.0 *
                                    ab[IDX_PCH3II] / 46.0 / Hnuclei + 31.0 *
                                    ab[IDX_PNH2II] / 47.0 / Hnuclei + 31.0 *
                                    ab[IDX_HPNII] / 46.0 / Hnuclei + 31.0 *
                                    ab[IDX_PC3HII] / 68.0 / Hnuclei + 31.0 *
                                    ab[IDX_PCH4II] / 47.0 / Hnuclei + 31.0 *
                                    ab[IDX_CCPII] / 55.0 / Hnuclei + 31.0 *
                                    ab[IDX_H2POII] / 49.0 / Hnuclei + 31.0 *
                                    ab[IDX_PC2H2II] / 57.0 / Hnuclei + 31.0 *
                                    ab[IDX_PC4HII] / 80.0 / Hnuclei + 31.0 *
                                    ab[IDX_PH3II] / 34.0 / Hnuclei + 31.0 *
                                    ab[IDX_CPII] / 43.0 / Hnuclei + 31.0 *
                                    ab[IDX_HC2PII] / 56.0 / Hnuclei + 31.0 *
                                    ab[IDX_PCH2II] / 45.0 / Hnuclei + 31.0 *
                                    ab[IDX_HCPII] / 44.0 / Hnuclei + 31.0 *
                                    ab[IDX_HPOII] / 48.0 / Hnuclei + 31.0 *
                                    ab[IDX_CH2PHI] / 46.0 / Hnuclei + 31.0 *
                                    ab[IDX_HPOI] / 48.0 / Hnuclei + 31.0 *
                                    ab[IDX_POII] / 47.0 / Hnuclei + 31.0 *
                                    ab[IDX_PH2I] / 33.0 / Hnuclei + 31.0 *
                                    ab[IDX_PNI] / 45.0 / Hnuclei + 31.0 *
                                    ab[IDX_C4PI] / 79.0 / Hnuclei + 31.0 *
                                    ab[IDX_HCPI] / 44.0 / Hnuclei + 31.0 *
                                    ab[IDX_C3PI] / 67.0 / Hnuclei + 31.0 *
                                    ab[IDX_PH2II] / 33.0 / Hnuclei + 31.0 *
                                    ab[IDX_POI] / 47.0 / Hnuclei + 31.0 *
                                    ab[IDX_HC2PI] / 56.0 / Hnuclei + 31.0 *
                                    ab[IDX_CCPI] / 55.0 / Hnuclei + 31.0 *
                                    ab[IDX_CPI] / 43.0 / Hnuclei + 31.0 *
                                    ab[IDX_PHI] / 32.0 / Hnuclei + 31.0 *
                                    ab[IDX_PHII] / 32.0 / Hnuclei + 31.0 *
                                    ab[IDX_PII] / 31.0 / Hnuclei + 31.0 *
                                    ab[IDX_PI] / 31.0 / Hnuclei;
    IJth(A, IDX_ELEM_P, IDX_ELEM_Fe) = 0.0;
    IJth(A, IDX_ELEM_P, IDX_ELEM_Mg) = 0.0;
    IJth(A, IDX_ELEM_P, IDX_ELEM_Na) = 0.0;
    IJth(A, IDX_ELEM_P, IDX_ELEM_Si) = 0.0;
    IJth(A, IDX_ELEM_P, IDX_ELEM_S) = 0.0;
    IJth(A, IDX_ELEM_P, IDX_ELEM_N) = 0.0 + 14.0 * ab[IDX_GPNI] / 45.0 / Hnuclei +
                                    14.0 * ab[IDX_PNII] / 45.0 / Hnuclei + 14.0
                                    * ab[IDX_PNH3II] / 48.0 / Hnuclei + 14.0 *
                                    ab[IDX_PNH2II] / 47.0 / Hnuclei + 14.0 *
                                    ab[IDX_HPNII] / 46.0 / Hnuclei + 14.0 *
                                    ab[IDX_PNI] / 45.0 / Hnuclei;
    IJth(A, IDX_ELEM_P, IDX_ELEM_O) = 0.0 + 16.0 * ab[IDX_GHPOI] / 48.0 / Hnuclei
                                    + 16.0 * ab[IDX_GPOI] / 47.0 / Hnuclei +
                                    16.0 * ab[IDX_H2POII] / 49.0 / Hnuclei +
                                    16.0 * ab[IDX_HPOII] / 48.0 / Hnuclei + 16.0
                                    * ab[IDX_HPOI] / 48.0 / Hnuclei + 16.0 *
                                    ab[IDX_POII] / 47.0 / Hnuclei + 16.0 *
                                    ab[IDX_POI] / 47.0 / Hnuclei;
    IJth(A, IDX_ELEM_P, IDX_ELEM_He) = 0.0;
    IJth(A, IDX_ELEM_P, IDX_ELEM_C) = 0.0 + 24.0 * ab[IDX_GHC2PI] / 56.0 / Hnuclei
                                    + 48.0 * ab[IDX_GC4PI] / 79.0 / Hnuclei +
                                    12.0 * ab[IDX_GCH2PHI] / 46.0 / Hnuclei +
                                    12.0 * ab[IDX_GHCPI] / 44.0 / Hnuclei + 36.0
                                    * ab[IDX_GC3PI] / 67.0 / Hnuclei + 24.0 *
                                    ab[IDX_GCCPI] / 55.0 / Hnuclei + 48.0 *
                                    ab[IDX_C4PII] / 79.0 / Hnuclei + 12.0 *
                                    ab[IDX_GCPI] / 43.0 / Hnuclei + 24.0 *
                                    ab[IDX_PC2H4II] / 59.0 / Hnuclei + 24.0 *
                                    ab[IDX_PC2H3II] / 58.0 / Hnuclei + 12.0 *
                                    ab[IDX_PCH3II] / 46.0 / Hnuclei + 36.0 *
                                    ab[IDX_PC3HII] / 68.0 / Hnuclei + 12.0 *
                                    ab[IDX_PCH4II] / 47.0 / Hnuclei + 24.0 *
                                    ab[IDX_CCPII] / 55.0 / Hnuclei + 24.0 *
                                    ab[IDX_PC2H2II] / 57.0 / Hnuclei + 48.0 *
                                    ab[IDX_PC4HII] / 80.0 / Hnuclei + 12.0 *
                                    ab[IDX_CPII] / 43.0 / Hnuclei + 24.0 *
                                    ab[IDX_HC2PII] / 56.0 / Hnuclei + 12.0 *
                                    ab[IDX_PCH2II] / 45.0 / Hnuclei + 12.0 *
                                    ab[IDX_HCPII] / 44.0 / Hnuclei + 12.0 *
                                    ab[IDX_CH2PHI] / 46.0 / Hnuclei + 48.0 *
                                    ab[IDX_C4PI] / 79.0 / Hnuclei + 12.0 *
                                    ab[IDX_HCPI] / 44.0 / Hnuclei + 36.0 *
                                    ab[IDX_C3PI] / 67.0 / Hnuclei + 24.0 *
                                    ab[IDX_HC2PI] / 56.0 / Hnuclei + 24.0 *
                                    ab[IDX_CCPI] / 55.0 / Hnuclei + 12.0 *
                                    ab[IDX_CPI] / 43.0 / Hnuclei;
    IJth(A, IDX_ELEM_P, IDX_ELEM_GRAIN) = 0.0;
    IJth(A, IDX_ELEM_P, IDX_ELEM_H) = 0.0 + 1.0 * ab[IDX_GHC2PI] / 56.0 / Hnuclei
                                    + 1.0 * ab[IDX_GHPOI] / 48.0 / Hnuclei + 3.0
                                    * ab[IDX_GCH2PHI] / 46.0 / Hnuclei + 1.0 *
                                    ab[IDX_GHCPI] / 44.0 / Hnuclei + 2.0 *
                                    ab[IDX_GPH2I] / 33.0 / Hnuclei + 1.0 *
                                    ab[IDX_GPHI] / 32.0 / Hnuclei + 4.0 *
                                    ab[IDX_PC2H4II] / 59.0 / Hnuclei + 3.0 *
                                    ab[IDX_PC2H3II] / 58.0 / Hnuclei + 3.0 *
                                    ab[IDX_PNH3II] / 48.0 / Hnuclei + 3.0 *
                                    ab[IDX_PCH3II] / 46.0 / Hnuclei + 2.0 *
                                    ab[IDX_PNH2II] / 47.0 / Hnuclei + 1.0 *
                                    ab[IDX_HPNII] / 46.0 / Hnuclei + 1.0 *
                                    ab[IDX_PC3HII] / 68.0 / Hnuclei + 4.0 *
                                    ab[IDX_PCH4II] / 47.0 / Hnuclei + 2.0 *
                                    ab[IDX_H2POII] / 49.0 / Hnuclei + 2.0 *
                                    ab[IDX_PC2H2II] / 57.0 / Hnuclei + 1.0 *
                                    ab[IDX_PC4HII] / 80.0 / Hnuclei + 3.0 *
                                    ab[IDX_PH3II] / 34.0 / Hnuclei + 1.0 *
                                    ab[IDX_HC2PII] / 56.0 / Hnuclei + 2.0 *
                                    ab[IDX_PCH2II] / 45.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCPII] / 44.0 / Hnuclei + 1.0 *
                                    ab[IDX_HPOII] / 48.0 / Hnuclei + 3.0 *
                                    ab[IDX_CH2PHI] / 46.0 / Hnuclei + 1.0 *
                                    ab[IDX_HPOI] / 48.0 / Hnuclei + 2.0 *
                                    ab[IDX_PH2I] / 33.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCPI] / 44.0 / Hnuclei + 2.0 *
                                    ab[IDX_PH2II] / 33.0 / Hnuclei + 1.0 *
                                    ab[IDX_HC2PI] / 56.0 / Hnuclei + 1.0 *
                                    ab[IDX_PHI] / 32.0 / Hnuclei + 1.0 *
                                    ab[IDX_PHII] / 32.0 / Hnuclei;
    IJth(A, IDX_ELEM_Fe, IDX_ELEM_F) = 0.0;
    IJth(A, IDX_ELEM_Fe, IDX_ELEM_Cl) = 0.0;
    IJth(A, IDX_ELEM_Fe, IDX_ELEM_P) = 0.0;
    IJth(A, IDX_ELEM_Fe, IDX_ELEM_Fe) = 0.0 + 56.0 * ab[IDX_GFeI] / 56.0 / Hnuclei +
                                    56.0 * ab[IDX_FeII] / 56.0 / Hnuclei + 56.0
                                    * ab[IDX_FeI] / 56.0 / Hnuclei;
    IJth(A, IDX_ELEM_Fe, IDX_ELEM_Mg) = 0.0;
    IJth(A, IDX_ELEM_Fe, IDX_ELEM_Na) = 0.0;
    IJth(A, IDX_ELEM_Fe, IDX_ELEM_Si) = 0.0;
    IJth(A, IDX_ELEM_Fe, IDX_ELEM_S) = 0.0;
    IJth(A, IDX_ELEM_Fe, IDX_ELEM_N) = 0.0;
    IJth(A, IDX_ELEM_Fe, IDX_ELEM_O) = 0.0;
    IJth(A, IDX_ELEM_Fe, IDX_ELEM_He) = 0.0;
    IJth(A, IDX_ELEM_Fe, IDX_ELEM_C) = 0.0;
    IJth(A, IDX_ELEM_Fe, IDX_ELEM_GRAIN) = 0.0;
    IJth(A, IDX_ELEM_Fe, IDX_ELEM_H) = 0.0;
    IJth(A, IDX_ELEM_Mg, IDX_ELEM_F) = 0.0;
    IJth(A, IDX_ELEM_Mg, IDX_ELEM_Cl) = 0.0;
    IJth(A, IDX_ELEM_Mg, IDX_ELEM_P) = 0.0;
    IJth(A, IDX_ELEM_Mg, IDX_ELEM_Fe) = 0.0;
    IJth(A, IDX_ELEM_Mg, IDX_ELEM_Mg) = 0.0 + 24.0 * ab[IDX_GMgI] / 24.0 / Hnuclei +
                                    24.0 * ab[IDX_MgII] / 24.0 / Hnuclei + 24.0
                                    * ab[IDX_MgI] / 24.0 / Hnuclei;
    IJth(A, IDX_ELEM_Mg, IDX_ELEM_Na) = 0.0;
    IJth(A, IDX_ELEM_Mg, IDX_ELEM_Si) = 0.0;
    IJth(A, IDX_ELEM_Mg, IDX_ELEM_S) = 0.0;
    IJth(A, IDX_ELEM_Mg, IDX_ELEM_N) = 0.0;
    IJth(A, IDX_ELEM_Mg, IDX_ELEM_O) = 0.0;
    IJth(A, IDX_ELEM_Mg, IDX_ELEM_He) = 0.0;
    IJth(A, IDX_ELEM_Mg, IDX_ELEM_C) = 0.0;
    IJth(A, IDX_ELEM_Mg, IDX_ELEM_GRAIN) = 0.0;
    IJth(A, IDX_ELEM_Mg, IDX_ELEM_H) = 0.0;
    IJth(A, IDX_ELEM_Na, IDX_ELEM_F) = 0.0;
    IJth(A, IDX_ELEM_Na, IDX_ELEM_Cl) = 0.0;
    IJth(A, IDX_ELEM_Na, IDX_ELEM_P) = 0.0;
    IJth(A, IDX_ELEM_Na, IDX_ELEM_Fe) = 0.0;
    IJth(A, IDX_ELEM_Na, IDX_ELEM_Mg) = 0.0;
    IJth(A, IDX_ELEM_Na, IDX_ELEM_Na) = 0.0 + 23.0 * ab[IDX_GNaI] / 23.0 / Hnuclei +
                                    23.0 * ab[IDX_NaII] / 23.0 / Hnuclei + 23.0
                                    * ab[IDX_NaI] / 23.0 / Hnuclei;
    IJth(A, IDX_ELEM_Na, IDX_ELEM_Si) = 0.0;
    IJth(A, IDX_ELEM_Na, IDX_ELEM_S) = 0.0;
    IJth(A, IDX_ELEM_Na, IDX_ELEM_N) = 0.0;
    IJth(A, IDX_ELEM_Na, IDX_ELEM_O) = 0.0;
    IJth(A, IDX_ELEM_Na, IDX_ELEM_He) = 0.0;
    IJth(A, IDX_ELEM_Na, IDX_ELEM_C) = 0.0;
    IJth(A, IDX_ELEM_Na, IDX_ELEM_GRAIN) = 0.0;
    IJth(A, IDX_ELEM_Na, IDX_ELEM_H) = 0.0;
    IJth(A, IDX_ELEM_Si, IDX_ELEM_F) = 0.0 + 19.0 * ab[IDX_SiFII] / 47.0 / Hnuclei;
    IJth(A, IDX_ELEM_Si, IDX_ELEM_Cl) = 0.0;
    IJth(A, IDX_ELEM_Si, IDX_ELEM_P) = 0.0;
    IJth(A, IDX_ELEM_Si, IDX_ELEM_Fe) = 0.0;
    IJth(A, IDX_ELEM_Si, IDX_ELEM_Mg) = 0.0;
    IJth(A, IDX_ELEM_Si, IDX_ELEM_Na) = 0.0;
    IJth(A, IDX_ELEM_Si, IDX_ELEM_Si) = 0.0 + 28.0 * ab[IDX_GHNSiI] / 43.0 / Hnuclei
                                    + 28.0 * ab[IDX_GSiC3HI] / 65.0 / Hnuclei +
                                    28.0 * ab[IDX_GSiCH3I] / 43.0 / Hnuclei +
                                    28.0 * ab[IDX_GSiO2I] / 60.0 / Hnuclei +
                                    28.0 * ab[IDX_GSiSI] / 60.0 / Hnuclei + 28.0
                                    * ab[IDX_GH2SiOI] / 46.0 / Hnuclei + 28.0 *
                                    ab[IDX_GSiC2HI] / 53.0 / Hnuclei + 28.0 *
                                    ab[IDX_GSiC2H2I] / 54.0 / Hnuclei + 28.0 *
                                    ab[IDX_GSiC4I] / 76.0 / Hnuclei + 28.0 *
                                    ab[IDX_GSiNCI] / 54.0 / Hnuclei + 28.0 *
                                    ab[IDX_GHCSiI] / 41.0 / Hnuclei + 28.0 *
                                    ab[IDX_GSiH4I] / 32.0 / Hnuclei + 28.0 *
                                    ab[IDX_GSiC3I] / 64.0 / Hnuclei + 28.0 *
                                    ab[IDX_GSiH3I] / 31.0 / Hnuclei + 28.0 *
                                    ab[IDX_GSiNI] / 42.0 / Hnuclei + 28.0 *
                                    ab[IDX_GSiOI] / 44.0 / Hnuclei + 28.0 *
                                    ab[IDX_GSiCH2I] / 42.0 / Hnuclei + 28.0 *
                                    ab[IDX_GSiHI] / 29.0 / Hnuclei + 28.0 *
                                    ab[IDX_GSiH2I] / 30.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiFII] / 47.0 / Hnuclei + 28.0 *
                                    ab[IDX_GSiCI] / 40.0 / Hnuclei + 28.0 *
                                    ab[IDX_GSiC2I] / 52.0 / Hnuclei + 28.0 *
                                    ab[IDX_HSiO2II] / 61.0 / Hnuclei + 28.0 *
                                    ab[IDX_H2SiOII] / 46.0 / Hnuclei + 28.0 *
                                    ab[IDX_H3SiOII] / 47.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiH4II] / 32.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiO2I] / 60.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiC3H2II] / 66.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiC4II] / 76.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiC4HII] / 77.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiH5II] / 33.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiNCHII] / 55.0 / Hnuclei + 28.0 *
                                    ab[IDX_HNSiII] / 43.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiCH4II] / 44.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiC2H3II] / 55.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiC3HI] / 65.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiCH3II] / 43.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiNH2II] / 44.0 / Hnuclei + 28.0 *
                                    ab[IDX_HSiSII] / 61.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiNII] / 42.0 / Hnuclei + 28.0 *
                                    ab[IDX_H2SiOI] / 46.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiC4I] / 76.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiNCI] / 54.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiNCII] / 54.0 / Hnuclei + 28.0 *
                                    ab[IDX_HNSiI] / 43.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiC2H2I] / 54.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiC3II] / 64.0 / Hnuclei + 28.0 *
                                    ab[IDX_GSiI] / 28.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiC3HII] / 65.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiCH3I] / 43.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiC2II] / 52.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiC2H2II] / 54.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiC2HI] / 53.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiC3I] / 64.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiCII] / 40.0 / Hnuclei + 28.0 *
                                    ab[IDX_HCSiI] / 41.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiCH2I] / 42.0 / Hnuclei + 28.0 *
                                    ab[IDX_HCSiII] / 41.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiC2I] / 52.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiNI] / 42.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiC2HII] / 53.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiH2I] / 30.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiH2II] / 30.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiH3II] / 31.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiCI] / 40.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiH3I] / 31.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiCH2II] / 42.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiHII] / 29.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiHI] / 29.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiH4I] / 32.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiSII] / 60.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiSI] / 60.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiOII] / 44.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiOHII] / 45.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiOI] / 44.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiII] / 28.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiI] / 28.0 / Hnuclei;
    IJth(A, IDX_ELEM_Si, IDX_ELEM_S) = 0.0 + 32.0 * ab[IDX_GSiSI] / 60.0 / Hnuclei
                                    + 32.0 * ab[IDX_HSiSII] / 61.0 / Hnuclei +
                                    32.0 * ab[IDX_SiSII] / 60.0 / Hnuclei + 32.0
                                    * ab[IDX_SiSI] / 60.0 / Hnuclei;
    IJth(A, IDX_ELEM_Si, IDX_ELEM_N) = 0.0 + 14.0 * ab[IDX_GHNSiI] / 43.0 / Hnuclei
                                    + 14.0 * ab[IDX_GSiNCI] / 54.0 / Hnuclei +
                                    14.0 * ab[IDX_GSiNI] / 42.0 / Hnuclei + 14.0
                                    * ab[IDX_SiNCHII] / 55.0 / Hnuclei + 14.0 *
                                    ab[IDX_HNSiII] / 43.0 / Hnuclei + 14.0 *
                                    ab[IDX_SiNH2II] / 44.0 / Hnuclei + 14.0 *
                                    ab[IDX_SiNII] / 42.0 / Hnuclei + 14.0 *
                                    ab[IDX_SiNCI] / 54.0 / Hnuclei + 14.0 *
                                    ab[IDX_SiNCII] / 54.0 / Hnuclei + 14.0 *
                                    ab[IDX_HNSiI] / 43.0 / Hnuclei + 14.0 *
                                    ab[IDX_SiNI] / 42.0 / Hnuclei;
    IJth(A, IDX_ELEM_Si, IDX_ELEM_O) = 0.0 + 32.0 * ab[IDX_GSiO2I] / 60.0 / Hnuclei
                                    + 16.0 * ab[IDX_GH2SiOI] / 46.0 / Hnuclei +
                                    16.0 * ab[IDX_GSiOI] / 44.0 / Hnuclei + 32.0
                                    * ab[IDX_HSiO2II] / 61.0 / Hnuclei + 16.0 *
                                    ab[IDX_H2SiOII] / 46.0 / Hnuclei + 16.0 *
                                    ab[IDX_H3SiOII] / 47.0 / Hnuclei + 32.0 *
                                    ab[IDX_SiO2I] / 60.0 / Hnuclei + 16.0 *
                                    ab[IDX_H2SiOI] / 46.0 / Hnuclei + 16.0 *
                                    ab[IDX_SiOII] / 44.0 / Hnuclei + 16.0 *
                                    ab[IDX_SiOHII] / 45.0 / Hnuclei + 16.0 *
                                    ab[IDX_SiOI] / 44.0 / Hnuclei;
    IJth(A, IDX_ELEM_Si, IDX_ELEM_He) = 0.0;
    IJth(A, IDX_ELEM_Si, IDX_ELEM_C) = 0.0 + 36.0 * ab[IDX_GSiC3HI] / 65.0 /
                                    Hnuclei + 12.0 * ab[IDX_GSiCH3I] / 43.0 /
                                    Hnuclei + 24.0 * ab[IDX_GSiC2HI] / 53.0 /
                                    Hnuclei + 24.0 * ab[IDX_GSiC2H2I] / 54.0 /
                                    Hnuclei + 48.0 * ab[IDX_GSiC4I] / 76.0 /
                                    Hnuclei + 12.0 * ab[IDX_GSiNCI] / 54.0 /
                                    Hnuclei + 12.0 * ab[IDX_GHCSiI] / 41.0 /
                                    Hnuclei + 36.0 * ab[IDX_GSiC3I] / 64.0 /
                                    Hnuclei + 12.0 * ab[IDX_GSiCH2I] / 42.0 /
                                    Hnuclei + 12.0 * ab[IDX_GSiCI] / 40.0 /
                                    Hnuclei + 24.0 * ab[IDX_GSiC2I] / 52.0 /
                                    Hnuclei + 36.0 * ab[IDX_SiC3H2II] / 66.0 /
                                    Hnuclei + 48.0 * ab[IDX_SiC4II] / 76.0 /
                                    Hnuclei + 48.0 * ab[IDX_SiC4HII] / 77.0 /
                                    Hnuclei + 12.0 * ab[IDX_SiNCHII] / 55.0 /
                                    Hnuclei + 12.0 * ab[IDX_SiCH4II] / 44.0 /
                                    Hnuclei + 24.0 * ab[IDX_SiC2H3II] / 55.0 /
                                    Hnuclei + 36.0 * ab[IDX_SiC3HI] / 65.0 /
                                    Hnuclei + 12.0 * ab[IDX_SiCH3II] / 43.0 /
                                    Hnuclei + 48.0 * ab[IDX_SiC4I] / 76.0 /
                                    Hnuclei + 12.0 * ab[IDX_SiNCI] / 54.0 /
                                    Hnuclei + 12.0 * ab[IDX_SiNCII] / 54.0 /
                                    Hnuclei + 24.0 * ab[IDX_SiC2H2I] / 54.0 /
                                    Hnuclei + 36.0 * ab[IDX_SiC3II] / 64.0 /
                                    Hnuclei + 36.0 * ab[IDX_SiC3HII] / 65.0 /
                                    Hnuclei + 12.0 * ab[IDX_SiCH3I] / 43.0 /
                                    Hnuclei + 24.0 * ab[IDX_SiC2II] / 52.0 /
                                    Hnuclei + 24.0 * ab[IDX_SiC2H2II] / 54.0 /
                                    Hnuclei + 24.0 * ab[IDX_SiC2HI] / 53.0 /
                                    Hnuclei + 36.0 * ab[IDX_SiC3I] / 64.0 /
                                    Hnuclei + 12.0 * ab[IDX_SiCII] / 40.0 /
                                    Hnuclei + 12.0 * ab[IDX_HCSiI] / 41.0 /
                                    Hnuclei + 12.0 * ab[IDX_SiCH2I] / 42.0 /
                                    Hnuclei + 12.0 * ab[IDX_HCSiII] / 41.0 /
                                    Hnuclei + 24.0 * ab[IDX_SiC2I] / 52.0 /
                                    Hnuclei + 24.0 * ab[IDX_SiC2HII] / 53.0 /
                                    Hnuclei + 12.0 * ab[IDX_SiCI] / 40.0 /
                                    Hnuclei + 12.0 * ab[IDX_SiCH2II] / 42.0 /
                                    Hnuclei;
    IJth(A, IDX_ELEM_Si, IDX_ELEM_GRAIN) = 0.0;
    IJth(A, IDX_ELEM_Si, IDX_ELEM_H) = 0.0 + 1.0 * ab[IDX_GHNSiI] / 43.0 / Hnuclei
                                    + 1.0 * ab[IDX_GSiC3HI] / 65.0 / Hnuclei +
                                    3.0 * ab[IDX_GSiCH3I] / 43.0 / Hnuclei + 2.0
                                    * ab[IDX_GH2SiOI] / 46.0 / Hnuclei + 1.0 *
                                    ab[IDX_GSiC2HI] / 53.0 / Hnuclei + 2.0 *
                                    ab[IDX_GSiC2H2I] / 54.0 / Hnuclei + 1.0 *
                                    ab[IDX_GHCSiI] / 41.0 / Hnuclei + 4.0 *
                                    ab[IDX_GSiH4I] / 32.0 / Hnuclei + 3.0 *
                                    ab[IDX_GSiH3I] / 31.0 / Hnuclei + 2.0 *
                                    ab[IDX_GSiCH2I] / 42.0 / Hnuclei + 1.0 *
                                    ab[IDX_GSiHI] / 29.0 / Hnuclei + 2.0 *
                                    ab[IDX_GSiH2I] / 30.0 / Hnuclei + 1.0 *
                                    ab[IDX_HSiO2II] / 61.0 / Hnuclei + 2.0 *
                                    ab[IDX_H2SiOII] / 46.0 / Hnuclei + 3.0 *
                                    ab[IDX_H3SiOII] / 47.0 / Hnuclei + 4.0 *
                                    ab[IDX_SiH4II] / 32.0 / Hnuclei + 2.0 *
                                    ab[IDX_SiC3H2II] / 66.0 / Hnuclei + 1.0 *
                                    ab[IDX_SiC4HII] / 77.0 / Hnuclei + 5.0 *
                                    ab[IDX_SiH5II] / 33.0 / Hnuclei + 1.0 *
                                    ab[IDX_SiNCHII] / 55.0 / Hnuclei + 1.0 *
                                    ab[IDX_HNSiII] / 43.0 / Hnuclei + 4.0 *
                                    ab[IDX_SiCH4II] / 44.0 / Hnuclei + 3.0 *
                                    ab[IDX_SiC2H3II] / 55.0 / Hnuclei + 1.0 *
                                    ab[IDX_SiC3HI] / 65.0 / Hnuclei + 3.0 *
                                    ab[IDX_SiCH3II] / 43.0 / Hnuclei + 2.0 *
                                    ab[IDX_SiNH2II] / 44.0 / Hnuclei + 1.0 *
                                    ab[IDX_HSiSII] / 61.0 / Hnuclei + 2.0 *
                                    ab[IDX_H2SiOI] / 46.0 / Hnuclei + 1.0 *
                                    ab[IDX_HNSiI] / 43.0 / Hnuclei + 2.0 *
                                    ab[IDX_SiC2H2I] / 54.0 / Hnuclei + 1.0 *
                                    ab[IDX_SiC3HII] / 65.0 / Hnuclei + 3.0 *
                                    ab[IDX_SiCH3I] / 43.0 / Hnuclei + 2.0 *
                                    ab[IDX_SiC2H2II] / 54.0 / Hnuclei + 1.0 *
                                    ab[IDX_SiC2HI] / 53.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCSiI] / 41.0 / Hnuclei + 2.0 *
                                    ab[IDX_SiCH2I] / 42.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCSiII] / 41.0 / Hnuclei + 1.0 *
                                    ab[IDX_SiC2HII] / 53.0 / Hnuclei + 2.0 *
                                    ab[IDX_SiH2I] / 30.0 / Hnuclei + 2.0 *
                                    ab[IDX_SiH2II] / 30.0 / Hnuclei + 3.0 *
                                    ab[IDX_SiH3II] / 31.0 / Hnuclei + 3.0 *
                                    ab[IDX_SiH3I] / 31.0 / Hnuclei + 2.0 *
                                    ab[IDX_SiCH2II] / 42.0 / Hnuclei + 1.0 *
                                    ab[IDX_SiHII] / 29.0 / Hnuclei + 1.0 *
                                    ab[IDX_SiHI] / 29.0 / Hnuclei + 4.0 *
                                    ab[IDX_SiH4I] / 32.0 / Hnuclei + 1.0 *
                                    ab[IDX_SiOHII] / 45.0 / Hnuclei;
    IJth(A, IDX_ELEM_S, IDX_ELEM_F) = 0.0;
    IJth(A, IDX_ELEM_S, IDX_ELEM_Cl) = 0.0;
    IJth(A, IDX_ELEM_S, IDX_ELEM_P) = 0.0;
    IJth(A, IDX_ELEM_S, IDX_ELEM_Fe) = 0.0;
    IJth(A, IDX_ELEM_S, IDX_ELEM_Mg) = 0.0;
    IJth(A, IDX_ELEM_S, IDX_ELEM_Na) = 0.0;
    IJth(A, IDX_ELEM_S, IDX_ELEM_Si) = 0.0 + 28.0 * ab[IDX_GSiSI] / 60.0 / Hnuclei
                                    + 28.0 * ab[IDX_HSiSII] / 61.0 / Hnuclei +
                                    28.0 * ab[IDX_SiSII] / 60.0 / Hnuclei + 28.0
                                    * ab[IDX_SiSI] / 60.0 / Hnuclei;
    IJth(A, IDX_ELEM_S, IDX_ELEM_S) = 0.0 + 128.0 * ab[IDX_GH2S2I] / 66.0 /
                                    Hnuclei + 32.0 * ab[IDX_GC4SI] / 80.0 /
                                    Hnuclei + 32.0 * ab[IDX_GSiSI] / 60.0 /
                                    Hnuclei + 128.0 * ab[IDX_GS2I] / 64.0 /
                                    Hnuclei + 32.0 * ab[IDX_GC3SI] / 68.0 /
                                    Hnuclei + 32.0 * ab[IDX_GH2SI] / 34.0 /
                                    Hnuclei + 128.0 * ab[IDX_GHS2I] / 65.0 /
                                    Hnuclei + 32.0 * ab[IDX_GSO2I] / 64.0 /
                                    Hnuclei + 32.0 * ab[IDX_GC2SI] / 56.0 /
                                    Hnuclei + 32.0 * ab[IDX_GH2CSI] / 46.0 /
                                    Hnuclei + 32.0 * ab[IDX_GHCSI] / 45.0 /
                                    Hnuclei + 32.0 * ab[IDX_GOCSI] / 60.0 /
                                    Hnuclei + 32.0 * ab[IDX_GSOI] / 48.0 /
                                    Hnuclei + 32.0 * ab[IDX_GNSI] / 46.0 /
                                    Hnuclei + 32.0 * ab[IDX_HNSII] / 47.0 /
                                    Hnuclei + 32.0 * ab[IDX_HSOII] / 49.0 /
                                    Hnuclei + 32.0 * ab[IDX_CH3CSII] / 59.0 /
                                    Hnuclei + 128.0 * ab[IDX_H3S2II] / 67.0 /
                                    Hnuclei + 32.0 * ab[IDX_C2SII] / 56.0 /
                                    Hnuclei + 32.0 * ab[IDX_C3SII] / 68.0 /
                                    Hnuclei + 32.0 * ab[IDX_HSO2II] / 65.0 /
                                    Hnuclei + 32.0 * ab[IDX_GCSI] / 44.0 /
                                    Hnuclei + 32.0 * ab[IDX_GHSI] / 33.0 /
                                    Hnuclei + 128.0 * ab[IDX_H2S2I] / 66.0 /
                                    Hnuclei + 32.0 * ab[IDX_HOCSII] / 61.0 /
                                    Hnuclei + 32.0 * ab[IDX_NSII] / 46.0 /
                                    Hnuclei + 32.0 * ab[IDX_HC4SII] / 81.0 /
                                    Hnuclei + 128.0 * ab[IDX_H2S2II] / 66.0 /
                                    Hnuclei + 32.0 * ab[IDX_HSiSII] / 61.0 /
                                    Hnuclei + 32.0 * ab[IDX_SO2II] / 64.0 /
                                    Hnuclei + 32.0 * ab[IDX_H3CSII] / 47.0 /
                                    Hnuclei + 32.0 * ab[IDX_HC3SII] / 69.0 /
                                    Hnuclei + 128.0 * ab[IDX_HS2II] / 65.0 /
                                    Hnuclei + 128.0 * ab[IDX_HS2I] / 65.0 /
                                    Hnuclei + 32.0 * ab[IDX_H2CSII] / 46.0 /
                                    Hnuclei + 32.0 * ab[IDX_H2CSI] / 46.0 /
                                    Hnuclei + 128.0 * ab[IDX_S2I] / 64.0 /
                                    Hnuclei + 128.0 * ab[IDX_S2II] / 64.0 /
                                    Hnuclei + 32.0 * ab[IDX_C3SI] / 68.0 /
                                    Hnuclei + 32.0 * ab[IDX_OCSII] / 60.0 /
                                    Hnuclei + 32.0 * ab[IDX_GSI] / 32.0 /
                                    Hnuclei + 32.0 * ab[IDX_HCSI] / 45.0 /
                                    Hnuclei + 32.0 * ab[IDX_NSI] / 46.0 /
                                    Hnuclei + 32.0 * ab[IDX_SO2I] / 64.0 /
                                    Hnuclei + 32.0 * ab[IDX_HCSII] / 45.0 /
                                    Hnuclei + 32.0 * ab[IDX_CSII] / 44.0 /
                                    Hnuclei + 32.0 * ab[IDX_H3SII] / 35.0 /
                                    Hnuclei + 32.0 * ab[IDX_C4SII] / 80.0 /
                                    Hnuclei + 32.0 * ab[IDX_HSII] / 33.0 /
                                    Hnuclei + 32.0 * ab[IDX_OCSI] / 60.0 /
                                    Hnuclei + 32.0 * ab[IDX_SM] / 32.0 / Hnuclei
                                    + 32.0 * ab[IDX_SiSII] / 60.0 / Hnuclei +
                                    32.0 * ab[IDX_HC2SII] / 57.0 / Hnuclei +
                                    32.0 * ab[IDX_C4SI] / 80.0 / Hnuclei + 32.0
                                    * ab[IDX_HSI] / 33.0 / Hnuclei + 32.0 *
                                    ab[IDX_CSI] / 44.0 / Hnuclei + 32.0 *
                                    ab[IDX_SiSI] / 60.0 / Hnuclei + 32.0 *
                                    ab[IDX_C2SI] / 56.0 / Hnuclei + 32.0 *
                                    ab[IDX_SOI] / 48.0 / Hnuclei + 32.0 *
                                    ab[IDX_SOII] / 48.0 / Hnuclei + 32.0 *
                                    ab[IDX_H2SII] / 34.0 / Hnuclei + 32.0 *
                                    ab[IDX_H2SI] / 34.0 / Hnuclei + 32.0 *
                                    ab[IDX_SII] / 32.0 / Hnuclei + 32.0 *
                                    ab[IDX_SI] / 32.0 / Hnuclei;
    IJth(A, IDX_ELEM_S, IDX_ELEM_N) = 0.0 + 14.0 * ab[IDX_GNSI] / 46.0 / Hnuclei +
                                    14.0 * ab[IDX_HNSII] / 47.0 / Hnuclei + 14.0
                                    * ab[IDX_NSII] / 46.0 / Hnuclei + 14.0 *
                                    ab[IDX_NSI] / 46.0 / Hnuclei;
    IJth(A, IDX_ELEM_S, IDX_ELEM_O) = 0.0 + 32.0 * ab[IDX_GSO2I] / 64.0 / Hnuclei
                                    + 16.0 * ab[IDX_GOCSI] / 60.0 / Hnuclei +
                                    16.0 * ab[IDX_GSOI] / 48.0 / Hnuclei + 16.0
                                    * ab[IDX_HSOII] / 49.0 / Hnuclei + 32.0 *
                                    ab[IDX_HSO2II] / 65.0 / Hnuclei + 16.0 *
                                    ab[IDX_HOCSII] / 61.0 / Hnuclei + 32.0 *
                                    ab[IDX_SO2II] / 64.0 / Hnuclei + 16.0 *
                                    ab[IDX_OCSII] / 60.0 / Hnuclei + 32.0 *
                                    ab[IDX_SO2I] / 64.0 / Hnuclei + 16.0 *
                                    ab[IDX_OCSI] / 60.0 / Hnuclei + 16.0 *
                                    ab[IDX_SOI] / 48.0 / Hnuclei + 16.0 *
                                    ab[IDX_SOII] / 48.0 / Hnuclei;
    IJth(A, IDX_ELEM_S, IDX_ELEM_He) = 0.0;
    IJth(A, IDX_ELEM_S, IDX_ELEM_C) = 0.0 + 48.0 * ab[IDX_GC4SI] / 80.0 / Hnuclei
                                    + 36.0 * ab[IDX_GC3SI] / 68.0 / Hnuclei +
                                    24.0 * ab[IDX_GC2SI] / 56.0 / Hnuclei + 12.0
                                    * ab[IDX_GH2CSI] / 46.0 / Hnuclei + 12.0 *
                                    ab[IDX_GHCSI] / 45.0 / Hnuclei + 12.0 *
                                    ab[IDX_GOCSI] / 60.0 / Hnuclei + 24.0 *
                                    ab[IDX_CH3CSII] / 59.0 / Hnuclei + 24.0 *
                                    ab[IDX_C2SII] / 56.0 / Hnuclei + 36.0 *
                                    ab[IDX_C3SII] / 68.0 / Hnuclei + 12.0 *
                                    ab[IDX_GCSI] / 44.0 / Hnuclei + 12.0 *
                                    ab[IDX_HOCSII] / 61.0 / Hnuclei + 48.0 *
                                    ab[IDX_HC4SII] / 81.0 / Hnuclei + 12.0 *
                                    ab[IDX_H3CSII] / 47.0 / Hnuclei + 36.0 *
                                    ab[IDX_HC3SII] / 69.0 / Hnuclei + 12.0 *
                                    ab[IDX_H2CSII] / 46.0 / Hnuclei + 12.0 *
                                    ab[IDX_H2CSI] / 46.0 / Hnuclei + 36.0 *
                                    ab[IDX_C3SI] / 68.0 / Hnuclei + 12.0 *
                                    ab[IDX_OCSII] / 60.0 / Hnuclei + 12.0 *
                                    ab[IDX_HCSI] / 45.0 / Hnuclei + 12.0 *
                                    ab[IDX_HCSII] / 45.0 / Hnuclei + 12.0 *
                                    ab[IDX_CSII] / 44.0 / Hnuclei + 48.0 *
                                    ab[IDX_C4SII] / 80.0 / Hnuclei + 12.0 *
                                    ab[IDX_OCSI] / 60.0 / Hnuclei + 24.0 *
                                    ab[IDX_HC2SII] / 57.0 / Hnuclei + 48.0 *
                                    ab[IDX_C4SI] / 80.0 / Hnuclei + 12.0 *
                                    ab[IDX_CSI] / 44.0 / Hnuclei + 24.0 *
                                    ab[IDX_C2SI] / 56.0 / Hnuclei;
    IJth(A, IDX_ELEM_S, IDX_ELEM_GRAIN) = 0.0;
    IJth(A, IDX_ELEM_S, IDX_ELEM_H) = 0.0 + 4.0 * ab[IDX_GH2S2I] / 66.0 / Hnuclei
                                    + 2.0 * ab[IDX_GH2SI] / 34.0 / Hnuclei + 2.0
                                    * ab[IDX_GHS2I] / 65.0 / Hnuclei + 2.0 *
                                    ab[IDX_GH2CSI] / 46.0 / Hnuclei + 1.0 *
                                    ab[IDX_GHCSI] / 45.0 / Hnuclei + 1.0 *
                                    ab[IDX_HNSII] / 47.0 / Hnuclei + 1.0 *
                                    ab[IDX_HSOII] / 49.0 / Hnuclei + 3.0 *
                                    ab[IDX_CH3CSII] / 59.0 / Hnuclei + 6.0 *
                                    ab[IDX_H3S2II] / 67.0 / Hnuclei + 1.0 *
                                    ab[IDX_HSO2II] / 65.0 / Hnuclei + 1.0 *
                                    ab[IDX_GHSI] / 33.0 / Hnuclei + 4.0 *
                                    ab[IDX_H2S2I] / 66.0 / Hnuclei + 1.0 *
                                    ab[IDX_HOCSII] / 61.0 / Hnuclei + 1.0 *
                                    ab[IDX_HC4SII] / 81.0 / Hnuclei + 4.0 *
                                    ab[IDX_H2S2II] / 66.0 / Hnuclei + 1.0 *
                                    ab[IDX_HSiSII] / 61.0 / Hnuclei + 3.0 *
                                    ab[IDX_H3CSII] / 47.0 / Hnuclei + 1.0 *
                                    ab[IDX_HC3SII] / 69.0 / Hnuclei + 2.0 *
                                    ab[IDX_HS2II] / 65.0 / Hnuclei + 2.0 *
                                    ab[IDX_HS2I] / 65.0 / Hnuclei + 2.0 *
                                    ab[IDX_H2CSII] / 46.0 / Hnuclei + 2.0 *
                                    ab[IDX_H2CSI] / 46.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCSI] / 45.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCSII] / 45.0 / Hnuclei + 3.0 *
                                    ab[IDX_H3SII] / 35.0 / Hnuclei + 1.0 *
                                    ab[IDX_HSII] / 33.0 / Hnuclei + 1.0 *
                                    ab[IDX_HC2SII] / 57.0 / Hnuclei + 1.0 *
                                    ab[IDX_HSI] / 33.0 / Hnuclei + 2.0 *
                                    ab[IDX_H2SII] / 34.0 / Hnuclei + 2.0 *
                                    ab[IDX_H2SI] / 34.0 / Hnuclei;
    IJth(A, IDX_ELEM_N, IDX_ELEM_F) = 0.0;
    IJth(A, IDX_ELEM_N, IDX_ELEM_Cl) = 0.0;
    IJth(A, IDX_ELEM_N, IDX_ELEM_P) = 0.0 + 31.0 * ab[IDX_GPNI] / 45.0 / Hnuclei +
                                    31.0 * ab[IDX_PNII] / 45.0 / Hnuclei + 31.0
                                    * ab[IDX_PNH3II] / 48.0 / Hnuclei + 31.0 *
                                    ab[IDX_PNH2II] / 47.0 / Hnuclei + 31.0 *
                                    ab[IDX_HPNII] / 46.0 / Hnuclei + 31.0 *
                                    ab[IDX_PNI] / 45.0 / Hnuclei;
    IJth(A, IDX_ELEM_N, IDX_ELEM_Fe) = 0.0;
    IJth(A, IDX_ELEM_N, IDX_ELEM_Mg) = 0.0;
    IJth(A, IDX_ELEM_N, IDX_ELEM_Na) = 0.0;
    IJth(A, IDX_ELEM_N, IDX_ELEM_Si) = 0.0 + 28.0 * ab[IDX_GHNSiI] / 43.0 / Hnuclei
                                    + 28.0 * ab[IDX_GSiNCI] / 54.0 / Hnuclei +
                                    28.0 * ab[IDX_GSiNI] / 42.0 / Hnuclei + 28.0
                                    * ab[IDX_SiNCHII] / 55.0 / Hnuclei + 28.0 *
                                    ab[IDX_HNSiII] / 43.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiNH2II] / 44.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiNII] / 42.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiNCI] / 54.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiNCII] / 54.0 / Hnuclei + 28.0 *
                                    ab[IDX_HNSiI] / 43.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiNI] / 42.0 / Hnuclei;
    IJth(A, IDX_ELEM_N, IDX_ELEM_S) = 0.0 + 32.0 * ab[IDX_GNSI] / 46.0 / Hnuclei +
                                    32.0 * ab[IDX_HNSII] / 47.0 / Hnuclei + 32.0
                                    * ab[IDX_NSII] / 46.0 / Hnuclei + 32.0 *
                                    ab[IDX_NSI] / 46.0 / Hnuclei;
    IJth(A, IDX_ELEM_N, IDX_ELEM_N) = 0.0 + 14.0 * ab[IDX_GCH3C3NI] / 65.0 /
                                    Hnuclei + 14.0 * ab[IDX_GCH3C5NI] / 89.0 /
                                    Hnuclei + 14.0 * ab[IDX_GCH3C7NI] / 113.0 /
                                    Hnuclei + 14.0 * ab[IDX_GHNSiI] / 43.0 /
                                    Hnuclei + 56.0 * ab[IDX_GNH2CNI] / 42.0 /
                                    Hnuclei + 14.0 * ab[IDX_GNO2I] / 46.0 /
                                    Hnuclei + 14.0 * ab[IDX_GPNI] / 45.0 /
                                    Hnuclei + 56.0 * ab[IDX_GNCCNI] / 52.0 /
                                    Hnuclei + 14.0 * ab[IDX_C2H4CNI] / 54.0 /
                                    Hnuclei + 14.0 * ab[IDX_GC4NI] / 62.0 /
                                    Hnuclei + 14.0 * ab[IDX_GHCNOI] / 43.0 /
                                    Hnuclei + 14.0 * ab[IDX_GHNC3I] / 51.0 /
                                    Hnuclei + 14.0 * ab[IDX_GHNCOI] / 43.0 /
                                    Hnuclei + 14.0 * ab[IDX_GHOCNI] / 43.0 /
                                    Hnuclei + 14.0 * ab[IDX_GHONCI] / 43.0 /
                                    Hnuclei + 56.0 * ab[IDX_GN2OI] / 44.0 /
                                    Hnuclei + 14.0 * ab[IDX_GSiNCI] / 54.0 /
                                    Hnuclei + 56.0 * ab[IDX_NCCNCH3II] / 67.0 /
                                    Hnuclei + 14.0 * ab[IDX_GC2H4CNI] / 54.0 /
                                    Hnuclei + 14.0 * ab[IDX_GCH2CNI] / 40.0 /
                                    Hnuclei + 14.0 * ab[IDX_GCH3CNI] / 41.0 /
                                    Hnuclei + 14.0 * ab[IDX_GH2CNI] / 28.0 /
                                    Hnuclei + 14.0 * ab[IDX_GHC5NI] / 75.0 /
                                    Hnuclei + 14.0 * ab[IDX_GHC7NI] / 99.0 /
                                    Hnuclei + 14.0 * ab[IDX_GHC9NI] / 123.0 /
                                    Hnuclei + 14.0 * ab[IDX_GHCCNI] / 39.0 /
                                    Hnuclei + 14.0 * ab[IDX_GSiNI] / 42.0 /
                                    Hnuclei + 14.0 * ab[IDX_HCCNI] / 39.0 /
                                    Hnuclei + 14.0 * ab[IDX_CH3C3NII] / 65.0 /
                                    Hnuclei + 14.0 * ab[IDX_GCH2CHCNI] / 53.0 /
                                    Hnuclei + 14.0 * ab[IDX_GHC3NI] / 51.0 /
                                    Hnuclei + 14.0 * ab[IDX_C5NII] / 74.0 /
                                    Hnuclei + 14.0 * ab[IDX_CH2CHCNII] / 53.0 /
                                    Hnuclei + 14.0 * ab[IDX_GC2H5CNI] / 55.0 /
                                    Hnuclei + 14.0 * ab[IDX_GC9NI] / 122.0 /
                                    Hnuclei + 14.0 * ab[IDX_GOCNI] / 42.0 /
                                    Hnuclei + 14.0 * ab[IDX_H2CNOII] / 44.0 /
                                    Hnuclei + 14.0 * ab[IDX_H2NCOII] / 44.0 /
                                    Hnuclei + 14.0 * ab[IDX_H2OCNII] / 44.0 /
                                    Hnuclei + 14.0 * ab[IDX_C7NII] / 98.0 /
                                    Hnuclei + 14.0 * ab[IDX_C9NII] / 122.0 /
                                    Hnuclei + 14.0 * ab[IDX_CH3NHII] / 30.0 /
                                    Hnuclei + 14.0 * ab[IDX_GC2NI] / 38.0 /
                                    Hnuclei + 14.0 * ab[IDX_GCH2NHI] / 29.0 /
                                    Hnuclei + 14.0 * ab[IDX_GCNOI] / 42.0 /
                                    Hnuclei + 14.0 * ab[IDX_GHNCI] / 27.0 /
                                    Hnuclei + 14.0 * ab[IDX_GNSI] / 46.0 /
                                    Hnuclei + 14.0 * ab[IDX_H2C4NII] / 64.0 /
                                    Hnuclei + 14.0 * ab[IDX_H2NOII] / 32.0 /
                                    Hnuclei + 14.0 * ab[IDX_HNSII] / 47.0 /
                                    Hnuclei + 56.0 * ab[IDX_NH2CNHII] / 43.0 /
                                    Hnuclei + 14.0 * ab[IDX_GHNOI] / 31.0 /
                                    Hnuclei + 56.0 * ab[IDX_GN2I] / 28.0 /
                                    Hnuclei + 14.0 * ab[IDX_HC4NII] / 63.0 /
                                    Hnuclei + 14.0 * ab[IDX_HCNOHII] / 44.0 /
                                    Hnuclei + 14.0 * ab[IDX_HNCOHII] / 44.0 /
                                    Hnuclei + 14.0 * ab[IDX_HOCNII] / 43.0 /
                                    Hnuclei + 14.0 * ab[IDX_PNII] / 45.0 /
                                    Hnuclei + 14.0 * ab[IDX_PNH3II] / 48.0 /
                                    Hnuclei + 14.0 * ab[IDX_GC5NI] / 74.0 /
                                    Hnuclei + 14.0 * ab[IDX_H2C7NII] / 100.0 /
                                    Hnuclei + 14.0 * ab[IDX_H2C9NII] / 124.0 /
                                    Hnuclei + 14.0 * ab[IDX_HONCII] / 43.0 /
                                    Hnuclei + 14.0 * ab[IDX_PNH2II] / 47.0 /
                                    Hnuclei + 14.0 * ab[IDX_C2H5CNHII] / 56.0 /
                                    Hnuclei + 14.0 * ab[IDX_CH2NH2II] / 30.0 /
                                    Hnuclei + 14.0 * ab[IDX_CH3C5NHII] / 90.0 /
                                    Hnuclei + 14.0 * ab[IDX_CH3C7NHII] / 114.0 /
                                    Hnuclei + 14.0 * ab[IDX_GC3NI] / 50.0 /
                                    Hnuclei + 14.0 * ab[IDX_GC7NI] / 98.0 /
                                    Hnuclei + 14.0 * ab[IDX_GHCNI] / 27.0 /
                                    Hnuclei + 14.0 * ab[IDX_H2CNI] / 28.0 /
                                    Hnuclei + 14.0 * ab[IDX_H3C5NII] / 77.0 /
                                    Hnuclei + 14.0 * ab[IDX_HCNOII] / 43.0 /
                                    Hnuclei + 14.0 * ab[IDX_GNH3I] / 17.0 /
                                    Hnuclei + 14.0 * ab[IDX_H3C9NII] / 125.0 /
                                    Hnuclei + 14.0 * ab[IDX_HPNII] / 46.0 /
                                    Hnuclei + 14.0 * ab[IDX_C4NI] / 62.0 /
                                    Hnuclei + 14.0 * ab[IDX_H3C7NII] / 101.0 /
                                    Hnuclei + 56.0 * ab[IDX_NH2CNI] / 42.0 /
                                    Hnuclei + 14.0 * ab[IDX_CH3C3NHII] / 66.0 /
                                    Hnuclei + 14.0 * ab[IDX_HC7NII] / 99.0 /
                                    Hnuclei + 14.0 * ab[IDX_HC9NII] / 123.0 /
                                    Hnuclei + 14.0 * ab[IDX_HNCOII] / 43.0 /
                                    Hnuclei + 14.0 * ab[IDX_NO2II] / 46.0 /
                                    Hnuclei + 14.0 * ab[IDX_C2H5CNI] / 55.0 /
                                    Hnuclei + 14.0 * ab[IDX_CH2CHCNHII] / 54.0 /
                                    Hnuclei + 56.0 * ab[IDX_HN2OII] / 45.0 /
                                    Hnuclei + 56.0 * ab[IDX_N2OII] / 44.0 /
                                    Hnuclei + 14.0 * ab[IDX_SiNCHII] / 55.0 /
                                    Hnuclei + 14.0 * ab[IDX_HNSiII] / 43.0 /
                                    Hnuclei + 14.0 * ab[IDX_NSII] / 46.0 /
                                    Hnuclei + 14.0 * ab[IDX_OCNII] / 42.0 /
                                    Hnuclei + 56.0 * ab[IDX_C2N2II] / 52.0 /
                                    Hnuclei + 14.0 * ab[IDX_CH3CNII] / 41.0 /
                                    Hnuclei + 14.0 * ab[IDX_HC5NHII] / 76.0 /
                                    Hnuclei + 14.0 * ab[IDX_SiNH2II] / 44.0 /
                                    Hnuclei + 14.0 * ab[IDX_C3NII] / 50.0 /
                                    Hnuclei + 14.0 * ab[IDX_CH2CNII] / 40.0 /
                                    Hnuclei + 14.0 * ab[IDX_HC5NII] / 75.0 /
                                    Hnuclei + 14.0 * ab[IDX_HONCI] / 43.0 /
                                    Hnuclei + 14.0 * ab[IDX_SiNII] / 42.0 /
                                    Hnuclei + 14.0 * ab[IDX_CH3C5NI] / 89.0 /
                                    Hnuclei + 14.0 * ab[IDX_CH3C7NI] / 113.0 /
                                    Hnuclei + 14.0 * ab[IDX_H2NCII] / 28.0 /
                                    Hnuclei + 14.0 * ab[IDX_SiNCI] / 54.0 /
                                    Hnuclei + 14.0 * ab[IDX_SiNCII] / 54.0 /
                                    Hnuclei + 14.0 * ab[IDX_CNOI] / 42.0 /
                                    Hnuclei + 14.0 * ab[IDX_HNSiI] / 43.0 /
                                    Hnuclei + 14.0 * ab[IDX_C2NHII] / 39.0 /
                                    Hnuclei + 14.0 * ab[IDX_HCNOI] / 43.0 /
                                    Hnuclei + 14.0 * ab[IDX_C4NII] / 62.0 /
                                    Hnuclei + 14.0 * ab[IDX_CH3C3NI] / 65.0 /
                                    Hnuclei + 14.0 * ab[IDX_HOCNI] / 43.0 /
                                    Hnuclei + 14.0 * ab[IDX_GNOI] / 30.0 /
                                    Hnuclei + 14.0 * ab[IDX_HC9NI] / 123.0 /
                                    Hnuclei + 56.0 * ab[IDX_NCCNHII] / 53.0 /
                                    Hnuclei + 14.0 * ab[IDX_NO2I] / 46.0 /
                                    Hnuclei + 14.0 * ab[IDX_PNI] / 45.0 /
                                    Hnuclei + 14.0 * ab[IDX_CH2CNI] / 40.0 /
                                    Hnuclei + 14.0 * ab[IDX_HNCOI] / 43.0 /
                                    Hnuclei + 14.0 * ab[IDX_C9NI] / 122.0 /
                                    Hnuclei + 14.0 * ab[IDX_GNH2I] / 16.0 /
                                    Hnuclei + 14.0 * ab[IDX_CH2NHI] / 29.0 /
                                    Hnuclei + 14.0 * ab[IDX_SiNI] / 42.0 /
                                    Hnuclei + 14.0 * ab[IDX_GNHI] / 15.0 /
                                    Hnuclei + 14.0 * ab[IDX_NSI] / 46.0 /
                                    Hnuclei + 14.0 * ab[IDX_CH2CHCNI] / 53.0 /
                                    Hnuclei + 14.0 * ab[IDX_CH3CNHII] / 42.0 /
                                    Hnuclei + 14.0 * ab[IDX_HC7NI] / 99.0 /
                                    Hnuclei + 14.0 * ab[IDX_C7NI] / 98.0 /
                                    Hnuclei + 14.0 * ab[IDX_HNOI] / 31.0 /
                                    Hnuclei + 56.0 * ab[IDX_N2OI] / 44.0 /
                                    Hnuclei + 14.0 * ab[IDX_C2NII] / 38.0 /
                                    Hnuclei + 14.0 * ab[IDX_HNC3I] / 51.0 /
                                    Hnuclei + 14.0 * ab[IDX_OCNI] / 42.0 /
                                    Hnuclei + 14.0 * ab[IDX_HC3NII] / 51.0 /
                                    Hnuclei + 14.0 * ab[IDX_HC5NI] / 75.0 /
                                    Hnuclei + 14.0 * ab[IDX_HC3NHII] / 52.0 /
                                    Hnuclei + 14.0 * ab[IDX_CH3CNI] / 41.0 /
                                    Hnuclei + 56.0 * ab[IDX_NCCNI] / 52.0 /
                                    Hnuclei + 14.0 * ab[IDX_HNOII] / 31.0 /
                                    Hnuclei + 14.0 * ab[IDX_GCNI] / 26.0 /
                                    Hnuclei + 14.0 * ab[IDX_CNII] / 26.0 /
                                    Hnuclei + 14.0 * ab[IDX_GNI] / 14.0 /
                                    Hnuclei + 56.0 * ab[IDX_N2II] / 28.0 /
                                    Hnuclei + 14.0 * ab[IDX_HCNII] / 27.0 /
                                    Hnuclei + 14.0 * ab[IDX_CNCII] / 38.0 /
                                    Hnuclei + 14.0 * ab[IDX_NHII] / 15.0 /
                                    Hnuclei + 14.0 * ab[IDX_NH2II] / 16.0 /
                                    Hnuclei + 14.0 * ab[IDX_C5NM] / 74.0 /
                                    Hnuclei + 14.0 * ab[IDX_HC3NI] / 51.0 /
                                    Hnuclei + 14.0 * ab[IDX_C2NI] / 38.0 /
                                    Hnuclei + 14.0 * ab[IDX_NH2I] / 16.0 /
                                    Hnuclei + 14.0 * ab[IDX_C5NI] / 74.0 /
                                    Hnuclei + 14.0 * ab[IDX_C3NM] / 50.0 /
                                    Hnuclei + 14.0 * ab[IDX_C3NI] / 50.0 /
                                    Hnuclei + 14.0 * ab[IDX_NHI] / 15.0 /
                                    Hnuclei + 14.0 * ab[IDX_CNM] / 26.0 /
                                    Hnuclei + 56.0 * ab[IDX_N2HII] / 29.0 /
                                    Hnuclei + 14.0 * ab[IDX_NII] / 14.0 /
                                    Hnuclei + 14.0 * ab[IDX_NOII] / 30.0 /
                                    Hnuclei + 14.0 * ab[IDX_HCNHII] / 28.0 /
                                    Hnuclei + 14.0 * ab[IDX_HNCI] / 27.0 /
                                    Hnuclei + 14.0 * ab[IDX_NH3II] / 17.0 /
                                    Hnuclei + 14.0 * ab[IDX_NH4II] / 18.0 /
                                    Hnuclei + 56.0 * ab[IDX_N2I] / 28.0 /
                                    Hnuclei + 14.0 * ab[IDX_NOI] / 30.0 /
                                    Hnuclei + 14.0 * ab[IDX_HCNI] / 27.0 /
                                    Hnuclei + 14.0 * ab[IDX_NH3I] / 17.0 /
                                    Hnuclei + 14.0 * ab[IDX_CNI] / 26.0 /
                                    Hnuclei + 14.0 * ab[IDX_NI] / 14.0 / Hnuclei;
    IJth(A, IDX_ELEM_N, IDX_ELEM_O) = 0.0 + 32.0 * ab[IDX_GNO2I] / 46.0 / Hnuclei
                                    + 16.0 * ab[IDX_GHCNOI] / 43.0 / Hnuclei +
                                    16.0 * ab[IDX_GHNCOI] / 43.0 / Hnuclei +
                                    16.0 * ab[IDX_GHOCNI] / 43.0 / Hnuclei +
                                    16.0 * ab[IDX_GHONCI] / 43.0 / Hnuclei +
                                    32.0 * ab[IDX_GN2OI] / 44.0 / Hnuclei + 16.0
                                    * ab[IDX_GOCNI] / 42.0 / Hnuclei + 16.0 *
                                    ab[IDX_H2CNOII] / 44.0 / Hnuclei + 16.0 *
                                    ab[IDX_H2NCOII] / 44.0 / Hnuclei + 16.0 *
                                    ab[IDX_H2OCNII] / 44.0 / Hnuclei + 16.0 *
                                    ab[IDX_GCNOI] / 42.0 / Hnuclei + 16.0 *
                                    ab[IDX_H2NOII] / 32.0 / Hnuclei + 16.0 *
                                    ab[IDX_GHNOI] / 31.0 / Hnuclei + 16.0 *
                                    ab[IDX_HCNOHII] / 44.0 / Hnuclei + 16.0 *
                                    ab[IDX_HNCOHII] / 44.0 / Hnuclei + 16.0 *
                                    ab[IDX_HOCNII] / 43.0 / Hnuclei + 16.0 *
                                    ab[IDX_HONCII] / 43.0 / Hnuclei + 16.0 *
                                    ab[IDX_HCNOII] / 43.0 / Hnuclei + 16.0 *
                                    ab[IDX_HNCOII] / 43.0 / Hnuclei + 32.0 *
                                    ab[IDX_NO2II] / 46.0 / Hnuclei + 32.0 *
                                    ab[IDX_HN2OII] / 45.0 / Hnuclei + 32.0 *
                                    ab[IDX_N2OII] / 44.0 / Hnuclei + 16.0 *
                                    ab[IDX_OCNII] / 42.0 / Hnuclei + 16.0 *
                                    ab[IDX_HONCI] / 43.0 / Hnuclei + 16.0 *
                                    ab[IDX_CNOI] / 42.0 / Hnuclei + 16.0 *
                                    ab[IDX_HCNOI] / 43.0 / Hnuclei + 16.0 *
                                    ab[IDX_HOCNI] / 43.0 / Hnuclei + 16.0 *
                                    ab[IDX_GNOI] / 30.0 / Hnuclei + 32.0 *
                                    ab[IDX_NO2I] / 46.0 / Hnuclei + 16.0 *
                                    ab[IDX_HNCOI] / 43.0 / Hnuclei + 16.0 *
                                    ab[IDX_HNOI] / 31.0 / Hnuclei + 32.0 *
                                    ab[IDX_N2OI] / 44.0 / Hnuclei + 16.0 *
                                    ab[IDX_OCNI] / 42.0 / Hnuclei + 16.0 *
                                    ab[IDX_HNOII] / 31.0 / Hnuclei + 16.0 *
                                    ab[IDX_NOII] / 30.0 / Hnuclei + 16.0 *
                                    ab[IDX_NOI] / 30.0 / Hnuclei;
    IJth(A, IDX_ELEM_N, IDX_ELEM_He) = 0.0;
    IJth(A, IDX_ELEM_N, IDX_ELEM_C) = 0.0 + 48.0 * ab[IDX_GCH3C3NI] / 65.0 /
                                    Hnuclei + 72.0 * ab[IDX_GCH3C5NI] / 89.0 /
                                    Hnuclei + 96.0 * ab[IDX_GCH3C7NI] / 113.0 /
                                    Hnuclei + 24.0 * ab[IDX_GNH2CNI] / 42.0 /
                                    Hnuclei + 48.0 * ab[IDX_GNCCNI] / 52.0 /
                                    Hnuclei + 36.0 * ab[IDX_C2H4CNI] / 54.0 /
                                    Hnuclei + 48.0 * ab[IDX_GC4NI] / 62.0 /
                                    Hnuclei + 12.0 * ab[IDX_GHCNOI] / 43.0 /
                                    Hnuclei + 36.0 * ab[IDX_GHNC3I] / 51.0 /
                                    Hnuclei + 12.0 * ab[IDX_GHNCOI] / 43.0 /
                                    Hnuclei + 12.0 * ab[IDX_GHOCNI] / 43.0 /
                                    Hnuclei + 12.0 * ab[IDX_GHONCI] / 43.0 /
                                    Hnuclei + 12.0 * ab[IDX_GSiNCI] / 54.0 /
                                    Hnuclei + 72.0 * ab[IDX_NCCNCH3II] / 67.0 /
                                    Hnuclei + 36.0 * ab[IDX_GC2H4CNI] / 54.0 /
                                    Hnuclei + 24.0 * ab[IDX_GCH2CNI] / 40.0 /
                                    Hnuclei + 24.0 * ab[IDX_GCH3CNI] / 41.0 /
                                    Hnuclei + 12.0 * ab[IDX_GH2CNI] / 28.0 /
                                    Hnuclei + 60.0 * ab[IDX_GHC5NI] / 75.0 /
                                    Hnuclei + 84.0 * ab[IDX_GHC7NI] / 99.0 /
                                    Hnuclei + 108.0 * ab[IDX_GHC9NI] / 123.0 /
                                    Hnuclei + 24.0 * ab[IDX_GHCCNI] / 39.0 /
                                    Hnuclei + 24.0 * ab[IDX_HCCNI] / 39.0 /
                                    Hnuclei + 48.0 * ab[IDX_CH3C3NII] / 65.0 /
                                    Hnuclei + 36.0 * ab[IDX_GCH2CHCNI] / 53.0 /
                                    Hnuclei + 36.0 * ab[IDX_GHC3NI] / 51.0 /
                                    Hnuclei + 60.0 * ab[IDX_C5NII] / 74.0 /
                                    Hnuclei + 36.0 * ab[IDX_CH2CHCNII] / 53.0 /
                                    Hnuclei + 36.0 * ab[IDX_GC2H5CNI] / 55.0 /
                                    Hnuclei + 108.0 * ab[IDX_GC9NI] / 122.0 /
                                    Hnuclei + 12.0 * ab[IDX_GOCNI] / 42.0 /
                                    Hnuclei + 12.0 * ab[IDX_H2CNOII] / 44.0 /
                                    Hnuclei + 12.0 * ab[IDX_H2NCOII] / 44.0 /
                                    Hnuclei + 12.0 * ab[IDX_H2OCNII] / 44.0 /
                                    Hnuclei + 84.0 * ab[IDX_C7NII] / 98.0 /
                                    Hnuclei + 108.0 * ab[IDX_C9NII] / 122.0 /
                                    Hnuclei + 12.0 * ab[IDX_CH3NHII] / 30.0 /
                                    Hnuclei + 24.0 * ab[IDX_GC2NI] / 38.0 /
                                    Hnuclei + 12.0 * ab[IDX_GCH2NHI] / 29.0 /
                                    Hnuclei + 12.0 * ab[IDX_GCNOI] / 42.0 /
                                    Hnuclei + 12.0 * ab[IDX_GHNCI] / 27.0 /
                                    Hnuclei + 48.0 * ab[IDX_H2C4NII] / 64.0 /
                                    Hnuclei + 24.0 * ab[IDX_NH2CNHII] / 43.0 /
                                    Hnuclei + 48.0 * ab[IDX_HC4NII] / 63.0 /
                                    Hnuclei + 12.0 * ab[IDX_HCNOHII] / 44.0 /
                                    Hnuclei + 12.0 * ab[IDX_HNCOHII] / 44.0 /
                                    Hnuclei + 12.0 * ab[IDX_HOCNII] / 43.0 /
                                    Hnuclei + 60.0 * ab[IDX_GC5NI] / 74.0 /
                                    Hnuclei + 84.0 * ab[IDX_H2C7NII] / 100.0 /
                                    Hnuclei + 108.0 * ab[IDX_H2C9NII] / 124.0 /
                                    Hnuclei + 12.0 * ab[IDX_HONCII] / 43.0 /
                                    Hnuclei + 36.0 * ab[IDX_C2H5CNHII] / 56.0 /
                                    Hnuclei + 12.0 * ab[IDX_CH2NH2II] / 30.0 /
                                    Hnuclei + 72.0 * ab[IDX_CH3C5NHII] / 90.0 /
                                    Hnuclei + 96.0 * ab[IDX_CH3C7NHII] / 114.0 /
                                    Hnuclei + 36.0 * ab[IDX_GC3NI] / 50.0 /
                                    Hnuclei + 84.0 * ab[IDX_GC7NI] / 98.0 /
                                    Hnuclei + 12.0 * ab[IDX_GHCNI] / 27.0 /
                                    Hnuclei + 12.0 * ab[IDX_H2CNI] / 28.0 /
                                    Hnuclei + 60.0 * ab[IDX_H3C5NII] / 77.0 /
                                    Hnuclei + 12.0 * ab[IDX_HCNOII] / 43.0 /
                                    Hnuclei + 108.0 * ab[IDX_H3C9NII] / 125.0 /
                                    Hnuclei + 48.0 * ab[IDX_C4NI] / 62.0 /
                                    Hnuclei + 84.0 * ab[IDX_H3C7NII] / 101.0 /
                                    Hnuclei + 24.0 * ab[IDX_NH2CNI] / 42.0 /
                                    Hnuclei + 48.0 * ab[IDX_CH3C3NHII] / 66.0 /
                                    Hnuclei + 84.0 * ab[IDX_HC7NII] / 99.0 /
                                    Hnuclei + 108.0 * ab[IDX_HC9NII] / 123.0 /
                                    Hnuclei + 12.0 * ab[IDX_HNCOII] / 43.0 /
                                    Hnuclei + 36.0 * ab[IDX_C2H5CNI] / 55.0 /
                                    Hnuclei + 36.0 * ab[IDX_CH2CHCNHII] / 54.0 /
                                    Hnuclei + 12.0 * ab[IDX_SiNCHII] / 55.0 /
                                    Hnuclei + 12.0 * ab[IDX_OCNII] / 42.0 /
                                    Hnuclei + 48.0 * ab[IDX_C2N2II] / 52.0 /
                                    Hnuclei + 24.0 * ab[IDX_CH3CNII] / 41.0 /
                                    Hnuclei + 60.0 * ab[IDX_HC5NHII] / 76.0 /
                                    Hnuclei + 36.0 * ab[IDX_C3NII] / 50.0 /
                                    Hnuclei + 24.0 * ab[IDX_CH2CNII] / 40.0 /
                                    Hnuclei + 60.0 * ab[IDX_HC5NII] / 75.0 /
                                    Hnuclei + 12.0 * ab[IDX_HONCI] / 43.0 /
                                    Hnuclei + 72.0 * ab[IDX_CH3C5NI] / 89.0 /
                                    Hnuclei + 96.0 * ab[IDX_CH3C7NI] / 113.0 /
                                    Hnuclei + 12.0 * ab[IDX_H2NCII] / 28.0 /
                                    Hnuclei + 12.0 * ab[IDX_SiNCI] / 54.0 /
                                    Hnuclei + 12.0 * ab[IDX_SiNCII] / 54.0 /
                                    Hnuclei + 12.0 * ab[IDX_CNOI] / 42.0 /
                                    Hnuclei + 24.0 * ab[IDX_C2NHII] / 39.0 /
                                    Hnuclei + 12.0 * ab[IDX_HCNOI] / 43.0 /
                                    Hnuclei + 48.0 * ab[IDX_C4NII] / 62.0 /
                                    Hnuclei + 48.0 * ab[IDX_CH3C3NI] / 65.0 /
                                    Hnuclei + 12.0 * ab[IDX_HOCNI] / 43.0 /
                                    Hnuclei + 108.0 * ab[IDX_HC9NI] / 123.0 /
                                    Hnuclei + 48.0 * ab[IDX_NCCNHII] / 53.0 /
                                    Hnuclei + 24.0 * ab[IDX_CH2CNI] / 40.0 /
                                    Hnuclei + 12.0 * ab[IDX_HNCOI] / 43.0 /
                                    Hnuclei + 108.0 * ab[IDX_C9NI] / 122.0 /
                                    Hnuclei + 12.0 * ab[IDX_CH2NHI] / 29.0 /
                                    Hnuclei + 36.0 * ab[IDX_CH2CHCNI] / 53.0 /
                                    Hnuclei + 24.0 * ab[IDX_CH3CNHII] / 42.0 /
                                    Hnuclei + 84.0 * ab[IDX_HC7NI] / 99.0 /
                                    Hnuclei + 84.0 * ab[IDX_C7NI] / 98.0 /
                                    Hnuclei + 24.0 * ab[IDX_C2NII] / 38.0 /
                                    Hnuclei + 36.0 * ab[IDX_HNC3I] / 51.0 /
                                    Hnuclei + 12.0 * ab[IDX_OCNI] / 42.0 /
                                    Hnuclei + 36.0 * ab[IDX_HC3NII] / 51.0 /
                                    Hnuclei + 60.0 * ab[IDX_HC5NI] / 75.0 /
                                    Hnuclei + 36.0 * ab[IDX_HC3NHII] / 52.0 /
                                    Hnuclei + 24.0 * ab[IDX_CH3CNI] / 41.0 /
                                    Hnuclei + 48.0 * ab[IDX_NCCNI] / 52.0 /
                                    Hnuclei + 12.0 * ab[IDX_GCNI] / 26.0 /
                                    Hnuclei + 12.0 * ab[IDX_CNII] / 26.0 /
                                    Hnuclei + 12.0 * ab[IDX_HCNII] / 27.0 /
                                    Hnuclei + 24.0 * ab[IDX_CNCII] / 38.0 /
                                    Hnuclei + 60.0 * ab[IDX_C5NM] / 74.0 /
                                    Hnuclei + 36.0 * ab[IDX_HC3NI] / 51.0 /
                                    Hnuclei + 24.0 * ab[IDX_C2NI] / 38.0 /
                                    Hnuclei + 60.0 * ab[IDX_C5NI] / 74.0 /
                                    Hnuclei + 36.0 * ab[IDX_C3NM] / 50.0 /
                                    Hnuclei + 36.0 * ab[IDX_C3NI] / 50.0 /
                                    Hnuclei + 12.0 * ab[IDX_CNM] / 26.0 /
                                    Hnuclei + 12.0 * ab[IDX_HCNHII] / 28.0 /
                                    Hnuclei + 12.0 * ab[IDX_HNCI] / 27.0 /
                                    Hnuclei + 12.0 * ab[IDX_HCNI] / 27.0 /
                                    Hnuclei + 12.0 * ab[IDX_CNI] / 26.0 /
                                    Hnuclei;
    IJth(A, IDX_ELEM_N, IDX_ELEM_GRAIN) = 0.0;
    IJth(A, IDX_ELEM_N, IDX_ELEM_H) = 0.0 + 3.0 * ab[IDX_GCH3C3NI] / 65.0 /
                                    Hnuclei + 3.0 * ab[IDX_GCH3C5NI] / 89.0 /
                                    Hnuclei + 3.0 * ab[IDX_GCH3C7NI] / 113.0 /
                                    Hnuclei + 1.0 * ab[IDX_GHNSiI] / 43.0 /
                                    Hnuclei + 4.0 * ab[IDX_GNH2CNI] / 42.0 /
                                    Hnuclei + 4.0 * ab[IDX_C2H4CNI] / 54.0 /
                                    Hnuclei + 1.0 * ab[IDX_GHCNOI] / 43.0 /
                                    Hnuclei + 1.0 * ab[IDX_GHNC3I] / 51.0 /
                                    Hnuclei + 1.0 * ab[IDX_GHNCOI] / 43.0 /
                                    Hnuclei + 1.0 * ab[IDX_GHOCNI] / 43.0 /
                                    Hnuclei + 1.0 * ab[IDX_GHONCI] / 43.0 /
                                    Hnuclei + 6.0 * ab[IDX_NCCNCH3II] / 67.0 /
                                    Hnuclei + 4.0 * ab[IDX_GC2H4CNI] / 54.0 /
                                    Hnuclei + 2.0 * ab[IDX_GCH2CNI] / 40.0 /
                                    Hnuclei + 3.0 * ab[IDX_GCH3CNI] / 41.0 /
                                    Hnuclei + 2.0 * ab[IDX_GH2CNI] / 28.0 /
                                    Hnuclei + 1.0 * ab[IDX_GHC5NI] / 75.0 /
                                    Hnuclei + 1.0 * ab[IDX_GHC7NI] / 99.0 /
                                    Hnuclei + 1.0 * ab[IDX_GHC9NI] / 123.0 /
                                    Hnuclei + 1.0 * ab[IDX_GHCCNI] / 39.0 /
                                    Hnuclei + 1.0 * ab[IDX_HCCNI] / 39.0 /
                                    Hnuclei + 3.0 * ab[IDX_CH3C3NII] / 65.0 /
                                    Hnuclei + 3.0 * ab[IDX_GCH2CHCNI] / 53.0 /
                                    Hnuclei + 1.0 * ab[IDX_GHC3NI] / 51.0 /
                                    Hnuclei + 3.0 * ab[IDX_CH2CHCNII] / 53.0 /
                                    Hnuclei + 5.0 * ab[IDX_GC2H5CNI] / 55.0 /
                                    Hnuclei + 2.0 * ab[IDX_H2CNOII] / 44.0 /
                                    Hnuclei + 2.0 * ab[IDX_H2NCOII] / 44.0 /
                                    Hnuclei + 2.0 * ab[IDX_H2OCNII] / 44.0 /
                                    Hnuclei + 4.0 * ab[IDX_CH3NHII] / 30.0 /
                                    Hnuclei + 3.0 * ab[IDX_GCH2NHI] / 29.0 /
                                    Hnuclei + 1.0 * ab[IDX_GHNCI] / 27.0 /
                                    Hnuclei + 2.0 * ab[IDX_H2C4NII] / 64.0 /
                                    Hnuclei + 2.0 * ab[IDX_H2NOII] / 32.0 /
                                    Hnuclei + 1.0 * ab[IDX_HNSII] / 47.0 /
                                    Hnuclei + 6.0 * ab[IDX_NH2CNHII] / 43.0 /
                                    Hnuclei + 1.0 * ab[IDX_GHNOI] / 31.0 /
                                    Hnuclei + 1.0 * ab[IDX_HC4NII] / 63.0 /
                                    Hnuclei + 2.0 * ab[IDX_HCNOHII] / 44.0 /
                                    Hnuclei + 2.0 * ab[IDX_HNCOHII] / 44.0 /
                                    Hnuclei + 1.0 * ab[IDX_HOCNII] / 43.0 /
                                    Hnuclei + 3.0 * ab[IDX_PNH3II] / 48.0 /
                                    Hnuclei + 2.0 * ab[IDX_H2C7NII] / 100.0 /
                                    Hnuclei + 2.0 * ab[IDX_H2C9NII] / 124.0 /
                                    Hnuclei + 1.0 * ab[IDX_HONCII] / 43.0 /
                                    Hnuclei + 2.0 * ab[IDX_PNH2II] / 47.0 /
                                    Hnuclei + 6.0 * ab[IDX_C2H5CNHII] / 56.0 /
                                    Hnuclei + 4.0 * ab[IDX_CH2NH2II] / 30.0 /
                                    Hnuclei + 4.0 * ab[IDX_CH3C5NHII] / 90.0 /
                                    Hnuclei + 4.0 * ab[IDX_CH3C7NHII] / 114.0 /
                                    Hnuclei + 1.0 * ab[IDX_GHCNI] / 27.0 /
                                    Hnuclei + 2.0 * ab[IDX_H2CNI] / 28.0 /
                                    Hnuclei + 3.0 * ab[IDX_H3C5NII] / 77.0 /
                                    Hnuclei + 1.0 * ab[IDX_HCNOII] / 43.0 /
                                    Hnuclei + 3.0 * ab[IDX_GNH3I] / 17.0 /
                                    Hnuclei + 3.0 * ab[IDX_H3C9NII] / 125.0 /
                                    Hnuclei + 1.0 * ab[IDX_HPNII] / 46.0 /
                                    Hnuclei + 3.0 * ab[IDX_H3C7NII] / 101.0 /
                                    Hnuclei + 4.0 * ab[IDX_NH2CNI] / 42.0 /
                                    Hnuclei + 4.0 * ab[IDX_CH3C3NHII] / 66.0 /
                                    Hnuclei + 1.0 * ab[IDX_HC7NII] / 99.0 /
                                    Hnuclei + 1.0 * ab[IDX_HC9NII] / 123.0 /
                                    Hnuclei + 1.0 * ab[IDX_HNCOII] / 43.0 /
                                    Hnuclei + 5.0 * ab[IDX_C2H5CNI] / 55.0 /
                                    Hnuclei + 4.0 * ab[IDX_CH2CHCNHII] / 54.0 /
                                    Hnuclei + 2.0 * ab[IDX_HN2OII] / 45.0 /
                                    Hnuclei + 1.0 * ab[IDX_SiNCHII] / 55.0 /
                                    Hnuclei + 1.0 * ab[IDX_HNSiII] / 43.0 /
                                    Hnuclei + 3.0 * ab[IDX_CH3CNII] / 41.0 /
                                    Hnuclei + 2.0 * ab[IDX_HC5NHII] / 76.0 /
                                    Hnuclei + 2.0 * ab[IDX_SiNH2II] / 44.0 /
                                    Hnuclei + 2.0 * ab[IDX_CH2CNII] / 40.0 /
                                    Hnuclei + 1.0 * ab[IDX_HC5NII] / 75.0 /
                                    Hnuclei + 1.0 * ab[IDX_HONCI] / 43.0 /
                                    Hnuclei + 3.0 * ab[IDX_CH3C5NI] / 89.0 /
                                    Hnuclei + 3.0 * ab[IDX_CH3C7NI] / 113.0 /
                                    Hnuclei + 2.0 * ab[IDX_H2NCII] / 28.0 /
                                    Hnuclei + 1.0 * ab[IDX_HNSiI] / 43.0 /
                                    Hnuclei + 1.0 * ab[IDX_C2NHII] / 39.0 /
                                    Hnuclei + 1.0 * ab[IDX_HCNOI] / 43.0 /
                                    Hnuclei + 3.0 * ab[IDX_CH3C3NI] / 65.0 /
                                    Hnuclei + 1.0 * ab[IDX_HOCNI] / 43.0 /
                                    Hnuclei + 1.0 * ab[IDX_HC9NI] / 123.0 /
                                    Hnuclei + 2.0 * ab[IDX_NCCNHII] / 53.0 /
                                    Hnuclei + 2.0 * ab[IDX_CH2CNI] / 40.0 /
                                    Hnuclei + 1.0 * ab[IDX_HNCOI] / 43.0 /
                                    Hnuclei + 2.0 * ab[IDX_GNH2I] / 16.0 /
                                    Hnuclei + 3.0 * ab[IDX_CH2NHI] / 29.0 /
                                    Hnuclei + 1.0 * ab[IDX_GNHI] / 15.0 /
                                    Hnuclei + 3.0 * ab[IDX_CH2CHCNI] / 53.0 /
                                    Hnuclei + 4.0 * ab[IDX_CH3CNHII] / 42.0 /
                                    Hnuclei + 1.0 * ab[IDX_HC7NI] / 99.0 /
                                    Hnuclei + 1.0 * ab[IDX_HNOI] / 31.0 /
                                    Hnuclei + 1.0 * ab[IDX_HNC3I] / 51.0 /
                                    Hnuclei + 1.0 * ab[IDX_HC3NII] / 51.0 /
                                    Hnuclei + 1.0 * ab[IDX_HC5NI] / 75.0 /
                                    Hnuclei + 2.0 * ab[IDX_HC3NHII] / 52.0 /
                                    Hnuclei + 3.0 * ab[IDX_CH3CNI] / 41.0 /
                                    Hnuclei + 1.0 * ab[IDX_HNOII] / 31.0 /
                                    Hnuclei + 1.0 * ab[IDX_HCNII] / 27.0 /
                                    Hnuclei + 1.0 * ab[IDX_NHII] / 15.0 /
                                    Hnuclei + 2.0 * ab[IDX_NH2II] / 16.0 /
                                    Hnuclei + 1.0 * ab[IDX_HC3NI] / 51.0 /
                                    Hnuclei + 2.0 * ab[IDX_NH2I] / 16.0 /
                                    Hnuclei + 1.0 * ab[IDX_NHI] / 15.0 / Hnuclei
                                    + 2.0 * ab[IDX_N2HII] / 29.0 / Hnuclei + 2.0
                                    * ab[IDX_HCNHII] / 28.0 / Hnuclei + 1.0 *
                                    ab[IDX_HNCI] / 27.0 / Hnuclei + 3.0 *
                                    ab[IDX_NH3II] / 17.0 / Hnuclei + 4.0 *
                                    ab[IDX_NH4II] / 18.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCNI] / 27.0 / Hnuclei + 3.0 *
                                    ab[IDX_NH3I] / 17.0 / Hnuclei;
    IJth(A, IDX_ELEM_O, IDX_ELEM_F) = 0.0;
    IJth(A, IDX_ELEM_O, IDX_ELEM_Cl) = 0.0 + 35.0 * ab[IDX_GClOI] / 51.0 / Hnuclei
                                    + 35.0 * ab[IDX_ClOII] / 51.0 / Hnuclei +
                                    35.0 * ab[IDX_ClOI] / 51.0 / Hnuclei;
    IJth(A, IDX_ELEM_O, IDX_ELEM_P) = 0.0 + 31.0 * ab[IDX_GHPOI] / 48.0 / Hnuclei
                                    + 31.0 * ab[IDX_GPOI] / 47.0 / Hnuclei +
                                    31.0 * ab[IDX_H2POII] / 49.0 / Hnuclei +
                                    31.0 * ab[IDX_HPOII] / 48.0 / Hnuclei + 31.0
                                    * ab[IDX_HPOI] / 48.0 / Hnuclei + 31.0 *
                                    ab[IDX_POII] / 47.0 / Hnuclei + 31.0 *
                                    ab[IDX_POI] / 47.0 / Hnuclei;
    IJth(A, IDX_ELEM_O, IDX_ELEM_Fe) = 0.0;
    IJth(A, IDX_ELEM_O, IDX_ELEM_Mg) = 0.0;
    IJth(A, IDX_ELEM_O, IDX_ELEM_Na) = 0.0;
    IJth(A, IDX_ELEM_O, IDX_ELEM_Si) = 0.0 + 56.0 * ab[IDX_GSiO2I] / 60.0 / Hnuclei
                                    + 28.0 * ab[IDX_GH2SiOI] / 46.0 / Hnuclei +
                                    28.0 * ab[IDX_GSiOI] / 44.0 / Hnuclei + 56.0
                                    * ab[IDX_HSiO2II] / 61.0 / Hnuclei + 28.0 *
                                    ab[IDX_H2SiOII] / 46.0 / Hnuclei + 28.0 *
                                    ab[IDX_H3SiOII] / 47.0 / Hnuclei + 56.0 *
                                    ab[IDX_SiO2I] / 60.0 / Hnuclei + 28.0 *
                                    ab[IDX_H2SiOI] / 46.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiOII] / 44.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiOHII] / 45.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiOI] / 44.0 / Hnuclei;
    IJth(A, IDX_ELEM_O, IDX_ELEM_S) = 0.0 + 64.0 * ab[IDX_GSO2I] / 64.0 / Hnuclei
                                    + 32.0 * ab[IDX_GOCSI] / 60.0 / Hnuclei +
                                    32.0 * ab[IDX_GSOI] / 48.0 / Hnuclei + 32.0
                                    * ab[IDX_HSOII] / 49.0 / Hnuclei + 64.0 *
                                    ab[IDX_HSO2II] / 65.0 / Hnuclei + 32.0 *
                                    ab[IDX_HOCSII] / 61.0 / Hnuclei + 64.0 *
                                    ab[IDX_SO2II] / 64.0 / Hnuclei + 32.0 *
                                    ab[IDX_OCSII] / 60.0 / Hnuclei + 64.0 *
                                    ab[IDX_SO2I] / 64.0 / Hnuclei + 32.0 *
                                    ab[IDX_OCSI] / 60.0 / Hnuclei + 32.0 *
                                    ab[IDX_SOI] / 48.0 / Hnuclei + 32.0 *
                                    ab[IDX_SOII] / 48.0 / Hnuclei;
    IJth(A, IDX_ELEM_O, IDX_ELEM_N) = 0.0 + 28.0 * ab[IDX_GNO2I] / 46.0 / Hnuclei
                                    + 14.0 * ab[IDX_GHCNOI] / 43.0 / Hnuclei +
                                    14.0 * ab[IDX_GHNCOI] / 43.0 / Hnuclei +
                                    14.0 * ab[IDX_GHOCNI] / 43.0 / Hnuclei +
                                    14.0 * ab[IDX_GHONCI] / 43.0 / Hnuclei +
                                    28.0 * ab[IDX_GN2OI] / 44.0 / Hnuclei + 14.0
                                    * ab[IDX_GOCNI] / 42.0 / Hnuclei + 14.0 *
                                    ab[IDX_H2CNOII] / 44.0 / Hnuclei + 14.0 *
                                    ab[IDX_H2NCOII] / 44.0 / Hnuclei + 14.0 *
                                    ab[IDX_H2OCNII] / 44.0 / Hnuclei + 14.0 *
                                    ab[IDX_GCNOI] / 42.0 / Hnuclei + 14.0 *
                                    ab[IDX_H2NOII] / 32.0 / Hnuclei + 14.0 *
                                    ab[IDX_GHNOI] / 31.0 / Hnuclei + 14.0 *
                                    ab[IDX_HCNOHII] / 44.0 / Hnuclei + 14.0 *
                                    ab[IDX_HNCOHII] / 44.0 / Hnuclei + 14.0 *
                                    ab[IDX_HOCNII] / 43.0 / Hnuclei + 14.0 *
                                    ab[IDX_HONCII] / 43.0 / Hnuclei + 14.0 *
                                    ab[IDX_HCNOII] / 43.0 / Hnuclei + 14.0 *
                                    ab[IDX_HNCOII] / 43.0 / Hnuclei + 28.0 *
                                    ab[IDX_NO2II] / 46.0 / Hnuclei + 28.0 *
                                    ab[IDX_HN2OII] / 45.0 / Hnuclei + 28.0 *
                                    ab[IDX_N2OII] / 44.0 / Hnuclei + 14.0 *
                                    ab[IDX_OCNII] / 42.0 / Hnuclei + 14.0 *
                                    ab[IDX_HONCI] / 43.0 / Hnuclei + 14.0 *
                                    ab[IDX_CNOI] / 42.0 / Hnuclei + 14.0 *
                                    ab[IDX_HCNOI] / 43.0 / Hnuclei + 14.0 *
                                    ab[IDX_HOCNI] / 43.0 / Hnuclei + 14.0 *
                                    ab[IDX_GNOI] / 30.0 / Hnuclei + 28.0 *
                                    ab[IDX_NO2I] / 46.0 / Hnuclei + 14.0 *
                                    ab[IDX_HNCOI] / 43.0 / Hnuclei + 14.0 *
                                    ab[IDX_HNOI] / 31.0 / Hnuclei + 28.0 *
                                    ab[IDX_N2OI] / 44.0 / Hnuclei + 14.0 *
                                    ab[IDX_OCNI] / 42.0 / Hnuclei + 14.0 *
                                    ab[IDX_HNOII] / 31.0 / Hnuclei + 14.0 *
                                    ab[IDX_NOII] / 30.0 / Hnuclei + 14.0 *
                                    ab[IDX_NOI] / 30.0 / Hnuclei;
    IJth(A, IDX_ELEM_O, IDX_ELEM_O) = 0.0 + 16.0 * ab[IDX_GClOI] / 51.0 / Hnuclei
                                    + 16.0 * ab[IDX_GHPOI] / 48.0 / Hnuclei +
                                    64.0 * ab[IDX_GNO2I] / 46.0 / Hnuclei + 64.0
                                    * ab[IDX_GSiO2I] / 60.0 / Hnuclei + 16.0 *
                                    ab[IDX_GH2SiOI] / 46.0 / Hnuclei + 64.0 *
                                    ab[IDX_GCH3COOHI] / 60.0 / Hnuclei + 16.0 *
                                    ab[IDX_GHCNOI] / 43.0 / Hnuclei + 16.0 *
                                    ab[IDX_GHNCOI] / 43.0 / Hnuclei + 16.0 *
                                    ab[IDX_GHOCNI] / 43.0 / Hnuclei + 16.0 *
                                    ab[IDX_GHONCI] / 43.0 / Hnuclei + 16.0 *
                                    ab[IDX_GN2OI] / 44.0 / Hnuclei + 16.0 *
                                    ab[IDX_GPOI] / 47.0 / Hnuclei + 16.0 *
                                    ab[IDX_HC2OI] / 41.0 / Hnuclei + 64.0 *
                                    ab[IDX_GH2O2I] / 34.0 / Hnuclei + 64.0 *
                                    ab[IDX_GSO2I] / 64.0 / Hnuclei + 16.0 *
                                    ab[IDX_GC3OI] / 52.0 / Hnuclei + 16.0 *
                                    ab[IDX_GHC2OI] / 41.0 / Hnuclei + 16.0 *
                                    ab[IDX_GSiOI] / 44.0 / Hnuclei + 16.0 *
                                    ab[IDX_C3H2OII] / 54.0 / Hnuclei + 16.0 *
                                    ab[IDX_ClOII] / 51.0 / Hnuclei + 16.0 *
                                    ab[IDX_GCH3COCH3I] / 58.0 / Hnuclei + 16.0 *
                                    ab[IDX_GCH3OCH3I] / 46.0 / Hnuclei + 16.0 *
                                    ab[IDX_H3C3OII] / 55.0 / Hnuclei + 16.0 *
                                    ab[IDX_GOCSI] / 60.0 / Hnuclei + 64.0 *
                                    ab[IDX_COOCH3II] / 59.0 / Hnuclei + 16.0 *
                                    ab[IDX_GC2H5OHI] / 46.0 / Hnuclei + 16.0 *
                                    ab[IDX_GC2OI] / 40.0 / Hnuclei + 64.0 *
                                    ab[IDX_GCO2I] / 44.0 / Hnuclei + 64.0 *
                                    ab[IDX_GO2HI] / 33.0 / Hnuclei + 16.0 *
                                    ab[IDX_GOCNI] / 42.0 / Hnuclei + 16.0 *
                                    ab[IDX_GSOI] / 48.0 / Hnuclei + 16.0 *
                                    ab[IDX_H2CNOII] / 44.0 / Hnuclei + 16.0 *
                                    ab[IDX_H2NCOII] / 44.0 / Hnuclei + 16.0 *
                                    ab[IDX_H2OCNII] / 44.0 / Hnuclei + 16.0 *
                                    ab[IDX_ClOI] / 51.0 / Hnuclei + 16.0 *
                                    ab[IDX_GCNOI] / 42.0 / Hnuclei + 16.0 *
                                    ab[IDX_H2NOII] / 32.0 / Hnuclei + 16.0 *
                                    ab[IDX_HOCII] / 29.0 / Hnuclei + 16.0 *
                                    ab[IDX_HSOII] / 49.0 / Hnuclei + 64.0 *
                                    ab[IDX_HSiO2II] / 61.0 / Hnuclei + 16.0 *
                                    ab[IDX_C2H5OHII] / 46.0 / Hnuclei + 16.0 *
                                    ab[IDX_C2OII] / 40.0 / Hnuclei + 64.0 *
                                    ab[IDX_CH2OHCOII] / 59.0 / Hnuclei + 16.0 *
                                    ab[IDX_GCH2COI] / 42.0 / Hnuclei + 16.0 *
                                    ab[IDX_GHNOI] / 31.0 / Hnuclei + 16.0 *
                                    ab[IDX_HCNOHII] / 44.0 / Hnuclei + 16.0 *
                                    ab[IDX_HNCOHII] / 44.0 / Hnuclei + 16.0 *
                                    ab[IDX_HOCNII] / 43.0 / Hnuclei + 64.0 *
                                    ab[IDX_H2O2I] / 34.0 / Hnuclei + 16.0 *
                                    ab[IDX_HONCII] / 43.0 / Hnuclei + 64.0 *
                                    ab[IDX_CH2OHCH2OII] / 61.0 / Hnuclei + 64.0
                                    * ab[IDX_GCOOCH3I] / 59.0 / Hnuclei + 16.0 *
                                    ab[IDX_H2SiOII] / 46.0 / Hnuclei + 16.0 *
                                    ab[IDX_HCNOII] / 43.0 / Hnuclei + 64.0 *
                                    ab[IDX_CH3COOHII] / 60.0 / Hnuclei + 64.0 *
                                    ab[IDX_H5C2O2II] / 61.0 / Hnuclei + 64.0 *
                                    ab[IDX_HCOOHII] / 46.0 / Hnuclei + 64.0 *
                                    ab[IDX_CH3COOH2II] / 61.0 / Hnuclei + 16.0 *
                                    ab[IDX_CH3OCH3II] / 46.0 / Hnuclei + 64.0 *
                                    ab[IDX_GCH2OHCOI] / 59.0 / Hnuclei + 64.0 *
                                    ab[IDX_GCOOHI] / 45.0 / Hnuclei + 64.0 *
                                    ab[IDX_GO2I] / 32.0 / Hnuclei + 16.0 *
                                    ab[IDX_H3SiOII] / 47.0 / Hnuclei + 64.0 *
                                    ab[IDX_HSO2II] / 65.0 / Hnuclei + 16.0 *
                                    ab[IDX_CH3OCH4II] / 47.0 / Hnuclei + 16.0 *
                                    ab[IDX_HNCOII] / 43.0 / Hnuclei + 64.0 *
                                    ab[IDX_NO2II] / 46.0 / Hnuclei + 64.0 *
                                    ab[IDX_SiO2I] / 60.0 / Hnuclei + 16.0 *
                                    ab[IDX_C2H5OH2II] / 47.0 / Hnuclei + 64.0 *
                                    ab[IDX_COOCH3I] / 59.0 / Hnuclei + 16.0 *
                                    ab[IDX_H2POII] / 49.0 / Hnuclei + 64.0 *
                                    ab[IDX_HCOOCH3II] / 60.0 / Hnuclei + 16.0 *
                                    ab[IDX_HN2OII] / 45.0 / Hnuclei + 16.0 *
                                    ab[IDX_N2OII] / 44.0 / Hnuclei + 16.0 *
                                    ab[IDX_C3OII] / 52.0 / Hnuclei + 64.0 *
                                    ab[IDX_CH2OHCHOII] / 60.0 / Hnuclei + 16.0 *
                                    ab[IDX_CH3CHOII] / 44.0 / Hnuclei + 64.0 *
                                    ab[IDX_GCH2OHCHOI] / 60.0 / Hnuclei + 64.0 *
                                    ab[IDX_GHCOOHI] / 46.0 / Hnuclei + 16.0 *
                                    ab[IDX_HOCSII] / 61.0 / Hnuclei + 16.0 *
                                    ab[IDX_OCNII] / 42.0 / Hnuclei + 16.0 *
                                    ab[IDX_GCH3COI] / 43.0 / Hnuclei + 64.0 *
                                    ab[IDX_GHCOOCH3I] / 60.0 / Hnuclei + 16.0 *
                                    ab[IDX_CH3COCH3II] / 58.0 / Hnuclei + 16.0 *
                                    ab[IDX_GCH2OHI] / 31.0 / Hnuclei + 16.0 *
                                    ab[IDX_GCH3OHI] / 32.0 / Hnuclei + 16.0 *
                                    ab[IDX_HONCI] / 43.0 / Hnuclei + 64.0 *
                                    ab[IDX_SO2II] / 64.0 / Hnuclei + 64.0 *
                                    ab[IDX_CH2OHCOI] / 59.0 / Hnuclei + 16.0 *
                                    ab[IDX_H2SiOI] / 46.0 / Hnuclei + 16.0 *
                                    ab[IDX_HC2OII] / 41.0 / Hnuclei + 16.0 *
                                    ab[IDX_CNOI] / 42.0 / Hnuclei + 16.0 *
                                    ab[IDX_GCH3OI] / 31.0 / Hnuclei + 16.0 *
                                    ab[IDX_GH2OI] / 18.0 / Hnuclei + 64.0 *
                                    ab[IDX_HCOOH2II] / 47.0 / Hnuclei + 16.0 *
                                    ab[IDX_HPOII] / 48.0 / Hnuclei + 64.0 *
                                    ab[IDX_CH2OHCHOI] / 60.0 / Hnuclei + 16.0 *
                                    ab[IDX_HC3OII] / 53.0 / Hnuclei + 16.0 *
                                    ab[IDX_HCNOI] / 43.0 / Hnuclei + 16.0 *
                                    ab[IDX_CH2COII] / 42.0 / Hnuclei + 16.0 *
                                    ab[IDX_GCH3CHOI] / 44.0 / Hnuclei + 16.0 *
                                    ab[IDX_GH2COI] / 30.0 / Hnuclei + 16.0 *
                                    ab[IDX_HOCNI] / 43.0 / Hnuclei + 16.0 *
                                    ab[IDX_HPOI] / 48.0 / Hnuclei + 16.0 *
                                    ab[IDX_CH3COI] / 43.0 / Hnuclei + 16.0 *
                                    ab[IDX_CH3COCH4II] / 59.0 / Hnuclei + 16.0 *
                                    ab[IDX_CH3OHII] / 32.0 / Hnuclei + 64.0 *
                                    ab[IDX_COOHI] / 45.0 / Hnuclei + 16.0 *
                                    ab[IDX_GNOI] / 30.0 / Hnuclei + 16.0 *
                                    ab[IDX_POII] / 47.0 / Hnuclei + 64.0 *
                                    ab[IDX_HCOOCH3I] / 60.0 / Hnuclei + 64.0 *
                                    ab[IDX_NO2I] / 46.0 / Hnuclei + 16.0 *
                                    ab[IDX_HNCOI] / 43.0 / Hnuclei + 64.0 *
                                    ab[IDX_CH3COOHI] / 60.0 / Hnuclei + 16.0 *
                                    ab[IDX_CH3OCH3I] / 46.0 / Hnuclei + 16.0 *
                                    ab[IDX_POI] / 47.0 / Hnuclei + 16.0 *
                                    ab[IDX_CH3OI] / 31.0 / Hnuclei + 16.0 *
                                    ab[IDX_GHCOI] / 29.0 / Hnuclei + 16.0 *
                                    ab[IDX_OCSII] / 60.0 / Hnuclei + 16.0 *
                                    ab[IDX_C2OI] / 40.0 / Hnuclei + 16.0 *
                                    ab[IDX_CH2OHI] / 31.0 / Hnuclei + 16.0 *
                                    ab[IDX_C3OI] / 52.0 / Hnuclei + 16.0 *
                                    ab[IDX_CH3CHOHII] / 45.0 / Hnuclei + 16.0 *
                                    ab[IDX_CH3COII] / 43.0 / Hnuclei + 64.0 *
                                    ab[IDX_HCOOHI] / 46.0 / Hnuclei + 64.0 *
                                    ab[IDX_O2HI] / 33.0 / Hnuclei + 64.0 *
                                    ab[IDX_SO2I] / 64.0 / Hnuclei + 16.0 *
                                    ab[IDX_C2H5OHI] / 46.0 / Hnuclei + 16.0 *
                                    ab[IDX_HNOI] / 31.0 / Hnuclei + 16.0 *
                                    ab[IDX_N2OI] / 44.0 / Hnuclei + 16.0 *
                                    ab[IDX_CH3OH2II] / 33.0 / Hnuclei + 16.0 *
                                    ab[IDX_CH2COI] / 42.0 / Hnuclei + 16.0 *
                                    ab[IDX_OCNI] / 42.0 / Hnuclei + 16.0 *
                                    ab[IDX_CH3COCH3I] / 58.0 / Hnuclei + 64.0 *
                                    ab[IDX_CO2II] / 44.0 / Hnuclei + 16.0 *
                                    ab[IDX_GOHI] / 17.0 / Hnuclei + 16.0 *
                                    ab[IDX_GCOI] / 28.0 / Hnuclei + 64.0 *
                                    ab[IDX_HCO2II] / 45.0 / Hnuclei + 16.0 *
                                    ab[IDX_CH3CHOI] / 44.0 / Hnuclei + 64.0 *
                                    ab[IDX_O2HII] / 33.0 / Hnuclei + 16.0 *
                                    ab[IDX_HNOII] / 31.0 / Hnuclei + 16.0 *
                                    ab[IDX_GOI] / 16.0 / Hnuclei + 64.0 *
                                    ab[IDX_O2M] / 32.0 / Hnuclei + 16.0 *
                                    ab[IDX_OCSI] / 60.0 / Hnuclei + 16.0 *
                                    ab[IDX_COII] / 28.0 / Hnuclei + 16.0 *
                                    ab[IDX_OHM] / 17.0 / Hnuclei + 16.0 *
                                    ab[IDX_OM] / 16.0 / Hnuclei + 16.0 *
                                    ab[IDX_CH3OHI] / 32.0 / Hnuclei + 16.0 *
                                    ab[IDX_H3COII] / 31.0 / Hnuclei + 16.0 *
                                    ab[IDX_SiOII] / 44.0 / Hnuclei + 16.0 *
                                    ab[IDX_SiOHII] / 45.0 / Hnuclei + 16.0 *
                                    ab[IDX_OHII] / 17.0 / Hnuclei + 16.0 *
                                    ab[IDX_H2OII] / 18.0 / Hnuclei + 64.0 *
                                    ab[IDX_CO2I] / 44.0 / Hnuclei + 64.0 *
                                    ab[IDX_O2II] / 32.0 / Hnuclei + 16.0 *
                                    ab[IDX_SiOI] / 44.0 / Hnuclei + 16.0 *
                                    ab[IDX_SOI] / 48.0 / Hnuclei + 16.0 *
                                    ab[IDX_SOII] / 48.0 / Hnuclei + 16.0 *
                                    ab[IDX_H2COII] / 30.0 / Hnuclei + 16.0 *
                                    ab[IDX_NOII] / 30.0 / Hnuclei + 16.0 *
                                    ab[IDX_OII] / 16.0 / Hnuclei + 64.0 *
                                    ab[IDX_O2I] / 32.0 / Hnuclei + 16.0 *
                                    ab[IDX_NOI] / 30.0 / Hnuclei + 16.0 *
                                    ab[IDX_HCOI] / 29.0 / Hnuclei + 16.0 *
                                    ab[IDX_H2COI] / 30.0 / Hnuclei + 16.0 *
                                    ab[IDX_OHI] / 17.0 / Hnuclei + 16.0 *
                                    ab[IDX_H3OII] / 19.0 / Hnuclei + 16.0 *
                                    ab[IDX_OI] / 16.0 / Hnuclei + 16.0 *
                                    ab[IDX_H2OI] / 18.0 / Hnuclei + 16.0 *
                                    ab[IDX_HCOII] / 29.0 / Hnuclei + 16.0 *
                                    ab[IDX_COI] / 28.0 / Hnuclei;
    IJth(A, IDX_ELEM_O, IDX_ELEM_He) = 0.0;
    IJth(A, IDX_ELEM_O, IDX_ELEM_C) = 0.0 + 48.0 * ab[IDX_GCH3COOHI] / 60.0 /
                                    Hnuclei + 12.0 * ab[IDX_GHCNOI] / 43.0 /
                                    Hnuclei + 12.0 * ab[IDX_GHNCOI] / 43.0 /
                                    Hnuclei + 12.0 * ab[IDX_GHOCNI] / 43.0 /
                                    Hnuclei + 12.0 * ab[IDX_GHONCI] / 43.0 /
                                    Hnuclei + 24.0 * ab[IDX_HC2OI] / 41.0 /
                                    Hnuclei + 36.0 * ab[IDX_GC3OI] / 52.0 /
                                    Hnuclei + 24.0 * ab[IDX_GHC2OI] / 41.0 /
                                    Hnuclei + 36.0 * ab[IDX_C3H2OII] / 54.0 /
                                    Hnuclei + 36.0 * ab[IDX_GCH3COCH3I] / 58.0 /
                                    Hnuclei + 24.0 * ab[IDX_GCH3OCH3I] / 46.0 /
                                    Hnuclei + 36.0 * ab[IDX_H3C3OII] / 55.0 /
                                    Hnuclei + 12.0 * ab[IDX_GOCSI] / 60.0 /
                                    Hnuclei + 48.0 * ab[IDX_COOCH3II] / 59.0 /
                                    Hnuclei + 24.0 * ab[IDX_GC2H5OHI] / 46.0 /
                                    Hnuclei + 24.0 * ab[IDX_GC2OI] / 40.0 /
                                    Hnuclei + 24.0 * ab[IDX_GCO2I] / 44.0 /
                                    Hnuclei + 12.0 * ab[IDX_GOCNI] / 42.0 /
                                    Hnuclei + 12.0 * ab[IDX_H2CNOII] / 44.0 /
                                    Hnuclei + 12.0 * ab[IDX_H2NCOII] / 44.0 /
                                    Hnuclei + 12.0 * ab[IDX_H2OCNII] / 44.0 /
                                    Hnuclei + 12.0 * ab[IDX_GCNOI] / 42.0 /
                                    Hnuclei + 12.0 * ab[IDX_HOCII] / 29.0 /
                                    Hnuclei + 24.0 * ab[IDX_C2H5OHII] / 46.0 /
                                    Hnuclei + 24.0 * ab[IDX_C2OII] / 40.0 /
                                    Hnuclei + 48.0 * ab[IDX_CH2OHCOII] / 59.0 /
                                    Hnuclei + 24.0 * ab[IDX_GCH2COI] / 42.0 /
                                    Hnuclei + 12.0 * ab[IDX_HCNOHII] / 44.0 /
                                    Hnuclei + 12.0 * ab[IDX_HNCOHII] / 44.0 /
                                    Hnuclei + 12.0 * ab[IDX_HOCNII] / 43.0 /
                                    Hnuclei + 12.0 * ab[IDX_HONCII] / 43.0 /
                                    Hnuclei + 48.0 * ab[IDX_CH2OHCH2OII] / 61.0
                                    / Hnuclei + 48.0 * ab[IDX_GCOOCH3I] / 59.0 /
                                    Hnuclei + 12.0 * ab[IDX_HCNOII] / 43.0 /
                                    Hnuclei + 48.0 * ab[IDX_CH3COOHII] / 60.0 /
                                    Hnuclei + 48.0 * ab[IDX_H5C2O2II] / 61.0 /
                                    Hnuclei + 24.0 * ab[IDX_HCOOHII] / 46.0 /
                                    Hnuclei + 48.0 * ab[IDX_CH3COOH2II] / 61.0 /
                                    Hnuclei + 24.0 * ab[IDX_CH3OCH3II] / 46.0 /
                                    Hnuclei + 48.0 * ab[IDX_GCH2OHCOI] / 59.0 /
                                    Hnuclei + 24.0 * ab[IDX_GCOOHI] / 45.0 /
                                    Hnuclei + 24.0 * ab[IDX_CH3OCH4II] / 47.0 /
                                    Hnuclei + 12.0 * ab[IDX_HNCOII] / 43.0 /
                                    Hnuclei + 24.0 * ab[IDX_C2H5OH2II] / 47.0 /
                                    Hnuclei + 48.0 * ab[IDX_COOCH3I] / 59.0 /
                                    Hnuclei + 48.0 * ab[IDX_HCOOCH3II] / 60.0 /
                                    Hnuclei + 36.0 * ab[IDX_C3OII] / 52.0 /
                                    Hnuclei + 48.0 * ab[IDX_CH2OHCHOII] / 60.0 /
                                    Hnuclei + 24.0 * ab[IDX_CH3CHOII] / 44.0 /
                                    Hnuclei + 48.0 * ab[IDX_GCH2OHCHOI] / 60.0 /
                                    Hnuclei + 24.0 * ab[IDX_GHCOOHI] / 46.0 /
                                    Hnuclei + 12.0 * ab[IDX_HOCSII] / 61.0 /
                                    Hnuclei + 12.0 * ab[IDX_OCNII] / 42.0 /
                                    Hnuclei + 24.0 * ab[IDX_GCH3COI] / 43.0 /
                                    Hnuclei + 48.0 * ab[IDX_GHCOOCH3I] / 60.0 /
                                    Hnuclei + 36.0 * ab[IDX_CH3COCH3II] / 58.0 /
                                    Hnuclei + 12.0 * ab[IDX_GCH2OHI] / 31.0 /
                                    Hnuclei + 12.0 * ab[IDX_GCH3OHI] / 32.0 /
                                    Hnuclei + 12.0 * ab[IDX_HONCI] / 43.0 /
                                    Hnuclei + 48.0 * ab[IDX_CH2OHCOI] / 59.0 /
                                    Hnuclei + 24.0 * ab[IDX_HC2OII] / 41.0 /
                                    Hnuclei + 12.0 * ab[IDX_CNOI] / 42.0 /
                                    Hnuclei + 12.0 * ab[IDX_GCH3OI] / 31.0 /
                                    Hnuclei + 24.0 * ab[IDX_HCOOH2II] / 47.0 /
                                    Hnuclei + 48.0 * ab[IDX_CH2OHCHOI] / 60.0 /
                                    Hnuclei + 36.0 * ab[IDX_HC3OII] / 53.0 /
                                    Hnuclei + 12.0 * ab[IDX_HCNOI] / 43.0 /
                                    Hnuclei + 24.0 * ab[IDX_CH2COII] / 42.0 /
                                    Hnuclei + 24.0 * ab[IDX_GCH3CHOI] / 44.0 /
                                    Hnuclei + 12.0 * ab[IDX_GH2COI] / 30.0 /
                                    Hnuclei + 12.0 * ab[IDX_HOCNI] / 43.0 /
                                    Hnuclei + 24.0 * ab[IDX_CH3COI] / 43.0 /
                                    Hnuclei + 36.0 * ab[IDX_CH3COCH4II] / 59.0 /
                                    Hnuclei + 12.0 * ab[IDX_CH3OHII] / 32.0 /
                                    Hnuclei + 24.0 * ab[IDX_COOHI] / 45.0 /
                                    Hnuclei + 48.0 * ab[IDX_HCOOCH3I] / 60.0 /
                                    Hnuclei + 12.0 * ab[IDX_HNCOI] / 43.0 /
                                    Hnuclei + 48.0 * ab[IDX_CH3COOHI] / 60.0 /
                                    Hnuclei + 24.0 * ab[IDX_CH3OCH3I] / 46.0 /
                                    Hnuclei + 12.0 * ab[IDX_CH3OI] / 31.0 /
                                    Hnuclei + 12.0 * ab[IDX_GHCOI] / 29.0 /
                                    Hnuclei + 12.0 * ab[IDX_OCSII] / 60.0 /
                                    Hnuclei + 24.0 * ab[IDX_C2OI] / 40.0 /
                                    Hnuclei + 12.0 * ab[IDX_CH2OHI] / 31.0 /
                                    Hnuclei + 36.0 * ab[IDX_C3OI] / 52.0 /
                                    Hnuclei + 24.0 * ab[IDX_CH3CHOHII] / 45.0 /
                                    Hnuclei + 24.0 * ab[IDX_CH3COII] / 43.0 /
                                    Hnuclei + 24.0 * ab[IDX_HCOOHI] / 46.0 /
                                    Hnuclei + 24.0 * ab[IDX_C2H5OHI] / 46.0 /
                                    Hnuclei + 12.0 * ab[IDX_CH3OH2II] / 33.0 /
                                    Hnuclei + 24.0 * ab[IDX_CH2COI] / 42.0 /
                                    Hnuclei + 12.0 * ab[IDX_OCNI] / 42.0 /
                                    Hnuclei + 36.0 * ab[IDX_CH3COCH3I] / 58.0 /
                                    Hnuclei + 24.0 * ab[IDX_CO2II] / 44.0 /
                                    Hnuclei + 12.0 * ab[IDX_GCOI] / 28.0 /
                                    Hnuclei + 24.0 * ab[IDX_HCO2II] / 45.0 /
                                    Hnuclei + 24.0 * ab[IDX_CH3CHOI] / 44.0 /
                                    Hnuclei + 12.0 * ab[IDX_OCSI] / 60.0 /
                                    Hnuclei + 12.0 * ab[IDX_COII] / 28.0 /
                                    Hnuclei + 12.0 * ab[IDX_CH3OHI] / 32.0 /
                                    Hnuclei + 12.0 * ab[IDX_H3COII] / 31.0 /
                                    Hnuclei + 24.0 * ab[IDX_CO2I] / 44.0 /
                                    Hnuclei + 12.0 * ab[IDX_H2COII] / 30.0 /
                                    Hnuclei + 12.0 * ab[IDX_HCOI] / 29.0 /
                                    Hnuclei + 12.0 * ab[IDX_H2COI] / 30.0 /
                                    Hnuclei + 12.0 * ab[IDX_HCOII] / 29.0 /
                                    Hnuclei + 12.0 * ab[IDX_COI] / 28.0 /
                                    Hnuclei;
    IJth(A, IDX_ELEM_O, IDX_ELEM_GRAIN) = 0.0;
    IJth(A, IDX_ELEM_O, IDX_ELEM_H) = 0.0 + 1.0 * ab[IDX_GHPOI] / 48.0 / Hnuclei +
                                    2.0 * ab[IDX_GH2SiOI] / 46.0 / Hnuclei + 8.0
                                    * ab[IDX_GCH3COOHI] / 60.0 / Hnuclei + 1.0 *
                                    ab[IDX_GHCNOI] / 43.0 / Hnuclei + 1.0 *
                                    ab[IDX_GHNCOI] / 43.0 / Hnuclei + 1.0 *
                                    ab[IDX_GHOCNI] / 43.0 / Hnuclei + 1.0 *
                                    ab[IDX_GHONCI] / 43.0 / Hnuclei + 1.0 *
                                    ab[IDX_HC2OI] / 41.0 / Hnuclei + 4.0 *
                                    ab[IDX_GH2O2I] / 34.0 / Hnuclei + 1.0 *
                                    ab[IDX_GHC2OI] / 41.0 / Hnuclei + 2.0 *
                                    ab[IDX_C3H2OII] / 54.0 / Hnuclei + 6.0 *
                                    ab[IDX_GCH3COCH3I] / 58.0 / Hnuclei + 6.0 *
                                    ab[IDX_GCH3OCH3I] / 46.0 / Hnuclei + 3.0 *
                                    ab[IDX_H3C3OII] / 55.0 / Hnuclei + 6.0 *
                                    ab[IDX_COOCH3II] / 59.0 / Hnuclei + 6.0 *
                                    ab[IDX_GC2H5OHI] / 46.0 / Hnuclei + 2.0 *
                                    ab[IDX_GO2HI] / 33.0 / Hnuclei + 2.0 *
                                    ab[IDX_H2CNOII] / 44.0 / Hnuclei + 2.0 *
                                    ab[IDX_H2NCOII] / 44.0 / Hnuclei + 2.0 *
                                    ab[IDX_H2OCNII] / 44.0 / Hnuclei + 2.0 *
                                    ab[IDX_H2NOII] / 32.0 / Hnuclei + 1.0 *
                                    ab[IDX_HOCII] / 29.0 / Hnuclei + 1.0 *
                                    ab[IDX_HSOII] / 49.0 / Hnuclei + 2.0 *
                                    ab[IDX_HSiO2II] / 61.0 / Hnuclei + 6.0 *
                                    ab[IDX_C2H5OHII] / 46.0 / Hnuclei + 6.0 *
                                    ab[IDX_CH2OHCOII] / 59.0 / Hnuclei + 2.0 *
                                    ab[IDX_GCH2COI] / 42.0 / Hnuclei + 1.0 *
                                    ab[IDX_GHNOI] / 31.0 / Hnuclei + 2.0 *
                                    ab[IDX_HCNOHII] / 44.0 / Hnuclei + 2.0 *
                                    ab[IDX_HNCOHII] / 44.0 / Hnuclei + 1.0 *
                                    ab[IDX_HOCNII] / 43.0 / Hnuclei + 4.0 *
                                    ab[IDX_H2O2I] / 34.0 / Hnuclei + 1.0 *
                                    ab[IDX_HONCII] / 43.0 / Hnuclei + 10.0 *
                                    ab[IDX_CH2OHCH2OII] / 61.0 / Hnuclei + 6.0 *
                                    ab[IDX_GCOOCH3I] / 59.0 / Hnuclei + 2.0 *
                                    ab[IDX_H2SiOII] / 46.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCNOII] / 43.0 / Hnuclei + 8.0 *
                                    ab[IDX_CH3COOHII] / 60.0 / Hnuclei + 10.0 *
                                    ab[IDX_H5C2O2II] / 61.0 / Hnuclei + 4.0 *
                                    ab[IDX_HCOOHII] / 46.0 / Hnuclei + 10.0 *
                                    ab[IDX_CH3COOH2II] / 61.0 / Hnuclei + 6.0 *
                                    ab[IDX_CH3OCH3II] / 46.0 / Hnuclei + 6.0 *
                                    ab[IDX_GCH2OHCOI] / 59.0 / Hnuclei + 2.0 *
                                    ab[IDX_GCOOHI] / 45.0 / Hnuclei + 3.0 *
                                    ab[IDX_H3SiOII] / 47.0 / Hnuclei + 2.0 *
                                    ab[IDX_HSO2II] / 65.0 / Hnuclei + 7.0 *
                                    ab[IDX_CH3OCH4II] / 47.0 / Hnuclei + 1.0 *
                                    ab[IDX_HNCOII] / 43.0 / Hnuclei + 7.0 *
                                    ab[IDX_C2H5OH2II] / 47.0 / Hnuclei + 6.0 *
                                    ab[IDX_COOCH3I] / 59.0 / Hnuclei + 2.0 *
                                    ab[IDX_H2POII] / 49.0 / Hnuclei + 8.0 *
                                    ab[IDX_HCOOCH3II] / 60.0 / Hnuclei + 1.0 *
                                    ab[IDX_HN2OII] / 45.0 / Hnuclei + 8.0 *
                                    ab[IDX_CH2OHCHOII] / 60.0 / Hnuclei + 4.0 *
                                    ab[IDX_CH3CHOII] / 44.0 / Hnuclei + 8.0 *
                                    ab[IDX_GCH2OHCHOI] / 60.0 / Hnuclei + 4.0 *
                                    ab[IDX_GHCOOHI] / 46.0 / Hnuclei + 1.0 *
                                    ab[IDX_HOCSII] / 61.0 / Hnuclei + 3.0 *
                                    ab[IDX_GCH3COI] / 43.0 / Hnuclei + 8.0 *
                                    ab[IDX_GHCOOCH3I] / 60.0 / Hnuclei + 6.0 *
                                    ab[IDX_CH3COCH3II] / 58.0 / Hnuclei + 3.0 *
                                    ab[IDX_GCH2OHI] / 31.0 / Hnuclei + 4.0 *
                                    ab[IDX_GCH3OHI] / 32.0 / Hnuclei + 1.0 *
                                    ab[IDX_HONCI] / 43.0 / Hnuclei + 6.0 *
                                    ab[IDX_CH2OHCOI] / 59.0 / Hnuclei + 2.0 *
                                    ab[IDX_H2SiOI] / 46.0 / Hnuclei + 1.0 *
                                    ab[IDX_HC2OII] / 41.0 / Hnuclei + 3.0 *
                                    ab[IDX_GCH3OI] / 31.0 / Hnuclei + 2.0 *
                                    ab[IDX_GH2OI] / 18.0 / Hnuclei + 6.0 *
                                    ab[IDX_HCOOH2II] / 47.0 / Hnuclei + 1.0 *
                                    ab[IDX_HPOII] / 48.0 / Hnuclei + 8.0 *
                                    ab[IDX_CH2OHCHOI] / 60.0 / Hnuclei + 1.0 *
                                    ab[IDX_HC3OII] / 53.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCNOI] / 43.0 / Hnuclei + 2.0 *
                                    ab[IDX_CH2COII] / 42.0 / Hnuclei + 4.0 *
                                    ab[IDX_GCH3CHOI] / 44.0 / Hnuclei + 2.0 *
                                    ab[IDX_GH2COI] / 30.0 / Hnuclei + 1.0 *
                                    ab[IDX_HOCNI] / 43.0 / Hnuclei + 1.0 *
                                    ab[IDX_HPOI] / 48.0 / Hnuclei + 3.0 *
                                    ab[IDX_CH3COI] / 43.0 / Hnuclei + 7.0 *
                                    ab[IDX_CH3COCH4II] / 59.0 / Hnuclei + 4.0 *
                                    ab[IDX_CH3OHII] / 32.0 / Hnuclei + 2.0 *
                                    ab[IDX_COOHI] / 45.0 / Hnuclei + 8.0 *
                                    ab[IDX_HCOOCH3I] / 60.0 / Hnuclei + 1.0 *
                                    ab[IDX_HNCOI] / 43.0 / Hnuclei + 8.0 *
                                    ab[IDX_CH3COOHI] / 60.0 / Hnuclei + 6.0 *
                                    ab[IDX_CH3OCH3I] / 46.0 / Hnuclei + 3.0 *
                                    ab[IDX_CH3OI] / 31.0 / Hnuclei + 1.0 *
                                    ab[IDX_GHCOI] / 29.0 / Hnuclei + 3.0 *
                                    ab[IDX_CH2OHI] / 31.0 / Hnuclei + 5.0 *
                                    ab[IDX_CH3CHOHII] / 45.0 / Hnuclei + 3.0 *
                                    ab[IDX_CH3COII] / 43.0 / Hnuclei + 4.0 *
                                    ab[IDX_HCOOHI] / 46.0 / Hnuclei + 2.0 *
                                    ab[IDX_O2HI] / 33.0 / Hnuclei + 6.0 *
                                    ab[IDX_C2H5OHI] / 46.0 / Hnuclei + 1.0 *
                                    ab[IDX_HNOI] / 31.0 / Hnuclei + 5.0 *
                                    ab[IDX_CH3OH2II] / 33.0 / Hnuclei + 2.0 *
                                    ab[IDX_CH2COI] / 42.0 / Hnuclei + 6.0 *
                                    ab[IDX_CH3COCH3I] / 58.0 / Hnuclei + 1.0 *
                                    ab[IDX_GOHI] / 17.0 / Hnuclei + 2.0 *
                                    ab[IDX_HCO2II] / 45.0 / Hnuclei + 4.0 *
                                    ab[IDX_CH3CHOI] / 44.0 / Hnuclei + 2.0 *
                                    ab[IDX_O2HII] / 33.0 / Hnuclei + 1.0 *
                                    ab[IDX_HNOII] / 31.0 / Hnuclei + 1.0 *
                                    ab[IDX_OHM] / 17.0 / Hnuclei + 4.0 *
                                    ab[IDX_CH3OHI] / 32.0 / Hnuclei + 3.0 *
                                    ab[IDX_H3COII] / 31.0 / Hnuclei + 1.0 *
                                    ab[IDX_SiOHII] / 45.0 / Hnuclei + 1.0 *
                                    ab[IDX_OHII] / 17.0 / Hnuclei + 2.0 *
                                    ab[IDX_H2OII] / 18.0 / Hnuclei + 2.0 *
                                    ab[IDX_H2COII] / 30.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCOI] / 29.0 / Hnuclei + 2.0 *
                                    ab[IDX_H2COI] / 30.0 / Hnuclei + 1.0 *
                                    ab[IDX_OHI] / 17.0 / Hnuclei + 3.0 *
                                    ab[IDX_H3OII] / 19.0 / Hnuclei + 2.0 *
                                    ab[IDX_H2OI] / 18.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCOII] / 29.0 / Hnuclei;
    IJth(A, IDX_ELEM_He, IDX_ELEM_F) = 0.0;
    IJth(A, IDX_ELEM_He, IDX_ELEM_Cl) = 0.0;
    IJth(A, IDX_ELEM_He, IDX_ELEM_P) = 0.0;
    IJth(A, IDX_ELEM_He, IDX_ELEM_Fe) = 0.0;
    IJth(A, IDX_ELEM_He, IDX_ELEM_Mg) = 0.0;
    IJth(A, IDX_ELEM_He, IDX_ELEM_Na) = 0.0;
    IJth(A, IDX_ELEM_He, IDX_ELEM_Si) = 0.0;
    IJth(A, IDX_ELEM_He, IDX_ELEM_S) = 0.0;
    IJth(A, IDX_ELEM_He, IDX_ELEM_N) = 0.0;
    IJth(A, IDX_ELEM_He, IDX_ELEM_O) = 0.0;
    IJth(A, IDX_ELEM_He, IDX_ELEM_He) = 0.0 + 4.0 * ab[IDX_GHeI] / 4.0 / Hnuclei +
                                    4.0 * ab[IDX_HeHII] / 5.0 / Hnuclei + 4.0 *
                                    ab[IDX_HeII] / 4.0 / Hnuclei + 4.0 *
                                    ab[IDX_HeI] / 4.0 / Hnuclei;
    IJth(A, IDX_ELEM_He, IDX_ELEM_C) = 0.0;
    IJth(A, IDX_ELEM_He, IDX_ELEM_GRAIN) = 0.0;
    IJth(A, IDX_ELEM_He, IDX_ELEM_H) = 0.0 + 1.0 * ab[IDX_HeHII] / 5.0 / Hnuclei;
    IJth(A, IDX_ELEM_C, IDX_ELEM_F) = 0.0 + 19.0 * ab[IDX_CFII] / 31.0 / Hnuclei;
    IJth(A, IDX_ELEM_C, IDX_ELEM_Cl) = 0.0 + 35.0 * ab[IDX_GCClI] / 47.0 / Hnuclei
                                    + 35.0 * ab[IDX_H2CClII] / 49.0 / Hnuclei +
                                    35.0 * ab[IDX_CClII] / 47.0 / Hnuclei + 35.0
                                    * ab[IDX_CClI] / 47.0 / Hnuclei;
    IJth(A, IDX_ELEM_C, IDX_ELEM_P) = 0.0 + 62.0 * ab[IDX_GHC2PI] / 56.0 / Hnuclei
                                    + 124.0 * ab[IDX_GC4PI] / 79.0 / Hnuclei +
                                    31.0 * ab[IDX_GCH2PHI] / 46.0 / Hnuclei +
                                    31.0 * ab[IDX_GHCPI] / 44.0 / Hnuclei + 93.0
                                    * ab[IDX_GC3PI] / 67.0 / Hnuclei + 62.0 *
                                    ab[IDX_GCCPI] / 55.0 / Hnuclei + 124.0 *
                                    ab[IDX_C4PII] / 79.0 / Hnuclei + 31.0 *
                                    ab[IDX_GCPI] / 43.0 / Hnuclei + 62.0 *
                                    ab[IDX_PC2H4II] / 59.0 / Hnuclei + 62.0 *
                                    ab[IDX_PC2H3II] / 58.0 / Hnuclei + 31.0 *
                                    ab[IDX_PCH3II] / 46.0 / Hnuclei + 93.0 *
                                    ab[IDX_PC3HII] / 68.0 / Hnuclei + 31.0 *
                                    ab[IDX_PCH4II] / 47.0 / Hnuclei + 62.0 *
                                    ab[IDX_CCPII] / 55.0 / Hnuclei + 62.0 *
                                    ab[IDX_PC2H2II] / 57.0 / Hnuclei + 124.0 *
                                    ab[IDX_PC4HII] / 80.0 / Hnuclei + 31.0 *
                                    ab[IDX_CPII] / 43.0 / Hnuclei + 62.0 *
                                    ab[IDX_HC2PII] / 56.0 / Hnuclei + 31.0 *
                                    ab[IDX_PCH2II] / 45.0 / Hnuclei + 31.0 *
                                    ab[IDX_HCPII] / 44.0 / Hnuclei + 31.0 *
                                    ab[IDX_CH2PHI] / 46.0 / Hnuclei + 124.0 *
                                    ab[IDX_C4PI] / 79.0 / Hnuclei + 31.0 *
                                    ab[IDX_HCPI] / 44.0 / Hnuclei + 93.0 *
                                    ab[IDX_C3PI] / 67.0 / Hnuclei + 62.0 *
                                    ab[IDX_HC2PI] / 56.0 / Hnuclei + 62.0 *
                                    ab[IDX_CCPI] / 55.0 / Hnuclei + 31.0 *
                                    ab[IDX_CPI] / 43.0 / Hnuclei;
    IJth(A, IDX_ELEM_C, IDX_ELEM_Fe) = 0.0;
    IJth(A, IDX_ELEM_C, IDX_ELEM_Mg) = 0.0;
    IJth(A, IDX_ELEM_C, IDX_ELEM_Na) = 0.0;
    IJth(A, IDX_ELEM_C, IDX_ELEM_Si) = 0.0 + 84.0 * ab[IDX_GSiC3HI] / 65.0 /
                                    Hnuclei + 28.0 * ab[IDX_GSiCH3I] / 43.0 /
                                    Hnuclei + 56.0 * ab[IDX_GSiC2HI] / 53.0 /
                                    Hnuclei + 56.0 * ab[IDX_GSiC2H2I] / 54.0 /
                                    Hnuclei + 112.0 * ab[IDX_GSiC4I] / 76.0 /
                                    Hnuclei + 28.0 * ab[IDX_GSiNCI] / 54.0 /
                                    Hnuclei + 28.0 * ab[IDX_GHCSiI] / 41.0 /
                                    Hnuclei + 84.0 * ab[IDX_GSiC3I] / 64.0 /
                                    Hnuclei + 28.0 * ab[IDX_GSiCH2I] / 42.0 /
                                    Hnuclei + 28.0 * ab[IDX_GSiCI] / 40.0 /
                                    Hnuclei + 56.0 * ab[IDX_GSiC2I] / 52.0 /
                                    Hnuclei + 84.0 * ab[IDX_SiC3H2II] / 66.0 /
                                    Hnuclei + 112.0 * ab[IDX_SiC4II] / 76.0 /
                                    Hnuclei + 112.0 * ab[IDX_SiC4HII] / 77.0 /
                                    Hnuclei + 28.0 * ab[IDX_SiNCHII] / 55.0 /
                                    Hnuclei + 28.0 * ab[IDX_SiCH4II] / 44.0 /
                                    Hnuclei + 56.0 * ab[IDX_SiC2H3II] / 55.0 /
                                    Hnuclei + 84.0 * ab[IDX_SiC3HI] / 65.0 /
                                    Hnuclei + 28.0 * ab[IDX_SiCH3II] / 43.0 /
                                    Hnuclei + 112.0 * ab[IDX_SiC4I] / 76.0 /
                                    Hnuclei + 28.0 * ab[IDX_SiNCI] / 54.0 /
                                    Hnuclei + 28.0 * ab[IDX_SiNCII] / 54.0 /
                                    Hnuclei + 56.0 * ab[IDX_SiC2H2I] / 54.0 /
                                    Hnuclei + 84.0 * ab[IDX_SiC3II] / 64.0 /
                                    Hnuclei + 84.0 * ab[IDX_SiC3HII] / 65.0 /
                                    Hnuclei + 28.0 * ab[IDX_SiCH3I] / 43.0 /
                                    Hnuclei + 56.0 * ab[IDX_SiC2II] / 52.0 /
                                    Hnuclei + 56.0 * ab[IDX_SiC2H2II] / 54.0 /
                                    Hnuclei + 56.0 * ab[IDX_SiC2HI] / 53.0 /
                                    Hnuclei + 84.0 * ab[IDX_SiC3I] / 64.0 /
                                    Hnuclei + 28.0 * ab[IDX_SiCII] / 40.0 /
                                    Hnuclei + 28.0 * ab[IDX_HCSiI] / 41.0 /
                                    Hnuclei + 28.0 * ab[IDX_SiCH2I] / 42.0 /
                                    Hnuclei + 28.0 * ab[IDX_HCSiII] / 41.0 /
                                    Hnuclei + 56.0 * ab[IDX_SiC2I] / 52.0 /
                                    Hnuclei + 56.0 * ab[IDX_SiC2HII] / 53.0 /
                                    Hnuclei + 28.0 * ab[IDX_SiCI] / 40.0 /
                                    Hnuclei + 28.0 * ab[IDX_SiCH2II] / 42.0 /
                                    Hnuclei;
    IJth(A, IDX_ELEM_C, IDX_ELEM_S) = 0.0 + 128.0 * ab[IDX_GC4SI] / 80.0 / Hnuclei
                                    + 96.0 * ab[IDX_GC3SI] / 68.0 / Hnuclei +
                                    64.0 * ab[IDX_GC2SI] / 56.0 / Hnuclei + 32.0
                                    * ab[IDX_GH2CSI] / 46.0 / Hnuclei + 32.0 *
                                    ab[IDX_GHCSI] / 45.0 / Hnuclei + 32.0 *
                                    ab[IDX_GOCSI] / 60.0 / Hnuclei + 64.0 *
                                    ab[IDX_CH3CSII] / 59.0 / Hnuclei + 64.0 *
                                    ab[IDX_C2SII] / 56.0 / Hnuclei + 96.0 *
                                    ab[IDX_C3SII] / 68.0 / Hnuclei + 32.0 *
                                    ab[IDX_GCSI] / 44.0 / Hnuclei + 32.0 *
                                    ab[IDX_HOCSII] / 61.0 / Hnuclei + 128.0 *
                                    ab[IDX_HC4SII] / 81.0 / Hnuclei + 32.0 *
                                    ab[IDX_H3CSII] / 47.0 / Hnuclei + 96.0 *
                                    ab[IDX_HC3SII] / 69.0 / Hnuclei + 32.0 *
                                    ab[IDX_H2CSII] / 46.0 / Hnuclei + 32.0 *
                                    ab[IDX_H2CSI] / 46.0 / Hnuclei + 96.0 *
                                    ab[IDX_C3SI] / 68.0 / Hnuclei + 32.0 *
                                    ab[IDX_OCSII] / 60.0 / Hnuclei + 32.0 *
                                    ab[IDX_HCSI] / 45.0 / Hnuclei + 32.0 *
                                    ab[IDX_HCSII] / 45.0 / Hnuclei + 32.0 *
                                    ab[IDX_CSII] / 44.0 / Hnuclei + 128.0 *
                                    ab[IDX_C4SII] / 80.0 / Hnuclei + 32.0 *
                                    ab[IDX_OCSI] / 60.0 / Hnuclei + 64.0 *
                                    ab[IDX_HC2SII] / 57.0 / Hnuclei + 128.0 *
                                    ab[IDX_C4SI] / 80.0 / Hnuclei + 32.0 *
                                    ab[IDX_CSI] / 44.0 / Hnuclei + 64.0 *
                                    ab[IDX_C2SI] / 56.0 / Hnuclei;
    IJth(A, IDX_ELEM_C, IDX_ELEM_N) = 0.0 + 56.0 * ab[IDX_GCH3C3NI] / 65.0 /
                                    Hnuclei + 84.0 * ab[IDX_GCH3C5NI] / 89.0 /
                                    Hnuclei + 112.0 * ab[IDX_GCH3C7NI] / 113.0 /
                                    Hnuclei + 28.0 * ab[IDX_GNH2CNI] / 42.0 /
                                    Hnuclei + 56.0 * ab[IDX_GNCCNI] / 52.0 /
                                    Hnuclei + 42.0 * ab[IDX_C2H4CNI] / 54.0 /
                                    Hnuclei + 56.0 * ab[IDX_GC4NI] / 62.0 /
                                    Hnuclei + 14.0 * ab[IDX_GHCNOI] / 43.0 /
                                    Hnuclei + 42.0 * ab[IDX_GHNC3I] / 51.0 /
                                    Hnuclei + 14.0 * ab[IDX_GHNCOI] / 43.0 /
                                    Hnuclei + 14.0 * ab[IDX_GHOCNI] / 43.0 /
                                    Hnuclei + 14.0 * ab[IDX_GHONCI] / 43.0 /
                                    Hnuclei + 14.0 * ab[IDX_GSiNCI] / 54.0 /
                                    Hnuclei + 84.0 * ab[IDX_NCCNCH3II] / 67.0 /
                                    Hnuclei + 42.0 * ab[IDX_GC2H4CNI] / 54.0 /
                                    Hnuclei + 28.0 * ab[IDX_GCH2CNI] / 40.0 /
                                    Hnuclei + 28.0 * ab[IDX_GCH3CNI] / 41.0 /
                                    Hnuclei + 14.0 * ab[IDX_GH2CNI] / 28.0 /
                                    Hnuclei + 70.0 * ab[IDX_GHC5NI] / 75.0 /
                                    Hnuclei + 98.0 * ab[IDX_GHC7NI] / 99.0 /
                                    Hnuclei + 126.0 * ab[IDX_GHC9NI] / 123.0 /
                                    Hnuclei + 28.0 * ab[IDX_GHCCNI] / 39.0 /
                                    Hnuclei + 28.0 * ab[IDX_HCCNI] / 39.0 /
                                    Hnuclei + 56.0 * ab[IDX_CH3C3NII] / 65.0 /
                                    Hnuclei + 42.0 * ab[IDX_GCH2CHCNI] / 53.0 /
                                    Hnuclei + 42.0 * ab[IDX_GHC3NI] / 51.0 /
                                    Hnuclei + 70.0 * ab[IDX_C5NII] / 74.0 /
                                    Hnuclei + 42.0 * ab[IDX_CH2CHCNII] / 53.0 /
                                    Hnuclei + 42.0 * ab[IDX_GC2H5CNI] / 55.0 /
                                    Hnuclei + 126.0 * ab[IDX_GC9NI] / 122.0 /
                                    Hnuclei + 14.0 * ab[IDX_GOCNI] / 42.0 /
                                    Hnuclei + 14.0 * ab[IDX_H2CNOII] / 44.0 /
                                    Hnuclei + 14.0 * ab[IDX_H2NCOII] / 44.0 /
                                    Hnuclei + 14.0 * ab[IDX_H2OCNII] / 44.0 /
                                    Hnuclei + 98.0 * ab[IDX_C7NII] / 98.0 /
                                    Hnuclei + 126.0 * ab[IDX_C9NII] / 122.0 /
                                    Hnuclei + 14.0 * ab[IDX_CH3NHII] / 30.0 /
                                    Hnuclei + 28.0 * ab[IDX_GC2NI] / 38.0 /
                                    Hnuclei + 14.0 * ab[IDX_GCH2NHI] / 29.0 /
                                    Hnuclei + 14.0 * ab[IDX_GCNOI] / 42.0 /
                                    Hnuclei + 14.0 * ab[IDX_GHNCI] / 27.0 /
                                    Hnuclei + 56.0 * ab[IDX_H2C4NII] / 64.0 /
                                    Hnuclei + 28.0 * ab[IDX_NH2CNHII] / 43.0 /
                                    Hnuclei + 56.0 * ab[IDX_HC4NII] / 63.0 /
                                    Hnuclei + 14.0 * ab[IDX_HCNOHII] / 44.0 /
                                    Hnuclei + 14.0 * ab[IDX_HNCOHII] / 44.0 /
                                    Hnuclei + 14.0 * ab[IDX_HOCNII] / 43.0 /
                                    Hnuclei + 70.0 * ab[IDX_GC5NI] / 74.0 /
                                    Hnuclei + 98.0 * ab[IDX_H2C7NII] / 100.0 /
                                    Hnuclei + 126.0 * ab[IDX_H2C9NII] / 124.0 /
                                    Hnuclei + 14.0 * ab[IDX_HONCII] / 43.0 /
                                    Hnuclei + 42.0 * ab[IDX_C2H5CNHII] / 56.0 /
                                    Hnuclei + 14.0 * ab[IDX_CH2NH2II] / 30.0 /
                                    Hnuclei + 84.0 * ab[IDX_CH3C5NHII] / 90.0 /
                                    Hnuclei + 112.0 * ab[IDX_CH3C7NHII] / 114.0
                                    / Hnuclei + 42.0 * ab[IDX_GC3NI] / 50.0 /
                                    Hnuclei + 98.0 * ab[IDX_GC7NI] / 98.0 /
                                    Hnuclei + 14.0 * ab[IDX_GHCNI] / 27.0 /
                                    Hnuclei + 14.0 * ab[IDX_H2CNI] / 28.0 /
                                    Hnuclei + 70.0 * ab[IDX_H3C5NII] / 77.0 /
                                    Hnuclei + 14.0 * ab[IDX_HCNOII] / 43.0 /
                                    Hnuclei + 126.0 * ab[IDX_H3C9NII] / 125.0 /
                                    Hnuclei + 56.0 * ab[IDX_C4NI] / 62.0 /
                                    Hnuclei + 98.0 * ab[IDX_H3C7NII] / 101.0 /
                                    Hnuclei + 28.0 * ab[IDX_NH2CNI] / 42.0 /
                                    Hnuclei + 56.0 * ab[IDX_CH3C3NHII] / 66.0 /
                                    Hnuclei + 98.0 * ab[IDX_HC7NII] / 99.0 /
                                    Hnuclei + 126.0 * ab[IDX_HC9NII] / 123.0 /
                                    Hnuclei + 14.0 * ab[IDX_HNCOII] / 43.0 /
                                    Hnuclei + 42.0 * ab[IDX_C2H5CNI] / 55.0 /
                                    Hnuclei + 42.0 * ab[IDX_CH2CHCNHII] / 54.0 /
                                    Hnuclei + 14.0 * ab[IDX_SiNCHII] / 55.0 /
                                    Hnuclei + 14.0 * ab[IDX_OCNII] / 42.0 /
                                    Hnuclei + 56.0 * ab[IDX_C2N2II] / 52.0 /
                                    Hnuclei + 28.0 * ab[IDX_CH3CNII] / 41.0 /
                                    Hnuclei + 70.0 * ab[IDX_HC5NHII] / 76.0 /
                                    Hnuclei + 42.0 * ab[IDX_C3NII] / 50.0 /
                                    Hnuclei + 28.0 * ab[IDX_CH2CNII] / 40.0 /
                                    Hnuclei + 70.0 * ab[IDX_HC5NII] / 75.0 /
                                    Hnuclei + 14.0 * ab[IDX_HONCI] / 43.0 /
                                    Hnuclei + 84.0 * ab[IDX_CH3C5NI] / 89.0 /
                                    Hnuclei + 112.0 * ab[IDX_CH3C7NI] / 113.0 /
                                    Hnuclei + 14.0 * ab[IDX_H2NCII] / 28.0 /
                                    Hnuclei + 14.0 * ab[IDX_SiNCI] / 54.0 /
                                    Hnuclei + 14.0 * ab[IDX_SiNCII] / 54.0 /
                                    Hnuclei + 14.0 * ab[IDX_CNOI] / 42.0 /
                                    Hnuclei + 28.0 * ab[IDX_C2NHII] / 39.0 /
                                    Hnuclei + 14.0 * ab[IDX_HCNOI] / 43.0 /
                                    Hnuclei + 56.0 * ab[IDX_C4NII] / 62.0 /
                                    Hnuclei + 56.0 * ab[IDX_CH3C3NI] / 65.0 /
                                    Hnuclei + 14.0 * ab[IDX_HOCNI] / 43.0 /
                                    Hnuclei + 126.0 * ab[IDX_HC9NI] / 123.0 /
                                    Hnuclei + 56.0 * ab[IDX_NCCNHII] / 53.0 /
                                    Hnuclei + 28.0 * ab[IDX_CH2CNI] / 40.0 /
                                    Hnuclei + 14.0 * ab[IDX_HNCOI] / 43.0 /
                                    Hnuclei + 126.0 * ab[IDX_C9NI] / 122.0 /
                                    Hnuclei + 14.0 * ab[IDX_CH2NHI] / 29.0 /
                                    Hnuclei + 42.0 * ab[IDX_CH2CHCNI] / 53.0 /
                                    Hnuclei + 28.0 * ab[IDX_CH3CNHII] / 42.0 /
                                    Hnuclei + 98.0 * ab[IDX_HC7NI] / 99.0 /
                                    Hnuclei + 98.0 * ab[IDX_C7NI] / 98.0 /
                                    Hnuclei + 28.0 * ab[IDX_C2NII] / 38.0 /
                                    Hnuclei + 42.0 * ab[IDX_HNC3I] / 51.0 /
                                    Hnuclei + 14.0 * ab[IDX_OCNI] / 42.0 /
                                    Hnuclei + 42.0 * ab[IDX_HC3NII] / 51.0 /
                                    Hnuclei + 70.0 * ab[IDX_HC5NI] / 75.0 /
                                    Hnuclei + 42.0 * ab[IDX_HC3NHII] / 52.0 /
                                    Hnuclei + 28.0 * ab[IDX_CH3CNI] / 41.0 /
                                    Hnuclei + 56.0 * ab[IDX_NCCNI] / 52.0 /
                                    Hnuclei + 14.0 * ab[IDX_GCNI] / 26.0 /
                                    Hnuclei + 14.0 * ab[IDX_CNII] / 26.0 /
                                    Hnuclei + 14.0 * ab[IDX_HCNII] / 27.0 /
                                    Hnuclei + 28.0 * ab[IDX_CNCII] / 38.0 /
                                    Hnuclei + 70.0 * ab[IDX_C5NM] / 74.0 /
                                    Hnuclei + 42.0 * ab[IDX_HC3NI] / 51.0 /
                                    Hnuclei + 28.0 * ab[IDX_C2NI] / 38.0 /
                                    Hnuclei + 70.0 * ab[IDX_C5NI] / 74.0 /
                                    Hnuclei + 42.0 * ab[IDX_C3NM] / 50.0 /
                                    Hnuclei + 42.0 * ab[IDX_C3NI] / 50.0 /
                                    Hnuclei + 14.0 * ab[IDX_CNM] / 26.0 /
                                    Hnuclei + 14.0 * ab[IDX_HCNHII] / 28.0 /
                                    Hnuclei + 14.0 * ab[IDX_HNCI] / 27.0 /
                                    Hnuclei + 14.0 * ab[IDX_HCNI] / 27.0 /
                                    Hnuclei + 14.0 * ab[IDX_CNI] / 26.0 /
                                    Hnuclei;
    IJth(A, IDX_ELEM_C, IDX_ELEM_O) = 0.0 + 64.0 * ab[IDX_GCH3COOHI] / 60.0 /
                                    Hnuclei + 16.0 * ab[IDX_GHCNOI] / 43.0 /
                                    Hnuclei + 16.0 * ab[IDX_GHNCOI] / 43.0 /
                                    Hnuclei + 16.0 * ab[IDX_GHOCNI] / 43.0 /
                                    Hnuclei + 16.0 * ab[IDX_GHONCI] / 43.0 /
                                    Hnuclei + 32.0 * ab[IDX_HC2OI] / 41.0 /
                                    Hnuclei + 48.0 * ab[IDX_GC3OI] / 52.0 /
                                    Hnuclei + 32.0 * ab[IDX_GHC2OI] / 41.0 /
                                    Hnuclei + 48.0 * ab[IDX_C3H2OII] / 54.0 /
                                    Hnuclei + 48.0 * ab[IDX_GCH3COCH3I] / 58.0 /
                                    Hnuclei + 32.0 * ab[IDX_GCH3OCH3I] / 46.0 /
                                    Hnuclei + 48.0 * ab[IDX_H3C3OII] / 55.0 /
                                    Hnuclei + 16.0 * ab[IDX_GOCSI] / 60.0 /
                                    Hnuclei + 64.0 * ab[IDX_COOCH3II] / 59.0 /
                                    Hnuclei + 32.0 * ab[IDX_GC2H5OHI] / 46.0 /
                                    Hnuclei + 32.0 * ab[IDX_GC2OI] / 40.0 /
                                    Hnuclei + 32.0 * ab[IDX_GCO2I] / 44.0 /
                                    Hnuclei + 16.0 * ab[IDX_GOCNI] / 42.0 /
                                    Hnuclei + 16.0 * ab[IDX_H2CNOII] / 44.0 /
                                    Hnuclei + 16.0 * ab[IDX_H2NCOII] / 44.0 /
                                    Hnuclei + 16.0 * ab[IDX_H2OCNII] / 44.0 /
                                    Hnuclei + 16.0 * ab[IDX_GCNOI] / 42.0 /
                                    Hnuclei + 16.0 * ab[IDX_HOCII] / 29.0 /
                                    Hnuclei + 32.0 * ab[IDX_C2H5OHII] / 46.0 /
                                    Hnuclei + 32.0 * ab[IDX_C2OII] / 40.0 /
                                    Hnuclei + 64.0 * ab[IDX_CH2OHCOII] / 59.0 /
                                    Hnuclei + 32.0 * ab[IDX_GCH2COI] / 42.0 /
                                    Hnuclei + 16.0 * ab[IDX_HCNOHII] / 44.0 /
                                    Hnuclei + 16.0 * ab[IDX_HNCOHII] / 44.0 /
                                    Hnuclei + 16.0 * ab[IDX_HOCNII] / 43.0 /
                                    Hnuclei + 16.0 * ab[IDX_HONCII] / 43.0 /
                                    Hnuclei + 64.0 * ab[IDX_CH2OHCH2OII] / 61.0
                                    / Hnuclei + 64.0 * ab[IDX_GCOOCH3I] / 59.0 /
                                    Hnuclei + 16.0 * ab[IDX_HCNOII] / 43.0 /
                                    Hnuclei + 64.0 * ab[IDX_CH3COOHII] / 60.0 /
                                    Hnuclei + 64.0 * ab[IDX_H5C2O2II] / 61.0 /
                                    Hnuclei + 32.0 * ab[IDX_HCOOHII] / 46.0 /
                                    Hnuclei + 64.0 * ab[IDX_CH3COOH2II] / 61.0 /
                                    Hnuclei + 32.0 * ab[IDX_CH3OCH3II] / 46.0 /
                                    Hnuclei + 64.0 * ab[IDX_GCH2OHCOI] / 59.0 /
                                    Hnuclei + 32.0 * ab[IDX_GCOOHI] / 45.0 /
                                    Hnuclei + 32.0 * ab[IDX_CH3OCH4II] / 47.0 /
                                    Hnuclei + 16.0 * ab[IDX_HNCOII] / 43.0 /
                                    Hnuclei + 32.0 * ab[IDX_C2H5OH2II] / 47.0 /
                                    Hnuclei + 64.0 * ab[IDX_COOCH3I] / 59.0 /
                                    Hnuclei + 64.0 * ab[IDX_HCOOCH3II] / 60.0 /
                                    Hnuclei + 48.0 * ab[IDX_C3OII] / 52.0 /
                                    Hnuclei + 64.0 * ab[IDX_CH2OHCHOII] / 60.0 /
                                    Hnuclei + 32.0 * ab[IDX_CH3CHOII] / 44.0 /
                                    Hnuclei + 64.0 * ab[IDX_GCH2OHCHOI] / 60.0 /
                                    Hnuclei + 32.0 * ab[IDX_GHCOOHI] / 46.0 /
                                    Hnuclei + 16.0 * ab[IDX_HOCSII] / 61.0 /
                                    Hnuclei + 16.0 * ab[IDX_OCNII] / 42.0 /
                                    Hnuclei + 32.0 * ab[IDX_GCH3COI] / 43.0 /
                                    Hnuclei + 64.0 * ab[IDX_GHCOOCH3I] / 60.0 /
                                    Hnuclei + 48.0 * ab[IDX_CH3COCH3II] / 58.0 /
                                    Hnuclei + 16.0 * ab[IDX_GCH2OHI] / 31.0 /
                                    Hnuclei + 16.0 * ab[IDX_GCH3OHI] / 32.0 /
                                    Hnuclei + 16.0 * ab[IDX_HONCI] / 43.0 /
                                    Hnuclei + 64.0 * ab[IDX_CH2OHCOI] / 59.0 /
                                    Hnuclei + 32.0 * ab[IDX_HC2OII] / 41.0 /
                                    Hnuclei + 16.0 * ab[IDX_CNOI] / 42.0 /
                                    Hnuclei + 16.0 * ab[IDX_GCH3OI] / 31.0 /
                                    Hnuclei + 32.0 * ab[IDX_HCOOH2II] / 47.0 /
                                    Hnuclei + 64.0 * ab[IDX_CH2OHCHOI] / 60.0 /
                                    Hnuclei + 48.0 * ab[IDX_HC3OII] / 53.0 /
                                    Hnuclei + 16.0 * ab[IDX_HCNOI] / 43.0 /
                                    Hnuclei + 32.0 * ab[IDX_CH2COII] / 42.0 /
                                    Hnuclei + 32.0 * ab[IDX_GCH3CHOI] / 44.0 /
                                    Hnuclei + 16.0 * ab[IDX_GH2COI] / 30.0 /
                                    Hnuclei + 16.0 * ab[IDX_HOCNI] / 43.0 /
                                    Hnuclei + 32.0 * ab[IDX_CH3COI] / 43.0 /
                                    Hnuclei + 48.0 * ab[IDX_CH3COCH4II] / 59.0 /
                                    Hnuclei + 16.0 * ab[IDX_CH3OHII] / 32.0 /
                                    Hnuclei + 32.0 * ab[IDX_COOHI] / 45.0 /
                                    Hnuclei + 64.0 * ab[IDX_HCOOCH3I] / 60.0 /
                                    Hnuclei + 16.0 * ab[IDX_HNCOI] / 43.0 /
                                    Hnuclei + 64.0 * ab[IDX_CH3COOHI] / 60.0 /
                                    Hnuclei + 32.0 * ab[IDX_CH3OCH3I] / 46.0 /
                                    Hnuclei + 16.0 * ab[IDX_CH3OI] / 31.0 /
                                    Hnuclei + 16.0 * ab[IDX_GHCOI] / 29.0 /
                                    Hnuclei + 16.0 * ab[IDX_OCSII] / 60.0 /
                                    Hnuclei + 32.0 * ab[IDX_C2OI] / 40.0 /
                                    Hnuclei + 16.0 * ab[IDX_CH2OHI] / 31.0 /
                                    Hnuclei + 48.0 * ab[IDX_C3OI] / 52.0 /
                                    Hnuclei + 32.0 * ab[IDX_CH3CHOHII] / 45.0 /
                                    Hnuclei + 32.0 * ab[IDX_CH3COII] / 43.0 /
                                    Hnuclei + 32.0 * ab[IDX_HCOOHI] / 46.0 /
                                    Hnuclei + 32.0 * ab[IDX_C2H5OHI] / 46.0 /
                                    Hnuclei + 16.0 * ab[IDX_CH3OH2II] / 33.0 /
                                    Hnuclei + 32.0 * ab[IDX_CH2COI] / 42.0 /
                                    Hnuclei + 16.0 * ab[IDX_OCNI] / 42.0 /
                                    Hnuclei + 48.0 * ab[IDX_CH3COCH3I] / 58.0 /
                                    Hnuclei + 32.0 * ab[IDX_CO2II] / 44.0 /
                                    Hnuclei + 16.0 * ab[IDX_GCOI] / 28.0 /
                                    Hnuclei + 32.0 * ab[IDX_HCO2II] / 45.0 /
                                    Hnuclei + 32.0 * ab[IDX_CH3CHOI] / 44.0 /
                                    Hnuclei + 16.0 * ab[IDX_OCSI] / 60.0 /
                                    Hnuclei + 16.0 * ab[IDX_COII] / 28.0 /
                                    Hnuclei + 16.0 * ab[IDX_CH3OHI] / 32.0 /
                                    Hnuclei + 16.0 * ab[IDX_H3COII] / 31.0 /
                                    Hnuclei + 32.0 * ab[IDX_CO2I] / 44.0 /
                                    Hnuclei + 16.0 * ab[IDX_H2COII] / 30.0 /
                                    Hnuclei + 16.0 * ab[IDX_HCOI] / 29.0 /
                                    Hnuclei + 16.0 * ab[IDX_H2COI] / 30.0 /
                                    Hnuclei + 16.0 * ab[IDX_HCOII] / 29.0 /
                                    Hnuclei + 16.0 * ab[IDX_COI] / 28.0 /
                                    Hnuclei;
    IJth(A, IDX_ELEM_C, IDX_ELEM_He) = 0.0;
    IJth(A, IDX_ELEM_C, IDX_ELEM_C) = 0.0 + 192.0 * ab[IDX_GC4H6I] / 54.0 /
                                    Hnuclei + 192.0 * ab[IDX_GC4SI] / 80.0 /
                                    Hnuclei + 12.0 * ab[IDX_GCClI] / 47.0 /
                                    Hnuclei + 192.0 * ab[IDX_GCH3C3NI] / 65.0 /
                                    Hnuclei + 300.0 * ab[IDX_GCH3C4HI] / 64.0 /
                                    Hnuclei + 432.0 * ab[IDX_GCH3C5NI] / 89.0 /
                                    Hnuclei + 588.0 * ab[IDX_GCH3C6HI] / 88.0 /
                                    Hnuclei + 768.0 * ab[IDX_GCH3C7NI] / 113.0 /
                                    Hnuclei + 48.0 * ab[IDX_GHC2PI] / 56.0 /
                                    Hnuclei + 12.0 * ab[IDX_GNH2CNI] / 42.0 /
                                    Hnuclei + 108.0 * ab[IDX_GSiC3HI] / 65.0 /
                                    Hnuclei + 12.0 * ab[IDX_GSiCH3I] / 43.0 /
                                    Hnuclei + 48.0 * ab[IDX_GNCCNI] / 52.0 /
                                    Hnuclei + 48.0 * ab[IDX_GSiC2HI] / 53.0 /
                                    Hnuclei + 108.0 * ab[IDX_C2H4CNI] / 54.0 /
                                    Hnuclei + 108.0 * ab[IDX_GC3SI] / 68.0 /
                                    Hnuclei + 192.0 * ab[IDX_GC4NI] / 62.0 /
                                    Hnuclei + 192.0 * ab[IDX_GC4PI] / 79.0 /
                                    Hnuclei + 12.0 * ab[IDX_GCH2PHI] / 46.0 /
                                    Hnuclei + 108.0 * ab[IDX_GCH3CHCH2I] / 42.0
                                    / Hnuclei + 48.0 * ab[IDX_GCH3COOHI] / 60.0
                                    / Hnuclei + 12.0 * ab[IDX_GHCNOI] / 43.0 /
                                    Hnuclei + 12.0 * ab[IDX_GHCPI] / 44.0 /
                                    Hnuclei + 108.0 * ab[IDX_GHNC3I] / 51.0 /
                                    Hnuclei + 12.0 * ab[IDX_GHNCOI] / 43.0 /
                                    Hnuclei + 12.0 * ab[IDX_GHOCNI] / 43.0 /
                                    Hnuclei + 12.0 * ab[IDX_GHONCI] / 43.0 /
                                    Hnuclei + 48.0 * ab[IDX_GSiC2H2I] / 54.0 /
                                    Hnuclei + 192.0 * ab[IDX_GSiC4I] / 76.0 /
                                    Hnuclei + 12.0 * ab[IDX_GSiNCI] / 54.0 /
                                    Hnuclei + 48.0 * ab[IDX_HC2OI] / 41.0 /
                                    Hnuclei + 108.0 * ab[IDX_GC3PI] / 67.0 /
                                    Hnuclei + 432.0 * ab[IDX_GC6H6I] / 78.0 /
                                    Hnuclei + 12.0 * ab[IDX_GHCSiI] / 41.0 /
                                    Hnuclei + 108.0 * ab[IDX_NCCNCH3II] / 67.0 /
                                    Hnuclei + 108.0 * ab[IDX_GC2H4CNI] / 54.0 /
                                    Hnuclei + 48.0 * ab[IDX_GC2SI] / 56.0 /
                                    Hnuclei + 108.0 * ab[IDX_GC3OI] / 52.0 /
                                    Hnuclei + 192.0 * ab[IDX_GC4H3I] / 51.0 /
                                    Hnuclei + 588.0 * ab[IDX_GC7H2I] / 86.0 /
                                    Hnuclei + 768.0 * ab[IDX_GC8H2I] / 98.0 /
                                    Hnuclei + 972.0 * ab[IDX_GC9H2I] / 110.0 /
                                    Hnuclei + 48.0 * ab[IDX_GCH2CNI] / 40.0 /
                                    Hnuclei + 48.0 * ab[IDX_GCH3CNI] / 41.0 /
                                    Hnuclei + 12.0 * ab[IDX_GH2CNI] / 28.0 /
                                    Hnuclei + 12.0 * ab[IDX_GH2CSI] / 46.0 /
                                    Hnuclei + 48.0 * ab[IDX_GHC2OI] / 41.0 /
                                    Hnuclei + 300.0 * ab[IDX_GHC5NI] / 75.0 /
                                    Hnuclei + 588.0 * ab[IDX_GHC7NI] / 99.0 /
                                    Hnuclei + 972.0 * ab[IDX_GHC9NI] / 123.0 /
                                    Hnuclei + 48.0 * ab[IDX_GHCCNI] / 39.0 /
                                    Hnuclei + 12.0 * ab[IDX_GHCSI] / 45.0 /
                                    Hnuclei + 108.0 * ab[IDX_GSiC3I] / 64.0 /
                                    Hnuclei + 48.0 * ab[IDX_HCCNI] / 39.0 /
                                    Hnuclei + 108.0 * ab[IDX_C3H2OII] / 54.0 /
                                    Hnuclei + 12.0 * ab[IDX_CFII] / 31.0 /
                                    Hnuclei + 48.0 * ab[IDX_GC2H6I] / 30.0 /
                                    Hnuclei + 108.0 * ab[IDX_GC3H2I] / 38.0 /
                                    Hnuclei + 300.0 * ab[IDX_GC5H2I] / 62.0 /
                                    Hnuclei + 48.0 * ab[IDX_GCCPI] / 55.0 /
                                    Hnuclei + 192.0 * ab[IDX_GCH2CHCCHI] / 52.0
                                    / Hnuclei + 108.0 * ab[IDX_GCH3COCH3I] /
                                    58.0 / Hnuclei + 48.0 * ab[IDX_GCH3OCH3I] /
                                    46.0 / Hnuclei + 12.0 * ab[IDX_GSiCH2I] /
                                    42.0 / Hnuclei + 108.0 * ab[IDX_H3C3OII] /
                                    55.0 / Hnuclei + 768.0 * ab[IDX_C8H5II] /
                                    101.0 / Hnuclei + 972.0 * ab[IDX_C9H5II] /
                                    113.0 / Hnuclei + 192.0 * ab[IDX_CH3C3NII] /
                                    65.0 / Hnuclei + 1200.0 * ab[IDX_GC10H2I] /
                                    122.0 / Hnuclei + 192.0 * ab[IDX_GC4H2I] /
                                    50.0 / Hnuclei + 108.0 * ab[IDX_GCH2CHCNI] /
                                    53.0 / Hnuclei + 108.0 * ab[IDX_GH2CCCI] /
                                    38.0 / Hnuclei + 108.0 * ab[IDX_GHC3NI] /
                                    51.0 / Hnuclei + 12.0 * ab[IDX_GOCSI] / 60.0
                                    / Hnuclei + 192.0 * ab[IDX_C4H6I] / 54.0 /
                                    Hnuclei + 192.0 * ab[IDX_C4PII] / 79.0 /
                                    Hnuclei + 300.0 * ab[IDX_C5NII] / 74.0 /
                                    Hnuclei + 432.0 * ab[IDX_C6H6II] / 78.0 /
                                    Hnuclei + 108.0 * ab[IDX_CH2CHCNII] / 53.0 /
                                    Hnuclei + 48.0 * ab[IDX_COOCH3II] / 59.0 /
                                    Hnuclei + 108.0 * ab[IDX_GC2H5CNI] / 55.0 /
                                    Hnuclei + 48.0 * ab[IDX_GC2H5OHI] / 46.0 /
                                    Hnuclei + 48.0 * ab[IDX_GC2OI] / 40.0 /
                                    Hnuclei + 432.0 * ab[IDX_GC6H2I] / 74.0 /
                                    Hnuclei + 972.0 * ab[IDX_GC9NI] / 122.0 /
                                    Hnuclei + 12.0 * ab[IDX_GCO2I] / 44.0 /
                                    Hnuclei + 12.0 * ab[IDX_GCPI] / 43.0 /
                                    Hnuclei + 12.0 * ab[IDX_GOCNI] / 42.0 /
                                    Hnuclei + 12.0 * ab[IDX_GSiCI] / 40.0 /
                                    Hnuclei + 12.0 * ab[IDX_H2CNOII] / 44.0 /
                                    Hnuclei + 12.0 * ab[IDX_H2NCOII] / 44.0 /
                                    Hnuclei + 12.0 * ab[IDX_H2OCNII] / 44.0 /
                                    Hnuclei + 48.0 * ab[IDX_PC2H4II] / 59.0 /
                                    Hnuclei + 588.0 * ab[IDX_C7NII] / 98.0 /
                                    Hnuclei + 972.0 * ab[IDX_C9NII] / 122.0 /
                                    Hnuclei + 12.0 * ab[IDX_CH3NHII] / 30.0 /
                                    Hnuclei + 1452.0 * ab[IDX_GC11I] / 132.0 /
                                    Hnuclei + 48.0 * ab[IDX_GC2NI] / 38.0 /
                                    Hnuclei + 108.0 * ab[IDX_GCH2CCH2I] / 40.0 /
                                    Hnuclei + 12.0 * ab[IDX_GCH2NHI] / 29.0 /
                                    Hnuclei + 108.0 * ab[IDX_GCH3CCHI] / 40.0 /
                                    Hnuclei + 12.0 * ab[IDX_GCNOI] / 42.0 /
                                    Hnuclei + 12.0 * ab[IDX_GHNCI] / 27.0 /
                                    Hnuclei + 48.0 * ab[IDX_GSiC2I] / 52.0 /
                                    Hnuclei + 192.0 * ab[IDX_H2C4NII] / 64.0 /
                                    Hnuclei + 12.0 * ab[IDX_H2CClII] / 49.0 /
                                    Hnuclei + 12.0 * ab[IDX_HOCII] / 29.0 /
                                    Hnuclei + 12.0 * ab[IDX_NH2CNHII] / 43.0 /
                                    Hnuclei + 48.0 * ab[IDX_C2H5OHII] / 46.0 /
                                    Hnuclei + 48.0 * ab[IDX_C2OII] / 40.0 /
                                    Hnuclei + 192.0 * ab[IDX_CH2CHCCHI] / 52.0 /
                                    Hnuclei + 48.0 * ab[IDX_CH2OHCOII] / 59.0 /
                                    Hnuclei + 108.0 * ab[IDX_GCH2CCHI] / 39.0 /
                                    Hnuclei + 48.0 * ab[IDX_GCH2COI] / 42.0 /
                                    Hnuclei + 192.0 * ab[IDX_HC4NII] / 63.0 /
                                    Hnuclei + 12.0 * ab[IDX_HCNOHII] / 44.0 /
                                    Hnuclei + 12.0 * ab[IDX_HNCOHII] / 44.0 /
                                    Hnuclei + 12.0 * ab[IDX_HOCNII] / 43.0 /
                                    Hnuclei + 48.0 * ab[IDX_PC2H3II] / 58.0 /
                                    Hnuclei + 48.0 * ab[IDX_CH3CSII] / 59.0 /
                                    Hnuclei + 300.0 * ab[IDX_GC5NI] / 74.0 /
                                    Hnuclei + 588.0 * ab[IDX_H2C7NII] / 100.0 /
                                    Hnuclei + 972.0 * ab[IDX_H2C9NII] / 124.0 /
                                    Hnuclei + 12.0 * ab[IDX_HONCII] / 43.0 /
                                    Hnuclei + 12.0 * ab[IDX_PCH3II] / 46.0 /
                                    Hnuclei + 1200.0 * ab[IDX_C10H3II] / 123.0 /
                                    Hnuclei + 108.0 * ab[IDX_C2H5CNHII] / 56.0 /
                                    Hnuclei + 12.0 * ab[IDX_CClII] / 47.0 /
                                    Hnuclei + 12.0 * ab[IDX_CH2NH2II] / 30.0 /
                                    Hnuclei + 48.0 * ab[IDX_CH2OHCH2OII] / 61.0
                                    / Hnuclei + 432.0 * ab[IDX_CH3C5NHII] / 90.0
                                    / Hnuclei + 768.0 * ab[IDX_CH3C7NHII] /
                                    114.0 / Hnuclei + 48.0 * ab[IDX_GC2H5I] /
                                    29.0 / Hnuclei + 108.0 * ab[IDX_GC3NI] /
                                    50.0 / Hnuclei + 588.0 * ab[IDX_GC7NI] /
                                    98.0 / Hnuclei + 48.0 * ab[IDX_GCOOCH3I] /
                                    59.0 / Hnuclei + 12.0 * ab[IDX_GHCNI] / 27.0
                                    / Hnuclei + 12.0 * ab[IDX_H2CNI] / 28.0 /
                                    Hnuclei + 300.0 * ab[IDX_H3C5NII] / 77.0 /
                                    Hnuclei + 12.0 * ab[IDX_HCNOII] / 43.0 /
                                    Hnuclei + 12.0 * ab[IDX_CClI] / 47.0 /
                                    Hnuclei + 48.0 * ab[IDX_CH3COOHII] / 60.0 /
                                    Hnuclei + 972.0 * ab[IDX_H3C9NII] / 125.0 /
                                    Hnuclei + 48.0 * ab[IDX_H5C2O2II] / 61.0 /
                                    Hnuclei + 12.0 * ab[IDX_HCOOHII] / 46.0 /
                                    Hnuclei + 48.0 * ab[IDX_C2SII] / 56.0 /
                                    Hnuclei + 108.0 * ab[IDX_C3SII] / 68.0 /
                                    Hnuclei + 192.0 * ab[IDX_C4NI] / 62.0 /
                                    Hnuclei + 48.0 * ab[IDX_CH3COOH2II] / 61.0 /
                                    Hnuclei + 48.0 * ab[IDX_CH3OCH3II] / 46.0 /
                                    Hnuclei + 1200.0 * ab[IDX_GC10I] / 120.0 /
                                    Hnuclei + 1200.0 * ab[IDX_GC10HI] / 121.0 /
                                    Hnuclei + 768.0 * ab[IDX_GC8HI] / 97.0 /
                                    Hnuclei + 48.0 * ab[IDX_GCH2OHCOI] / 59.0 /
                                    Hnuclei + 12.0 * ab[IDX_GCOOHI] / 45.0 /
                                    Hnuclei + 588.0 * ab[IDX_H3C7NII] / 101.0 /
                                    Hnuclei + 12.0 * ab[IDX_NH2CNI] / 42.0 /
                                    Hnuclei + 192.0 * ab[IDX_CH3C3NHII] / 66.0 /
                                    Hnuclei + 48.0 * ab[IDX_CH3OCH4II] / 47.0 /
                                    Hnuclei + 48.0 * ab[IDX_GC2H4I] / 28.0 /
                                    Hnuclei + 972.0 * ab[IDX_GC9I] / 108.0 /
                                    Hnuclei + 972.0 * ab[IDX_GC9HI] / 109.0 /
                                    Hnuclei + 12.0 * ab[IDX_GCSI] / 44.0 /
                                    Hnuclei + 588.0 * ab[IDX_HC7NII] / 99.0 /
                                    Hnuclei + 972.0 * ab[IDX_HC9NII] / 123.0 /
                                    Hnuclei + 12.0 * ab[IDX_HNCOII] / 43.0 /
                                    Hnuclei + 108.0 * ab[IDX_PC3HII] / 68.0 /
                                    Hnuclei + 12.0 * ab[IDX_PCH4II] / 47.0 /
                                    Hnuclei + 108.0 * ab[IDX_C2H5CNI] / 55.0 /
                                    Hnuclei + 48.0 * ab[IDX_C2H5OH2II] / 47.0 /
                                    Hnuclei + 48.0 * ab[IDX_C2H7II] / 31.0 /
                                    Hnuclei + 48.0 * ab[IDX_CCPII] / 55.0 /
                                    Hnuclei + 108.0 * ab[IDX_CH2CHCNHII] / 54.0
                                    / Hnuclei + 48.0 * ab[IDX_COOCH3I] / 59.0 /
                                    Hnuclei + 48.0 * ab[IDX_HCOOCH3II] / 60.0 /
                                    Hnuclei + 48.0 * ab[IDX_PC2H2II] / 57.0 /
                                    Hnuclei + 108.0 * ab[IDX_SiC3H2II] / 66.0 /
                                    Hnuclei + 192.0 * ab[IDX_SiC4II] / 76.0 /
                                    Hnuclei + 192.0 * ab[IDX_SiC4HII] / 77.0 /
                                    Hnuclei + 12.0 * ab[IDX_SiNCHII] / 55.0 /
                                    Hnuclei + 1452.0 * ab[IDX_C11I] / 132.0 /
                                    Hnuclei + 108.0 * ab[IDX_C3OII] / 52.0 /
                                    Hnuclei + 192.0 * ab[IDX_C4H5II] / 53.0 /
                                    Hnuclei + 192.0 * ab[IDX_C4H7II] / 55.0 /
                                    Hnuclei + 48.0 * ab[IDX_CH2OHCHOII] / 60.0 /
                                    Hnuclei + 48.0 * ab[IDX_CH3CHOII] / 44.0 /
                                    Hnuclei + 48.0 * ab[IDX_GCH2OHCHOI] / 60.0 /
                                    Hnuclei + 12.0 * ab[IDX_GHCOOHI] / 46.0 /
                                    Hnuclei + 12.0 * ab[IDX_HOCSII] / 61.0 /
                                    Hnuclei + 12.0 * ab[IDX_OCNII] / 42.0 /
                                    Hnuclei + 192.0 * ab[IDX_PC4HII] / 80.0 /
                                    Hnuclei + 12.0 * ab[IDX_SiCH4II] / 44.0 /
                                    Hnuclei + 1200.0 * ab[IDX_C10II] / 120.0 /
                                    Hnuclei + 1200.0 * ab[IDX_C10H2II] / 122.0 /
                                    Hnuclei + 48.0 * ab[IDX_C2N2II] / 52.0 /
                                    Hnuclei + 48.0 * ab[IDX_CH3CNII] / 41.0 /
                                    Hnuclei + 48.0 * ab[IDX_GC2H2I] / 26.0 /
                                    Hnuclei + 300.0 * ab[IDX_GC5I] / 60.0 /
                                    Hnuclei + 588.0 * ab[IDX_GC7HI] / 85.0 /
                                    Hnuclei + 48.0 * ab[IDX_GCH3COI] / 43.0 /
                                    Hnuclei + 48.0 * ab[IDX_GHCOOCH3I] / 60.0 /
                                    Hnuclei + 192.0 * ab[IDX_HC4SII] / 81.0 /
                                    Hnuclei + 300.0 * ab[IDX_HC5NHII] / 76.0 /
                                    Hnuclei + 48.0 * ab[IDX_SiC2H3II] / 55.0 /
                                    Hnuclei + 108.0 * ab[IDX_SiC3HI] / 65.0 /
                                    Hnuclei + 12.0 * ab[IDX_SiCH3II] / 43.0 /
                                    Hnuclei + 1452.0 * ab[IDX_C11II] / 132.0 /
                                    Hnuclei + 108.0 * ab[IDX_C3NII] / 50.0 /
                                    Hnuclei + 192.0 * ab[IDX_C4H4II] / 52.0 /
                                    Hnuclei + 48.0 * ab[IDX_CH2CNII] / 40.0 /
                                    Hnuclei + 108.0 * ab[IDX_CH3COCH3II] / 58.0
                                    / Hnuclei + 12.0 * ab[IDX_CPII] / 43.0 /
                                    Hnuclei + 432.0 * ab[IDX_GC6I] / 72.0 /
                                    Hnuclei + 768.0 * ab[IDX_GC8I] / 96.0 /
                                    Hnuclei + 12.0 * ab[IDX_GCH2OHI] / 31.0 /
                                    Hnuclei + 12.0 * ab[IDX_GCH3OHI] / 32.0 /
                                    Hnuclei + 48.0 * ab[IDX_HC2PII] / 56.0 /
                                    Hnuclei + 300.0 * ab[IDX_HC5NII] / 75.0 /
                                    Hnuclei + 12.0 * ab[IDX_HONCI] / 43.0 /
                                    Hnuclei + 12.0 * ab[IDX_PCH2II] / 45.0 /
                                    Hnuclei + 972.0 * ab[IDX_C9II] / 108.0 /
                                    Hnuclei + 48.0 * ab[IDX_CH2OHCOI] / 59.0 /
                                    Hnuclei + 432.0 * ab[IDX_CH3C5NI] / 89.0 /
                                    Hnuclei + 768.0 * ab[IDX_CH3C7NI] / 113.0 /
                                    Hnuclei + 108.0 * ab[IDX_GC3HI] / 37.0 /
                                    Hnuclei + 192.0 * ab[IDX_GC4HI] / 49.0 /
                                    Hnuclei + 300.0 * ab[IDX_GC5HI] / 61.0 /
                                    Hnuclei + 432.0 * ab[IDX_GC6HI] / 73.0 /
                                    Hnuclei + 12.0 * ab[IDX_H2NCII] / 28.0 /
                                    Hnuclei + 12.0 * ab[IDX_H3CSII] / 47.0 /
                                    Hnuclei + 48.0 * ab[IDX_HC2OII] / 41.0 /
                                    Hnuclei + 12.0 * ab[IDX_HCPII] / 44.0 /
                                    Hnuclei + 192.0 * ab[IDX_SiC4I] / 76.0 /
                                    Hnuclei + 12.0 * ab[IDX_SiNCI] / 54.0 /
                                    Hnuclei + 12.0 * ab[IDX_SiNCII] / 54.0 /
                                    Hnuclei + 12.0 * ab[IDX_CNOI] / 42.0 /
                                    Hnuclei + 48.0 * ab[IDX_GC2H3I] / 27.0 /
                                    Hnuclei + 12.0 * ab[IDX_GCH3OI] / 31.0 /
                                    Hnuclei + 108.0 * ab[IDX_HC3SII] / 69.0 /
                                    Hnuclei + 12.0 * ab[IDX_HCOOH2II] / 47.0 /
                                    Hnuclei + 48.0 * ab[IDX_SiC2H2I] / 54.0 /
                                    Hnuclei + 108.0 * ab[IDX_SiC3II] / 64.0 /
                                    Hnuclei + 48.0 * ab[IDX_C2NHII] / 39.0 /
                                    Hnuclei + 108.0 * ab[IDX_C3H6II] / 42.0 /
                                    Hnuclei + 432.0 * ab[IDX_C6H7II] / 79.0 /
                                    Hnuclei + 768.0 * ab[IDX_C8H4II] / 100.0 /
                                    Hnuclei + 48.0 * ab[IDX_CH2OHCHOI] / 60.0 /
                                    Hnuclei + 588.0 * ab[IDX_GC7I] / 84.0 /
                                    Hnuclei + 108.0 * ab[IDX_HC3OII] / 53.0 /
                                    Hnuclei + 12.0 * ab[IDX_HCNOI] / 43.0 /
                                    Hnuclei + 108.0 * ab[IDX_SiC3HII] / 65.0 /
                                    Hnuclei + 12.0 * ab[IDX_SiCH3I] / 43.0 /
                                    Hnuclei + 108.0 * ab[IDX_C3H7II] / 43.0 /
                                    Hnuclei + 192.0 * ab[IDX_C4NII] / 62.0 /
                                    Hnuclei + 768.0 * ab[IDX_C8II] / 96.0 /
                                    Hnuclei + 972.0 * ab[IDX_C9H4II] / 112.0 /
                                    Hnuclei + 48.0 * ab[IDX_CH2COII] / 42.0 /
                                    Hnuclei + 12.0 * ab[IDX_CH2PHI] / 46.0 /
                                    Hnuclei + 192.0 * ab[IDX_CH3C3NI] / 65.0 /
                                    Hnuclei + 192.0 * ab[IDX_GC4I] / 48.0 /
                                    Hnuclei + 48.0 * ab[IDX_GCH3CHOI] / 44.0 /
                                    Hnuclei + 12.0 * ab[IDX_GH2COI] / 30.0 /
                                    Hnuclei + 12.0 * ab[IDX_HOCNI] / 43.0 /
                                    Hnuclei + 48.0 * ab[IDX_SiC2II] / 52.0 /
                                    Hnuclei + 48.0 * ab[IDX_SiC2H2II] / 54.0 /
                                    Hnuclei + 432.0 * ab[IDX_C6H4II] / 76.0 /
                                    Hnuclei + 48.0 * ab[IDX_CH3COI] / 43.0 /
                                    Hnuclei + 108.0 * ab[IDX_CH3COCH4II] / 59.0
                                    / Hnuclei + 12.0 * ab[IDX_CH3OHII] / 32.0 /
                                    Hnuclei + 12.0 * ab[IDX_COOHI] / 45.0 /
                                    Hnuclei + 588.0 * ab[IDX_C7H4II] / 88.0 /
                                    Hnuclei + 588.0 * ab[IDX_C7H5II] / 89.0 /
                                    Hnuclei + 588.0 * ab[IDX_CH3C6HI] / 88.0 /
                                    Hnuclei + 12.0 * ab[IDX_H2CSII] / 46.0 /
                                    Hnuclei + 972.0 * ab[IDX_HC9NI] / 123.0 /
                                    Hnuclei + 48.0 * ab[IDX_HCOOCH3I] / 60.0 /
                                    Hnuclei + 48.0 * ab[IDX_NCCNHII] / 53.0 /
                                    Hnuclei + 48.0 * ab[IDX_SiC2HI] / 53.0 /
                                    Hnuclei + 108.0 * ab[IDX_SiC3I] / 64.0 /
                                    Hnuclei + 192.0 * ab[IDX_C4PI] / 79.0 /
                                    Hnuclei + 588.0 * ab[IDX_C7II] / 84.0 /
                                    Hnuclei + 48.0 * ab[IDX_CH2CNI] / 40.0 /
                                    Hnuclei + 300.0 * ab[IDX_CH3C4HII] / 64.0 /
                                    Hnuclei + 12.0 * ab[IDX_GCH4I] / 16.0 /
                                    Hnuclei + 12.0 * ab[IDX_HCPI] / 44.0 /
                                    Hnuclei + 12.0 * ab[IDX_HNCOI] / 43.0 /
                                    Hnuclei + 108.0 * ab[IDX_C3PI] / 67.0 /
                                    Hnuclei + 972.0 * ab[IDX_C9NI] / 122.0 /
                                    Hnuclei + 300.0 * ab[IDX_CH3C4HI] / 64.0 /
                                    Hnuclei + 48.0 * ab[IDX_CH3COOHI] / 60.0 /
                                    Hnuclei + 48.0 * ab[IDX_CH3OCH3I] / 46.0 /
                                    Hnuclei + 48.0 * ab[IDX_GC2HI] / 25.0 /
                                    Hnuclei + 12.0 * ab[IDX_H2CSI] / 46.0 /
                                    Hnuclei + 12.0 * ab[IDX_SiCII] / 40.0 /
                                    Hnuclei + 108.0 * ab[IDX_C3SI] / 68.0 /
                                    Hnuclei + 432.0 * ab[IDX_C6H5II] / 77.0 /
                                    Hnuclei + 12.0 * ab[IDX_CH2NHI] / 29.0 /
                                    Hnuclei + 12.0 * ab[IDX_CH3OI] / 31.0 /
                                    Hnuclei + 12.0 * ab[IDX_GHCOI] / 29.0 /
                                    Hnuclei + 48.0 * ab[IDX_HC2PI] / 56.0 /
                                    Hnuclei + 12.0 * ab[IDX_HCSiI] / 41.0 /
                                    Hnuclei + 12.0 * ab[IDX_SiCH2I] / 42.0 /
                                    Hnuclei + 1200.0 * ab[IDX_C10HII] / 121.0 /
                                    Hnuclei + 12.0 * ab[IDX_HCSiII] / 41.0 /
                                    Hnuclei + 12.0 * ab[IDX_OCSII] / 60.0 /
                                    Hnuclei + 48.0 * ab[IDX_SiC2I] / 52.0 /
                                    Hnuclei + 48.0 * ab[IDX_C2H6II] / 30.0 /
                                    Hnuclei + 48.0 * ab[IDX_C2OI] / 40.0 /
                                    Hnuclei + 768.0 * ab[IDX_C8HII] / 97.0 /
                                    Hnuclei + 12.0 * ab[IDX_CH2OHI] / 31.0 /
                                    Hnuclei + 300.0 * ab[IDX_C5H5II] / 65.0 /
                                    Hnuclei + 12.0 * ab[IDX_HCSI] / 45.0 /
                                    Hnuclei + 48.0 * ab[IDX_SiC2HII] / 53.0 /
                                    Hnuclei + 108.0 * ab[IDX_C3OI] / 52.0 /
                                    Hnuclei + 432.0 * ab[IDX_C6HII] / 73.0 /
                                    Hnuclei + 972.0 * ab[IDX_C9H2I] / 110.0 /
                                    Hnuclei + 108.0 * ab[IDX_CH2CHCNI] / 53.0 /
                                    Hnuclei + 48.0 * ab[IDX_CH3CHOHII] / 45.0 /
                                    Hnuclei + 48.0 * ab[IDX_CH3CNHII] / 42.0 /
                                    Hnuclei + 48.0 * ab[IDX_CH3COII] / 43.0 /
                                    Hnuclei + 588.0 * ab[IDX_HC7NI] / 99.0 /
                                    Hnuclei + 432.0 * ab[IDX_C6II] / 72.0 /
                                    Hnuclei + 432.0 * ab[IDX_C6H6I] / 78.0 /
                                    Hnuclei + 588.0 * ab[IDX_C7NI] / 98.0 /
                                    Hnuclei + 12.0 * ab[IDX_HCOOHI] / 46.0 /
                                    Hnuclei + 1200.0 * ab[IDX_C10H2I] / 122.0 /
                                    Hnuclei + 192.0 * ab[IDX_C4II] / 48.0 /
                                    Hnuclei + 48.0 * ab[IDX_CCPI] / 55.0 /
                                    Hnuclei + 12.0 * ab[IDX_CPI] / 43.0 /
                                    Hnuclei + 108.0 * ab[IDX_GC3I] / 36.0 /
                                    Hnuclei + 12.0 * ab[IDX_SiCI] / 40.0 /
                                    Hnuclei + 48.0 * ab[IDX_C2H5OHI] / 46.0 /
                                    Hnuclei + 12.0 * ab[IDX_CH3OH2II] / 33.0 /
                                    Hnuclei + 48.0 * ab[IDX_C2NII] / 38.0 /
                                    Hnuclei + 300.0 * ab[IDX_C5II] / 60.0 /
                                    Hnuclei + 432.0 * ab[IDX_C6H3II] / 75.0 /
                                    Hnuclei + 768.0 * ab[IDX_C8H3II] / 99.0 /
                                    Hnuclei + 972.0 * ab[IDX_C9HII] / 109.0 /
                                    Hnuclei + 972.0 * ab[IDX_C9H3II] / 111.0 /
                                    Hnuclei + 48.0 * ab[IDX_CH2COI] / 42.0 /
                                    Hnuclei + 108.0 * ab[IDX_HNC3I] / 51.0 /
                                    Hnuclei + 12.0 * ab[IDX_OCNI] / 42.0 /
                                    Hnuclei + 12.0 * ab[IDX_SiCH2II] / 42.0 /
                                    Hnuclei + 108.0 * ab[IDX_CH3COCH3I] / 58.0 /
                                    Hnuclei + 108.0 * ab[IDX_HC3NII] / 51.0 /
                                    Hnuclei + 768.0 * ab[IDX_C8H2I] / 98.0 /
                                    Hnuclei + 12.0 * ab[IDX_CO2II] / 44.0 /
                                    Hnuclei + 300.0 * ab[IDX_HC5NI] / 75.0 /
                                    Hnuclei + 588.0 * ab[IDX_C7HII] / 85.0 /
                                    Hnuclei + 972.0 * ab[IDX_C9H2II] / 110.0 /
                                    Hnuclei + 588.0 * ab[IDX_C7H2I] / 86.0 /
                                    Hnuclei + 12.0 * ab[IDX_GCOI] / 28.0 /
                                    Hnuclei + 12.0 * ab[IDX_HCSII] / 45.0 /
                                    Hnuclei + 108.0 * ab[IDX_C3H4II] / 40.0 /
                                    Hnuclei + 768.0 * ab[IDX_C8H2II] / 98.0 /
                                    Hnuclei + 12.0 * ab[IDX_GCH2I] / 14.0 /
                                    Hnuclei + 12.0 * ab[IDX_HCO2II] / 45.0 /
                                    Hnuclei + 300.0 * ab[IDX_C5H3II] / 63.0 /
                                    Hnuclei + 108.0 * ab[IDX_HC3NHII] / 52.0 /
                                    Hnuclei + 300.0 * ab[IDX_C5HII] / 61.0 /
                                    Hnuclei + 48.0 * ab[IDX_CH3CHOI] / 44.0 /
                                    Hnuclei + 12.0 * ab[IDX_CSII] / 44.0 /
                                    Hnuclei + 108.0 * ab[IDX_C3H5II] / 41.0 /
                                    Hnuclei + 588.0 * ab[IDX_C7H3II] / 87.0 /
                                    Hnuclei + 48.0 * ab[IDX_CH3CNI] / 41.0 /
                                    Hnuclei + 48.0 * ab[IDX_NCCNI] / 52.0 /
                                    Hnuclei + 588.0 * ab[IDX_C7H2II] / 86.0 /
                                    Hnuclei + 48.0 * ab[IDX_GC2I] / 24.0 /
                                    Hnuclei + 108.0 * ab[IDX_C3II] / 36.0 /
                                    Hnuclei + 432.0 * ab[IDX_C6H2II] / 74.0 /
                                    Hnuclei + 12.0 * ab[IDX_CH4II] / 16.0 /
                                    Hnuclei + 108.0 * ab[IDX_H2CCCI] / 38.0 /
                                    Hnuclei + 432.0 * ab[IDX_C6H2I] / 74.0 /
                                    Hnuclei + 12.0 * ab[IDX_GCNI] / 26.0 /
                                    Hnuclei + 48.0 * ab[IDX_C2H5II] / 29.0 /
                                    Hnuclei + 300.0 * ab[IDX_C5H2I] / 62.0 /
                                    Hnuclei + 300.0 * ab[IDX_C5H2II] / 62.0 /
                                    Hnuclei + 108.0 * ab[IDX_CH2CCH2I] / 40.0 /
                                    Hnuclei + 12.0 * ab[IDX_CNII] / 26.0 /
                                    Hnuclei + 48.0 * ab[IDX_C2H6I] / 30.0 /
                                    Hnuclei + 108.0 * ab[IDX_C3H2I] / 38.0 /
                                    Hnuclei + 48.0 * ab[IDX_C2H5I] / 29.0 /
                                    Hnuclei + 192.0 * ab[IDX_C4SII] / 80.0 /
                                    Hnuclei + 12.0 * ab[IDX_HCNII] / 27.0 /
                                    Hnuclei + 108.0 * ab[IDX_CH3CHCH2I] / 42.0 /
                                    Hnuclei + 48.0 * ab[IDX_C2II] / 24.0 /
                                    Hnuclei + 192.0 * ab[IDX_C4HII] / 49.0 /
                                    Hnuclei + 48.0 * ab[IDX_CNCII] / 38.0 /
                                    Hnuclei + 12.0 * ab[IDX_CHM] / 13.0 /
                                    Hnuclei + 12.0 * ab[IDX_OCSI] / 60.0 /
                                    Hnuclei + 12.0 * ab[IDX_GCH3I] / 15.0 /
                                    Hnuclei + 48.0 * ab[IDX_C2HM] / 25.0 /
                                    Hnuclei + 192.0 * ab[IDX_C4HM] / 49.0 /
                                    Hnuclei + 12.0 * ab[IDX_COII] / 28.0 /
                                    Hnuclei + 108.0 * ab[IDX_CH3CCHI] / 40.0 /
                                    Hnuclei + 12.0 * ab[IDX_CH5II] / 17.0 /
                                    Hnuclei + 48.0 * ab[IDX_HC2SII] / 57.0 /
                                    Hnuclei + 1200.0 * ab[IDX_C10HM] / 121.0 /
                                    Hnuclei + 108.0 * ab[IDX_C3HM] / 37.0 /
                                    Hnuclei + 192.0 * ab[IDX_C4SI] / 80.0 /
                                    Hnuclei + 432.0 * ab[IDX_C6HM] / 73.0 /
                                    Hnuclei + 768.0 * ab[IDX_C8HM] / 97.0 /
                                    Hnuclei + 588.0 * ab[IDX_C7HM] / 85.0 /
                                    Hnuclei + 972.0 * ab[IDX_C9HM] / 109.0 /
                                    Hnuclei + 1200.0 * ab[IDX_C10M] / 120.0 /
                                    Hnuclei + 300.0 * ab[IDX_C5HM] / 61.0 /
                                    Hnuclei + 300.0 * ab[IDX_C5NM] / 74.0 /
                                    Hnuclei + 12.0 * ab[IDX_CM] / 12.0 / Hnuclei
                                    + 12.0 * ab[IDX_CSI] / 44.0 / Hnuclei +
                                    972.0 * ab[IDX_C9M] / 108.0 / Hnuclei + 12.0
                                    * ab[IDX_CH3OHI] / 32.0 / Hnuclei + 108.0 *
                                    ab[IDX_HC3NI] / 51.0 / Hnuclei + 108.0 *
                                    ab[IDX_C3HII] / 37.0 / Hnuclei + 108.0 *
                                    ab[IDX_C3H3II] / 39.0 / Hnuclei + 768.0 *
                                    ab[IDX_C8M] / 96.0 / Hnuclei + 48.0 *
                                    ab[IDX_C2M] / 24.0 / Hnuclei + 48.0 *
                                    ab[IDX_C2SI] / 56.0 / Hnuclei + 108.0 *
                                    ab[IDX_C3M] / 36.0 / Hnuclei + 192.0 *
                                    ab[IDX_C4M] / 48.0 / Hnuclei + 432.0 *
                                    ab[IDX_C6M] / 72.0 / Hnuclei + 12.0 *
                                    ab[IDX_H3COII] / 31.0 / Hnuclei + 300.0 *
                                    ab[IDX_C5M] / 60.0 / Hnuclei + 588.0 *
                                    ab[IDX_C7M] / 84.0 / Hnuclei + 48.0 *
                                    ab[IDX_C2NI] / 38.0 / Hnuclei + 12.0 *
                                    ab[IDX_GCHI] / 13.0 / Hnuclei + 192.0 *
                                    ab[IDX_C4H3I] / 51.0 / Hnuclei + 48.0 *
                                    ab[IDX_C2HII] / 25.0 / Hnuclei + 108.0 *
                                    ab[IDX_C3H2II] / 38.0 / Hnuclei + 300.0 *
                                    ab[IDX_C5NI] / 74.0 / Hnuclei + 12.0 *
                                    ab[IDX_CH2II] / 14.0 / Hnuclei + 108.0 *
                                    ab[IDX_C3NM] / 50.0 / Hnuclei + 108.0 *
                                    ab[IDX_C3NI] / 50.0 / Hnuclei + 12.0 *
                                    ab[IDX_CHII] / 13.0 / Hnuclei + 12.0 *
                                    ab[IDX_CO2I] / 44.0 / Hnuclei + 1200.0 *
                                    ab[IDX_C10HI] / 121.0 / Hnuclei + 972.0 *
                                    ab[IDX_C9HI] / 109.0 / Hnuclei + 1200.0 *
                                    ab[IDX_C10I] / 120.0 / Hnuclei + 768.0 *
                                    ab[IDX_C8HI] / 97.0 / Hnuclei + 588.0 *
                                    ab[IDX_C7I] / 84.0 / Hnuclei + 972.0 *
                                    ab[IDX_C9I] / 108.0 / Hnuclei + 768.0 *
                                    ab[IDX_C8I] / 96.0 / Hnuclei + 12.0 *
                                    ab[IDX_GCI] / 12.0 / Hnuclei + 300.0 *
                                    ab[IDX_C5I] / 60.0 / Hnuclei + 432.0 *
                                    ab[IDX_C6I] / 72.0 / Hnuclei + 12.0 *
                                    ab[IDX_CNM] / 26.0 / Hnuclei + 192.0 *
                                    ab[IDX_C4I] / 48.0 / Hnuclei + 588.0 *
                                    ab[IDX_C7HI] / 85.0 / Hnuclei + 48.0 *
                                    ab[IDX_C2H4II] / 28.0 / Hnuclei + 192.0 *
                                    ab[IDX_C4H3II] / 51.0 / Hnuclei + 108.0 *
                                    ab[IDX_CH2CCHI] / 39.0 / Hnuclei + 108.0 *
                                    ab[IDX_CH2CCHII] / 39.0 / Hnuclei + 432.0 *
                                    ab[IDX_C6HI] / 73.0 / Hnuclei + 300.0 *
                                    ab[IDX_C5HI] / 61.0 / Hnuclei + 108.0 *
                                    ab[IDX_C3HI] / 37.0 / Hnuclei + 12.0 *
                                    ab[IDX_H2COII] / 30.0 / Hnuclei + 108.0 *
                                    ab[IDX_C3I] / 36.0 / Hnuclei + 192.0 *
                                    ab[IDX_C4HI] / 49.0 / Hnuclei + 192.0 *
                                    ab[IDX_C4H2I] / 50.0 / Hnuclei + 192.0 *
                                    ab[IDX_C4H2II] / 50.0 / Hnuclei + 12.0 *
                                    ab[IDX_HCNHII] / 28.0 / Hnuclei + 12.0 *
                                    ab[IDX_CH2I] / 14.0 / Hnuclei + 12.0 *
                                    ab[IDX_HNCI] / 27.0 / Hnuclei + 48.0 *
                                    ab[IDX_C2H4I] / 28.0 / Hnuclei + 48.0 *
                                    ab[IDX_C2H3I] / 27.0 / Hnuclei + 48.0 *
                                    ab[IDX_C2H3II] / 27.0 / Hnuclei + 12.0 *
                                    ab[IDX_HCOI] / 29.0 / Hnuclei + 12.0 *
                                    ab[IDX_CH4I] / 16.0 / Hnuclei + 48.0 *
                                    ab[IDX_C2HI] / 25.0 / Hnuclei + 12.0 *
                                    ab[IDX_H2COI] / 30.0 / Hnuclei + 12.0 *
                                    ab[IDX_HCNI] / 27.0 / Hnuclei + 12.0 *
                                    ab[IDX_CHI] / 13.0 / Hnuclei + 48.0 *
                                    ab[IDX_C2I] / 24.0 / Hnuclei + 48.0 *
                                    ab[IDX_C2H2II] / 26.0 / Hnuclei + 12.0 *
                                    ab[IDX_CH3II] / 15.0 / Hnuclei + 48.0 *
                                    ab[IDX_C2H2I] / 26.0 / Hnuclei + 12.0 *
                                    ab[IDX_CNI] / 26.0 / Hnuclei + 12.0 *
                                    ab[IDX_CH3I] / 15.0 / Hnuclei + 12.0 *
                                    ab[IDX_CII] / 12.0 / Hnuclei + 12.0 *
                                    ab[IDX_CI] / 12.0 / Hnuclei + 12.0 *
                                    ab[IDX_HCOII] / 29.0 / Hnuclei + 12.0 *
                                    ab[IDX_COI] / 28.0 / Hnuclei;
    IJth(A, IDX_ELEM_C, IDX_ELEM_GRAIN) = 0.0;
    IJth(A, IDX_ELEM_C, IDX_ELEM_H) = 0.0 + 24.0 * ab[IDX_GC4H6I] / 54.0 / Hnuclei
                                    + 12.0 * ab[IDX_GCH3C3NI] / 65.0 / Hnuclei +
                                    20.0 * ab[IDX_GCH3C4HI] / 64.0 / Hnuclei +
                                    18.0 * ab[IDX_GCH3C5NI] / 89.0 / Hnuclei +
                                    28.0 * ab[IDX_GCH3C6HI] / 88.0 / Hnuclei +
                                    24.0 * ab[IDX_GCH3C7NI] / 113.0 / Hnuclei +
                                    2.0 * ab[IDX_GHC2PI] / 56.0 / Hnuclei + 2.0
                                    * ab[IDX_GNH2CNI] / 42.0 / Hnuclei + 3.0 *
                                    ab[IDX_GSiC3HI] / 65.0 / Hnuclei + 3.0 *
                                    ab[IDX_GSiCH3I] / 43.0 / Hnuclei + 2.0 *
                                    ab[IDX_GSiC2HI] / 53.0 / Hnuclei + 12.0 *
                                    ab[IDX_C2H4CNI] / 54.0 / Hnuclei + 3.0 *
                                    ab[IDX_GCH2PHI] / 46.0 / Hnuclei + 18.0 *
                                    ab[IDX_GCH3CHCH2I] / 42.0 / Hnuclei + 8.0 *
                                    ab[IDX_GCH3COOHI] / 60.0 / Hnuclei + 1.0 *
                                    ab[IDX_GHCNOI] / 43.0 / Hnuclei + 1.0 *
                                    ab[IDX_GHCPI] / 44.0 / Hnuclei + 3.0 *
                                    ab[IDX_GHNC3I] / 51.0 / Hnuclei + 1.0 *
                                    ab[IDX_GHNCOI] / 43.0 / Hnuclei + 1.0 *
                                    ab[IDX_GHOCNI] / 43.0 / Hnuclei + 1.0 *
                                    ab[IDX_GHONCI] / 43.0 / Hnuclei + 4.0 *
                                    ab[IDX_GSiC2H2I] / 54.0 / Hnuclei + 2.0 *
                                    ab[IDX_HC2OI] / 41.0 / Hnuclei + 36.0 *
                                    ab[IDX_GC6H6I] / 78.0 / Hnuclei + 1.0 *
                                    ab[IDX_GHCSiI] / 41.0 / Hnuclei + 9.0 *
                                    ab[IDX_NCCNCH3II] / 67.0 / Hnuclei + 12.0 *
                                    ab[IDX_GC2H4CNI] / 54.0 / Hnuclei + 12.0 *
                                    ab[IDX_GC4H3I] / 51.0 / Hnuclei + 14.0 *
                                    ab[IDX_GC7H2I] / 86.0 / Hnuclei + 16.0 *
                                    ab[IDX_GC8H2I] / 98.0 / Hnuclei + 18.0 *
                                    ab[IDX_GC9H2I] / 110.0 / Hnuclei + 4.0 *
                                    ab[IDX_GCH2CNI] / 40.0 / Hnuclei + 6.0 *
                                    ab[IDX_GCH3CNI] / 41.0 / Hnuclei + 2.0 *
                                    ab[IDX_GH2CNI] / 28.0 / Hnuclei + 2.0 *
                                    ab[IDX_GH2CSI] / 46.0 / Hnuclei + 2.0 *
                                    ab[IDX_GHC2OI] / 41.0 / Hnuclei + 5.0 *
                                    ab[IDX_GHC5NI] / 75.0 / Hnuclei + 7.0 *
                                    ab[IDX_GHC7NI] / 99.0 / Hnuclei + 9.0 *
                                    ab[IDX_GHC9NI] / 123.0 / Hnuclei + 2.0 *
                                    ab[IDX_GHCCNI] / 39.0 / Hnuclei + 1.0 *
                                    ab[IDX_GHCSI] / 45.0 / Hnuclei + 2.0 *
                                    ab[IDX_HCCNI] / 39.0 / Hnuclei + 6.0 *
                                    ab[IDX_C3H2OII] / 54.0 / Hnuclei + 12.0 *
                                    ab[IDX_GC2H6I] / 30.0 / Hnuclei + 6.0 *
                                    ab[IDX_GC3H2I] / 38.0 / Hnuclei + 10.0 *
                                    ab[IDX_GC5H2I] / 62.0 / Hnuclei + 16.0 *
                                    ab[IDX_GCH2CHCCHI] / 52.0 / Hnuclei + 18.0 *
                                    ab[IDX_GCH3COCH3I] / 58.0 / Hnuclei + 12.0 *
                                    ab[IDX_GCH3OCH3I] / 46.0 / Hnuclei + 2.0 *
                                    ab[IDX_GSiCH2I] / 42.0 / Hnuclei + 9.0 *
                                    ab[IDX_H3C3OII] / 55.0 / Hnuclei + 40.0 *
                                    ab[IDX_C8H5II] / 101.0 / Hnuclei + 45.0 *
                                    ab[IDX_C9H5II] / 113.0 / Hnuclei + 12.0 *
                                    ab[IDX_CH3C3NII] / 65.0 / Hnuclei + 20.0 *
                                    ab[IDX_GC10H2I] / 122.0 / Hnuclei + 8.0 *
                                    ab[IDX_GC4H2I] / 50.0 / Hnuclei + 9.0 *
                                    ab[IDX_GCH2CHCNI] / 53.0 / Hnuclei + 6.0 *
                                    ab[IDX_GH2CCCI] / 38.0 / Hnuclei + 3.0 *
                                    ab[IDX_GHC3NI] / 51.0 / Hnuclei + 24.0 *
                                    ab[IDX_C4H6I] / 54.0 / Hnuclei + 36.0 *
                                    ab[IDX_C6H6II] / 78.0 / Hnuclei + 9.0 *
                                    ab[IDX_CH2CHCNII] / 53.0 / Hnuclei + 6.0 *
                                    ab[IDX_COOCH3II] / 59.0 / Hnuclei + 15.0 *
                                    ab[IDX_GC2H5CNI] / 55.0 / Hnuclei + 12.0 *
                                    ab[IDX_GC2H5OHI] / 46.0 / Hnuclei + 12.0 *
                                    ab[IDX_GC6H2I] / 74.0 / Hnuclei + 2.0 *
                                    ab[IDX_H2CNOII] / 44.0 / Hnuclei + 2.0 *
                                    ab[IDX_H2NCOII] / 44.0 / Hnuclei + 2.0 *
                                    ab[IDX_H2OCNII] / 44.0 / Hnuclei + 8.0 *
                                    ab[IDX_PC2H4II] / 59.0 / Hnuclei + 4.0 *
                                    ab[IDX_CH3NHII] / 30.0 / Hnuclei + 12.0 *
                                    ab[IDX_GCH2CCH2I] / 40.0 / Hnuclei + 3.0 *
                                    ab[IDX_GCH2NHI] / 29.0 / Hnuclei + 12.0 *
                                    ab[IDX_GCH3CCHI] / 40.0 / Hnuclei + 1.0 *
                                    ab[IDX_GHNCI] / 27.0 / Hnuclei + 8.0 *
                                    ab[IDX_H2C4NII] / 64.0 / Hnuclei + 2.0 *
                                    ab[IDX_H2CClII] / 49.0 / Hnuclei + 1.0 *
                                    ab[IDX_HOCII] / 29.0 / Hnuclei + 3.0 *
                                    ab[IDX_NH2CNHII] / 43.0 / Hnuclei + 12.0 *
                                    ab[IDX_C2H5OHII] / 46.0 / Hnuclei + 16.0 *
                                    ab[IDX_CH2CHCCHI] / 52.0 / Hnuclei + 6.0 *
                                    ab[IDX_CH2OHCOII] / 59.0 / Hnuclei + 9.0 *
                                    ab[IDX_GCH2CCHI] / 39.0 / Hnuclei + 4.0 *
                                    ab[IDX_GCH2COI] / 42.0 / Hnuclei + 4.0 *
                                    ab[IDX_HC4NII] / 63.0 / Hnuclei + 2.0 *
                                    ab[IDX_HCNOHII] / 44.0 / Hnuclei + 2.0 *
                                    ab[IDX_HNCOHII] / 44.0 / Hnuclei + 1.0 *
                                    ab[IDX_HOCNII] / 43.0 / Hnuclei + 6.0 *
                                    ab[IDX_PC2H3II] / 58.0 / Hnuclei + 6.0 *
                                    ab[IDX_CH3CSII] / 59.0 / Hnuclei + 14.0 *
                                    ab[IDX_H2C7NII] / 100.0 / Hnuclei + 18.0 *
                                    ab[IDX_H2C9NII] / 124.0 / Hnuclei + 1.0 *
                                    ab[IDX_HONCII] / 43.0 / Hnuclei + 3.0 *
                                    ab[IDX_PCH3II] / 46.0 / Hnuclei + 30.0 *
                                    ab[IDX_C10H3II] / 123.0 / Hnuclei + 18.0 *
                                    ab[IDX_C2H5CNHII] / 56.0 / Hnuclei + 4.0 *
                                    ab[IDX_CH2NH2II] / 30.0 / Hnuclei + 10.0 *
                                    ab[IDX_CH2OHCH2OII] / 61.0 / Hnuclei + 24.0
                                    * ab[IDX_CH3C5NHII] / 90.0 / Hnuclei + 32.0
                                    * ab[IDX_CH3C7NHII] / 114.0 / Hnuclei + 10.0
                                    * ab[IDX_GC2H5I] / 29.0 / Hnuclei + 6.0 *
                                    ab[IDX_GCOOCH3I] / 59.0 / Hnuclei + 1.0 *
                                    ab[IDX_GHCNI] / 27.0 / Hnuclei + 2.0 *
                                    ab[IDX_H2CNI] / 28.0 / Hnuclei + 15.0 *
                                    ab[IDX_H3C5NII] / 77.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCNOII] / 43.0 / Hnuclei + 8.0 *
                                    ab[IDX_CH3COOHII] / 60.0 / Hnuclei + 27.0 *
                                    ab[IDX_H3C9NII] / 125.0 / Hnuclei + 10.0 *
                                    ab[IDX_H5C2O2II] / 61.0 / Hnuclei + 2.0 *
                                    ab[IDX_HCOOHII] / 46.0 / Hnuclei + 10.0 *
                                    ab[IDX_CH3COOH2II] / 61.0 / Hnuclei + 12.0 *
                                    ab[IDX_CH3OCH3II] / 46.0 / Hnuclei + 10.0 *
                                    ab[IDX_GC10HI] / 121.0 / Hnuclei + 8.0 *
                                    ab[IDX_GC8HI] / 97.0 / Hnuclei + 6.0 *
                                    ab[IDX_GCH2OHCOI] / 59.0 / Hnuclei + 1.0 *
                                    ab[IDX_GCOOHI] / 45.0 / Hnuclei + 21.0 *
                                    ab[IDX_H3C7NII] / 101.0 / Hnuclei + 2.0 *
                                    ab[IDX_NH2CNI] / 42.0 / Hnuclei + 16.0 *
                                    ab[IDX_CH3C3NHII] / 66.0 / Hnuclei + 14.0 *
                                    ab[IDX_CH3OCH4II] / 47.0 / Hnuclei + 8.0 *
                                    ab[IDX_GC2H4I] / 28.0 / Hnuclei + 9.0 *
                                    ab[IDX_GC9HI] / 109.0 / Hnuclei + 7.0 *
                                    ab[IDX_HC7NII] / 99.0 / Hnuclei + 9.0 *
                                    ab[IDX_HC9NII] / 123.0 / Hnuclei + 1.0 *
                                    ab[IDX_HNCOII] / 43.0 / Hnuclei + 3.0 *
                                    ab[IDX_PC3HII] / 68.0 / Hnuclei + 4.0 *
                                    ab[IDX_PCH4II] / 47.0 / Hnuclei + 15.0 *
                                    ab[IDX_C2H5CNI] / 55.0 / Hnuclei + 14.0 *
                                    ab[IDX_C2H5OH2II] / 47.0 / Hnuclei + 14.0 *
                                    ab[IDX_C2H7II] / 31.0 / Hnuclei + 12.0 *
                                    ab[IDX_CH2CHCNHII] / 54.0 / Hnuclei + 6.0 *
                                    ab[IDX_COOCH3I] / 59.0 / Hnuclei + 8.0 *
                                    ab[IDX_HCOOCH3II] / 60.0 / Hnuclei + 4.0 *
                                    ab[IDX_PC2H2II] / 57.0 / Hnuclei + 6.0 *
                                    ab[IDX_SiC3H2II] / 66.0 / Hnuclei + 4.0 *
                                    ab[IDX_SiC4HII] / 77.0 / Hnuclei + 1.0 *
                                    ab[IDX_SiNCHII] / 55.0 / Hnuclei + 20.0 *
                                    ab[IDX_C4H5II] / 53.0 / Hnuclei + 28.0 *
                                    ab[IDX_C4H7II] / 55.0 / Hnuclei + 8.0 *
                                    ab[IDX_CH2OHCHOII] / 60.0 / Hnuclei + 8.0 *
                                    ab[IDX_CH3CHOII] / 44.0 / Hnuclei + 8.0 *
                                    ab[IDX_GCH2OHCHOI] / 60.0 / Hnuclei + 2.0 *
                                    ab[IDX_GHCOOHI] / 46.0 / Hnuclei + 1.0 *
                                    ab[IDX_HOCSII] / 61.0 / Hnuclei + 4.0 *
                                    ab[IDX_PC4HII] / 80.0 / Hnuclei + 4.0 *
                                    ab[IDX_SiCH4II] / 44.0 / Hnuclei + 20.0 *
                                    ab[IDX_C10H2II] / 122.0 / Hnuclei + 6.0 *
                                    ab[IDX_CH3CNII] / 41.0 / Hnuclei + 4.0 *
                                    ab[IDX_GC2H2I] / 26.0 / Hnuclei + 7.0 *
                                    ab[IDX_GC7HI] / 85.0 / Hnuclei + 6.0 *
                                    ab[IDX_GCH3COI] / 43.0 / Hnuclei + 8.0 *
                                    ab[IDX_GHCOOCH3I] / 60.0 / Hnuclei + 4.0 *
                                    ab[IDX_HC4SII] / 81.0 / Hnuclei + 10.0 *
                                    ab[IDX_HC5NHII] / 76.0 / Hnuclei + 6.0 *
                                    ab[IDX_SiC2H3II] / 55.0 / Hnuclei + 3.0 *
                                    ab[IDX_SiC3HI] / 65.0 / Hnuclei + 3.0 *
                                    ab[IDX_SiCH3II] / 43.0 / Hnuclei + 16.0 *
                                    ab[IDX_C4H4II] / 52.0 / Hnuclei + 4.0 *
                                    ab[IDX_CH2CNII] / 40.0 / Hnuclei + 18.0 *
                                    ab[IDX_CH3COCH3II] / 58.0 / Hnuclei + 3.0 *
                                    ab[IDX_GCH2OHI] / 31.0 / Hnuclei + 4.0 *
                                    ab[IDX_GCH3OHI] / 32.0 / Hnuclei + 2.0 *
                                    ab[IDX_HC2PII] / 56.0 / Hnuclei + 5.0 *
                                    ab[IDX_HC5NII] / 75.0 / Hnuclei + 1.0 *
                                    ab[IDX_HONCI] / 43.0 / Hnuclei + 2.0 *
                                    ab[IDX_PCH2II] / 45.0 / Hnuclei + 6.0 *
                                    ab[IDX_CH2OHCOI] / 59.0 / Hnuclei + 18.0 *
                                    ab[IDX_CH3C5NI] / 89.0 / Hnuclei + 24.0 *
                                    ab[IDX_CH3C7NI] / 113.0 / Hnuclei + 3.0 *
                                    ab[IDX_GC3HI] / 37.0 / Hnuclei + 4.0 *
                                    ab[IDX_GC4HI] / 49.0 / Hnuclei + 5.0 *
                                    ab[IDX_GC5HI] / 61.0 / Hnuclei + 6.0 *
                                    ab[IDX_GC6HI] / 73.0 / Hnuclei + 2.0 *
                                    ab[IDX_H2NCII] / 28.0 / Hnuclei + 3.0 *
                                    ab[IDX_H3CSII] / 47.0 / Hnuclei + 2.0 *
                                    ab[IDX_HC2OII] / 41.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCPII] / 44.0 / Hnuclei + 6.0 *
                                    ab[IDX_GC2H3I] / 27.0 / Hnuclei + 3.0 *
                                    ab[IDX_GCH3OI] / 31.0 / Hnuclei + 3.0 *
                                    ab[IDX_HC3SII] / 69.0 / Hnuclei + 3.0 *
                                    ab[IDX_HCOOH2II] / 47.0 / Hnuclei + 4.0 *
                                    ab[IDX_SiC2H2I] / 54.0 / Hnuclei + 2.0 *
                                    ab[IDX_C2NHII] / 39.0 / Hnuclei + 18.0 *
                                    ab[IDX_C3H6II] / 42.0 / Hnuclei + 42.0 *
                                    ab[IDX_C6H7II] / 79.0 / Hnuclei + 32.0 *
                                    ab[IDX_C8H4II] / 100.0 / Hnuclei + 8.0 *
                                    ab[IDX_CH2OHCHOI] / 60.0 / Hnuclei + 3.0 *
                                    ab[IDX_HC3OII] / 53.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCNOI] / 43.0 / Hnuclei + 3.0 *
                                    ab[IDX_SiC3HII] / 65.0 / Hnuclei + 3.0 *
                                    ab[IDX_SiCH3I] / 43.0 / Hnuclei + 21.0 *
                                    ab[IDX_C3H7II] / 43.0 / Hnuclei + 36.0 *
                                    ab[IDX_C9H4II] / 112.0 / Hnuclei + 4.0 *
                                    ab[IDX_CH2COII] / 42.0 / Hnuclei + 3.0 *
                                    ab[IDX_CH2PHI] / 46.0 / Hnuclei + 12.0 *
                                    ab[IDX_CH3C3NI] / 65.0 / Hnuclei + 8.0 *
                                    ab[IDX_GCH3CHOI] / 44.0 / Hnuclei + 2.0 *
                                    ab[IDX_GH2COI] / 30.0 / Hnuclei + 1.0 *
                                    ab[IDX_HOCNI] / 43.0 / Hnuclei + 4.0 *
                                    ab[IDX_SiC2H2II] / 54.0 / Hnuclei + 24.0 *
                                    ab[IDX_C6H4II] / 76.0 / Hnuclei + 6.0 *
                                    ab[IDX_CH3COI] / 43.0 / Hnuclei + 21.0 *
                                    ab[IDX_CH3COCH4II] / 59.0 / Hnuclei + 4.0 *
                                    ab[IDX_CH3OHII] / 32.0 / Hnuclei + 1.0 *
                                    ab[IDX_COOHI] / 45.0 / Hnuclei + 28.0 *
                                    ab[IDX_C7H4II] / 88.0 / Hnuclei + 35.0 *
                                    ab[IDX_C7H5II] / 89.0 / Hnuclei + 28.0 *
                                    ab[IDX_CH3C6HI] / 88.0 / Hnuclei + 2.0 *
                                    ab[IDX_H2CSII] / 46.0 / Hnuclei + 9.0 *
                                    ab[IDX_HC9NI] / 123.0 / Hnuclei + 8.0 *
                                    ab[IDX_HCOOCH3I] / 60.0 / Hnuclei + 2.0 *
                                    ab[IDX_NCCNHII] / 53.0 / Hnuclei + 2.0 *
                                    ab[IDX_SiC2HI] / 53.0 / Hnuclei + 4.0 *
                                    ab[IDX_CH2CNI] / 40.0 / Hnuclei + 20.0 *
                                    ab[IDX_CH3C4HII] / 64.0 / Hnuclei + 4.0 *
                                    ab[IDX_GCH4I] / 16.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCPI] / 44.0 / Hnuclei + 1.0 *
                                    ab[IDX_HNCOI] / 43.0 / Hnuclei + 20.0 *
                                    ab[IDX_CH3C4HI] / 64.0 / Hnuclei + 8.0 *
                                    ab[IDX_CH3COOHI] / 60.0 / Hnuclei + 12.0 *
                                    ab[IDX_CH3OCH3I] / 46.0 / Hnuclei + 2.0 *
                                    ab[IDX_GC2HI] / 25.0 / Hnuclei + 2.0 *
                                    ab[IDX_H2CSI] / 46.0 / Hnuclei + 30.0 *
                                    ab[IDX_C6H5II] / 77.0 / Hnuclei + 3.0 *
                                    ab[IDX_CH2NHI] / 29.0 / Hnuclei + 3.0 *
                                    ab[IDX_CH3OI] / 31.0 / Hnuclei + 1.0 *
                                    ab[IDX_GHCOI] / 29.0 / Hnuclei + 2.0 *
                                    ab[IDX_HC2PI] / 56.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCSiI] / 41.0 / Hnuclei + 2.0 *
                                    ab[IDX_SiCH2I] / 42.0 / Hnuclei + 10.0 *
                                    ab[IDX_C10HII] / 121.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCSiII] / 41.0 / Hnuclei + 12.0 *
                                    ab[IDX_C2H6II] / 30.0 / Hnuclei + 8.0 *
                                    ab[IDX_C8HII] / 97.0 / Hnuclei + 3.0 *
                                    ab[IDX_CH2OHI] / 31.0 / Hnuclei + 25.0 *
                                    ab[IDX_C5H5II] / 65.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCSI] / 45.0 / Hnuclei + 2.0 *
                                    ab[IDX_SiC2HII] / 53.0 / Hnuclei + 6.0 *
                                    ab[IDX_C6HII] / 73.0 / Hnuclei + 18.0 *
                                    ab[IDX_C9H2I] / 110.0 / Hnuclei + 9.0 *
                                    ab[IDX_CH2CHCNI] / 53.0 / Hnuclei + 10.0 *
                                    ab[IDX_CH3CHOHII] / 45.0 / Hnuclei + 8.0 *
                                    ab[IDX_CH3CNHII] / 42.0 / Hnuclei + 6.0 *
                                    ab[IDX_CH3COII] / 43.0 / Hnuclei + 7.0 *
                                    ab[IDX_HC7NI] / 99.0 / Hnuclei + 36.0 *
                                    ab[IDX_C6H6I] / 78.0 / Hnuclei + 2.0 *
                                    ab[IDX_HCOOHI] / 46.0 / Hnuclei + 20.0 *
                                    ab[IDX_C10H2I] / 122.0 / Hnuclei + 12.0 *
                                    ab[IDX_C2H5OHI] / 46.0 / Hnuclei + 5.0 *
                                    ab[IDX_CH3OH2II] / 33.0 / Hnuclei + 18.0 *
                                    ab[IDX_C6H3II] / 75.0 / Hnuclei + 24.0 *
                                    ab[IDX_C8H3II] / 99.0 / Hnuclei + 9.0 *
                                    ab[IDX_C9HII] / 109.0 / Hnuclei + 27.0 *
                                    ab[IDX_C9H3II] / 111.0 / Hnuclei + 4.0 *
                                    ab[IDX_CH2COI] / 42.0 / Hnuclei + 3.0 *
                                    ab[IDX_HNC3I] / 51.0 / Hnuclei + 2.0 *
                                    ab[IDX_SiCH2II] / 42.0 / Hnuclei + 18.0 *
                                    ab[IDX_CH3COCH3I] / 58.0 / Hnuclei + 3.0 *
                                    ab[IDX_HC3NII] / 51.0 / Hnuclei + 16.0 *
                                    ab[IDX_C8H2I] / 98.0 / Hnuclei + 5.0 *
                                    ab[IDX_HC5NI] / 75.0 / Hnuclei + 7.0 *
                                    ab[IDX_C7HII] / 85.0 / Hnuclei + 18.0 *
                                    ab[IDX_C9H2II] / 110.0 / Hnuclei + 14.0 *
                                    ab[IDX_C7H2I] / 86.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCSII] / 45.0 / Hnuclei + 12.0 *
                                    ab[IDX_C3H4II] / 40.0 / Hnuclei + 16.0 *
                                    ab[IDX_C8H2II] / 98.0 / Hnuclei + 2.0 *
                                    ab[IDX_GCH2I] / 14.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCO2II] / 45.0 / Hnuclei + 15.0 *
                                    ab[IDX_C5H3II] / 63.0 / Hnuclei + 6.0 *
                                    ab[IDX_HC3NHII] / 52.0 / Hnuclei + 5.0 *
                                    ab[IDX_C5HII] / 61.0 / Hnuclei + 8.0 *
                                    ab[IDX_CH3CHOI] / 44.0 / Hnuclei + 15.0 *
                                    ab[IDX_C3H5II] / 41.0 / Hnuclei + 21.0 *
                                    ab[IDX_C7H3II] / 87.0 / Hnuclei + 6.0 *
                                    ab[IDX_CH3CNI] / 41.0 / Hnuclei + 14.0 *
                                    ab[IDX_C7H2II] / 86.0 / Hnuclei + 12.0 *
                                    ab[IDX_C6H2II] / 74.0 / Hnuclei + 4.0 *
                                    ab[IDX_CH4II] / 16.0 / Hnuclei + 6.0 *
                                    ab[IDX_H2CCCI] / 38.0 / Hnuclei + 12.0 *
                                    ab[IDX_C6H2I] / 74.0 / Hnuclei + 10.0 *
                                    ab[IDX_C2H5II] / 29.0 / Hnuclei + 10.0 *
                                    ab[IDX_C5H2I] / 62.0 / Hnuclei + 10.0 *
                                    ab[IDX_C5H2II] / 62.0 / Hnuclei + 12.0 *
                                    ab[IDX_CH2CCH2I] / 40.0 / Hnuclei + 12.0 *
                                    ab[IDX_C2H6I] / 30.0 / Hnuclei + 6.0 *
                                    ab[IDX_C3H2I] / 38.0 / Hnuclei + 10.0 *
                                    ab[IDX_C2H5I] / 29.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCNII] / 27.0 / Hnuclei + 18.0 *
                                    ab[IDX_CH3CHCH2I] / 42.0 / Hnuclei + 4.0 *
                                    ab[IDX_C4HII] / 49.0 / Hnuclei + 1.0 *
                                    ab[IDX_CHM] / 13.0 / Hnuclei + 3.0 *
                                    ab[IDX_GCH3I] / 15.0 / Hnuclei + 2.0 *
                                    ab[IDX_C2HM] / 25.0 / Hnuclei + 4.0 *
                                    ab[IDX_C4HM] / 49.0 / Hnuclei + 12.0 *
                                    ab[IDX_CH3CCHI] / 40.0 / Hnuclei + 5.0 *
                                    ab[IDX_CH5II] / 17.0 / Hnuclei + 2.0 *
                                    ab[IDX_HC2SII] / 57.0 / Hnuclei + 10.0 *
                                    ab[IDX_C10HM] / 121.0 / Hnuclei + 3.0 *
                                    ab[IDX_C3HM] / 37.0 / Hnuclei + 6.0 *
                                    ab[IDX_C6HM] / 73.0 / Hnuclei + 8.0 *
                                    ab[IDX_C8HM] / 97.0 / Hnuclei + 7.0 *
                                    ab[IDX_C7HM] / 85.0 / Hnuclei + 9.0 *
                                    ab[IDX_C9HM] / 109.0 / Hnuclei + 5.0 *
                                    ab[IDX_C5HM] / 61.0 / Hnuclei + 4.0 *
                                    ab[IDX_CH3OHI] / 32.0 / Hnuclei + 3.0 *
                                    ab[IDX_HC3NI] / 51.0 / Hnuclei + 3.0 *
                                    ab[IDX_C3HII] / 37.0 / Hnuclei + 9.0 *
                                    ab[IDX_C3H3II] / 39.0 / Hnuclei + 3.0 *
                                    ab[IDX_H3COII] / 31.0 / Hnuclei + 1.0 *
                                    ab[IDX_GCHI] / 13.0 / Hnuclei + 12.0 *
                                    ab[IDX_C4H3I] / 51.0 / Hnuclei + 2.0 *
                                    ab[IDX_C2HII] / 25.0 / Hnuclei + 6.0 *
                                    ab[IDX_C3H2II] / 38.0 / Hnuclei + 2.0 *
                                    ab[IDX_CH2II] / 14.0 / Hnuclei + 1.0 *
                                    ab[IDX_CHII] / 13.0 / Hnuclei + 10.0 *
                                    ab[IDX_C10HI] / 121.0 / Hnuclei + 9.0 *
                                    ab[IDX_C9HI] / 109.0 / Hnuclei + 8.0 *
                                    ab[IDX_C8HI] / 97.0 / Hnuclei + 7.0 *
                                    ab[IDX_C7HI] / 85.0 / Hnuclei + 8.0 *
                                    ab[IDX_C2H4II] / 28.0 / Hnuclei + 12.0 *
                                    ab[IDX_C4H3II] / 51.0 / Hnuclei + 9.0 *
                                    ab[IDX_CH2CCHI] / 39.0 / Hnuclei + 9.0 *
                                    ab[IDX_CH2CCHII] / 39.0 / Hnuclei + 6.0 *
                                    ab[IDX_C6HI] / 73.0 / Hnuclei + 5.0 *
                                    ab[IDX_C5HI] / 61.0 / Hnuclei + 3.0 *
                                    ab[IDX_C3HI] / 37.0 / Hnuclei + 2.0 *
                                    ab[IDX_H2COII] / 30.0 / Hnuclei + 4.0 *
                                    ab[IDX_C4HI] / 49.0 / Hnuclei + 8.0 *
                                    ab[IDX_C4H2I] / 50.0 / Hnuclei + 8.0 *
                                    ab[IDX_C4H2II] / 50.0 / Hnuclei + 2.0 *
                                    ab[IDX_HCNHII] / 28.0 / Hnuclei + 2.0 *
                                    ab[IDX_CH2I] / 14.0 / Hnuclei + 1.0 *
                                    ab[IDX_HNCI] / 27.0 / Hnuclei + 8.0 *
                                    ab[IDX_C2H4I] / 28.0 / Hnuclei + 6.0 *
                                    ab[IDX_C2H3I] / 27.0 / Hnuclei + 6.0 *
                                    ab[IDX_C2H3II] / 27.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCOI] / 29.0 / Hnuclei + 4.0 *
                                    ab[IDX_CH4I] / 16.0 / Hnuclei + 2.0 *
                                    ab[IDX_C2HI] / 25.0 / Hnuclei + 2.0 *
                                    ab[IDX_H2COI] / 30.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCNI] / 27.0 / Hnuclei + 1.0 *
                                    ab[IDX_CHI] / 13.0 / Hnuclei + 4.0 *
                                    ab[IDX_C2H2II] / 26.0 / Hnuclei + 3.0 *
                                    ab[IDX_CH3II] / 15.0 / Hnuclei + 4.0 *
                                    ab[IDX_C2H2I] / 26.0 / Hnuclei + 3.0 *
                                    ab[IDX_CH3I] / 15.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCOII] / 29.0 / Hnuclei;
    IJth(A, IDX_ELEM_GRAIN, IDX_ELEM_F) = 0.0;
    IJth(A, IDX_ELEM_GRAIN, IDX_ELEM_Cl) = 0.0;
    IJth(A, IDX_ELEM_GRAIN, IDX_ELEM_P) = 0.0;
    IJth(A, IDX_ELEM_GRAIN, IDX_ELEM_Fe) = 0.0;
    IJth(A, IDX_ELEM_GRAIN, IDX_ELEM_Mg) = 0.0;
    IJth(A, IDX_ELEM_GRAIN, IDX_ELEM_Na) = 0.0;
    IJth(A, IDX_ELEM_GRAIN, IDX_ELEM_Si) = 0.0;
    IJth(A, IDX_ELEM_GRAIN, IDX_ELEM_S) = 0.0;
    IJth(A, IDX_ELEM_GRAIN, IDX_ELEM_N) = 0.0;
    IJth(A, IDX_ELEM_GRAIN, IDX_ELEM_O) = 0.0;
    IJth(A, IDX_ELEM_GRAIN, IDX_ELEM_He) = 0.0;
    IJth(A, IDX_ELEM_GRAIN, IDX_ELEM_C) = 0.0;
    IJth(A, IDX_ELEM_GRAIN, IDX_ELEM_GRAIN) = 0.0 + 0.0 * ab[IDX_GRAINM] / 0.0 / Hnuclei +
                                    0.0 * ab[IDX_GRAIN0I] / 0.0 / Hnuclei;
    IJth(A, IDX_ELEM_GRAIN, IDX_ELEM_H) = 0.0;
    IJth(A, IDX_ELEM_H, IDX_ELEM_F) = 0.0 + 19.0 * ab[IDX_GHFI] / 20.0 / Hnuclei +
                                    19.0 * ab[IDX_HFII] / 20.0 / Hnuclei + 38.0
                                    * ab[IDX_H2FII] / 21.0 / Hnuclei + 19.0 *
                                    ab[IDX_HFI] / 20.0 / Hnuclei;
    IJth(A, IDX_ELEM_H, IDX_ELEM_Cl) = 0.0 + 35.0 * ab[IDX_GHClI] / 36.0 / Hnuclei
                                    + 70.0 * ab[IDX_H2CClII] / 49.0 / Hnuclei +
                                    35.0 * ab[IDX_HClII] / 36.0 / Hnuclei + 70.0
                                    * ab[IDX_H2ClII] / 37.0 / Hnuclei + 35.0 *
                                    ab[IDX_HClI] / 36.0 / Hnuclei;
    IJth(A, IDX_ELEM_H, IDX_ELEM_P) = 0.0 + 31.0 * ab[IDX_GHC2PI] / 56.0 / Hnuclei
                                    + 31.0 * ab[IDX_GHPOI] / 48.0 / Hnuclei +
                                    93.0 * ab[IDX_GCH2PHI] / 46.0 / Hnuclei +
                                    31.0 * ab[IDX_GHCPI] / 44.0 / Hnuclei + 62.0
                                    * ab[IDX_GPH2I] / 33.0 / Hnuclei + 31.0 *
                                    ab[IDX_GPHI] / 32.0 / Hnuclei + 124.0 *
                                    ab[IDX_PC2H4II] / 59.0 / Hnuclei + 93.0 *
                                    ab[IDX_PC2H3II] / 58.0 / Hnuclei + 93.0 *
                                    ab[IDX_PNH3II] / 48.0 / Hnuclei + 93.0 *
                                    ab[IDX_PCH3II] / 46.0 / Hnuclei + 62.0 *
                                    ab[IDX_PNH2II] / 47.0 / Hnuclei + 31.0 *
                                    ab[IDX_HPNII] / 46.0 / Hnuclei + 31.0 *
                                    ab[IDX_PC3HII] / 68.0 / Hnuclei + 124.0 *
                                    ab[IDX_PCH4II] / 47.0 / Hnuclei + 62.0 *
                                    ab[IDX_H2POII] / 49.0 / Hnuclei + 62.0 *
                                    ab[IDX_PC2H2II] / 57.0 / Hnuclei + 31.0 *
                                    ab[IDX_PC4HII] / 80.0 / Hnuclei + 93.0 *
                                    ab[IDX_PH3II] / 34.0 / Hnuclei + 31.0 *
                                    ab[IDX_HC2PII] / 56.0 / Hnuclei + 62.0 *
                                    ab[IDX_PCH2II] / 45.0 / Hnuclei + 31.0 *
                                    ab[IDX_HCPII] / 44.0 / Hnuclei + 31.0 *
                                    ab[IDX_HPOII] / 48.0 / Hnuclei + 93.0 *
                                    ab[IDX_CH2PHI] / 46.0 / Hnuclei + 31.0 *
                                    ab[IDX_HPOI] / 48.0 / Hnuclei + 62.0 *
                                    ab[IDX_PH2I] / 33.0 / Hnuclei + 31.0 *
                                    ab[IDX_HCPI] / 44.0 / Hnuclei + 62.0 *
                                    ab[IDX_PH2II] / 33.0 / Hnuclei + 31.0 *
                                    ab[IDX_HC2PI] / 56.0 / Hnuclei + 31.0 *
                                    ab[IDX_PHI] / 32.0 / Hnuclei + 31.0 *
                                    ab[IDX_PHII] / 32.0 / Hnuclei;
    IJth(A, IDX_ELEM_H, IDX_ELEM_Fe) = 0.0;
    IJth(A, IDX_ELEM_H, IDX_ELEM_Mg) = 0.0;
    IJth(A, IDX_ELEM_H, IDX_ELEM_Na) = 0.0;
    IJth(A, IDX_ELEM_H, IDX_ELEM_Si) = 0.0 + 28.0 * ab[IDX_GHNSiI] / 43.0 / Hnuclei
                                    + 28.0 * ab[IDX_GSiC3HI] / 65.0 / Hnuclei +
                                    84.0 * ab[IDX_GSiCH3I] / 43.0 / Hnuclei +
                                    56.0 * ab[IDX_GH2SiOI] / 46.0 / Hnuclei +
                                    28.0 * ab[IDX_GSiC2HI] / 53.0 / Hnuclei +
                                    56.0 * ab[IDX_GSiC2H2I] / 54.0 / Hnuclei +
                                    28.0 * ab[IDX_GHCSiI] / 41.0 / Hnuclei +
                                    112.0 * ab[IDX_GSiH4I] / 32.0 / Hnuclei +
                                    84.0 * ab[IDX_GSiH3I] / 31.0 / Hnuclei +
                                    56.0 * ab[IDX_GSiCH2I] / 42.0 / Hnuclei +
                                    28.0 * ab[IDX_GSiHI] / 29.0 / Hnuclei + 56.0
                                    * ab[IDX_GSiH2I] / 30.0 / Hnuclei + 28.0 *
                                    ab[IDX_HSiO2II] / 61.0 / Hnuclei + 56.0 *
                                    ab[IDX_H2SiOII] / 46.0 / Hnuclei + 84.0 *
                                    ab[IDX_H3SiOII] / 47.0 / Hnuclei + 112.0 *
                                    ab[IDX_SiH4II] / 32.0 / Hnuclei + 56.0 *
                                    ab[IDX_SiC3H2II] / 66.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiC4HII] / 77.0 / Hnuclei + 140.0 *
                                    ab[IDX_SiH5II] / 33.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiNCHII] / 55.0 / Hnuclei + 28.0 *
                                    ab[IDX_HNSiII] / 43.0 / Hnuclei + 112.0 *
                                    ab[IDX_SiCH4II] / 44.0 / Hnuclei + 84.0 *
                                    ab[IDX_SiC2H3II] / 55.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiC3HI] / 65.0 / Hnuclei + 84.0 *
                                    ab[IDX_SiCH3II] / 43.0 / Hnuclei + 56.0 *
                                    ab[IDX_SiNH2II] / 44.0 / Hnuclei + 28.0 *
                                    ab[IDX_HSiSII] / 61.0 / Hnuclei + 56.0 *
                                    ab[IDX_H2SiOI] / 46.0 / Hnuclei + 28.0 *
                                    ab[IDX_HNSiI] / 43.0 / Hnuclei + 56.0 *
                                    ab[IDX_SiC2H2I] / 54.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiC3HII] / 65.0 / Hnuclei + 84.0 *
                                    ab[IDX_SiCH3I] / 43.0 / Hnuclei + 56.0 *
                                    ab[IDX_SiC2H2II] / 54.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiC2HI] / 53.0 / Hnuclei + 28.0 *
                                    ab[IDX_HCSiI] / 41.0 / Hnuclei + 56.0 *
                                    ab[IDX_SiCH2I] / 42.0 / Hnuclei + 28.0 *
                                    ab[IDX_HCSiII] / 41.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiC2HII] / 53.0 / Hnuclei + 56.0 *
                                    ab[IDX_SiH2I] / 30.0 / Hnuclei + 56.0 *
                                    ab[IDX_SiH2II] / 30.0 / Hnuclei + 84.0 *
                                    ab[IDX_SiH3II] / 31.0 / Hnuclei + 84.0 *
                                    ab[IDX_SiH3I] / 31.0 / Hnuclei + 56.0 *
                                    ab[IDX_SiCH2II] / 42.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiHII] / 29.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiHI] / 29.0 / Hnuclei + 112.0 *
                                    ab[IDX_SiH4I] / 32.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiOHII] / 45.0 / Hnuclei;
    IJth(A, IDX_ELEM_H, IDX_ELEM_S) = 0.0 + 128.0 * ab[IDX_GH2S2I] / 66.0 /
                                    Hnuclei + 64.0 * ab[IDX_GH2SI] / 34.0 /
                                    Hnuclei + 64.0 * ab[IDX_GHS2I] / 65.0 /
                                    Hnuclei + 64.0 * ab[IDX_GH2CSI] / 46.0 /
                                    Hnuclei + 32.0 * ab[IDX_GHCSI] / 45.0 /
                                    Hnuclei + 32.0 * ab[IDX_HNSII] / 47.0 /
                                    Hnuclei + 32.0 * ab[IDX_HSOII] / 49.0 /
                                    Hnuclei + 96.0 * ab[IDX_CH3CSII] / 59.0 /
                                    Hnuclei + 192.0 * ab[IDX_H3S2II] / 67.0 /
                                    Hnuclei + 32.0 * ab[IDX_HSO2II] / 65.0 /
                                    Hnuclei + 32.0 * ab[IDX_GHSI] / 33.0 /
                                    Hnuclei + 128.0 * ab[IDX_H2S2I] / 66.0 /
                                    Hnuclei + 32.0 * ab[IDX_HOCSII] / 61.0 /
                                    Hnuclei + 32.0 * ab[IDX_HC4SII] / 81.0 /
                                    Hnuclei + 128.0 * ab[IDX_H2S2II] / 66.0 /
                                    Hnuclei + 32.0 * ab[IDX_HSiSII] / 61.0 /
                                    Hnuclei + 96.0 * ab[IDX_H3CSII] / 47.0 /
                                    Hnuclei + 32.0 * ab[IDX_HC3SII] / 69.0 /
                                    Hnuclei + 64.0 * ab[IDX_HS2II] / 65.0 /
                                    Hnuclei + 64.0 * ab[IDX_HS2I] / 65.0 /
                                    Hnuclei + 64.0 * ab[IDX_H2CSII] / 46.0 /
                                    Hnuclei + 64.0 * ab[IDX_H2CSI] / 46.0 /
                                    Hnuclei + 32.0 * ab[IDX_HCSI] / 45.0 /
                                    Hnuclei + 32.0 * ab[IDX_HCSII] / 45.0 /
                                    Hnuclei + 96.0 * ab[IDX_H3SII] / 35.0 /
                                    Hnuclei + 32.0 * ab[IDX_HSII] / 33.0 /
                                    Hnuclei + 32.0 * ab[IDX_HC2SII] / 57.0 /
                                    Hnuclei + 32.0 * ab[IDX_HSI] / 33.0 /
                                    Hnuclei + 64.0 * ab[IDX_H2SII] / 34.0 /
                                    Hnuclei + 64.0 * ab[IDX_H2SI] / 34.0 /
                                    Hnuclei;
    IJth(A, IDX_ELEM_H, IDX_ELEM_N) = 0.0 + 42.0 * ab[IDX_GCH3C3NI] / 65.0 /
                                    Hnuclei + 42.0 * ab[IDX_GCH3C5NI] / 89.0 /
                                    Hnuclei + 42.0 * ab[IDX_GCH3C7NI] / 113.0 /
                                    Hnuclei + 14.0 * ab[IDX_GHNSiI] / 43.0 /
                                    Hnuclei + 56.0 * ab[IDX_GNH2CNI] / 42.0 /
                                    Hnuclei + 56.0 * ab[IDX_C2H4CNI] / 54.0 /
                                    Hnuclei + 14.0 * ab[IDX_GHCNOI] / 43.0 /
                                    Hnuclei + 14.0 * ab[IDX_GHNC3I] / 51.0 /
                                    Hnuclei + 14.0 * ab[IDX_GHNCOI] / 43.0 /
                                    Hnuclei + 14.0 * ab[IDX_GHOCNI] / 43.0 /
                                    Hnuclei + 14.0 * ab[IDX_GHONCI] / 43.0 /
                                    Hnuclei + 84.0 * ab[IDX_NCCNCH3II] / 67.0 /
                                    Hnuclei + 56.0 * ab[IDX_GC2H4CNI] / 54.0 /
                                    Hnuclei + 28.0 * ab[IDX_GCH2CNI] / 40.0 /
                                    Hnuclei + 42.0 * ab[IDX_GCH3CNI] / 41.0 /
                                    Hnuclei + 28.0 * ab[IDX_GH2CNI] / 28.0 /
                                    Hnuclei + 14.0 * ab[IDX_GHC5NI] / 75.0 /
                                    Hnuclei + 14.0 * ab[IDX_GHC7NI] / 99.0 /
                                    Hnuclei + 14.0 * ab[IDX_GHC9NI] / 123.0 /
                                    Hnuclei + 14.0 * ab[IDX_GHCCNI] / 39.0 /
                                    Hnuclei + 14.0 * ab[IDX_HCCNI] / 39.0 /
                                    Hnuclei + 42.0 * ab[IDX_CH3C3NII] / 65.0 /
                                    Hnuclei + 42.0 * ab[IDX_GCH2CHCNI] / 53.0 /
                                    Hnuclei + 14.0 * ab[IDX_GHC3NI] / 51.0 /
                                    Hnuclei + 42.0 * ab[IDX_CH2CHCNII] / 53.0 /
                                    Hnuclei + 70.0 * ab[IDX_GC2H5CNI] / 55.0 /
                                    Hnuclei + 28.0 * ab[IDX_H2CNOII] / 44.0 /
                                    Hnuclei + 28.0 * ab[IDX_H2NCOII] / 44.0 /
                                    Hnuclei + 28.0 * ab[IDX_H2OCNII] / 44.0 /
                                    Hnuclei + 56.0 * ab[IDX_CH3NHII] / 30.0 /
                                    Hnuclei + 42.0 * ab[IDX_GCH2NHI] / 29.0 /
                                    Hnuclei + 14.0 * ab[IDX_GHNCI] / 27.0 /
                                    Hnuclei + 28.0 * ab[IDX_H2C4NII] / 64.0 /
                                    Hnuclei + 28.0 * ab[IDX_H2NOII] / 32.0 /
                                    Hnuclei + 14.0 * ab[IDX_HNSII] / 47.0 /
                                    Hnuclei + 84.0 * ab[IDX_NH2CNHII] / 43.0 /
                                    Hnuclei + 14.0 * ab[IDX_GHNOI] / 31.0 /
                                    Hnuclei + 14.0 * ab[IDX_HC4NII] / 63.0 /
                                    Hnuclei + 28.0 * ab[IDX_HCNOHII] / 44.0 /
                                    Hnuclei + 28.0 * ab[IDX_HNCOHII] / 44.0 /
                                    Hnuclei + 14.0 * ab[IDX_HOCNII] / 43.0 /
                                    Hnuclei + 42.0 * ab[IDX_PNH3II] / 48.0 /
                                    Hnuclei + 28.0 * ab[IDX_H2C7NII] / 100.0 /
                                    Hnuclei + 28.0 * ab[IDX_H2C9NII] / 124.0 /
                                    Hnuclei + 14.0 * ab[IDX_HONCII] / 43.0 /
                                    Hnuclei + 28.0 * ab[IDX_PNH2II] / 47.0 /
                                    Hnuclei + 84.0 * ab[IDX_C2H5CNHII] / 56.0 /
                                    Hnuclei + 56.0 * ab[IDX_CH2NH2II] / 30.0 /
                                    Hnuclei + 56.0 * ab[IDX_CH3C5NHII] / 90.0 /
                                    Hnuclei + 56.0 * ab[IDX_CH3C7NHII] / 114.0 /
                                    Hnuclei + 14.0 * ab[IDX_GHCNI] / 27.0 /
                                    Hnuclei + 28.0 * ab[IDX_H2CNI] / 28.0 /
                                    Hnuclei + 42.0 * ab[IDX_H3C5NII] / 77.0 /
                                    Hnuclei + 14.0 * ab[IDX_HCNOII] / 43.0 /
                                    Hnuclei + 42.0 * ab[IDX_GNH3I] / 17.0 /
                                    Hnuclei + 42.0 * ab[IDX_H3C9NII] / 125.0 /
                                    Hnuclei + 14.0 * ab[IDX_HPNII] / 46.0 /
                                    Hnuclei + 42.0 * ab[IDX_H3C7NII] / 101.0 /
                                    Hnuclei + 56.0 * ab[IDX_NH2CNI] / 42.0 /
                                    Hnuclei + 56.0 * ab[IDX_CH3C3NHII] / 66.0 /
                                    Hnuclei + 14.0 * ab[IDX_HC7NII] / 99.0 /
                                    Hnuclei + 14.0 * ab[IDX_HC9NII] / 123.0 /
                                    Hnuclei + 14.0 * ab[IDX_HNCOII] / 43.0 /
                                    Hnuclei + 70.0 * ab[IDX_C2H5CNI] / 55.0 /
                                    Hnuclei + 56.0 * ab[IDX_CH2CHCNHII] / 54.0 /
                                    Hnuclei + 28.0 * ab[IDX_HN2OII] / 45.0 /
                                    Hnuclei + 14.0 * ab[IDX_SiNCHII] / 55.0 /
                                    Hnuclei + 14.0 * ab[IDX_HNSiII] / 43.0 /
                                    Hnuclei + 42.0 * ab[IDX_CH3CNII] / 41.0 /
                                    Hnuclei + 28.0 * ab[IDX_HC5NHII] / 76.0 /
                                    Hnuclei + 28.0 * ab[IDX_SiNH2II] / 44.0 /
                                    Hnuclei + 28.0 * ab[IDX_CH2CNII] / 40.0 /
                                    Hnuclei + 14.0 * ab[IDX_HC5NII] / 75.0 /
                                    Hnuclei + 14.0 * ab[IDX_HONCI] / 43.0 /
                                    Hnuclei + 42.0 * ab[IDX_CH3C5NI] / 89.0 /
                                    Hnuclei + 42.0 * ab[IDX_CH3C7NI] / 113.0 /
                                    Hnuclei + 28.0 * ab[IDX_H2NCII] / 28.0 /
                                    Hnuclei + 14.0 * ab[IDX_HNSiI] / 43.0 /
                                    Hnuclei + 14.0 * ab[IDX_C2NHII] / 39.0 /
                                    Hnuclei + 14.0 * ab[IDX_HCNOI] / 43.0 /
                                    Hnuclei + 42.0 * ab[IDX_CH3C3NI] / 65.0 /
                                    Hnuclei + 14.0 * ab[IDX_HOCNI] / 43.0 /
                                    Hnuclei + 14.0 * ab[IDX_HC9NI] / 123.0 /
                                    Hnuclei + 28.0 * ab[IDX_NCCNHII] / 53.0 /
                                    Hnuclei + 28.0 * ab[IDX_CH2CNI] / 40.0 /
                                    Hnuclei + 14.0 * ab[IDX_HNCOI] / 43.0 /
                                    Hnuclei + 28.0 * ab[IDX_GNH2I] / 16.0 /
                                    Hnuclei + 42.0 * ab[IDX_CH2NHI] / 29.0 /
                                    Hnuclei + 14.0 * ab[IDX_GNHI] / 15.0 /
                                    Hnuclei + 42.0 * ab[IDX_CH2CHCNI] / 53.0 /
                                    Hnuclei + 56.0 * ab[IDX_CH3CNHII] / 42.0 /
                                    Hnuclei + 14.0 * ab[IDX_HC7NI] / 99.0 /
                                    Hnuclei + 14.0 * ab[IDX_HNOI] / 31.0 /
                                    Hnuclei + 14.0 * ab[IDX_HNC3I] / 51.0 /
                                    Hnuclei + 14.0 * ab[IDX_HC3NII] / 51.0 /
                                    Hnuclei + 14.0 * ab[IDX_HC5NI] / 75.0 /
                                    Hnuclei + 28.0 * ab[IDX_HC3NHII] / 52.0 /
                                    Hnuclei + 42.0 * ab[IDX_CH3CNI] / 41.0 /
                                    Hnuclei + 14.0 * ab[IDX_HNOII] / 31.0 /
                                    Hnuclei + 14.0 * ab[IDX_HCNII] / 27.0 /
                                    Hnuclei + 14.0 * ab[IDX_NHII] / 15.0 /
                                    Hnuclei + 28.0 * ab[IDX_NH2II] / 16.0 /
                                    Hnuclei + 14.0 * ab[IDX_HC3NI] / 51.0 /
                                    Hnuclei + 28.0 * ab[IDX_NH2I] / 16.0 /
                                    Hnuclei + 14.0 * ab[IDX_NHI] / 15.0 /
                                    Hnuclei + 28.0 * ab[IDX_N2HII] / 29.0 /
                                    Hnuclei + 28.0 * ab[IDX_HCNHII] / 28.0 /
                                    Hnuclei + 14.0 * ab[IDX_HNCI] / 27.0 /
                                    Hnuclei + 42.0 * ab[IDX_NH3II] / 17.0 /
                                    Hnuclei + 56.0 * ab[IDX_NH4II] / 18.0 /
                                    Hnuclei + 14.0 * ab[IDX_HCNI] / 27.0 /
                                    Hnuclei + 42.0 * ab[IDX_NH3I] / 17.0 /
                                    Hnuclei;
    IJth(A, IDX_ELEM_H, IDX_ELEM_O) = 0.0 + 16.0 * ab[IDX_GHPOI] / 48.0 / Hnuclei
                                    + 32.0 * ab[IDX_GH2SiOI] / 46.0 / Hnuclei +
                                    128.0 * ab[IDX_GCH3COOHI] / 60.0 / Hnuclei +
                                    16.0 * ab[IDX_GHCNOI] / 43.0 / Hnuclei +
                                    16.0 * ab[IDX_GHNCOI] / 43.0 / Hnuclei +
                                    16.0 * ab[IDX_GHOCNI] / 43.0 / Hnuclei +
                                    16.0 * ab[IDX_GHONCI] / 43.0 / Hnuclei +
                                    16.0 * ab[IDX_HC2OI] / 41.0 / Hnuclei + 64.0
                                    * ab[IDX_GH2O2I] / 34.0 / Hnuclei + 16.0 *
                                    ab[IDX_GHC2OI] / 41.0 / Hnuclei + 32.0 *
                                    ab[IDX_C3H2OII] / 54.0 / Hnuclei + 96.0 *
                                    ab[IDX_GCH3COCH3I] / 58.0 / Hnuclei + 96.0 *
                                    ab[IDX_GCH3OCH3I] / 46.0 / Hnuclei + 48.0 *
                                    ab[IDX_H3C3OII] / 55.0 / Hnuclei + 96.0 *
                                    ab[IDX_COOCH3II] / 59.0 / Hnuclei + 96.0 *
                                    ab[IDX_GC2H5OHI] / 46.0 / Hnuclei + 32.0 *
                                    ab[IDX_GO2HI] / 33.0 / Hnuclei + 32.0 *
                                    ab[IDX_H2CNOII] / 44.0 / Hnuclei + 32.0 *
                                    ab[IDX_H2NCOII] / 44.0 / Hnuclei + 32.0 *
                                    ab[IDX_H2OCNII] / 44.0 / Hnuclei + 32.0 *
                                    ab[IDX_H2NOII] / 32.0 / Hnuclei + 16.0 *
                                    ab[IDX_HOCII] / 29.0 / Hnuclei + 16.0 *
                                    ab[IDX_HSOII] / 49.0 / Hnuclei + 32.0 *
                                    ab[IDX_HSiO2II] / 61.0 / Hnuclei + 96.0 *
                                    ab[IDX_C2H5OHII] / 46.0 / Hnuclei + 96.0 *
                                    ab[IDX_CH2OHCOII] / 59.0 / Hnuclei + 32.0 *
                                    ab[IDX_GCH2COI] / 42.0 / Hnuclei + 16.0 *
                                    ab[IDX_GHNOI] / 31.0 / Hnuclei + 32.0 *
                                    ab[IDX_HCNOHII] / 44.0 / Hnuclei + 32.0 *
                                    ab[IDX_HNCOHII] / 44.0 / Hnuclei + 16.0 *
                                    ab[IDX_HOCNII] / 43.0 / Hnuclei + 64.0 *
                                    ab[IDX_H2O2I] / 34.0 / Hnuclei + 16.0 *
                                    ab[IDX_HONCII] / 43.0 / Hnuclei + 160.0 *
                                    ab[IDX_CH2OHCH2OII] / 61.0 / Hnuclei + 96.0
                                    * ab[IDX_GCOOCH3I] / 59.0 / Hnuclei + 32.0 *
                                    ab[IDX_H2SiOII] / 46.0 / Hnuclei + 16.0 *
                                    ab[IDX_HCNOII] / 43.0 / Hnuclei + 128.0 *
                                    ab[IDX_CH3COOHII] / 60.0 / Hnuclei + 160.0 *
                                    ab[IDX_H5C2O2II] / 61.0 / Hnuclei + 64.0 *
                                    ab[IDX_HCOOHII] / 46.0 / Hnuclei + 160.0 *
                                    ab[IDX_CH3COOH2II] / 61.0 / Hnuclei + 96.0 *
                                    ab[IDX_CH3OCH3II] / 46.0 / Hnuclei + 96.0 *
                                    ab[IDX_GCH2OHCOI] / 59.0 / Hnuclei + 32.0 *
                                    ab[IDX_GCOOHI] / 45.0 / Hnuclei + 48.0 *
                                    ab[IDX_H3SiOII] / 47.0 / Hnuclei + 32.0 *
                                    ab[IDX_HSO2II] / 65.0 / Hnuclei + 112.0 *
                                    ab[IDX_CH3OCH4II] / 47.0 / Hnuclei + 16.0 *
                                    ab[IDX_HNCOII] / 43.0 / Hnuclei + 112.0 *
                                    ab[IDX_C2H5OH2II] / 47.0 / Hnuclei + 96.0 *
                                    ab[IDX_COOCH3I] / 59.0 / Hnuclei + 32.0 *
                                    ab[IDX_H2POII] / 49.0 / Hnuclei + 128.0 *
                                    ab[IDX_HCOOCH3II] / 60.0 / Hnuclei + 16.0 *
                                    ab[IDX_HN2OII] / 45.0 / Hnuclei + 128.0 *
                                    ab[IDX_CH2OHCHOII] / 60.0 / Hnuclei + 64.0 *
                                    ab[IDX_CH3CHOII] / 44.0 / Hnuclei + 128.0 *
                                    ab[IDX_GCH2OHCHOI] / 60.0 / Hnuclei + 64.0 *
                                    ab[IDX_GHCOOHI] / 46.0 / Hnuclei + 16.0 *
                                    ab[IDX_HOCSII] / 61.0 / Hnuclei + 48.0 *
                                    ab[IDX_GCH3COI] / 43.0 / Hnuclei + 128.0 *
                                    ab[IDX_GHCOOCH3I] / 60.0 / Hnuclei + 96.0 *
                                    ab[IDX_CH3COCH3II] / 58.0 / Hnuclei + 48.0 *
                                    ab[IDX_GCH2OHI] / 31.0 / Hnuclei + 64.0 *
                                    ab[IDX_GCH3OHI] / 32.0 / Hnuclei + 16.0 *
                                    ab[IDX_HONCI] / 43.0 / Hnuclei + 96.0 *
                                    ab[IDX_CH2OHCOI] / 59.0 / Hnuclei + 32.0 *
                                    ab[IDX_H2SiOI] / 46.0 / Hnuclei + 16.0 *
                                    ab[IDX_HC2OII] / 41.0 / Hnuclei + 48.0 *
                                    ab[IDX_GCH3OI] / 31.0 / Hnuclei + 32.0 *
                                    ab[IDX_GH2OI] / 18.0 / Hnuclei + 96.0 *
                                    ab[IDX_HCOOH2II] / 47.0 / Hnuclei + 16.0 *
                                    ab[IDX_HPOII] / 48.0 / Hnuclei + 128.0 *
                                    ab[IDX_CH2OHCHOI] / 60.0 / Hnuclei + 16.0 *
                                    ab[IDX_HC3OII] / 53.0 / Hnuclei + 16.0 *
                                    ab[IDX_HCNOI] / 43.0 / Hnuclei + 32.0 *
                                    ab[IDX_CH2COII] / 42.0 / Hnuclei + 64.0 *
                                    ab[IDX_GCH3CHOI] / 44.0 / Hnuclei + 32.0 *
                                    ab[IDX_GH2COI] / 30.0 / Hnuclei + 16.0 *
                                    ab[IDX_HOCNI] / 43.0 / Hnuclei + 16.0 *
                                    ab[IDX_HPOI] / 48.0 / Hnuclei + 48.0 *
                                    ab[IDX_CH3COI] / 43.0 / Hnuclei + 112.0 *
                                    ab[IDX_CH3COCH4II] / 59.0 / Hnuclei + 64.0 *
                                    ab[IDX_CH3OHII] / 32.0 / Hnuclei + 32.0 *
                                    ab[IDX_COOHI] / 45.0 / Hnuclei + 128.0 *
                                    ab[IDX_HCOOCH3I] / 60.0 / Hnuclei + 16.0 *
                                    ab[IDX_HNCOI] / 43.0 / Hnuclei + 128.0 *
                                    ab[IDX_CH3COOHI] / 60.0 / Hnuclei + 96.0 *
                                    ab[IDX_CH3OCH3I] / 46.0 / Hnuclei + 48.0 *
                                    ab[IDX_CH3OI] / 31.0 / Hnuclei + 16.0 *
                                    ab[IDX_GHCOI] / 29.0 / Hnuclei + 48.0 *
                                    ab[IDX_CH2OHI] / 31.0 / Hnuclei + 80.0 *
                                    ab[IDX_CH3CHOHII] / 45.0 / Hnuclei + 48.0 *
                                    ab[IDX_CH3COII] / 43.0 / Hnuclei + 64.0 *
                                    ab[IDX_HCOOHI] / 46.0 / Hnuclei + 32.0 *
                                    ab[IDX_O2HI] / 33.0 / Hnuclei + 96.0 *
                                    ab[IDX_C2H5OHI] / 46.0 / Hnuclei + 16.0 *
                                    ab[IDX_HNOI] / 31.0 / Hnuclei + 80.0 *
                                    ab[IDX_CH3OH2II] / 33.0 / Hnuclei + 32.0 *
                                    ab[IDX_CH2COI] / 42.0 / Hnuclei + 96.0 *
                                    ab[IDX_CH3COCH3I] / 58.0 / Hnuclei + 16.0 *
                                    ab[IDX_GOHI] / 17.0 / Hnuclei + 32.0 *
                                    ab[IDX_HCO2II] / 45.0 / Hnuclei + 64.0 *
                                    ab[IDX_CH3CHOI] / 44.0 / Hnuclei + 32.0 *
                                    ab[IDX_O2HII] / 33.0 / Hnuclei + 16.0 *
                                    ab[IDX_HNOII] / 31.0 / Hnuclei + 16.0 *
                                    ab[IDX_OHM] / 17.0 / Hnuclei + 64.0 *
                                    ab[IDX_CH3OHI] / 32.0 / Hnuclei + 48.0 *
                                    ab[IDX_H3COII] / 31.0 / Hnuclei + 16.0 *
                                    ab[IDX_SiOHII] / 45.0 / Hnuclei + 16.0 *
                                    ab[IDX_OHII] / 17.0 / Hnuclei + 32.0 *
                                    ab[IDX_H2OII] / 18.0 / Hnuclei + 32.0 *
                                    ab[IDX_H2COII] / 30.0 / Hnuclei + 16.0 *
                                    ab[IDX_HCOI] / 29.0 / Hnuclei + 32.0 *
                                    ab[IDX_H2COI] / 30.0 / Hnuclei + 16.0 *
                                    ab[IDX_OHI] / 17.0 / Hnuclei + 48.0 *
                                    ab[IDX_H3OII] / 19.0 / Hnuclei + 32.0 *
                                    ab[IDX_H2OI] / 18.0 / Hnuclei + 16.0 *
                                    ab[IDX_HCOII] / 29.0 / Hnuclei;
    IJth(A, IDX_ELEM_H, IDX_ELEM_He) = 0.0 + 4.0 * ab[IDX_HeHII] / 5.0 / Hnuclei;
    IJth(A, IDX_ELEM_H, IDX_ELEM_C) = 0.0 + 288.0 * ab[IDX_GC4H6I] / 54.0 /
                                    Hnuclei + 144.0 * ab[IDX_GCH3C3NI] / 65.0 /
                                    Hnuclei + 240.0 * ab[IDX_GCH3C4HI] / 64.0 /
                                    Hnuclei + 216.0 * ab[IDX_GCH3C5NI] / 89.0 /
                                    Hnuclei + 336.0 * ab[IDX_GCH3C6HI] / 88.0 /
                                    Hnuclei + 288.0 * ab[IDX_GCH3C7NI] / 113.0 /
                                    Hnuclei + 24.0 * ab[IDX_GHC2PI] / 56.0 /
                                    Hnuclei + 24.0 * ab[IDX_GNH2CNI] / 42.0 /
                                    Hnuclei + 36.0 * ab[IDX_GSiC3HI] / 65.0 /
                                    Hnuclei + 36.0 * ab[IDX_GSiCH3I] / 43.0 /
                                    Hnuclei + 24.0 * ab[IDX_GSiC2HI] / 53.0 /
                                    Hnuclei + 144.0 * ab[IDX_C2H4CNI] / 54.0 /
                                    Hnuclei + 36.0 * ab[IDX_GCH2PHI] / 46.0 /
                                    Hnuclei + 216.0 * ab[IDX_GCH3CHCH2I] / 42.0
                                    / Hnuclei + 96.0 * ab[IDX_GCH3COOHI] / 60.0
                                    / Hnuclei + 12.0 * ab[IDX_GHCNOI] / 43.0 /
                                    Hnuclei + 12.0 * ab[IDX_GHCPI] / 44.0 /
                                    Hnuclei + 36.0 * ab[IDX_GHNC3I] / 51.0 /
                                    Hnuclei + 12.0 * ab[IDX_GHNCOI] / 43.0 /
                                    Hnuclei + 12.0 * ab[IDX_GHOCNI] / 43.0 /
                                    Hnuclei + 12.0 * ab[IDX_GHONCI] / 43.0 /
                                    Hnuclei + 48.0 * ab[IDX_GSiC2H2I] / 54.0 /
                                    Hnuclei + 24.0 * ab[IDX_HC2OI] / 41.0 /
                                    Hnuclei + 432.0 * ab[IDX_GC6H6I] / 78.0 /
                                    Hnuclei + 12.0 * ab[IDX_GHCSiI] / 41.0 /
                                    Hnuclei + 108.0 * ab[IDX_NCCNCH3II] / 67.0 /
                                    Hnuclei + 144.0 * ab[IDX_GC2H4CNI] / 54.0 /
                                    Hnuclei + 144.0 * ab[IDX_GC4H3I] / 51.0 /
                                    Hnuclei + 168.0 * ab[IDX_GC7H2I] / 86.0 /
                                    Hnuclei + 192.0 * ab[IDX_GC8H2I] / 98.0 /
                                    Hnuclei + 216.0 * ab[IDX_GC9H2I] / 110.0 /
                                    Hnuclei + 48.0 * ab[IDX_GCH2CNI] / 40.0 /
                                    Hnuclei + 72.0 * ab[IDX_GCH3CNI] / 41.0 /
                                    Hnuclei + 24.0 * ab[IDX_GH2CNI] / 28.0 /
                                    Hnuclei + 24.0 * ab[IDX_GH2CSI] / 46.0 /
                                    Hnuclei + 24.0 * ab[IDX_GHC2OI] / 41.0 /
                                    Hnuclei + 60.0 * ab[IDX_GHC5NI] / 75.0 /
                                    Hnuclei + 84.0 * ab[IDX_GHC7NI] / 99.0 /
                                    Hnuclei + 108.0 * ab[IDX_GHC9NI] / 123.0 /
                                    Hnuclei + 24.0 * ab[IDX_GHCCNI] / 39.0 /
                                    Hnuclei + 12.0 * ab[IDX_GHCSI] / 45.0 /
                                    Hnuclei + 24.0 * ab[IDX_HCCNI] / 39.0 /
                                    Hnuclei + 72.0 * ab[IDX_C3H2OII] / 54.0 /
                                    Hnuclei + 144.0 * ab[IDX_GC2H6I] / 30.0 /
                                    Hnuclei + 72.0 * ab[IDX_GC3H2I] / 38.0 /
                                    Hnuclei + 120.0 * ab[IDX_GC5H2I] / 62.0 /
                                    Hnuclei + 192.0 * ab[IDX_GCH2CHCCHI] / 52.0
                                    / Hnuclei + 216.0 * ab[IDX_GCH3COCH3I] /
                                    58.0 / Hnuclei + 144.0 * ab[IDX_GCH3OCH3I] /
                                    46.0 / Hnuclei + 24.0 * ab[IDX_GSiCH2I] /
                                    42.0 / Hnuclei + 108.0 * ab[IDX_H3C3OII] /
                                    55.0 / Hnuclei + 480.0 * ab[IDX_C8H5II] /
                                    101.0 / Hnuclei + 540.0 * ab[IDX_C9H5II] /
                                    113.0 / Hnuclei + 144.0 * ab[IDX_CH3C3NII] /
                                    65.0 / Hnuclei + 240.0 * ab[IDX_GC10H2I] /
                                    122.0 / Hnuclei + 96.0 * ab[IDX_GC4H2I] /
                                    50.0 / Hnuclei + 108.0 * ab[IDX_GCH2CHCNI] /
                                    53.0 / Hnuclei + 72.0 * ab[IDX_GH2CCCI] /
                                    38.0 / Hnuclei + 36.0 * ab[IDX_GHC3NI] /
                                    51.0 / Hnuclei + 288.0 * ab[IDX_C4H6I] /
                                    54.0 / Hnuclei + 432.0 * ab[IDX_C6H6II] /
                                    78.0 / Hnuclei + 108.0 * ab[IDX_CH2CHCNII] /
                                    53.0 / Hnuclei + 72.0 * ab[IDX_COOCH3II] /
                                    59.0 / Hnuclei + 180.0 * ab[IDX_GC2H5CNI] /
                                    55.0 / Hnuclei + 144.0 * ab[IDX_GC2H5OHI] /
                                    46.0 / Hnuclei + 144.0 * ab[IDX_GC6H2I] /
                                    74.0 / Hnuclei + 24.0 * ab[IDX_H2CNOII] /
                                    44.0 / Hnuclei + 24.0 * ab[IDX_H2NCOII] /
                                    44.0 / Hnuclei + 24.0 * ab[IDX_H2OCNII] /
                                    44.0 / Hnuclei + 96.0 * ab[IDX_PC2H4II] /
                                    59.0 / Hnuclei + 48.0 * ab[IDX_CH3NHII] /
                                    30.0 / Hnuclei + 144.0 * ab[IDX_GCH2CCH2I] /
                                    40.0 / Hnuclei + 36.0 * ab[IDX_GCH2NHI] /
                                    29.0 / Hnuclei + 144.0 * ab[IDX_GCH3CCHI] /
                                    40.0 / Hnuclei + 12.0 * ab[IDX_GHNCI] / 27.0
                                    / Hnuclei + 96.0 * ab[IDX_H2C4NII] / 64.0 /
                                    Hnuclei + 24.0 * ab[IDX_H2CClII] / 49.0 /
                                    Hnuclei + 12.0 * ab[IDX_HOCII] / 29.0 /
                                    Hnuclei + 36.0 * ab[IDX_NH2CNHII] / 43.0 /
                                    Hnuclei + 144.0 * ab[IDX_C2H5OHII] / 46.0 /
                                    Hnuclei + 192.0 * ab[IDX_CH2CHCCHI] / 52.0 /
                                    Hnuclei + 72.0 * ab[IDX_CH2OHCOII] / 59.0 /
                                    Hnuclei + 108.0 * ab[IDX_GCH2CCHI] / 39.0 /
                                    Hnuclei + 48.0 * ab[IDX_GCH2COI] / 42.0 /
                                    Hnuclei + 48.0 * ab[IDX_HC4NII] / 63.0 /
                                    Hnuclei + 24.0 * ab[IDX_HCNOHII] / 44.0 /
                                    Hnuclei + 24.0 * ab[IDX_HNCOHII] / 44.0 /
                                    Hnuclei + 12.0 * ab[IDX_HOCNII] / 43.0 /
                                    Hnuclei + 72.0 * ab[IDX_PC2H3II] / 58.0 /
                                    Hnuclei + 72.0 * ab[IDX_CH3CSII] / 59.0 /
                                    Hnuclei + 168.0 * ab[IDX_H2C7NII] / 100.0 /
                                    Hnuclei + 216.0 * ab[IDX_H2C9NII] / 124.0 /
                                    Hnuclei + 12.0 * ab[IDX_HONCII] / 43.0 /
                                    Hnuclei + 36.0 * ab[IDX_PCH3II] / 46.0 /
                                    Hnuclei + 360.0 * ab[IDX_C10H3II] / 123.0 /
                                    Hnuclei + 216.0 * ab[IDX_C2H5CNHII] / 56.0 /
                                    Hnuclei + 48.0 * ab[IDX_CH2NH2II] / 30.0 /
                                    Hnuclei + 120.0 * ab[IDX_CH2OHCH2OII] / 61.0
                                    / Hnuclei + 288.0 * ab[IDX_CH3C5NHII] / 90.0
                                    / Hnuclei + 384.0 * ab[IDX_CH3C7NHII] /
                                    114.0 / Hnuclei + 120.0 * ab[IDX_GC2H5I] /
                                    29.0 / Hnuclei + 72.0 * ab[IDX_GCOOCH3I] /
                                    59.0 / Hnuclei + 12.0 * ab[IDX_GHCNI] / 27.0
                                    / Hnuclei + 24.0 * ab[IDX_H2CNI] / 28.0 /
                                    Hnuclei + 180.0 * ab[IDX_H3C5NII] / 77.0 /
                                    Hnuclei + 12.0 * ab[IDX_HCNOII] / 43.0 /
                                    Hnuclei + 96.0 * ab[IDX_CH3COOHII] / 60.0 /
                                    Hnuclei + 324.0 * ab[IDX_H3C9NII] / 125.0 /
                                    Hnuclei + 120.0 * ab[IDX_H5C2O2II] / 61.0 /
                                    Hnuclei + 24.0 * ab[IDX_HCOOHII] / 46.0 /
                                    Hnuclei + 120.0 * ab[IDX_CH3COOH2II] / 61.0
                                    / Hnuclei + 144.0 * ab[IDX_CH3OCH3II] / 46.0
                                    / Hnuclei + 120.0 * ab[IDX_GC10HI] / 121.0 /
                                    Hnuclei + 96.0 * ab[IDX_GC8HI] / 97.0 /
                                    Hnuclei + 72.0 * ab[IDX_GCH2OHCOI] / 59.0 /
                                    Hnuclei + 12.0 * ab[IDX_GCOOHI] / 45.0 /
                                    Hnuclei + 252.0 * ab[IDX_H3C7NII] / 101.0 /
                                    Hnuclei + 24.0 * ab[IDX_NH2CNI] / 42.0 /
                                    Hnuclei + 192.0 * ab[IDX_CH3C3NHII] / 66.0 /
                                    Hnuclei + 168.0 * ab[IDX_CH3OCH4II] / 47.0 /
                                    Hnuclei + 96.0 * ab[IDX_GC2H4I] / 28.0 /
                                    Hnuclei + 108.0 * ab[IDX_GC9HI] / 109.0 /
                                    Hnuclei + 84.0 * ab[IDX_HC7NII] / 99.0 /
                                    Hnuclei + 108.0 * ab[IDX_HC9NII] / 123.0 /
                                    Hnuclei + 12.0 * ab[IDX_HNCOII] / 43.0 /
                                    Hnuclei + 36.0 * ab[IDX_PC3HII] / 68.0 /
                                    Hnuclei + 48.0 * ab[IDX_PCH4II] / 47.0 /
                                    Hnuclei + 180.0 * ab[IDX_C2H5CNI] / 55.0 /
                                    Hnuclei + 168.0 * ab[IDX_C2H5OH2II] / 47.0 /
                                    Hnuclei + 168.0 * ab[IDX_C2H7II] / 31.0 /
                                    Hnuclei + 144.0 * ab[IDX_CH2CHCNHII] / 54.0
                                    / Hnuclei + 72.0 * ab[IDX_COOCH3I] / 59.0 /
                                    Hnuclei + 96.0 * ab[IDX_HCOOCH3II] / 60.0 /
                                    Hnuclei + 48.0 * ab[IDX_PC2H2II] / 57.0 /
                                    Hnuclei + 72.0 * ab[IDX_SiC3H2II] / 66.0 /
                                    Hnuclei + 48.0 * ab[IDX_SiC4HII] / 77.0 /
                                    Hnuclei + 12.0 * ab[IDX_SiNCHII] / 55.0 /
                                    Hnuclei + 240.0 * ab[IDX_C4H5II] / 53.0 /
                                    Hnuclei + 336.0 * ab[IDX_C4H7II] / 55.0 /
                                    Hnuclei + 96.0 * ab[IDX_CH2OHCHOII] / 60.0 /
                                    Hnuclei + 96.0 * ab[IDX_CH3CHOII] / 44.0 /
                                    Hnuclei + 96.0 * ab[IDX_GCH2OHCHOI] / 60.0 /
                                    Hnuclei + 24.0 * ab[IDX_GHCOOHI] / 46.0 /
                                    Hnuclei + 12.0 * ab[IDX_HOCSII] / 61.0 /
                                    Hnuclei + 48.0 * ab[IDX_PC4HII] / 80.0 /
                                    Hnuclei + 48.0 * ab[IDX_SiCH4II] / 44.0 /
                                    Hnuclei + 240.0 * ab[IDX_C10H2II] / 122.0 /
                                    Hnuclei + 72.0 * ab[IDX_CH3CNII] / 41.0 /
                                    Hnuclei + 48.0 * ab[IDX_GC2H2I] / 26.0 /
                                    Hnuclei + 84.0 * ab[IDX_GC7HI] / 85.0 /
                                    Hnuclei + 72.0 * ab[IDX_GCH3COI] / 43.0 /
                                    Hnuclei + 96.0 * ab[IDX_GHCOOCH3I] / 60.0 /
                                    Hnuclei + 48.0 * ab[IDX_HC4SII] / 81.0 /
                                    Hnuclei + 120.0 * ab[IDX_HC5NHII] / 76.0 /
                                    Hnuclei + 72.0 * ab[IDX_SiC2H3II] / 55.0 /
                                    Hnuclei + 36.0 * ab[IDX_SiC3HI] / 65.0 /
                                    Hnuclei + 36.0 * ab[IDX_SiCH3II] / 43.0 /
                                    Hnuclei + 192.0 * ab[IDX_C4H4II] / 52.0 /
                                    Hnuclei + 48.0 * ab[IDX_CH2CNII] / 40.0 /
                                    Hnuclei + 216.0 * ab[IDX_CH3COCH3II] / 58.0
                                    / Hnuclei + 36.0 * ab[IDX_GCH2OHI] / 31.0 /
                                    Hnuclei + 48.0 * ab[IDX_GCH3OHI] / 32.0 /
                                    Hnuclei + 24.0 * ab[IDX_HC2PII] / 56.0 /
                                    Hnuclei + 60.0 * ab[IDX_HC5NII] / 75.0 /
                                    Hnuclei + 12.0 * ab[IDX_HONCI] / 43.0 /
                                    Hnuclei + 24.0 * ab[IDX_PCH2II] / 45.0 /
                                    Hnuclei + 72.0 * ab[IDX_CH2OHCOI] / 59.0 /
                                    Hnuclei + 216.0 * ab[IDX_CH3C5NI] / 89.0 /
                                    Hnuclei + 288.0 * ab[IDX_CH3C7NI] / 113.0 /
                                    Hnuclei + 36.0 * ab[IDX_GC3HI] / 37.0 /
                                    Hnuclei + 48.0 * ab[IDX_GC4HI] / 49.0 /
                                    Hnuclei + 60.0 * ab[IDX_GC5HI] / 61.0 /
                                    Hnuclei + 72.0 * ab[IDX_GC6HI] / 73.0 /
                                    Hnuclei + 24.0 * ab[IDX_H2NCII] / 28.0 /
                                    Hnuclei + 36.0 * ab[IDX_H3CSII] / 47.0 /
                                    Hnuclei + 24.0 * ab[IDX_HC2OII] / 41.0 /
                                    Hnuclei + 12.0 * ab[IDX_HCPII] / 44.0 /
                                    Hnuclei + 72.0 * ab[IDX_GC2H3I] / 27.0 /
                                    Hnuclei + 36.0 * ab[IDX_GCH3OI] / 31.0 /
                                    Hnuclei + 36.0 * ab[IDX_HC3SII] / 69.0 /
                                    Hnuclei + 36.0 * ab[IDX_HCOOH2II] / 47.0 /
                                    Hnuclei + 48.0 * ab[IDX_SiC2H2I] / 54.0 /
                                    Hnuclei + 24.0 * ab[IDX_C2NHII] / 39.0 /
                                    Hnuclei + 216.0 * ab[IDX_C3H6II] / 42.0 /
                                    Hnuclei + 504.0 * ab[IDX_C6H7II] / 79.0 /
                                    Hnuclei + 384.0 * ab[IDX_C8H4II] / 100.0 /
                                    Hnuclei + 96.0 * ab[IDX_CH2OHCHOI] / 60.0 /
                                    Hnuclei + 36.0 * ab[IDX_HC3OII] / 53.0 /
                                    Hnuclei + 12.0 * ab[IDX_HCNOI] / 43.0 /
                                    Hnuclei + 36.0 * ab[IDX_SiC3HII] / 65.0 /
                                    Hnuclei + 36.0 * ab[IDX_SiCH3I] / 43.0 /
                                    Hnuclei + 252.0 * ab[IDX_C3H7II] / 43.0 /
                                    Hnuclei + 432.0 * ab[IDX_C9H4II] / 112.0 /
                                    Hnuclei + 48.0 * ab[IDX_CH2COII] / 42.0 /
                                    Hnuclei + 36.0 * ab[IDX_CH2PHI] / 46.0 /
                                    Hnuclei + 144.0 * ab[IDX_CH3C3NI] / 65.0 /
                                    Hnuclei + 96.0 * ab[IDX_GCH3CHOI] / 44.0 /
                                    Hnuclei + 24.0 * ab[IDX_GH2COI] / 30.0 /
                                    Hnuclei + 12.0 * ab[IDX_HOCNI] / 43.0 /
                                    Hnuclei + 48.0 * ab[IDX_SiC2H2II] / 54.0 /
                                    Hnuclei + 288.0 * ab[IDX_C6H4II] / 76.0 /
                                    Hnuclei + 72.0 * ab[IDX_CH3COI] / 43.0 /
                                    Hnuclei + 252.0 * ab[IDX_CH3COCH4II] / 59.0
                                    / Hnuclei + 48.0 * ab[IDX_CH3OHII] / 32.0 /
                                    Hnuclei + 12.0 * ab[IDX_COOHI] / 45.0 /
                                    Hnuclei + 336.0 * ab[IDX_C7H4II] / 88.0 /
                                    Hnuclei + 420.0 * ab[IDX_C7H5II] / 89.0 /
                                    Hnuclei + 336.0 * ab[IDX_CH3C6HI] / 88.0 /
                                    Hnuclei + 24.0 * ab[IDX_H2CSII] / 46.0 /
                                    Hnuclei + 108.0 * ab[IDX_HC9NI] / 123.0 /
                                    Hnuclei + 96.0 * ab[IDX_HCOOCH3I] / 60.0 /
                                    Hnuclei + 24.0 * ab[IDX_NCCNHII] / 53.0 /
                                    Hnuclei + 24.0 * ab[IDX_SiC2HI] / 53.0 /
                                    Hnuclei + 48.0 * ab[IDX_CH2CNI] / 40.0 /
                                    Hnuclei + 240.0 * ab[IDX_CH3C4HII] / 64.0 /
                                    Hnuclei + 48.0 * ab[IDX_GCH4I] / 16.0 /
                                    Hnuclei + 12.0 * ab[IDX_HCPI] / 44.0 /
                                    Hnuclei + 12.0 * ab[IDX_HNCOI] / 43.0 /
                                    Hnuclei + 240.0 * ab[IDX_CH3C4HI] / 64.0 /
                                    Hnuclei + 96.0 * ab[IDX_CH3COOHI] / 60.0 /
                                    Hnuclei + 144.0 * ab[IDX_CH3OCH3I] / 46.0 /
                                    Hnuclei + 24.0 * ab[IDX_GC2HI] / 25.0 /
                                    Hnuclei + 24.0 * ab[IDX_H2CSI] / 46.0 /
                                    Hnuclei + 360.0 * ab[IDX_C6H5II] / 77.0 /
                                    Hnuclei + 36.0 * ab[IDX_CH2NHI] / 29.0 /
                                    Hnuclei + 36.0 * ab[IDX_CH3OI] / 31.0 /
                                    Hnuclei + 12.0 * ab[IDX_GHCOI] / 29.0 /
                                    Hnuclei + 24.0 * ab[IDX_HC2PI] / 56.0 /
                                    Hnuclei + 12.0 * ab[IDX_HCSiI] / 41.0 /
                                    Hnuclei + 24.0 * ab[IDX_SiCH2I] / 42.0 /
                                    Hnuclei + 120.0 * ab[IDX_C10HII] / 121.0 /
                                    Hnuclei + 12.0 * ab[IDX_HCSiII] / 41.0 /
                                    Hnuclei + 144.0 * ab[IDX_C2H6II] / 30.0 /
                                    Hnuclei + 96.0 * ab[IDX_C8HII] / 97.0 /
                                    Hnuclei + 36.0 * ab[IDX_CH2OHI] / 31.0 /
                                    Hnuclei + 300.0 * ab[IDX_C5H5II] / 65.0 /
                                    Hnuclei + 12.0 * ab[IDX_HCSI] / 45.0 /
                                    Hnuclei + 24.0 * ab[IDX_SiC2HII] / 53.0 /
                                    Hnuclei + 72.0 * ab[IDX_C6HII] / 73.0 /
                                    Hnuclei + 216.0 * ab[IDX_C9H2I] / 110.0 /
                                    Hnuclei + 108.0 * ab[IDX_CH2CHCNI] / 53.0 /
                                    Hnuclei + 120.0 * ab[IDX_CH3CHOHII] / 45.0 /
                                    Hnuclei + 96.0 * ab[IDX_CH3CNHII] / 42.0 /
                                    Hnuclei + 72.0 * ab[IDX_CH3COII] / 43.0 /
                                    Hnuclei + 84.0 * ab[IDX_HC7NI] / 99.0 /
                                    Hnuclei + 432.0 * ab[IDX_C6H6I] / 78.0 /
                                    Hnuclei + 24.0 * ab[IDX_HCOOHI] / 46.0 /
                                    Hnuclei + 240.0 * ab[IDX_C10H2I] / 122.0 /
                                    Hnuclei + 144.0 * ab[IDX_C2H5OHI] / 46.0 /
                                    Hnuclei + 60.0 * ab[IDX_CH3OH2II] / 33.0 /
                                    Hnuclei + 216.0 * ab[IDX_C6H3II] / 75.0 /
                                    Hnuclei + 288.0 * ab[IDX_C8H3II] / 99.0 /
                                    Hnuclei + 108.0 * ab[IDX_C9HII] / 109.0 /
                                    Hnuclei + 324.0 * ab[IDX_C9H3II] / 111.0 /
                                    Hnuclei + 48.0 * ab[IDX_CH2COI] / 42.0 /
                                    Hnuclei + 36.0 * ab[IDX_HNC3I] / 51.0 /
                                    Hnuclei + 24.0 * ab[IDX_SiCH2II] / 42.0 /
                                    Hnuclei + 216.0 * ab[IDX_CH3COCH3I] / 58.0 /
                                    Hnuclei + 36.0 * ab[IDX_HC3NII] / 51.0 /
                                    Hnuclei + 192.0 * ab[IDX_C8H2I] / 98.0 /
                                    Hnuclei + 60.0 * ab[IDX_HC5NI] / 75.0 /
                                    Hnuclei + 84.0 * ab[IDX_C7HII] / 85.0 /
                                    Hnuclei + 216.0 * ab[IDX_C9H2II] / 110.0 /
                                    Hnuclei + 168.0 * ab[IDX_C7H2I] / 86.0 /
                                    Hnuclei + 12.0 * ab[IDX_HCSII] / 45.0 /
                                    Hnuclei + 144.0 * ab[IDX_C3H4II] / 40.0 /
                                    Hnuclei + 192.0 * ab[IDX_C8H2II] / 98.0 /
                                    Hnuclei + 24.0 * ab[IDX_GCH2I] / 14.0 /
                                    Hnuclei + 12.0 * ab[IDX_HCO2II] / 45.0 /
                                    Hnuclei + 180.0 * ab[IDX_C5H3II] / 63.0 /
                                    Hnuclei + 72.0 * ab[IDX_HC3NHII] / 52.0 /
                                    Hnuclei + 60.0 * ab[IDX_C5HII] / 61.0 /
                                    Hnuclei + 96.0 * ab[IDX_CH3CHOI] / 44.0 /
                                    Hnuclei + 180.0 * ab[IDX_C3H5II] / 41.0 /
                                    Hnuclei + 252.0 * ab[IDX_C7H3II] / 87.0 /
                                    Hnuclei + 72.0 * ab[IDX_CH3CNI] / 41.0 /
                                    Hnuclei + 168.0 * ab[IDX_C7H2II] / 86.0 /
                                    Hnuclei + 144.0 * ab[IDX_C6H2II] / 74.0 /
                                    Hnuclei + 48.0 * ab[IDX_CH4II] / 16.0 /
                                    Hnuclei + 72.0 * ab[IDX_H2CCCI] / 38.0 /
                                    Hnuclei + 144.0 * ab[IDX_C6H2I] / 74.0 /
                                    Hnuclei + 120.0 * ab[IDX_C2H5II] / 29.0 /
                                    Hnuclei + 120.0 * ab[IDX_C5H2I] / 62.0 /
                                    Hnuclei + 120.0 * ab[IDX_C5H2II] / 62.0 /
                                    Hnuclei + 144.0 * ab[IDX_CH2CCH2I] / 40.0 /
                                    Hnuclei + 144.0 * ab[IDX_C2H6I] / 30.0 /
                                    Hnuclei + 72.0 * ab[IDX_C3H2I] / 38.0 /
                                    Hnuclei + 120.0 * ab[IDX_C2H5I] / 29.0 /
                                    Hnuclei + 12.0 * ab[IDX_HCNII] / 27.0 /
                                    Hnuclei + 216.0 * ab[IDX_CH3CHCH2I] / 42.0 /
                                    Hnuclei + 48.0 * ab[IDX_C4HII] / 49.0 /
                                    Hnuclei + 12.0 * ab[IDX_CHM] / 13.0 /
                                    Hnuclei + 36.0 * ab[IDX_GCH3I] / 15.0 /
                                    Hnuclei + 24.0 * ab[IDX_C2HM] / 25.0 /
                                    Hnuclei + 48.0 * ab[IDX_C4HM] / 49.0 /
                                    Hnuclei + 144.0 * ab[IDX_CH3CCHI] / 40.0 /
                                    Hnuclei + 60.0 * ab[IDX_CH5II] / 17.0 /
                                    Hnuclei + 24.0 * ab[IDX_HC2SII] / 57.0 /
                                    Hnuclei + 120.0 * ab[IDX_C10HM] / 121.0 /
                                    Hnuclei + 36.0 * ab[IDX_C3HM] / 37.0 /
                                    Hnuclei + 72.0 * ab[IDX_C6HM] / 73.0 /
                                    Hnuclei + 96.0 * ab[IDX_C8HM] / 97.0 /
                                    Hnuclei + 84.0 * ab[IDX_C7HM] / 85.0 /
                                    Hnuclei + 108.0 * ab[IDX_C9HM] / 109.0 /
                                    Hnuclei + 60.0 * ab[IDX_C5HM] / 61.0 /
                                    Hnuclei + 48.0 * ab[IDX_CH3OHI] / 32.0 /
                                    Hnuclei + 36.0 * ab[IDX_HC3NI] / 51.0 /
                                    Hnuclei + 36.0 * ab[IDX_C3HII] / 37.0 /
                                    Hnuclei + 108.0 * ab[IDX_C3H3II] / 39.0 /
                                    Hnuclei + 36.0 * ab[IDX_H3COII] / 31.0 /
                                    Hnuclei + 12.0 * ab[IDX_GCHI] / 13.0 /
                                    Hnuclei + 144.0 * ab[IDX_C4H3I] / 51.0 /
                                    Hnuclei + 24.0 * ab[IDX_C2HII] / 25.0 /
                                    Hnuclei + 72.0 * ab[IDX_C3H2II] / 38.0 /
                                    Hnuclei + 24.0 * ab[IDX_CH2II] / 14.0 /
                                    Hnuclei + 12.0 * ab[IDX_CHII] / 13.0 /
                                    Hnuclei + 120.0 * ab[IDX_C10HI] / 121.0 /
                                    Hnuclei + 108.0 * ab[IDX_C9HI] / 109.0 /
                                    Hnuclei + 96.0 * ab[IDX_C8HI] / 97.0 /
                                    Hnuclei + 84.0 * ab[IDX_C7HI] / 85.0 /
                                    Hnuclei + 96.0 * ab[IDX_C2H4II] / 28.0 /
                                    Hnuclei + 144.0 * ab[IDX_C4H3II] / 51.0 /
                                    Hnuclei + 108.0 * ab[IDX_CH2CCHI] / 39.0 /
                                    Hnuclei + 108.0 * ab[IDX_CH2CCHII] / 39.0 /
                                    Hnuclei + 72.0 * ab[IDX_C6HI] / 73.0 /
                                    Hnuclei + 60.0 * ab[IDX_C5HI] / 61.0 /
                                    Hnuclei + 36.0 * ab[IDX_C3HI] / 37.0 /
                                    Hnuclei + 24.0 * ab[IDX_H2COII] / 30.0 /
                                    Hnuclei + 48.0 * ab[IDX_C4HI] / 49.0 /
                                    Hnuclei + 96.0 * ab[IDX_C4H2I] / 50.0 /
                                    Hnuclei + 96.0 * ab[IDX_C4H2II] / 50.0 /
                                    Hnuclei + 24.0 * ab[IDX_HCNHII] / 28.0 /
                                    Hnuclei + 24.0 * ab[IDX_CH2I] / 14.0 /
                                    Hnuclei + 12.0 * ab[IDX_HNCI] / 27.0 /
                                    Hnuclei + 96.0 * ab[IDX_C2H4I] / 28.0 /
                                    Hnuclei + 72.0 * ab[IDX_C2H3I] / 27.0 /
                                    Hnuclei + 72.0 * ab[IDX_C2H3II] / 27.0 /
                                    Hnuclei + 12.0 * ab[IDX_HCOI] / 29.0 /
                                    Hnuclei + 48.0 * ab[IDX_CH4I] / 16.0 /
                                    Hnuclei + 24.0 * ab[IDX_C2HI] / 25.0 /
                                    Hnuclei + 24.0 * ab[IDX_H2COI] / 30.0 /
                                    Hnuclei + 12.0 * ab[IDX_HCNI] / 27.0 /
                                    Hnuclei + 12.0 * ab[IDX_CHI] / 13.0 /
                                    Hnuclei + 48.0 * ab[IDX_C2H2II] / 26.0 /
                                    Hnuclei + 36.0 * ab[IDX_CH3II] / 15.0 /
                                    Hnuclei + 48.0 * ab[IDX_C2H2I] / 26.0 /
                                    Hnuclei + 36.0 * ab[IDX_CH3I] / 15.0 /
                                    Hnuclei + 12.0 * ab[IDX_HCOII] / 29.0 /
                                    Hnuclei;
    IJth(A, IDX_ELEM_H, IDX_ELEM_GRAIN) = 0.0;
    IJth(A, IDX_ELEM_H, IDX_ELEM_H) = 0.0 + 4.0 * ab[IDX_GH2S2I] / 66.0 / Hnuclei
                                    + 36.0 * ab[IDX_GC4H6I] / 54.0 / Hnuclei +
                                    9.0 * ab[IDX_GCH3C3NI] / 65.0 / Hnuclei +
                                    16.0 * ab[IDX_GCH3C4HI] / 64.0 / Hnuclei +
                                    9.0 * ab[IDX_GCH3C5NI] / 89.0 / Hnuclei +
                                    16.0 * ab[IDX_GCH3C6HI] / 88.0 / Hnuclei +
                                    9.0 * ab[IDX_GCH3C7NI] / 113.0 / Hnuclei +
                                    1.0 * ab[IDX_GHC2PI] / 56.0 / Hnuclei + 1.0
                                    * ab[IDX_GHClI] / 36.0 / Hnuclei + 1.0 *
                                    ab[IDX_GHFI] / 20.0 / Hnuclei + 1.0 *
                                    ab[IDX_GHNSiI] / 43.0 / Hnuclei + 1.0 *
                                    ab[IDX_GHPOI] / 48.0 / Hnuclei + 4.0 *
                                    ab[IDX_GNH2CNI] / 42.0 / Hnuclei + 1.0 *
                                    ab[IDX_GSiC3HI] / 65.0 / Hnuclei + 9.0 *
                                    ab[IDX_GSiCH3I] / 43.0 / Hnuclei + 4.0 *
                                    ab[IDX_GH2SiOI] / 46.0 / Hnuclei + 1.0 *
                                    ab[IDX_GSiC2HI] / 53.0 / Hnuclei + 16.0 *
                                    ab[IDX_C2H4CNI] / 54.0 / Hnuclei + 9.0 *
                                    ab[IDX_GCH2PHI] / 46.0 / Hnuclei + 36.0 *
                                    ab[IDX_GCH3CHCH2I] / 42.0 / Hnuclei + 16.0 *
                                    ab[IDX_GCH3COOHI] / 60.0 / Hnuclei + 4.0 *
                                    ab[IDX_GH2SI] / 34.0 / Hnuclei + 1.0 *
                                    ab[IDX_GHCNOI] / 43.0 / Hnuclei + 1.0 *
                                    ab[IDX_GHCPI] / 44.0 / Hnuclei + 1.0 *
                                    ab[IDX_GHNC3I] / 51.0 / Hnuclei + 1.0 *
                                    ab[IDX_GHNCOI] / 43.0 / Hnuclei + 1.0 *
                                    ab[IDX_GHOCNI] / 43.0 / Hnuclei + 1.0 *
                                    ab[IDX_GHONCI] / 43.0 / Hnuclei + 1.0 *
                                    ab[IDX_GHS2I] / 65.0 / Hnuclei + 4.0 *
                                    ab[IDX_GPH2I] / 33.0 / Hnuclei + 4.0 *
                                    ab[IDX_GSiC2H2I] / 54.0 / Hnuclei + 1.0 *
                                    ab[IDX_HC2OI] / 41.0 / Hnuclei + 36.0 *
                                    ab[IDX_GC6H6I] / 78.0 / Hnuclei + 4.0 *
                                    ab[IDX_GH2O2I] / 34.0 / Hnuclei + 1.0 *
                                    ab[IDX_GHCSiI] / 41.0 / Hnuclei + 1.0 *
                                    ab[IDX_GPHI] / 32.0 / Hnuclei + 16.0 *
                                    ab[IDX_GSiH4I] / 32.0 / Hnuclei + 9.0 *
                                    ab[IDX_NCCNCH3II] / 67.0 / Hnuclei + 16.0 *
                                    ab[IDX_GC2H4CNI] / 54.0 / Hnuclei + 9.0 *
                                    ab[IDX_GC4H3I] / 51.0 / Hnuclei + 4.0 *
                                    ab[IDX_GC7H2I] / 86.0 / Hnuclei + 4.0 *
                                    ab[IDX_GC8H2I] / 98.0 / Hnuclei + 4.0 *
                                    ab[IDX_GC9H2I] / 110.0 / Hnuclei + 4.0 *
                                    ab[IDX_GCH2CNI] / 40.0 / Hnuclei + 9.0 *
                                    ab[IDX_GCH3CNI] / 41.0 / Hnuclei + 4.0 *
                                    ab[IDX_GH2CNI] / 28.0 / Hnuclei + 4.0 *
                                    ab[IDX_GH2CSI] / 46.0 / Hnuclei + 1.0 *
                                    ab[IDX_GHC2OI] / 41.0 / Hnuclei + 1.0 *
                                    ab[IDX_GHC5NI] / 75.0 / Hnuclei + 1.0 *
                                    ab[IDX_GHC7NI] / 99.0 / Hnuclei + 1.0 *
                                    ab[IDX_GHC9NI] / 123.0 / Hnuclei + 1.0 *
                                    ab[IDX_GHCCNI] / 39.0 / Hnuclei + 1.0 *
                                    ab[IDX_GHCSI] / 45.0 / Hnuclei + 9.0 *
                                    ab[IDX_GSiH3I] / 31.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCCNI] / 39.0 / Hnuclei + 4.0 *
                                    ab[IDX_C3H2OII] / 54.0 / Hnuclei + 36.0 *
                                    ab[IDX_GC2H6I] / 30.0 / Hnuclei + 4.0 *
                                    ab[IDX_GC3H2I] / 38.0 / Hnuclei + 4.0 *
                                    ab[IDX_GC5H2I] / 62.0 / Hnuclei + 16.0 *
                                    ab[IDX_GCH2CHCCHI] / 52.0 / Hnuclei + 36.0 *
                                    ab[IDX_GCH3COCH3I] / 58.0 / Hnuclei + 36.0 *
                                    ab[IDX_GCH3OCH3I] / 46.0 / Hnuclei + 4.0 *
                                    ab[IDX_GSiCH2I] / 42.0 / Hnuclei + 1.0 *
                                    ab[IDX_GSiHI] / 29.0 / Hnuclei + 4.0 *
                                    ab[IDX_GSiH2I] / 30.0 / Hnuclei + 9.0 *
                                    ab[IDX_H3C3OII] / 55.0 / Hnuclei + 1.0 *
                                    ab[IDX_HFII] / 20.0 / Hnuclei + 25.0 *
                                    ab[IDX_C8H5II] / 101.0 / Hnuclei + 25.0 *
                                    ab[IDX_C9H5II] / 113.0 / Hnuclei + 9.0 *
                                    ab[IDX_CH3C3NII] / 65.0 / Hnuclei + 4.0 *
                                    ab[IDX_GC10H2I] / 122.0 / Hnuclei + 4.0 *
                                    ab[IDX_GC4H2I] / 50.0 / Hnuclei + 9.0 *
                                    ab[IDX_GCH2CHCNI] / 53.0 / Hnuclei + 4.0 *
                                    ab[IDX_GH2CCCI] / 38.0 / Hnuclei + 1.0 *
                                    ab[IDX_GHC3NI] / 51.0 / Hnuclei + 4.0 *
                                    ab[IDX_H2FII] / 21.0 / Hnuclei + 36.0 *
                                    ab[IDX_C4H6I] / 54.0 / Hnuclei + 36.0 *
                                    ab[IDX_C6H6II] / 78.0 / Hnuclei + 9.0 *
                                    ab[IDX_CH2CHCNII] / 53.0 / Hnuclei + 9.0 *
                                    ab[IDX_COOCH3II] / 59.0 / Hnuclei + 25.0 *
                                    ab[IDX_GC2H5CNI] / 55.0 / Hnuclei + 36.0 *
                                    ab[IDX_GC2H5OHI] / 46.0 / Hnuclei + 4.0 *
                                    ab[IDX_GC6H2I] / 74.0 / Hnuclei + 1.0 *
                                    ab[IDX_GO2HI] / 33.0 / Hnuclei + 4.0 *
                                    ab[IDX_H2CNOII] / 44.0 / Hnuclei + 4.0 *
                                    ab[IDX_H2NCOII] / 44.0 / Hnuclei + 4.0 *
                                    ab[IDX_H2OCNII] / 44.0 / Hnuclei + 16.0 *
                                    ab[IDX_PC2H4II] / 59.0 / Hnuclei + 16.0 *
                                    ab[IDX_CH3NHII] / 30.0 / Hnuclei + 16.0 *
                                    ab[IDX_GCH2CCH2I] / 40.0 / Hnuclei + 9.0 *
                                    ab[IDX_GCH2NHI] / 29.0 / Hnuclei + 16.0 *
                                    ab[IDX_GCH3CCHI] / 40.0 / Hnuclei + 1.0 *
                                    ab[IDX_GHNCI] / 27.0 / Hnuclei + 4.0 *
                                    ab[IDX_H2C4NII] / 64.0 / Hnuclei + 4.0 *
                                    ab[IDX_H2CClII] / 49.0 / Hnuclei + 4.0 *
                                    ab[IDX_H2NOII] / 32.0 / Hnuclei + 1.0 *
                                    ab[IDX_HClII] / 36.0 / Hnuclei + 1.0 *
                                    ab[IDX_HNSII] / 47.0 / Hnuclei + 1.0 *
                                    ab[IDX_HOCII] / 29.0 / Hnuclei + 1.0 *
                                    ab[IDX_HSOII] / 49.0 / Hnuclei + 1.0 *
                                    ab[IDX_HSiO2II] / 61.0 / Hnuclei + 9.0 *
                                    ab[IDX_NH2CNHII] / 43.0 / Hnuclei + 36.0 *
                                    ab[IDX_C2H5OHII] / 46.0 / Hnuclei + 16.0 *
                                    ab[IDX_CH2CHCCHI] / 52.0 / Hnuclei + 9.0 *
                                    ab[IDX_CH2OHCOII] / 59.0 / Hnuclei + 9.0 *
                                    ab[IDX_GCH2CCHI] / 39.0 / Hnuclei + 4.0 *
                                    ab[IDX_GCH2COI] / 42.0 / Hnuclei + 1.0 *
                                    ab[IDX_GHNOI] / 31.0 / Hnuclei + 1.0 *
                                    ab[IDX_HC4NII] / 63.0 / Hnuclei + 4.0 *
                                    ab[IDX_HCNOHII] / 44.0 / Hnuclei + 4.0 *
                                    ab[IDX_HNCOHII] / 44.0 / Hnuclei + 1.0 *
                                    ab[IDX_HOCNII] / 43.0 / Hnuclei + 1.0 *
                                    ab[IDX_HeHII] / 5.0 / Hnuclei + 9.0 *
                                    ab[IDX_PC2H3II] / 58.0 / Hnuclei + 9.0 *
                                    ab[IDX_PNH3II] / 48.0 / Hnuclei + 9.0 *
                                    ab[IDX_CH3CSII] / 59.0 / Hnuclei + 4.0 *
                                    ab[IDX_H2C7NII] / 100.0 / Hnuclei + 4.0 *
                                    ab[IDX_H2C9NII] / 124.0 / Hnuclei + 4.0 *
                                    ab[IDX_H2O2I] / 34.0 / Hnuclei + 9.0 *
                                    ab[IDX_H3S2II] / 67.0 / Hnuclei + 1.0 *
                                    ab[IDX_HONCII] / 43.0 / Hnuclei + 9.0 *
                                    ab[IDX_PCH3II] / 46.0 / Hnuclei + 4.0 *
                                    ab[IDX_PNH2II] / 47.0 / Hnuclei + 9.0 *
                                    ab[IDX_C10H3II] / 123.0 / Hnuclei + 36.0 *
                                    ab[IDX_C2H5CNHII] / 56.0 / Hnuclei + 16.0 *
                                    ab[IDX_CH2NH2II] / 30.0 / Hnuclei + 25.0 *
                                    ab[IDX_CH2OHCH2OII] / 61.0 / Hnuclei + 16.0
                                    * ab[IDX_CH3C5NHII] / 90.0 / Hnuclei + 16.0
                                    * ab[IDX_CH3C7NHII] / 114.0 / Hnuclei + 25.0
                                    * ab[IDX_GC2H5I] / 29.0 / Hnuclei + 9.0 *
                                    ab[IDX_GCOOCH3I] / 59.0 / Hnuclei + 1.0 *
                                    ab[IDX_GHCNI] / 27.0 / Hnuclei + 4.0 *
                                    ab[IDX_H2CNI] / 28.0 / Hnuclei + 4.0 *
                                    ab[IDX_H2SiOII] / 46.0 / Hnuclei + 9.0 *
                                    ab[IDX_H3C5NII] / 77.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCNOII] / 43.0 / Hnuclei + 16.0 *
                                    ab[IDX_CH3COOHII] / 60.0 / Hnuclei + 9.0 *
                                    ab[IDX_GNH3I] / 17.0 / Hnuclei + 4.0 *
                                    ab[IDX_H2ClII] / 37.0 / Hnuclei + 9.0 *
                                    ab[IDX_H3C9NII] / 125.0 / Hnuclei + 25.0 *
                                    ab[IDX_H5C2O2II] / 61.0 / Hnuclei + 4.0 *
                                    ab[IDX_HCOOHII] / 46.0 / Hnuclei + 1.0 *
                                    ab[IDX_HPNII] / 46.0 / Hnuclei + 25.0 *
                                    ab[IDX_CH3COOH2II] / 61.0 / Hnuclei + 36.0 *
                                    ab[IDX_CH3OCH3II] / 46.0 / Hnuclei + 1.0 *
                                    ab[IDX_GC10HI] / 121.0 / Hnuclei + 1.0 *
                                    ab[IDX_GC8HI] / 97.0 / Hnuclei + 9.0 *
                                    ab[IDX_GCH2OHCOI] / 59.0 / Hnuclei + 1.0 *
                                    ab[IDX_GCOOHI] / 45.0 / Hnuclei + 9.0 *
                                    ab[IDX_H3C7NII] / 101.0 / Hnuclei + 9.0 *
                                    ab[IDX_H3SiOII] / 47.0 / Hnuclei + 1.0 *
                                    ab[IDX_HSO2II] / 65.0 / Hnuclei + 4.0 *
                                    ab[IDX_NH2CNI] / 42.0 / Hnuclei + 16.0 *
                                    ab[IDX_CH3C3NHII] / 66.0 / Hnuclei + 49.0 *
                                    ab[IDX_CH3OCH4II] / 47.0 / Hnuclei + 16.0 *
                                    ab[IDX_GC2H4I] / 28.0 / Hnuclei + 1.0 *
                                    ab[IDX_GC9HI] / 109.0 / Hnuclei + 1.0 *
                                    ab[IDX_HC7NII] / 99.0 / Hnuclei + 1.0 *
                                    ab[IDX_HC9NII] / 123.0 / Hnuclei + 1.0 *
                                    ab[IDX_HNCOII] / 43.0 / Hnuclei + 1.0 *
                                    ab[IDX_PC3HII] / 68.0 / Hnuclei + 16.0 *
                                    ab[IDX_PCH4II] / 47.0 / Hnuclei + 16.0 *
                                    ab[IDX_SiH4II] / 32.0 / Hnuclei + 25.0 *
                                    ab[IDX_C2H5CNI] / 55.0 / Hnuclei + 49.0 *
                                    ab[IDX_C2H5OH2II] / 47.0 / Hnuclei + 49.0 *
                                    ab[IDX_C2H7II] / 31.0 / Hnuclei + 16.0 *
                                    ab[IDX_CH2CHCNHII] / 54.0 / Hnuclei + 9.0 *
                                    ab[IDX_COOCH3I] / 59.0 / Hnuclei + 1.0 *
                                    ab[IDX_GHSI] / 33.0 / Hnuclei + 4.0 *
                                    ab[IDX_H2POII] / 49.0 / Hnuclei + 16.0 *
                                    ab[IDX_HCOOCH3II] / 60.0 / Hnuclei + 1.0 *
                                    ab[IDX_HN2OII] / 45.0 / Hnuclei + 4.0 *
                                    ab[IDX_PC2H2II] / 57.0 / Hnuclei + 4.0 *
                                    ab[IDX_SiC3H2II] / 66.0 / Hnuclei + 1.0 *
                                    ab[IDX_SiC4HII] / 77.0 / Hnuclei + 25.0 *
                                    ab[IDX_SiH5II] / 33.0 / Hnuclei + 1.0 *
                                    ab[IDX_SiNCHII] / 55.0 / Hnuclei + 25.0 *
                                    ab[IDX_C4H5II] / 53.0 / Hnuclei + 49.0 *
                                    ab[IDX_C4H7II] / 55.0 / Hnuclei + 16.0 *
                                    ab[IDX_CH2OHCHOII] / 60.0 / Hnuclei + 16.0 *
                                    ab[IDX_CH3CHOII] / 44.0 / Hnuclei + 16.0 *
                                    ab[IDX_GCH2OHCHOI] / 60.0 / Hnuclei + 4.0 *
                                    ab[IDX_GHCOOHI] / 46.0 / Hnuclei + 4.0 *
                                    ab[IDX_H2S2I] / 66.0 / Hnuclei + 1.0 *
                                    ab[IDX_HNSiII] / 43.0 / Hnuclei + 1.0 *
                                    ab[IDX_HOCSII] / 61.0 / Hnuclei + 1.0 *
                                    ab[IDX_PC4HII] / 80.0 / Hnuclei + 9.0 *
                                    ab[IDX_PH3II] / 34.0 / Hnuclei + 16.0 *
                                    ab[IDX_SiCH4II] / 44.0 / Hnuclei + 4.0 *
                                    ab[IDX_C10H2II] / 122.0 / Hnuclei + 9.0 *
                                    ab[IDX_CH3CNII] / 41.0 / Hnuclei + 4.0 *
                                    ab[IDX_GC2H2I] / 26.0 / Hnuclei + 1.0 *
                                    ab[IDX_GC7HI] / 85.0 / Hnuclei + 9.0 *
                                    ab[IDX_GCH3COI] / 43.0 / Hnuclei + 16.0 *
                                    ab[IDX_GHCOOCH3I] / 60.0 / Hnuclei + 1.0 *
                                    ab[IDX_HC4SII] / 81.0 / Hnuclei + 4.0 *
                                    ab[IDX_HC5NHII] / 76.0 / Hnuclei + 9.0 *
                                    ab[IDX_SiC2H3II] / 55.0 / Hnuclei + 1.0 *
                                    ab[IDX_SiC3HI] / 65.0 / Hnuclei + 9.0 *
                                    ab[IDX_SiCH3II] / 43.0 / Hnuclei + 4.0 *
                                    ab[IDX_SiNH2II] / 44.0 / Hnuclei + 16.0 *
                                    ab[IDX_C4H4II] / 52.0 / Hnuclei + 4.0 *
                                    ab[IDX_CH2CNII] / 40.0 / Hnuclei + 36.0 *
                                    ab[IDX_CH3COCH3II] / 58.0 / Hnuclei + 9.0 *
                                    ab[IDX_GCH2OHI] / 31.0 / Hnuclei + 16.0 *
                                    ab[IDX_GCH3OHI] / 32.0 / Hnuclei + 4.0 *
                                    ab[IDX_H2S2II] / 66.0 / Hnuclei + 1.0 *
                                    ab[IDX_HC2PII] / 56.0 / Hnuclei + 1.0 *
                                    ab[IDX_HC5NII] / 75.0 / Hnuclei + 1.0 *
                                    ab[IDX_HFI] / 20.0 / Hnuclei + 1.0 *
                                    ab[IDX_HONCI] / 43.0 / Hnuclei + 1.0 *
                                    ab[IDX_HSiSII] / 61.0 / Hnuclei + 4.0 *
                                    ab[IDX_PCH2II] / 45.0 / Hnuclei + 9.0 *
                                    ab[IDX_CH2OHCOI] / 59.0 / Hnuclei + 9.0 *
                                    ab[IDX_CH3C5NI] / 89.0 / Hnuclei + 9.0 *
                                    ab[IDX_CH3C7NI] / 113.0 / Hnuclei + 1.0 *
                                    ab[IDX_GC3HI] / 37.0 / Hnuclei + 1.0 *
                                    ab[IDX_GC4HI] / 49.0 / Hnuclei + 1.0 *
                                    ab[IDX_GC5HI] / 61.0 / Hnuclei + 1.0 *
                                    ab[IDX_GC6HI] / 73.0 / Hnuclei + 4.0 *
                                    ab[IDX_H2NCII] / 28.0 / Hnuclei + 4.0 *
                                    ab[IDX_H2SiOI] / 46.0 / Hnuclei + 9.0 *
                                    ab[IDX_H3CSII] / 47.0 / Hnuclei + 1.0 *
                                    ab[IDX_HC2OII] / 41.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCPII] / 44.0 / Hnuclei + 9.0 *
                                    ab[IDX_GC2H3I] / 27.0 / Hnuclei + 9.0 *
                                    ab[IDX_GCH3OI] / 31.0 / Hnuclei + 4.0 *
                                    ab[IDX_GH2OI] / 18.0 / Hnuclei + 1.0 *
                                    ab[IDX_HC3SII] / 69.0 / Hnuclei + 9.0 *
                                    ab[IDX_HCOOH2II] / 47.0 / Hnuclei + 1.0 *
                                    ab[IDX_HNSiI] / 43.0 / Hnuclei + 1.0 *
                                    ab[IDX_HPOII] / 48.0 / Hnuclei + 1.0 *
                                    ab[IDX_HS2II] / 65.0 / Hnuclei + 4.0 *
                                    ab[IDX_SiC2H2I] / 54.0 / Hnuclei + 1.0 *
                                    ab[IDX_C2NHII] / 39.0 / Hnuclei + 36.0 *
                                    ab[IDX_C3H6II] / 42.0 / Hnuclei + 49.0 *
                                    ab[IDX_C6H7II] / 79.0 / Hnuclei + 16.0 *
                                    ab[IDX_C8H4II] / 100.0 / Hnuclei + 16.0 *
                                    ab[IDX_CH2OHCHOI] / 60.0 / Hnuclei + 1.0 *
                                    ab[IDX_HC3OII] / 53.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCNOI] / 43.0 / Hnuclei + 1.0 *
                                    ab[IDX_SiC3HII] / 65.0 / Hnuclei + 9.0 *
                                    ab[IDX_SiCH3I] / 43.0 / Hnuclei + 49.0 *
                                    ab[IDX_C3H7II] / 43.0 / Hnuclei + 16.0 *
                                    ab[IDX_C9H4II] / 112.0 / Hnuclei + 4.0 *
                                    ab[IDX_CH2COII] / 42.0 / Hnuclei + 9.0 *
                                    ab[IDX_CH2PHI] / 46.0 / Hnuclei + 9.0 *
                                    ab[IDX_CH3C3NI] / 65.0 / Hnuclei + 16.0 *
                                    ab[IDX_GCH3CHOI] / 44.0 / Hnuclei + 4.0 *
                                    ab[IDX_GH2COI] / 30.0 / Hnuclei + 1.0 *
                                    ab[IDX_HOCNI] / 43.0 / Hnuclei + 1.0 *
                                    ab[IDX_HPOI] / 48.0 / Hnuclei + 1.0 *
                                    ab[IDX_HS2I] / 65.0 / Hnuclei + 4.0 *
                                    ab[IDX_SiC2H2II] / 54.0 / Hnuclei + 16.0 *
                                    ab[IDX_C6H4II] / 76.0 / Hnuclei + 9.0 *
                                    ab[IDX_CH3COI] / 43.0 / Hnuclei + 49.0 *
                                    ab[IDX_CH3COCH4II] / 59.0 / Hnuclei + 16.0 *
                                    ab[IDX_CH3OHII] / 32.0 / Hnuclei + 1.0 *
                                    ab[IDX_COOHI] / 45.0 / Hnuclei + 16.0 *
                                    ab[IDX_C7H4II] / 88.0 / Hnuclei + 25.0 *
                                    ab[IDX_C7H5II] / 89.0 / Hnuclei + 16.0 *
                                    ab[IDX_CH3C6HI] / 88.0 / Hnuclei + 4.0 *
                                    ab[IDX_H2CSII] / 46.0 / Hnuclei + 1.0 *
                                    ab[IDX_HC9NI] / 123.0 / Hnuclei + 16.0 *
                                    ab[IDX_HCOOCH3I] / 60.0 / Hnuclei + 1.0 *
                                    ab[IDX_NCCNHII] / 53.0 / Hnuclei + 4.0 *
                                    ab[IDX_PH2I] / 33.0 / Hnuclei + 1.0 *
                                    ab[IDX_SiC2HI] / 53.0 / Hnuclei + 4.0 *
                                    ab[IDX_CH2CNI] / 40.0 / Hnuclei + 16.0 *
                                    ab[IDX_CH3C4HII] / 64.0 / Hnuclei + 16.0 *
                                    ab[IDX_GCH4I] / 16.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCPI] / 44.0 / Hnuclei + 1.0 *
                                    ab[IDX_HClI] / 36.0 / Hnuclei + 1.0 *
                                    ab[IDX_HNCOI] / 43.0 / Hnuclei + 16.0 *
                                    ab[IDX_CH3C4HI] / 64.0 / Hnuclei + 16.0 *
                                    ab[IDX_CH3COOHI] / 60.0 / Hnuclei + 36.0 *
                                    ab[IDX_CH3OCH3I] / 46.0 / Hnuclei + 1.0 *
                                    ab[IDX_GC2HI] / 25.0 / Hnuclei + 4.0 *
                                    ab[IDX_GNH2I] / 16.0 / Hnuclei + 4.0 *
                                    ab[IDX_H2CSI] / 46.0 / Hnuclei + 4.0 *
                                    ab[IDX_PH2II] / 33.0 / Hnuclei + 25.0 *
                                    ab[IDX_C6H5II] / 77.0 / Hnuclei + 9.0 *
                                    ab[IDX_CH2NHI] / 29.0 / Hnuclei + 9.0 *
                                    ab[IDX_CH3OI] / 31.0 / Hnuclei + 1.0 *
                                    ab[IDX_GHCOI] / 29.0 / Hnuclei + 1.0 *
                                    ab[IDX_HC2PI] / 56.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCSiI] / 41.0 / Hnuclei + 4.0 *
                                    ab[IDX_SiCH2I] / 42.0 / Hnuclei + 1.0 *
                                    ab[IDX_C10HII] / 121.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCSiII] / 41.0 / Hnuclei + 36.0 *
                                    ab[IDX_C2H6II] / 30.0 / Hnuclei + 1.0 *
                                    ab[IDX_C8HII] / 97.0 / Hnuclei + 9.0 *
                                    ab[IDX_CH2OHI] / 31.0 / Hnuclei + 25.0 *
                                    ab[IDX_C5H5II] / 65.0 / Hnuclei + 1.0 *
                                    ab[IDX_GNHI] / 15.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCSI] / 45.0 / Hnuclei + 1.0 *
                                    ab[IDX_SiC2HII] / 53.0 / Hnuclei + 4.0 *
                                    ab[IDX_SiH2I] / 30.0 / Hnuclei + 1.0 *
                                    ab[IDX_C6HII] / 73.0 / Hnuclei + 4.0 *
                                    ab[IDX_C9H2I] / 110.0 / Hnuclei + 9.0 *
                                    ab[IDX_CH2CHCNI] / 53.0 / Hnuclei + 25.0 *
                                    ab[IDX_CH3CHOHII] / 45.0 / Hnuclei + 16.0 *
                                    ab[IDX_CH3CNHII] / 42.0 / Hnuclei + 9.0 *
                                    ab[IDX_CH3COII] / 43.0 / Hnuclei + 1.0 *
                                    ab[IDX_HC7NI] / 99.0 / Hnuclei + 36.0 *
                                    ab[IDX_C6H6I] / 78.0 / Hnuclei + 4.0 *
                                    ab[IDX_HCOOHI] / 46.0 / Hnuclei + 4.0 *
                                    ab[IDX_SiH2II] / 30.0 / Hnuclei + 9.0 *
                                    ab[IDX_SiH3II] / 31.0 / Hnuclei + 4.0 *
                                    ab[IDX_C10H2I] / 122.0 / Hnuclei + 1.0 *
                                    ab[IDX_O2HI] / 33.0 / Hnuclei + 36.0 *
                                    ab[IDX_C2H5OHI] / 46.0 / Hnuclei + 1.0 *
                                    ab[IDX_HNOI] / 31.0 / Hnuclei + 9.0 *
                                    ab[IDX_SiH3I] / 31.0 / Hnuclei + 25.0 *
                                    ab[IDX_CH3OH2II] / 33.0 / Hnuclei + 9.0 *
                                    ab[IDX_C6H3II] / 75.0 / Hnuclei + 9.0 *
                                    ab[IDX_C8H3II] / 99.0 / Hnuclei + 1.0 *
                                    ab[IDX_C9HII] / 109.0 / Hnuclei + 9.0 *
                                    ab[IDX_C9H3II] / 111.0 / Hnuclei + 4.0 *
                                    ab[IDX_CH2COI] / 42.0 / Hnuclei + 1.0 *
                                    ab[IDX_HNC3I] / 51.0 / Hnuclei + 4.0 *
                                    ab[IDX_SiCH2II] / 42.0 / Hnuclei + 1.0 *
                                    ab[IDX_SiHII] / 29.0 / Hnuclei + 36.0 *
                                    ab[IDX_CH3COCH3I] / 58.0 / Hnuclei + 1.0 *
                                    ab[IDX_HC3NII] / 51.0 / Hnuclei + 4.0 *
                                    ab[IDX_C8H2I] / 98.0 / Hnuclei + 1.0 *
                                    ab[IDX_GOHI] / 17.0 / Hnuclei + 1.0 *
                                    ab[IDX_HC5NI] / 75.0 / Hnuclei + 1.0 *
                                    ab[IDX_SiHI] / 29.0 / Hnuclei + 16.0 *
                                    ab[IDX_SiH4I] / 32.0 / Hnuclei + 1.0 *
                                    ab[IDX_C7HII] / 85.0 / Hnuclei + 4.0 *
                                    ab[IDX_C9H2II] / 110.0 / Hnuclei + 1.0 *
                                    ab[IDX_PHI] / 32.0 / Hnuclei + 4.0 *
                                    ab[IDX_C7H2I] / 86.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCSII] / 45.0 / Hnuclei + 1.0 *
                                    ab[IDX_PHII] / 32.0 / Hnuclei + 16.0 *
                                    ab[IDX_C3H4II] / 40.0 / Hnuclei + 4.0 *
                                    ab[IDX_C8H2II] / 98.0 / Hnuclei + 4.0 *
                                    ab[IDX_GCH2I] / 14.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCO2II] / 45.0 / Hnuclei + 9.0 *
                                    ab[IDX_C5H3II] / 63.0 / Hnuclei + 4.0 *
                                    ab[IDX_HC3NHII] / 52.0 / Hnuclei + 1.0 *
                                    ab[IDX_C5HII] / 61.0 / Hnuclei + 16.0 *
                                    ab[IDX_CH3CHOI] / 44.0 / Hnuclei + 25.0 *
                                    ab[IDX_C3H5II] / 41.0 / Hnuclei + 9.0 *
                                    ab[IDX_C7H3II] / 87.0 / Hnuclei + 9.0 *
                                    ab[IDX_CH3CNI] / 41.0 / Hnuclei + 4.0 *
                                    ab[IDX_C7H2II] / 86.0 / Hnuclei + 9.0 *
                                    ab[IDX_H3SII] / 35.0 / Hnuclei + 1.0 *
                                    ab[IDX_O2HII] / 33.0 / Hnuclei + 4.0 *
                                    ab[IDX_C6H2II] / 74.0 / Hnuclei + 16.0 *
                                    ab[IDX_CH4II] / 16.0 / Hnuclei + 1.0 *
                                    ab[IDX_HNOII] / 31.0 / Hnuclei + 4.0 *
                                    ab[IDX_H2CCCI] / 38.0 / Hnuclei + 4.0 *
                                    ab[IDX_C6H2I] / 74.0 / Hnuclei + 25.0 *
                                    ab[IDX_C2H5II] / 29.0 / Hnuclei + 4.0 *
                                    ab[IDX_C5H2I] / 62.0 / Hnuclei + 4.0 *
                                    ab[IDX_C5H2II] / 62.0 / Hnuclei + 16.0 *
                                    ab[IDX_CH2CCH2I] / 40.0 / Hnuclei + 36.0 *
                                    ab[IDX_C2H6I] / 30.0 / Hnuclei + 4.0 *
                                    ab[IDX_C3H2I] / 38.0 / Hnuclei + 25.0 *
                                    ab[IDX_C2H5I] / 29.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCNII] / 27.0 / Hnuclei + 36.0 *
                                    ab[IDX_CH3CHCH2I] / 42.0 / Hnuclei + 1.0 *
                                    ab[IDX_C4HII] / 49.0 / Hnuclei + 1.0 *
                                    ab[IDX_HSII] / 33.0 / Hnuclei + 1.0 *
                                    ab[IDX_NHII] / 15.0 / Hnuclei + 1.0 *
                                    ab[IDX_CHM] / 13.0 / Hnuclei + 4.0 *
                                    ab[IDX_H2II] / 2.0 / Hnuclei + 9.0 *
                                    ab[IDX_GCH3I] / 15.0 / Hnuclei + 4.0 *
                                    ab[IDX_NH2II] / 16.0 / Hnuclei + 1.0 *
                                    ab[IDX_C2HM] / 25.0 / Hnuclei + 1.0 *
                                    ab[IDX_C4HM] / 49.0 / Hnuclei + 16.0 *
                                    ab[IDX_CH3CCHI] / 40.0 / Hnuclei + 25.0 *
                                    ab[IDX_CH5II] / 17.0 / Hnuclei + 1.0 *
                                    ab[IDX_HC2SII] / 57.0 / Hnuclei + 1.0 *
                                    ab[IDX_OHM] / 17.0 / Hnuclei + 1.0 *
                                    ab[IDX_C10HM] / 121.0 / Hnuclei + 1.0 *
                                    ab[IDX_C3HM] / 37.0 / Hnuclei + 1.0 *
                                    ab[IDX_C6HM] / 73.0 / Hnuclei + 1.0 *
                                    ab[IDX_C8HM] / 97.0 / Hnuclei + 1.0 *
                                    ab[IDX_C7HM] / 85.0 / Hnuclei + 1.0 *
                                    ab[IDX_C9HM] / 109.0 / Hnuclei + 1.0 *
                                    ab[IDX_C5HM] / 61.0 / Hnuclei + 1.0 *
                                    ab[IDX_HSI] / 33.0 / Hnuclei + 16.0 *
                                    ab[IDX_CH3OHI] / 32.0 / Hnuclei + 1.0 *
                                    ab[IDX_HM] / 1.0 / Hnuclei + 1.0 *
                                    ab[IDX_HC3NI] / 51.0 / Hnuclei + 1.0 *
                                    ab[IDX_C3HII] / 37.0 / Hnuclei + 9.0 *
                                    ab[IDX_C3H3II] / 39.0 / Hnuclei + 9.0 *
                                    ab[IDX_H3COII] / 31.0 / Hnuclei + 1.0 *
                                    ab[IDX_GCHI] / 13.0 / Hnuclei + 1.0 *
                                    ab[IDX_SiOHII] / 45.0 / Hnuclei + 4.0 *
                                    ab[IDX_NH2I] / 16.0 / Hnuclei + 9.0 *
                                    ab[IDX_C4H3I] / 51.0 / Hnuclei + 1.0 *
                                    ab[IDX_C2HII] / 25.0 / Hnuclei + 4.0 *
                                    ab[IDX_C3H2II] / 38.0 / Hnuclei + 4.0 *
                                    ab[IDX_CH2II] / 14.0 / Hnuclei + 1.0 *
                                    ab[IDX_OHII] / 17.0 / Hnuclei + 4.0 *
                                    ab[IDX_H2OII] / 18.0 / Hnuclei + 4.0 *
                                    ab[IDX_GH2I] / 2.0 / Hnuclei + 1.0 *
                                    ab[IDX_CHII] / 13.0 / Hnuclei + 1.0 *
                                    ab[IDX_C10HI] / 121.0 / Hnuclei + 1.0 *
                                    ab[IDX_C9HI] / 109.0 / Hnuclei + 1.0 *
                                    ab[IDX_C8HI] / 97.0 / Hnuclei + 1.0 *
                                    ab[IDX_NHI] / 15.0 / Hnuclei + 1.0 *
                                    ab[IDX_C7HI] / 85.0 / Hnuclei + 16.0 *
                                    ab[IDX_C2H4II] / 28.0 / Hnuclei + 9.0 *
                                    ab[IDX_C4H3II] / 51.0 / Hnuclei + 1.0 *
                                    ab[IDX_N2HII] / 29.0 / Hnuclei + 9.0 *
                                    ab[IDX_CH2CCHI] / 39.0 / Hnuclei + 9.0 *
                                    ab[IDX_CH2CCHII] / 39.0 / Hnuclei + 1.0 *
                                    ab[IDX_C6HI] / 73.0 / Hnuclei + 1.0 *
                                    ab[IDX_C5HI] / 61.0 / Hnuclei + 4.0 *
                                    ab[IDX_H2SII] / 34.0 / Hnuclei + 1.0 *
                                    ab[IDX_C3HI] / 37.0 / Hnuclei + 4.0 *
                                    ab[IDX_H2COII] / 30.0 / Hnuclei + 1.0 *
                                    ab[IDX_C4HI] / 49.0 / Hnuclei + 4.0 *
                                    ab[IDX_C4H2I] / 50.0 / Hnuclei + 4.0 *
                                    ab[IDX_C4H2II] / 50.0 / Hnuclei + 4.0 *
                                    ab[IDX_HCNHII] / 28.0 / Hnuclei + 4.0 *
                                    ab[IDX_CH2I] / 14.0 / Hnuclei + 1.0 *
                                    ab[IDX_HNCI] / 27.0 / Hnuclei + 9.0 *
                                    ab[IDX_NH3II] / 17.0 / Hnuclei + 16.0 *
                                    ab[IDX_C2H4I] / 28.0 / Hnuclei + 9.0 *
                                    ab[IDX_C2H3I] / 27.0 / Hnuclei + 16.0 *
                                    ab[IDX_NH4II] / 18.0 / Hnuclei + 4.0 *
                                    ab[IDX_H2SI] / 34.0 / Hnuclei + 9.0 *
                                    ab[IDX_C2H3II] / 27.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCOI] / 29.0 / Hnuclei + 16.0 *
                                    ab[IDX_CH4I] / 16.0 / Hnuclei + 1.0 *
                                    ab[IDX_C2HI] / 25.0 / Hnuclei + 4.0 *
                                    ab[IDX_H2COI] / 30.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCNI] / 27.0 / Hnuclei + 1.0 *
                                    ab[IDX_CHI] / 13.0 / Hnuclei + 1.0 *
                                    ab[IDX_OHI] / 17.0 / Hnuclei + 9.0 *
                                    ab[IDX_NH3I] / 17.0 / Hnuclei + 4.0 *
                                    ab[IDX_C2H2II] / 26.0 / Hnuclei + 9.0 *
                                    ab[IDX_CH3II] / 15.0 / Hnuclei + 4.0 *
                                    ab[IDX_C2H2I] / 26.0 / Hnuclei + 1.0 *
                                    ab[IDX_GHI] / 1.0 / Hnuclei + 9.0 *
                                    ab[IDX_CH3I] / 15.0 / Hnuclei + 9.0 *
                                    ab[IDX_H3OII] / 19.0 / Hnuclei + 4.0 *
                                    ab[IDX_H2OI] / 18.0 / Hnuclei + 1.0 *
                                    ab[IDX_HII] / 1.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCOII] / 29.0 / Hnuclei + 9.0 *
                                    ab[IDX_H3II] / 3.0 / Hnuclei + 4.0 *
                                    ab[IDX_H2I] / 2.0 / Hnuclei + 1.0 *
                                    ab[IDX_HI] / 1.0 / Hnuclei;
        // clang-format on

    return NAUNET_SUCCESS;
}

// clang-format off
__host__ int RenormAbundance(realtype *rptr, realtype *ab) {
    
    ab[IDX_GFeI] = ab[IDX_GFeI] * (56.0 * rptr[IDX_ELEM_Fe] / 56.0);
    ab[IDX_GHeI] = ab[IDX_GHeI] * (4.0 * rptr[IDX_ELEM_He] / 4.0);
    ab[IDX_GMgI] = ab[IDX_GMgI] * (24.0 * rptr[IDX_ELEM_Mg] / 24.0);
    ab[IDX_GNaI] = ab[IDX_GNaI] * (23.0 * rptr[IDX_ELEM_Na] / 23.0);
    ab[IDX_GH2S2I] = ab[IDX_GH2S2I] * (64.0 * rptr[IDX_ELEM_S] / 66.0 + 2.0 * rptr[IDX_ELEM_H] / 66.0);
    ab[IDX_GC4H6I] = ab[IDX_GC4H6I] * (48.0 * rptr[IDX_ELEM_C] / 54.0 + 6.0 * rptr[IDX_ELEM_H] / 54.0);
    ab[IDX_GC4SI] = ab[IDX_GC4SI] * (32.0 * rptr[IDX_ELEM_S] / 80.0 + 48.0 * rptr[IDX_ELEM_C] / 80.0);
    ab[IDX_GCClI] = ab[IDX_GCClI] * (35.0 * rptr[IDX_ELEM_Cl] / 47.0 + 12.0 * rptr[IDX_ELEM_C] / 47.0);
    ab[IDX_GCH3C3NI] = ab[IDX_GCH3C3NI] * (14.0 * rptr[IDX_ELEM_N] / 65.0 + 48.0 * rptr[IDX_ELEM_C] / 65.0 + 3.0 * rptr[IDX_ELEM_H] / 65.0);
    ab[IDX_GCH3C4HI] = ab[IDX_GCH3C4HI] * (60.0 * rptr[IDX_ELEM_C] / 64.0 + 4.0 * rptr[IDX_ELEM_H] / 64.0);
    ab[IDX_GCH3C5NI] = ab[IDX_GCH3C5NI] * (14.0 * rptr[IDX_ELEM_N] / 89.0 + 72.0 * rptr[IDX_ELEM_C] / 89.0 + 3.0 * rptr[IDX_ELEM_H] / 89.0);
    ab[IDX_GCH3C6HI] = ab[IDX_GCH3C6HI] * (84.0 * rptr[IDX_ELEM_C] / 88.0 + 4.0 * rptr[IDX_ELEM_H] / 88.0);
    ab[IDX_GCH3C7NI] = ab[IDX_GCH3C7NI] * (14.0 * rptr[IDX_ELEM_N] / 113.0 + 96.0 * rptr[IDX_ELEM_C] / 113.0 + 3.0 * rptr[IDX_ELEM_H] / 113.0);
    ab[IDX_GClOI] = ab[IDX_GClOI] * (35.0 * rptr[IDX_ELEM_Cl] / 51.0 + 16.0 * rptr[IDX_ELEM_O] / 51.0);
    ab[IDX_GHC2PI] = ab[IDX_GHC2PI] * (31.0 * rptr[IDX_ELEM_P] / 56.0 + 24.0 * rptr[IDX_ELEM_C] / 56.0 + 1.0 * rptr[IDX_ELEM_H] / 56.0);
    ab[IDX_GHClI] = ab[IDX_GHClI] * (35.0 * rptr[IDX_ELEM_Cl] / 36.0 + 1.0 * rptr[IDX_ELEM_H] / 36.0);
    ab[IDX_GHFI] = ab[IDX_GHFI] * (19.0 * rptr[IDX_ELEM_F] / 20.0 + 1.0 * rptr[IDX_ELEM_H] / 20.0);
    ab[IDX_GHNSiI] = ab[IDX_GHNSiI] * (28.0 * rptr[IDX_ELEM_Si] / 43.0 + 14.0 * rptr[IDX_ELEM_N] / 43.0 + 1.0 * rptr[IDX_ELEM_H] / 43.0);
    ab[IDX_GHPOI] = ab[IDX_GHPOI] * (31.0 * rptr[IDX_ELEM_P] / 48.0 + 16.0 * rptr[IDX_ELEM_O] / 48.0 + 1.0 * rptr[IDX_ELEM_H] / 48.0);
    ab[IDX_GNH2CNI] = ab[IDX_GNH2CNI] * (28.0 * rptr[IDX_ELEM_N] / 42.0 + 12.0 * rptr[IDX_ELEM_C] / 42.0 + 2.0 * rptr[IDX_ELEM_H] / 42.0);
    ab[IDX_GNO2I] = ab[IDX_GNO2I] * (14.0 * rptr[IDX_ELEM_N] / 46.0 + 32.0 * rptr[IDX_ELEM_O] / 46.0);
    ab[IDX_GPNI] = ab[IDX_GPNI] * (31.0 * rptr[IDX_ELEM_P] / 45.0 + 14.0 * rptr[IDX_ELEM_N] / 45.0);
    ab[IDX_GSiC3HI] = ab[IDX_GSiC3HI] * (28.0 * rptr[IDX_ELEM_Si] / 65.0 + 36.0 * rptr[IDX_ELEM_C] / 65.0 + 1.0 * rptr[IDX_ELEM_H] / 65.0);
    ab[IDX_GSiCH3I] = ab[IDX_GSiCH3I] * (28.0 * rptr[IDX_ELEM_Si] / 43.0 + 12.0 * rptr[IDX_ELEM_C] / 43.0 + 3.0 * rptr[IDX_ELEM_H] / 43.0);
    ab[IDX_GSiO2I] = ab[IDX_GSiO2I] * (28.0 * rptr[IDX_ELEM_Si] / 60.0 + 32.0 * rptr[IDX_ELEM_O] / 60.0);
    ab[IDX_GSiSI] = ab[IDX_GSiSI] * (28.0 * rptr[IDX_ELEM_Si] / 60.0 + 32.0 * rptr[IDX_ELEM_S] / 60.0);
    ab[IDX_GFI] = ab[IDX_GFI] * (19.0 * rptr[IDX_ELEM_F] / 19.0);
    ab[IDX_GH2SiOI] = ab[IDX_GH2SiOI] * (28.0 * rptr[IDX_ELEM_Si] / 46.0 + 16.0 * rptr[IDX_ELEM_O] / 46.0 + 2.0 * rptr[IDX_ELEM_H] / 46.0);
    ab[IDX_GNCCNI] = ab[IDX_GNCCNI] * (28.0 * rptr[IDX_ELEM_N] / 52.0 + 24.0 * rptr[IDX_ELEM_C] / 52.0);
    ab[IDX_GS2I] = ab[IDX_GS2I] * (64.0 * rptr[IDX_ELEM_S] / 64.0);
    ab[IDX_GSiC2HI] = ab[IDX_GSiC2HI] * (28.0 * rptr[IDX_ELEM_Si] / 53.0 + 24.0 * rptr[IDX_ELEM_C] / 53.0 + 1.0 * rptr[IDX_ELEM_H] / 53.0);
    ab[IDX_C2H4CNI] = ab[IDX_C2H4CNI] * (14.0 * rptr[IDX_ELEM_N] / 54.0 + 36.0 * rptr[IDX_ELEM_C] / 54.0 + 4.0 * rptr[IDX_ELEM_H] / 54.0);
    ab[IDX_GC3SI] = ab[IDX_GC3SI] * (32.0 * rptr[IDX_ELEM_S] / 68.0 + 36.0 * rptr[IDX_ELEM_C] / 68.0);
    ab[IDX_GC4NI] = ab[IDX_GC4NI] * (14.0 * rptr[IDX_ELEM_N] / 62.0 + 48.0 * rptr[IDX_ELEM_C] / 62.0);
    ab[IDX_GC4PI] = ab[IDX_GC4PI] * (31.0 * rptr[IDX_ELEM_P] / 79.0 + 48.0 * rptr[IDX_ELEM_C] / 79.0);
    ab[IDX_GCH2PHI] = ab[IDX_GCH2PHI] * (31.0 * rptr[IDX_ELEM_P] / 46.0 + 12.0 * rptr[IDX_ELEM_C] / 46.0 + 3.0 * rptr[IDX_ELEM_H] / 46.0);
    ab[IDX_GCH3CHCH2I] = ab[IDX_GCH3CHCH2I] * (36.0 * rptr[IDX_ELEM_C] / 42.0 + 6.0 * rptr[IDX_ELEM_H] / 42.0);
    ab[IDX_GCH3COOHI] = ab[IDX_GCH3COOHI] * (32.0 * rptr[IDX_ELEM_O] / 60.0 + 24.0 * rptr[IDX_ELEM_C] / 60.0 + 4.0 * rptr[IDX_ELEM_H] / 60.0);
    ab[IDX_GH2SI] = ab[IDX_GH2SI] * (32.0 * rptr[IDX_ELEM_S] / 34.0 + 2.0 * rptr[IDX_ELEM_H] / 34.0);
    ab[IDX_GHCNOI] = ab[IDX_GHCNOI] * (14.0 * rptr[IDX_ELEM_N] / 43.0 + 16.0 * rptr[IDX_ELEM_O] / 43.0 + 12.0 * rptr[IDX_ELEM_C] / 43.0 + 1.0 * rptr[IDX_ELEM_H] / 43.0);
    ab[IDX_GHCPI] = ab[IDX_GHCPI] * (31.0 * rptr[IDX_ELEM_P] / 44.0 + 12.0 * rptr[IDX_ELEM_C] / 44.0 + 1.0 * rptr[IDX_ELEM_H] / 44.0);
    ab[IDX_GHNC3I] = ab[IDX_GHNC3I] * (14.0 * rptr[IDX_ELEM_N] / 51.0 + 36.0 * rptr[IDX_ELEM_C] / 51.0 + 1.0 * rptr[IDX_ELEM_H] / 51.0);
    ab[IDX_GHNCOI] = ab[IDX_GHNCOI] * (14.0 * rptr[IDX_ELEM_N] / 43.0 + 16.0 * rptr[IDX_ELEM_O] / 43.0 + 12.0 * rptr[IDX_ELEM_C] / 43.0 + 1.0 * rptr[IDX_ELEM_H] / 43.0);
    ab[IDX_GHOCNI] = ab[IDX_GHOCNI] * (14.0 * rptr[IDX_ELEM_N] / 43.0 + 16.0 * rptr[IDX_ELEM_O] / 43.0 + 12.0 * rptr[IDX_ELEM_C] / 43.0 + 1.0 * rptr[IDX_ELEM_H] / 43.0);
    ab[IDX_GHONCI] = ab[IDX_GHONCI] * (14.0 * rptr[IDX_ELEM_N] / 43.0 + 16.0 * rptr[IDX_ELEM_O] / 43.0 + 12.0 * rptr[IDX_ELEM_C] / 43.0 + 1.0 * rptr[IDX_ELEM_H] / 43.0);
    ab[IDX_GHS2I] = ab[IDX_GHS2I] * (64.0 * rptr[IDX_ELEM_S] / 65.0 + 1.0 * rptr[IDX_ELEM_H] / 65.0);
    ab[IDX_GN2OI] = ab[IDX_GN2OI] * (28.0 * rptr[IDX_ELEM_N] / 44.0 + 16.0 * rptr[IDX_ELEM_O] / 44.0);
    ab[IDX_GPH2I] = ab[IDX_GPH2I] * (31.0 * rptr[IDX_ELEM_P] / 33.0 + 2.0 * rptr[IDX_ELEM_H] / 33.0);
    ab[IDX_GPOI] = ab[IDX_GPOI] * (31.0 * rptr[IDX_ELEM_P] / 47.0 + 16.0 * rptr[IDX_ELEM_O] / 47.0);
    ab[IDX_GSiC2H2I] = ab[IDX_GSiC2H2I] * (28.0 * rptr[IDX_ELEM_Si] / 54.0 + 24.0 * rptr[IDX_ELEM_C] / 54.0 + 2.0 * rptr[IDX_ELEM_H] / 54.0);
    ab[IDX_GSiC4I] = ab[IDX_GSiC4I] * (28.0 * rptr[IDX_ELEM_Si] / 76.0 + 48.0 * rptr[IDX_ELEM_C] / 76.0);
    ab[IDX_GSiNCI] = ab[IDX_GSiNCI] * (28.0 * rptr[IDX_ELEM_Si] / 54.0 + 14.0 * rptr[IDX_ELEM_N] / 54.0 + 12.0 * rptr[IDX_ELEM_C] / 54.0);
    ab[IDX_HC2OI] = ab[IDX_HC2OI] * (16.0 * rptr[IDX_ELEM_O] / 41.0 + 24.0 * rptr[IDX_ELEM_C] / 41.0 + 1.0 * rptr[IDX_ELEM_H] / 41.0);
    ab[IDX_GC3PI] = ab[IDX_GC3PI] * (31.0 * rptr[IDX_ELEM_P] / 67.0 + 36.0 * rptr[IDX_ELEM_C] / 67.0);
    ab[IDX_GC6H6I] = ab[IDX_GC6H6I] * (72.0 * rptr[IDX_ELEM_C] / 78.0 + 6.0 * rptr[IDX_ELEM_H] / 78.0);
    ab[IDX_GH2O2I] = ab[IDX_GH2O2I] * (32.0 * rptr[IDX_ELEM_O] / 34.0 + 2.0 * rptr[IDX_ELEM_H] / 34.0);
    ab[IDX_GHCSiI] = ab[IDX_GHCSiI] * (28.0 * rptr[IDX_ELEM_Si] / 41.0 + 12.0 * rptr[IDX_ELEM_C] / 41.0 + 1.0 * rptr[IDX_ELEM_H] / 41.0);
    ab[IDX_GPHI] = ab[IDX_GPHI] * (31.0 * rptr[IDX_ELEM_P] / 32.0 + 1.0 * rptr[IDX_ELEM_H] / 32.0);
    ab[IDX_GSO2I] = ab[IDX_GSO2I] * (32.0 * rptr[IDX_ELEM_S] / 64.0 + 32.0 * rptr[IDX_ELEM_O] / 64.0);
    ab[IDX_GSiH4I] = ab[IDX_GSiH4I] * (28.0 * rptr[IDX_ELEM_Si] / 32.0 + 4.0 * rptr[IDX_ELEM_H] / 32.0);
    ab[IDX_NCCNCH3II] = ab[IDX_NCCNCH3II] * (28.0 * rptr[IDX_ELEM_N] / 67.0 + 36.0 * rptr[IDX_ELEM_C] / 67.0 + 3.0 * rptr[IDX_ELEM_H] / 67.0);
    ab[IDX_GC2H4CNI] = ab[IDX_GC2H4CNI] * (14.0 * rptr[IDX_ELEM_N] / 54.0 + 36.0 * rptr[IDX_ELEM_C] / 54.0 + 4.0 * rptr[IDX_ELEM_H] / 54.0);
    ab[IDX_GC2SI] = ab[IDX_GC2SI] * (32.0 * rptr[IDX_ELEM_S] / 56.0 + 24.0 * rptr[IDX_ELEM_C] / 56.0);
    ab[IDX_GC3OI] = ab[IDX_GC3OI] * (16.0 * rptr[IDX_ELEM_O] / 52.0 + 36.0 * rptr[IDX_ELEM_C] / 52.0);
    ab[IDX_GC4H3I] = ab[IDX_GC4H3I] * (48.0 * rptr[IDX_ELEM_C] / 51.0 + 3.0 * rptr[IDX_ELEM_H] / 51.0);
    ab[IDX_GC7H2I] = ab[IDX_GC7H2I] * (84.0 * rptr[IDX_ELEM_C] / 86.0 + 2.0 * rptr[IDX_ELEM_H] / 86.0);
    ab[IDX_GC8H2I] = ab[IDX_GC8H2I] * (96.0 * rptr[IDX_ELEM_C] / 98.0 + 2.0 * rptr[IDX_ELEM_H] / 98.0);
    ab[IDX_GC9H2I] = ab[IDX_GC9H2I] * (108.0 * rptr[IDX_ELEM_C] / 110.0 + 2.0 * rptr[IDX_ELEM_H] / 110.0);
    ab[IDX_GCH2CNI] = ab[IDX_GCH2CNI] * (14.0 * rptr[IDX_ELEM_N] / 40.0 + 24.0 * rptr[IDX_ELEM_C] / 40.0 + 2.0 * rptr[IDX_ELEM_H] / 40.0);
    ab[IDX_GCH3CNI] = ab[IDX_GCH3CNI] * (14.0 * rptr[IDX_ELEM_N] / 41.0 + 24.0 * rptr[IDX_ELEM_C] / 41.0 + 3.0 * rptr[IDX_ELEM_H] / 41.0);
    ab[IDX_GH2CNI] = ab[IDX_GH2CNI] * (14.0 * rptr[IDX_ELEM_N] / 28.0 + 12.0 * rptr[IDX_ELEM_C] / 28.0 + 2.0 * rptr[IDX_ELEM_H] / 28.0);
    ab[IDX_GH2CSI] = ab[IDX_GH2CSI] * (32.0 * rptr[IDX_ELEM_S] / 46.0 + 12.0 * rptr[IDX_ELEM_C] / 46.0 + 2.0 * rptr[IDX_ELEM_H] / 46.0);
    ab[IDX_GHC2OI] = ab[IDX_GHC2OI] * (16.0 * rptr[IDX_ELEM_O] / 41.0 + 24.0 * rptr[IDX_ELEM_C] / 41.0 + 1.0 * rptr[IDX_ELEM_H] / 41.0);
    ab[IDX_GHC5NI] = ab[IDX_GHC5NI] * (14.0 * rptr[IDX_ELEM_N] / 75.0 + 60.0 * rptr[IDX_ELEM_C] / 75.0 + 1.0 * rptr[IDX_ELEM_H] / 75.0);
    ab[IDX_GHC7NI] = ab[IDX_GHC7NI] * (14.0 * rptr[IDX_ELEM_N] / 99.0 + 84.0 * rptr[IDX_ELEM_C] / 99.0 + 1.0 * rptr[IDX_ELEM_H] / 99.0);
    ab[IDX_GHC9NI] = ab[IDX_GHC9NI] * (14.0 * rptr[IDX_ELEM_N] / 123.0 + 108.0 * rptr[IDX_ELEM_C] / 123.0 + 1.0 * rptr[IDX_ELEM_H] / 123.0);
    ab[IDX_GHCCNI] = ab[IDX_GHCCNI] * (14.0 * rptr[IDX_ELEM_N] / 39.0 + 24.0 * rptr[IDX_ELEM_C] / 39.0 + 1.0 * rptr[IDX_ELEM_H] / 39.0);
    ab[IDX_GHCSI] = ab[IDX_GHCSI] * (32.0 * rptr[IDX_ELEM_S] / 45.0 + 12.0 * rptr[IDX_ELEM_C] / 45.0 + 1.0 * rptr[IDX_ELEM_H] / 45.0);
    ab[IDX_GSiC3I] = ab[IDX_GSiC3I] * (28.0 * rptr[IDX_ELEM_Si] / 64.0 + 36.0 * rptr[IDX_ELEM_C] / 64.0);
    ab[IDX_GSiH3I] = ab[IDX_GSiH3I] * (28.0 * rptr[IDX_ELEM_Si] / 31.0 + 3.0 * rptr[IDX_ELEM_H] / 31.0);
    ab[IDX_GSiNI] = ab[IDX_GSiNI] * (28.0 * rptr[IDX_ELEM_Si] / 42.0 + 14.0 * rptr[IDX_ELEM_N] / 42.0);
    ab[IDX_GSiOI] = ab[IDX_GSiOI] * (28.0 * rptr[IDX_ELEM_Si] / 44.0 + 16.0 * rptr[IDX_ELEM_O] / 44.0);
    ab[IDX_HCCNI] = ab[IDX_HCCNI] * (14.0 * rptr[IDX_ELEM_N] / 39.0 + 24.0 * rptr[IDX_ELEM_C] / 39.0 + 1.0 * rptr[IDX_ELEM_H] / 39.0);
    ab[IDX_C3H2OII] = ab[IDX_C3H2OII] * (16.0 * rptr[IDX_ELEM_O] / 54.0 + 36.0 * rptr[IDX_ELEM_C] / 54.0 + 2.0 * rptr[IDX_ELEM_H] / 54.0);
    ab[IDX_CFII] = ab[IDX_CFII] * (19.0 * rptr[IDX_ELEM_F] / 31.0 + 12.0 * rptr[IDX_ELEM_C] / 31.0);
    ab[IDX_ClOII] = ab[IDX_ClOII] * (35.0 * rptr[IDX_ELEM_Cl] / 51.0 + 16.0 * rptr[IDX_ELEM_O] / 51.0);
    ab[IDX_GC2H6I] = ab[IDX_GC2H6I] * (24.0 * rptr[IDX_ELEM_C] / 30.0 + 6.0 * rptr[IDX_ELEM_H] / 30.0);
    ab[IDX_GC3H2I] = ab[IDX_GC3H2I] * (36.0 * rptr[IDX_ELEM_C] / 38.0 + 2.0 * rptr[IDX_ELEM_H] / 38.0);
    ab[IDX_GC5H2I] = ab[IDX_GC5H2I] * (60.0 * rptr[IDX_ELEM_C] / 62.0 + 2.0 * rptr[IDX_ELEM_H] / 62.0);
    ab[IDX_GCCPI] = ab[IDX_GCCPI] * (31.0 * rptr[IDX_ELEM_P] / 55.0 + 24.0 * rptr[IDX_ELEM_C] / 55.0);
    ab[IDX_GCH2CHCCHI] = ab[IDX_GCH2CHCCHI] * (48.0 * rptr[IDX_ELEM_C] / 52.0 + 4.0 * rptr[IDX_ELEM_H] / 52.0);
    ab[IDX_GCH3COCH3I] = ab[IDX_GCH3COCH3I] * (16.0 * rptr[IDX_ELEM_O] / 58.0 + 36.0 * rptr[IDX_ELEM_C] / 58.0 + 6.0 * rptr[IDX_ELEM_H] / 58.0);
    ab[IDX_GCH3OCH3I] = ab[IDX_GCH3OCH3I] * (16.0 * rptr[IDX_ELEM_O] / 46.0 + 24.0 * rptr[IDX_ELEM_C] / 46.0 + 6.0 * rptr[IDX_ELEM_H] / 46.0);
    ab[IDX_GClI] = ab[IDX_GClI] * (35.0 * rptr[IDX_ELEM_Cl] / 35.0);
    ab[IDX_GSiCH2I] = ab[IDX_GSiCH2I] * (28.0 * rptr[IDX_ELEM_Si] / 42.0 + 12.0 * rptr[IDX_ELEM_C] / 42.0 + 2.0 * rptr[IDX_ELEM_H] / 42.0);
    ab[IDX_GSiHI] = ab[IDX_GSiHI] * (28.0 * rptr[IDX_ELEM_Si] / 29.0 + 1.0 * rptr[IDX_ELEM_H] / 29.0);
    ab[IDX_GSiH2I] = ab[IDX_GSiH2I] * (28.0 * rptr[IDX_ELEM_Si] / 30.0 + 2.0 * rptr[IDX_ELEM_H] / 30.0);
    ab[IDX_H3C3OII] = ab[IDX_H3C3OII] * (16.0 * rptr[IDX_ELEM_O] / 55.0 + 36.0 * rptr[IDX_ELEM_C] / 55.0 + 3.0 * rptr[IDX_ELEM_H] / 55.0);
    ab[IDX_HFII] = ab[IDX_HFII] * (19.0 * rptr[IDX_ELEM_F] / 20.0 + 1.0 * rptr[IDX_ELEM_H] / 20.0);
    ab[IDX_SiFII] = ab[IDX_SiFII] * (19.0 * rptr[IDX_ELEM_F] / 47.0 + 28.0 * rptr[IDX_ELEM_Si] / 47.0);
    ab[IDX_C8H5II] = ab[IDX_C8H5II] * (96.0 * rptr[IDX_ELEM_C] / 101.0 + 5.0 * rptr[IDX_ELEM_H] / 101.0);
    ab[IDX_C9H5II] = ab[IDX_C9H5II] * (108.0 * rptr[IDX_ELEM_C] / 113.0 + 5.0 * rptr[IDX_ELEM_H] / 113.0);
    ab[IDX_CH3C3NII] = ab[IDX_CH3C3NII] * (14.0 * rptr[IDX_ELEM_N] / 65.0 + 48.0 * rptr[IDX_ELEM_C] / 65.0 + 3.0 * rptr[IDX_ELEM_H] / 65.0);
    ab[IDX_GC10H2I] = ab[IDX_GC10H2I] * (120.0 * rptr[IDX_ELEM_C] / 122.0 + 2.0 * rptr[IDX_ELEM_H] / 122.0);
    ab[IDX_GC4H2I] = ab[IDX_GC4H2I] * (48.0 * rptr[IDX_ELEM_C] / 50.0 + 2.0 * rptr[IDX_ELEM_H] / 50.0);
    ab[IDX_GCH2CHCNI] = ab[IDX_GCH2CHCNI] * (14.0 * rptr[IDX_ELEM_N] / 53.0 + 36.0 * rptr[IDX_ELEM_C] / 53.0 + 3.0 * rptr[IDX_ELEM_H] / 53.0);
    ab[IDX_GH2CCCI] = ab[IDX_GH2CCCI] * (36.0 * rptr[IDX_ELEM_C] / 38.0 + 2.0 * rptr[IDX_ELEM_H] / 38.0);
    ab[IDX_GHC3NI] = ab[IDX_GHC3NI] * (14.0 * rptr[IDX_ELEM_N] / 51.0 + 36.0 * rptr[IDX_ELEM_C] / 51.0 + 1.0 * rptr[IDX_ELEM_H] / 51.0);
    ab[IDX_GOCSI] = ab[IDX_GOCSI] * (32.0 * rptr[IDX_ELEM_S] / 60.0 + 16.0 * rptr[IDX_ELEM_O] / 60.0 + 12.0 * rptr[IDX_ELEM_C] / 60.0);
    ab[IDX_H2FII] = ab[IDX_H2FII] * (19.0 * rptr[IDX_ELEM_F] / 21.0 + 2.0 * rptr[IDX_ELEM_H] / 21.0);
    ab[IDX_C4H6I] = ab[IDX_C4H6I] * (48.0 * rptr[IDX_ELEM_C] / 54.0 + 6.0 * rptr[IDX_ELEM_H] / 54.0);
    ab[IDX_C4PII] = ab[IDX_C4PII] * (31.0 * rptr[IDX_ELEM_P] / 79.0 + 48.0 * rptr[IDX_ELEM_C] / 79.0);
    ab[IDX_C5NII] = ab[IDX_C5NII] * (14.0 * rptr[IDX_ELEM_N] / 74.0 + 60.0 * rptr[IDX_ELEM_C] / 74.0);
    ab[IDX_C6H6II] = ab[IDX_C6H6II] * (72.0 * rptr[IDX_ELEM_C] / 78.0 + 6.0 * rptr[IDX_ELEM_H] / 78.0);
    ab[IDX_CH2CHCNII] = ab[IDX_CH2CHCNII] * (14.0 * rptr[IDX_ELEM_N] / 53.0 + 36.0 * rptr[IDX_ELEM_C] / 53.0 + 3.0 * rptr[IDX_ELEM_H] / 53.0);
    ab[IDX_COOCH3II] = ab[IDX_COOCH3II] * (32.0 * rptr[IDX_ELEM_O] / 59.0 + 24.0 * rptr[IDX_ELEM_C] / 59.0 + 3.0 * rptr[IDX_ELEM_H] / 59.0);
    ab[IDX_FII] = ab[IDX_FII] * (19.0 * rptr[IDX_ELEM_F] / 19.0);
    ab[IDX_GC2H5CNI] = ab[IDX_GC2H5CNI] * (14.0 * rptr[IDX_ELEM_N] / 55.0 + 36.0 * rptr[IDX_ELEM_C] / 55.0 + 5.0 * rptr[IDX_ELEM_H] / 55.0);
    ab[IDX_GC2H5OHI] = ab[IDX_GC2H5OHI] * (16.0 * rptr[IDX_ELEM_O] / 46.0 + 24.0 * rptr[IDX_ELEM_C] / 46.0 + 6.0 * rptr[IDX_ELEM_H] / 46.0);
    ab[IDX_GC2OI] = ab[IDX_GC2OI] * (16.0 * rptr[IDX_ELEM_O] / 40.0 + 24.0 * rptr[IDX_ELEM_C] / 40.0);
    ab[IDX_GC6H2I] = ab[IDX_GC6H2I] * (72.0 * rptr[IDX_ELEM_C] / 74.0 + 2.0 * rptr[IDX_ELEM_H] / 74.0);
    ab[IDX_GC9NI] = ab[IDX_GC9NI] * (14.0 * rptr[IDX_ELEM_N] / 122.0 + 108.0 * rptr[IDX_ELEM_C] / 122.0);
    ab[IDX_GCO2I] = ab[IDX_GCO2I] * (32.0 * rptr[IDX_ELEM_O] / 44.0 + 12.0 * rptr[IDX_ELEM_C] / 44.0);
    ab[IDX_GCPI] = ab[IDX_GCPI] * (31.0 * rptr[IDX_ELEM_P] / 43.0 + 12.0 * rptr[IDX_ELEM_C] / 43.0);
    ab[IDX_GO2HI] = ab[IDX_GO2HI] * (32.0 * rptr[IDX_ELEM_O] / 33.0 + 1.0 * rptr[IDX_ELEM_H] / 33.0);
    ab[IDX_GOCNI] = ab[IDX_GOCNI] * (14.0 * rptr[IDX_ELEM_N] / 42.0 + 16.0 * rptr[IDX_ELEM_O] / 42.0 + 12.0 * rptr[IDX_ELEM_C] / 42.0);
    ab[IDX_GSOI] = ab[IDX_GSOI] * (32.0 * rptr[IDX_ELEM_S] / 48.0 + 16.0 * rptr[IDX_ELEM_O] / 48.0);
    ab[IDX_GSiCI] = ab[IDX_GSiCI] * (28.0 * rptr[IDX_ELEM_Si] / 40.0 + 12.0 * rptr[IDX_ELEM_C] / 40.0);
    ab[IDX_H2CNOII] = ab[IDX_H2CNOII] * (14.0 * rptr[IDX_ELEM_N] / 44.0 + 16.0 * rptr[IDX_ELEM_O] / 44.0 + 12.0 * rptr[IDX_ELEM_C] / 44.0 + 2.0 * rptr[IDX_ELEM_H] / 44.0);
    ab[IDX_H2NCOII] = ab[IDX_H2NCOII] * (14.0 * rptr[IDX_ELEM_N] / 44.0 + 16.0 * rptr[IDX_ELEM_O] / 44.0 + 12.0 * rptr[IDX_ELEM_C] / 44.0 + 2.0 * rptr[IDX_ELEM_H] / 44.0);
    ab[IDX_H2OCNII] = ab[IDX_H2OCNII] * (14.0 * rptr[IDX_ELEM_N] / 44.0 + 16.0 * rptr[IDX_ELEM_O] / 44.0 + 12.0 * rptr[IDX_ELEM_C] / 44.0 + 2.0 * rptr[IDX_ELEM_H] / 44.0);
    ab[IDX_PC2H4II] = ab[IDX_PC2H4II] * (31.0 * rptr[IDX_ELEM_P] / 59.0 + 24.0 * rptr[IDX_ELEM_C] / 59.0 + 4.0 * rptr[IDX_ELEM_H] / 59.0);
    ab[IDX_C7NII] = ab[IDX_C7NII] * (14.0 * rptr[IDX_ELEM_N] / 98.0 + 84.0 * rptr[IDX_ELEM_C] / 98.0);
    ab[IDX_C9NII] = ab[IDX_C9NII] * (14.0 * rptr[IDX_ELEM_N] / 122.0 + 108.0 * rptr[IDX_ELEM_C] / 122.0);
    ab[IDX_CH3NHII] = ab[IDX_CH3NHII] * (14.0 * rptr[IDX_ELEM_N] / 30.0 + 12.0 * rptr[IDX_ELEM_C] / 30.0 + 4.0 * rptr[IDX_ELEM_H] / 30.0);
    ab[IDX_ClOI] = ab[IDX_ClOI] * (35.0 * rptr[IDX_ELEM_Cl] / 51.0 + 16.0 * rptr[IDX_ELEM_O] / 51.0);
    ab[IDX_GC11I] = ab[IDX_GC11I] * (132.0 * rptr[IDX_ELEM_C] / 132.0);
    ab[IDX_GC2NI] = ab[IDX_GC2NI] * (14.0 * rptr[IDX_ELEM_N] / 38.0 + 24.0 * rptr[IDX_ELEM_C] / 38.0);
    ab[IDX_GCH2CCH2I] = ab[IDX_GCH2CCH2I] * (36.0 * rptr[IDX_ELEM_C] / 40.0 + 4.0 * rptr[IDX_ELEM_H] / 40.0);
    ab[IDX_GCH2NHI] = ab[IDX_GCH2NHI] * (14.0 * rptr[IDX_ELEM_N] / 29.0 + 12.0 * rptr[IDX_ELEM_C] / 29.0 + 3.0 * rptr[IDX_ELEM_H] / 29.0);
    ab[IDX_GCH3CCHI] = ab[IDX_GCH3CCHI] * (36.0 * rptr[IDX_ELEM_C] / 40.0 + 4.0 * rptr[IDX_ELEM_H] / 40.0);
    ab[IDX_GCNOI] = ab[IDX_GCNOI] * (14.0 * rptr[IDX_ELEM_N] / 42.0 + 16.0 * rptr[IDX_ELEM_O] / 42.0 + 12.0 * rptr[IDX_ELEM_C] / 42.0);
    ab[IDX_GHNCI] = ab[IDX_GHNCI] * (14.0 * rptr[IDX_ELEM_N] / 27.0 + 12.0 * rptr[IDX_ELEM_C] / 27.0 + 1.0 * rptr[IDX_ELEM_H] / 27.0);
    ab[IDX_GNSI] = ab[IDX_GNSI] * (32.0 * rptr[IDX_ELEM_S] / 46.0 + 14.0 * rptr[IDX_ELEM_N] / 46.0);
    ab[IDX_GSiC2I] = ab[IDX_GSiC2I] * (28.0 * rptr[IDX_ELEM_Si] / 52.0 + 24.0 * rptr[IDX_ELEM_C] / 52.0);
    ab[IDX_H2C4NII] = ab[IDX_H2C4NII] * (14.0 * rptr[IDX_ELEM_N] / 64.0 + 48.0 * rptr[IDX_ELEM_C] / 64.0 + 2.0 * rptr[IDX_ELEM_H] / 64.0);
    ab[IDX_H2CClII] = ab[IDX_H2CClII] * (35.0 * rptr[IDX_ELEM_Cl] / 49.0 + 12.0 * rptr[IDX_ELEM_C] / 49.0 + 2.0 * rptr[IDX_ELEM_H] / 49.0);
    ab[IDX_H2NOII] = ab[IDX_H2NOII] * (14.0 * rptr[IDX_ELEM_N] / 32.0 + 16.0 * rptr[IDX_ELEM_O] / 32.0 + 2.0 * rptr[IDX_ELEM_H] / 32.0);
    ab[IDX_HClII] = ab[IDX_HClII] * (35.0 * rptr[IDX_ELEM_Cl] / 36.0 + 1.0 * rptr[IDX_ELEM_H] / 36.0);
    ab[IDX_HNSII] = ab[IDX_HNSII] * (32.0 * rptr[IDX_ELEM_S] / 47.0 + 14.0 * rptr[IDX_ELEM_N] / 47.0 + 1.0 * rptr[IDX_ELEM_H] / 47.0);
    ab[IDX_HOCII] = ab[IDX_HOCII] * (16.0 * rptr[IDX_ELEM_O] / 29.0 + 12.0 * rptr[IDX_ELEM_C] / 29.0 + 1.0 * rptr[IDX_ELEM_H] / 29.0);
    ab[IDX_HSOII] = ab[IDX_HSOII] * (32.0 * rptr[IDX_ELEM_S] / 49.0 + 16.0 * rptr[IDX_ELEM_O] / 49.0 + 1.0 * rptr[IDX_ELEM_H] / 49.0);
    ab[IDX_HSiO2II] = ab[IDX_HSiO2II] * (28.0 * rptr[IDX_ELEM_Si] / 61.0 + 32.0 * rptr[IDX_ELEM_O] / 61.0 + 1.0 * rptr[IDX_ELEM_H] / 61.0);
    ab[IDX_NH2CNHII] = ab[IDX_NH2CNHII] * (28.0 * rptr[IDX_ELEM_N] / 43.0 + 12.0 * rptr[IDX_ELEM_C] / 43.0 + 3.0 * rptr[IDX_ELEM_H] / 43.0);
    ab[IDX_C2H5OHII] = ab[IDX_C2H5OHII] * (16.0 * rptr[IDX_ELEM_O] / 46.0 + 24.0 * rptr[IDX_ELEM_C] / 46.0 + 6.0 * rptr[IDX_ELEM_H] / 46.0);
    ab[IDX_C2OII] = ab[IDX_C2OII] * (16.0 * rptr[IDX_ELEM_O] / 40.0 + 24.0 * rptr[IDX_ELEM_C] / 40.0);
    ab[IDX_CH2CHCCHI] = ab[IDX_CH2CHCCHI] * (48.0 * rptr[IDX_ELEM_C] / 52.0 + 4.0 * rptr[IDX_ELEM_H] / 52.0);
    ab[IDX_CH2OHCOII] = ab[IDX_CH2OHCOII] * (32.0 * rptr[IDX_ELEM_O] / 59.0 + 24.0 * rptr[IDX_ELEM_C] / 59.0 + 3.0 * rptr[IDX_ELEM_H] / 59.0);
    ab[IDX_GCH2CCHI] = ab[IDX_GCH2CCHI] * (36.0 * rptr[IDX_ELEM_C] / 39.0 + 3.0 * rptr[IDX_ELEM_H] / 39.0);
    ab[IDX_GCH2COI] = ab[IDX_GCH2COI] * (16.0 * rptr[IDX_ELEM_O] / 42.0 + 24.0 * rptr[IDX_ELEM_C] / 42.0 + 2.0 * rptr[IDX_ELEM_H] / 42.0);
    ab[IDX_GHNOI] = ab[IDX_GHNOI] * (14.0 * rptr[IDX_ELEM_N] / 31.0 + 16.0 * rptr[IDX_ELEM_O] / 31.0 + 1.0 * rptr[IDX_ELEM_H] / 31.0);
    ab[IDX_GN2I] = ab[IDX_GN2I] * (28.0 * rptr[IDX_ELEM_N] / 28.0);
    ab[IDX_HC4NII] = ab[IDX_HC4NII] * (14.0 * rptr[IDX_ELEM_N] / 63.0 + 48.0 * rptr[IDX_ELEM_C] / 63.0 + 1.0 * rptr[IDX_ELEM_H] / 63.0);
    ab[IDX_HCNOHII] = ab[IDX_HCNOHII] * (14.0 * rptr[IDX_ELEM_N] / 44.0 + 16.0 * rptr[IDX_ELEM_O] / 44.0 + 12.0 * rptr[IDX_ELEM_C] / 44.0 + 2.0 * rptr[IDX_ELEM_H] / 44.0);
    ab[IDX_HNCOHII] = ab[IDX_HNCOHII] * (14.0 * rptr[IDX_ELEM_N] / 44.0 + 16.0 * rptr[IDX_ELEM_O] / 44.0 + 12.0 * rptr[IDX_ELEM_C] / 44.0 + 2.0 * rptr[IDX_ELEM_H] / 44.0);
    ab[IDX_HOCNII] = ab[IDX_HOCNII] * (14.0 * rptr[IDX_ELEM_N] / 43.0 + 16.0 * rptr[IDX_ELEM_O] / 43.0 + 12.0 * rptr[IDX_ELEM_C] / 43.0 + 1.0 * rptr[IDX_ELEM_H] / 43.0);
    ab[IDX_HeHII] = ab[IDX_HeHII] * (4.0 * rptr[IDX_ELEM_He] / 5.0 + 1.0 * rptr[IDX_ELEM_H] / 5.0);
    ab[IDX_PC2H3II] = ab[IDX_PC2H3II] * (31.0 * rptr[IDX_ELEM_P] / 58.0 + 24.0 * rptr[IDX_ELEM_C] / 58.0 + 3.0 * rptr[IDX_ELEM_H] / 58.0);
    ab[IDX_PNII] = ab[IDX_PNII] * (31.0 * rptr[IDX_ELEM_P] / 45.0 + 14.0 * rptr[IDX_ELEM_N] / 45.0);
    ab[IDX_PNH3II] = ab[IDX_PNH3II] * (31.0 * rptr[IDX_ELEM_P] / 48.0 + 14.0 * rptr[IDX_ELEM_N] / 48.0 + 3.0 * rptr[IDX_ELEM_H] / 48.0);
    ab[IDX_CH3CSII] = ab[IDX_CH3CSII] * (32.0 * rptr[IDX_ELEM_S] / 59.0 + 24.0 * rptr[IDX_ELEM_C] / 59.0 + 3.0 * rptr[IDX_ELEM_H] / 59.0);
    ab[IDX_GC5NI] = ab[IDX_GC5NI] * (14.0 * rptr[IDX_ELEM_N] / 74.0 + 60.0 * rptr[IDX_ELEM_C] / 74.0);
    ab[IDX_GPI] = ab[IDX_GPI] * (31.0 * rptr[IDX_ELEM_P] / 31.0);
    ab[IDX_H2C7NII] = ab[IDX_H2C7NII] * (14.0 * rptr[IDX_ELEM_N] / 100.0 + 84.0 * rptr[IDX_ELEM_C] / 100.0 + 2.0 * rptr[IDX_ELEM_H] / 100.0);
    ab[IDX_H2C9NII] = ab[IDX_H2C9NII] * (14.0 * rptr[IDX_ELEM_N] / 124.0 + 108.0 * rptr[IDX_ELEM_C] / 124.0 + 2.0 * rptr[IDX_ELEM_H] / 124.0);
    ab[IDX_H2O2I] = ab[IDX_H2O2I] * (32.0 * rptr[IDX_ELEM_O] / 34.0 + 2.0 * rptr[IDX_ELEM_H] / 34.0);
    ab[IDX_H3S2II] = ab[IDX_H3S2II] * (64.0 * rptr[IDX_ELEM_S] / 67.0 + 3.0 * rptr[IDX_ELEM_H] / 67.0);
    ab[IDX_HONCII] = ab[IDX_HONCII] * (14.0 * rptr[IDX_ELEM_N] / 43.0 + 16.0 * rptr[IDX_ELEM_O] / 43.0 + 12.0 * rptr[IDX_ELEM_C] / 43.0 + 1.0 * rptr[IDX_ELEM_H] / 43.0);
    ab[IDX_PCH3II] = ab[IDX_PCH3II] * (31.0 * rptr[IDX_ELEM_P] / 46.0 + 12.0 * rptr[IDX_ELEM_C] / 46.0 + 3.0 * rptr[IDX_ELEM_H] / 46.0);
    ab[IDX_PNH2II] = ab[IDX_PNH2II] * (31.0 * rptr[IDX_ELEM_P] / 47.0 + 14.0 * rptr[IDX_ELEM_N] / 47.0 + 2.0 * rptr[IDX_ELEM_H] / 47.0);
    ab[IDX_C10H3II] = ab[IDX_C10H3II] * (120.0 * rptr[IDX_ELEM_C] / 123.0 + 3.0 * rptr[IDX_ELEM_H] / 123.0);
    ab[IDX_C2H5CNHII] = ab[IDX_C2H5CNHII] * (14.0 * rptr[IDX_ELEM_N] / 56.0 + 36.0 * rptr[IDX_ELEM_C] / 56.0 + 6.0 * rptr[IDX_ELEM_H] / 56.0);
    ab[IDX_CClII] = ab[IDX_CClII] * (35.0 * rptr[IDX_ELEM_Cl] / 47.0 + 12.0 * rptr[IDX_ELEM_C] / 47.0);
    ab[IDX_CH2NH2II] = ab[IDX_CH2NH2II] * (14.0 * rptr[IDX_ELEM_N] / 30.0 + 12.0 * rptr[IDX_ELEM_C] / 30.0 + 4.0 * rptr[IDX_ELEM_H] / 30.0);
    ab[IDX_CH2OHCH2OII] = ab[IDX_CH2OHCH2OII] * (32.0 * rptr[IDX_ELEM_O] / 61.0 + 24.0 * rptr[IDX_ELEM_C] / 61.0 + 5.0 * rptr[IDX_ELEM_H] / 61.0);
    ab[IDX_CH3C5NHII] = ab[IDX_CH3C5NHII] * (14.0 * rptr[IDX_ELEM_N] / 90.0 + 72.0 * rptr[IDX_ELEM_C] / 90.0 + 4.0 * rptr[IDX_ELEM_H] / 90.0);
    ab[IDX_CH3C7NHII] = ab[IDX_CH3C7NHII] * (14.0 * rptr[IDX_ELEM_N] / 114.0 + 96.0 * rptr[IDX_ELEM_C] / 114.0 + 4.0 * rptr[IDX_ELEM_H] / 114.0);
    ab[IDX_GC2H5I] = ab[IDX_GC2H5I] * (24.0 * rptr[IDX_ELEM_C] / 29.0 + 5.0 * rptr[IDX_ELEM_H] / 29.0);
    ab[IDX_GC3NI] = ab[IDX_GC3NI] * (14.0 * rptr[IDX_ELEM_N] / 50.0 + 36.0 * rptr[IDX_ELEM_C] / 50.0);
    ab[IDX_GC7NI] = ab[IDX_GC7NI] * (14.0 * rptr[IDX_ELEM_N] / 98.0 + 84.0 * rptr[IDX_ELEM_C] / 98.0);
    ab[IDX_GCOOCH3I] = ab[IDX_GCOOCH3I] * (32.0 * rptr[IDX_ELEM_O] / 59.0 + 24.0 * rptr[IDX_ELEM_C] / 59.0 + 3.0 * rptr[IDX_ELEM_H] / 59.0);
    ab[IDX_GHCNI] = ab[IDX_GHCNI] * (14.0 * rptr[IDX_ELEM_N] / 27.0 + 12.0 * rptr[IDX_ELEM_C] / 27.0 + 1.0 * rptr[IDX_ELEM_H] / 27.0);
    ab[IDX_H2CNI] = ab[IDX_H2CNI] * (14.0 * rptr[IDX_ELEM_N] / 28.0 + 12.0 * rptr[IDX_ELEM_C] / 28.0 + 2.0 * rptr[IDX_ELEM_H] / 28.0);
    ab[IDX_H2SiOII] = ab[IDX_H2SiOII] * (28.0 * rptr[IDX_ELEM_Si] / 46.0 + 16.0 * rptr[IDX_ELEM_O] / 46.0 + 2.0 * rptr[IDX_ELEM_H] / 46.0);
    ab[IDX_H3C5NII] = ab[IDX_H3C5NII] * (14.0 * rptr[IDX_ELEM_N] / 77.0 + 60.0 * rptr[IDX_ELEM_C] / 77.0 + 3.0 * rptr[IDX_ELEM_H] / 77.0);
    ab[IDX_HCNOII] = ab[IDX_HCNOII] * (14.0 * rptr[IDX_ELEM_N] / 43.0 + 16.0 * rptr[IDX_ELEM_O] / 43.0 + 12.0 * rptr[IDX_ELEM_C] / 43.0 + 1.0 * rptr[IDX_ELEM_H] / 43.0);
    ab[IDX_CClI] = ab[IDX_CClI] * (35.0 * rptr[IDX_ELEM_Cl] / 47.0 + 12.0 * rptr[IDX_ELEM_C] / 47.0);
    ab[IDX_CH3COOHII] = ab[IDX_CH3COOHII] * (32.0 * rptr[IDX_ELEM_O] / 60.0 + 24.0 * rptr[IDX_ELEM_C] / 60.0 + 4.0 * rptr[IDX_ELEM_H] / 60.0);
    ab[IDX_ClII] = ab[IDX_ClII] * (35.0 * rptr[IDX_ELEM_Cl] / 35.0);
    ab[IDX_GNH3I] = ab[IDX_GNH3I] * (14.0 * rptr[IDX_ELEM_N] / 17.0 + 3.0 * rptr[IDX_ELEM_H] / 17.0);
    ab[IDX_H2ClII] = ab[IDX_H2ClII] * (35.0 * rptr[IDX_ELEM_Cl] / 37.0 + 2.0 * rptr[IDX_ELEM_H] / 37.0);
    ab[IDX_H3C9NII] = ab[IDX_H3C9NII] * (14.0 * rptr[IDX_ELEM_N] / 125.0 + 108.0 * rptr[IDX_ELEM_C] / 125.0 + 3.0 * rptr[IDX_ELEM_H] / 125.0);
    ab[IDX_H5C2O2II] = ab[IDX_H5C2O2II] * (32.0 * rptr[IDX_ELEM_O] / 61.0 + 24.0 * rptr[IDX_ELEM_C] / 61.0 + 5.0 * rptr[IDX_ELEM_H] / 61.0);
    ab[IDX_HCOOHII] = ab[IDX_HCOOHII] * (32.0 * rptr[IDX_ELEM_O] / 46.0 + 12.0 * rptr[IDX_ELEM_C] / 46.0 + 2.0 * rptr[IDX_ELEM_H] / 46.0);
    ab[IDX_HPNII] = ab[IDX_HPNII] * (31.0 * rptr[IDX_ELEM_P] / 46.0 + 14.0 * rptr[IDX_ELEM_N] / 46.0 + 1.0 * rptr[IDX_ELEM_H] / 46.0);
    ab[IDX_C2SII] = ab[IDX_C2SII] * (32.0 * rptr[IDX_ELEM_S] / 56.0 + 24.0 * rptr[IDX_ELEM_C] / 56.0);
    ab[IDX_C3SII] = ab[IDX_C3SII] * (32.0 * rptr[IDX_ELEM_S] / 68.0 + 36.0 * rptr[IDX_ELEM_C] / 68.0);
    ab[IDX_C4NI] = ab[IDX_C4NI] * (14.0 * rptr[IDX_ELEM_N] / 62.0 + 48.0 * rptr[IDX_ELEM_C] / 62.0);
    ab[IDX_CH3COOH2II] = ab[IDX_CH3COOH2II] * (32.0 * rptr[IDX_ELEM_O] / 61.0 + 24.0 * rptr[IDX_ELEM_C] / 61.0 + 5.0 * rptr[IDX_ELEM_H] / 61.0);
    ab[IDX_CH3OCH3II] = ab[IDX_CH3OCH3II] * (16.0 * rptr[IDX_ELEM_O] / 46.0 + 24.0 * rptr[IDX_ELEM_C] / 46.0 + 6.0 * rptr[IDX_ELEM_H] / 46.0);
    ab[IDX_GC10I] = ab[IDX_GC10I] * (120.0 * rptr[IDX_ELEM_C] / 120.0);
    ab[IDX_GC10HI] = ab[IDX_GC10HI] * (120.0 * rptr[IDX_ELEM_C] / 121.0 + 1.0 * rptr[IDX_ELEM_H] / 121.0);
    ab[IDX_GC8HI] = ab[IDX_GC8HI] * (96.0 * rptr[IDX_ELEM_C] / 97.0 + 1.0 * rptr[IDX_ELEM_H] / 97.0);
    ab[IDX_GCH2OHCOI] = ab[IDX_GCH2OHCOI] * (32.0 * rptr[IDX_ELEM_O] / 59.0 + 24.0 * rptr[IDX_ELEM_C] / 59.0 + 3.0 * rptr[IDX_ELEM_H] / 59.0);
    ab[IDX_GCOOHI] = ab[IDX_GCOOHI] * (32.0 * rptr[IDX_ELEM_O] / 45.0 + 12.0 * rptr[IDX_ELEM_C] / 45.0 + 1.0 * rptr[IDX_ELEM_H] / 45.0);
    ab[IDX_GO2I] = ab[IDX_GO2I] * (32.0 * rptr[IDX_ELEM_O] / 32.0);
    ab[IDX_H3C7NII] = ab[IDX_H3C7NII] * (14.0 * rptr[IDX_ELEM_N] / 101.0 + 84.0 * rptr[IDX_ELEM_C] / 101.0 + 3.0 * rptr[IDX_ELEM_H] / 101.0);
    ab[IDX_H3SiOII] = ab[IDX_H3SiOII] * (28.0 * rptr[IDX_ELEM_Si] / 47.0 + 16.0 * rptr[IDX_ELEM_O] / 47.0 + 3.0 * rptr[IDX_ELEM_H] / 47.0);
    ab[IDX_HSO2II] = ab[IDX_HSO2II] * (32.0 * rptr[IDX_ELEM_S] / 65.0 + 32.0 * rptr[IDX_ELEM_O] / 65.0 + 1.0 * rptr[IDX_ELEM_H] / 65.0);
    ab[IDX_NH2CNI] = ab[IDX_NH2CNI] * (28.0 * rptr[IDX_ELEM_N] / 42.0 + 12.0 * rptr[IDX_ELEM_C] / 42.0 + 2.0 * rptr[IDX_ELEM_H] / 42.0);
    ab[IDX_CH3C3NHII] = ab[IDX_CH3C3NHII] * (14.0 * rptr[IDX_ELEM_N] / 66.0 + 48.0 * rptr[IDX_ELEM_C] / 66.0 + 4.0 * rptr[IDX_ELEM_H] / 66.0);
    ab[IDX_CH3OCH4II] = ab[IDX_CH3OCH4II] * (16.0 * rptr[IDX_ELEM_O] / 47.0 + 24.0 * rptr[IDX_ELEM_C] / 47.0 + 7.0 * rptr[IDX_ELEM_H] / 47.0);
    ab[IDX_GC2H4I] = ab[IDX_GC2H4I] * (24.0 * rptr[IDX_ELEM_C] / 28.0 + 4.0 * rptr[IDX_ELEM_H] / 28.0);
    ab[IDX_GC9I] = ab[IDX_GC9I] * (108.0 * rptr[IDX_ELEM_C] / 108.0);
    ab[IDX_GC9HI] = ab[IDX_GC9HI] * (108.0 * rptr[IDX_ELEM_C] / 109.0 + 1.0 * rptr[IDX_ELEM_H] / 109.0);
    ab[IDX_GCSI] = ab[IDX_GCSI] * (32.0 * rptr[IDX_ELEM_S] / 44.0 + 12.0 * rptr[IDX_ELEM_C] / 44.0);
    ab[IDX_HC7NII] = ab[IDX_HC7NII] * (14.0 * rptr[IDX_ELEM_N] / 99.0 + 84.0 * rptr[IDX_ELEM_C] / 99.0 + 1.0 * rptr[IDX_ELEM_H] / 99.0);
    ab[IDX_HC9NII] = ab[IDX_HC9NII] * (14.0 * rptr[IDX_ELEM_N] / 123.0 + 108.0 * rptr[IDX_ELEM_C] / 123.0 + 1.0 * rptr[IDX_ELEM_H] / 123.0);
    ab[IDX_HNCOII] = ab[IDX_HNCOII] * (14.0 * rptr[IDX_ELEM_N] / 43.0 + 16.0 * rptr[IDX_ELEM_O] / 43.0 + 12.0 * rptr[IDX_ELEM_C] / 43.0 + 1.0 * rptr[IDX_ELEM_H] / 43.0);
    ab[IDX_NO2II] = ab[IDX_NO2II] * (14.0 * rptr[IDX_ELEM_N] / 46.0 + 32.0 * rptr[IDX_ELEM_O] / 46.0);
    ab[IDX_PC3HII] = ab[IDX_PC3HII] * (31.0 * rptr[IDX_ELEM_P] / 68.0 + 36.0 * rptr[IDX_ELEM_C] / 68.0 + 1.0 * rptr[IDX_ELEM_H] / 68.0);
    ab[IDX_PCH4II] = ab[IDX_PCH4II] * (31.0 * rptr[IDX_ELEM_P] / 47.0 + 12.0 * rptr[IDX_ELEM_C] / 47.0 + 4.0 * rptr[IDX_ELEM_H] / 47.0);
    ab[IDX_SiH4II] = ab[IDX_SiH4II] * (28.0 * rptr[IDX_ELEM_Si] / 32.0 + 4.0 * rptr[IDX_ELEM_H] / 32.0);
    ab[IDX_SiO2I] = ab[IDX_SiO2I] * (28.0 * rptr[IDX_ELEM_Si] / 60.0 + 32.0 * rptr[IDX_ELEM_O] / 60.0);
    ab[IDX_C2H5CNI] = ab[IDX_C2H5CNI] * (14.0 * rptr[IDX_ELEM_N] / 55.0 + 36.0 * rptr[IDX_ELEM_C] / 55.0 + 5.0 * rptr[IDX_ELEM_H] / 55.0);
    ab[IDX_C2H5OH2II] = ab[IDX_C2H5OH2II] * (16.0 * rptr[IDX_ELEM_O] / 47.0 + 24.0 * rptr[IDX_ELEM_C] / 47.0 + 7.0 * rptr[IDX_ELEM_H] / 47.0);
    ab[IDX_C2H7II] = ab[IDX_C2H7II] * (24.0 * rptr[IDX_ELEM_C] / 31.0 + 7.0 * rptr[IDX_ELEM_H] / 31.0);
    ab[IDX_CCPII] = ab[IDX_CCPII] * (31.0 * rptr[IDX_ELEM_P] / 55.0 + 24.0 * rptr[IDX_ELEM_C] / 55.0);
    ab[IDX_CH2CHCNHII] = ab[IDX_CH2CHCNHII] * (14.0 * rptr[IDX_ELEM_N] / 54.0 + 36.0 * rptr[IDX_ELEM_C] / 54.0 + 4.0 * rptr[IDX_ELEM_H] / 54.0);
    ab[IDX_COOCH3I] = ab[IDX_COOCH3I] * (32.0 * rptr[IDX_ELEM_O] / 59.0 + 24.0 * rptr[IDX_ELEM_C] / 59.0 + 3.0 * rptr[IDX_ELEM_H] / 59.0);
    ab[IDX_FI] = ab[IDX_FI] * (19.0 * rptr[IDX_ELEM_F] / 19.0);
    ab[IDX_GHSI] = ab[IDX_GHSI] * (32.0 * rptr[IDX_ELEM_S] / 33.0 + 1.0 * rptr[IDX_ELEM_H] / 33.0);
    ab[IDX_H2POII] = ab[IDX_H2POII] * (31.0 * rptr[IDX_ELEM_P] / 49.0 + 16.0 * rptr[IDX_ELEM_O] / 49.0 + 2.0 * rptr[IDX_ELEM_H] / 49.0);
    ab[IDX_HCOOCH3II] = ab[IDX_HCOOCH3II] * (32.0 * rptr[IDX_ELEM_O] / 60.0 + 24.0 * rptr[IDX_ELEM_C] / 60.0 + 4.0 * rptr[IDX_ELEM_H] / 60.0);
    ab[IDX_HN2OII] = ab[IDX_HN2OII] * (28.0 * rptr[IDX_ELEM_N] / 45.0 + 16.0 * rptr[IDX_ELEM_O] / 45.0 + 1.0 * rptr[IDX_ELEM_H] / 45.0);
    ab[IDX_N2OII] = ab[IDX_N2OII] * (28.0 * rptr[IDX_ELEM_N] / 44.0 + 16.0 * rptr[IDX_ELEM_O] / 44.0);
    ab[IDX_PC2H2II] = ab[IDX_PC2H2II] * (31.0 * rptr[IDX_ELEM_P] / 57.0 + 24.0 * rptr[IDX_ELEM_C] / 57.0 + 2.0 * rptr[IDX_ELEM_H] / 57.0);
    ab[IDX_SiC3H2II] = ab[IDX_SiC3H2II] * (28.0 * rptr[IDX_ELEM_Si] / 66.0 + 36.0 * rptr[IDX_ELEM_C] / 66.0 + 2.0 * rptr[IDX_ELEM_H] / 66.0);
    ab[IDX_SiC4II] = ab[IDX_SiC4II] * (28.0 * rptr[IDX_ELEM_Si] / 76.0 + 48.0 * rptr[IDX_ELEM_C] / 76.0);
    ab[IDX_SiC4HII] = ab[IDX_SiC4HII] * (28.0 * rptr[IDX_ELEM_Si] / 77.0 + 48.0 * rptr[IDX_ELEM_C] / 77.0 + 1.0 * rptr[IDX_ELEM_H] / 77.0);
    ab[IDX_SiH5II] = ab[IDX_SiH5II] * (28.0 * rptr[IDX_ELEM_Si] / 33.0 + 5.0 * rptr[IDX_ELEM_H] / 33.0);
    ab[IDX_SiNCHII] = ab[IDX_SiNCHII] * (28.0 * rptr[IDX_ELEM_Si] / 55.0 + 14.0 * rptr[IDX_ELEM_N] / 55.0 + 12.0 * rptr[IDX_ELEM_C] / 55.0 + 1.0 * rptr[IDX_ELEM_H] / 55.0);
    ab[IDX_C11I] = ab[IDX_C11I] * (132.0 * rptr[IDX_ELEM_C] / 132.0);
    ab[IDX_C3OII] = ab[IDX_C3OII] * (16.0 * rptr[IDX_ELEM_O] / 52.0 + 36.0 * rptr[IDX_ELEM_C] / 52.0);
    ab[IDX_C4H5II] = ab[IDX_C4H5II] * (48.0 * rptr[IDX_ELEM_C] / 53.0 + 5.0 * rptr[IDX_ELEM_H] / 53.0);
    ab[IDX_C4H7II] = ab[IDX_C4H7II] * (48.0 * rptr[IDX_ELEM_C] / 55.0 + 7.0 * rptr[IDX_ELEM_H] / 55.0);
    ab[IDX_CH2OHCHOII] = ab[IDX_CH2OHCHOII] * (32.0 * rptr[IDX_ELEM_O] / 60.0 + 24.0 * rptr[IDX_ELEM_C] / 60.0 + 4.0 * rptr[IDX_ELEM_H] / 60.0);
    ab[IDX_CH3CHOII] = ab[IDX_CH3CHOII] * (16.0 * rptr[IDX_ELEM_O] / 44.0 + 24.0 * rptr[IDX_ELEM_C] / 44.0 + 4.0 * rptr[IDX_ELEM_H] / 44.0);
    ab[IDX_GCH2OHCHOI] = ab[IDX_GCH2OHCHOI] * (32.0 * rptr[IDX_ELEM_O] / 60.0 + 24.0 * rptr[IDX_ELEM_C] / 60.0 + 4.0 * rptr[IDX_ELEM_H] / 60.0);
    ab[IDX_GHCOOHI] = ab[IDX_GHCOOHI] * (32.0 * rptr[IDX_ELEM_O] / 46.0 + 12.0 * rptr[IDX_ELEM_C] / 46.0 + 2.0 * rptr[IDX_ELEM_H] / 46.0);
    ab[IDX_H2S2I] = ab[IDX_H2S2I] * (64.0 * rptr[IDX_ELEM_S] / 66.0 + 2.0 * rptr[IDX_ELEM_H] / 66.0);
    ab[IDX_HNSiII] = ab[IDX_HNSiII] * (28.0 * rptr[IDX_ELEM_Si] / 43.0 + 14.0 * rptr[IDX_ELEM_N] / 43.0 + 1.0 * rptr[IDX_ELEM_H] / 43.0);
    ab[IDX_HOCSII] = ab[IDX_HOCSII] * (32.0 * rptr[IDX_ELEM_S] / 61.0 + 16.0 * rptr[IDX_ELEM_O] / 61.0 + 12.0 * rptr[IDX_ELEM_C] / 61.0 + 1.0 * rptr[IDX_ELEM_H] / 61.0);
    ab[IDX_NSII] = ab[IDX_NSII] * (32.0 * rptr[IDX_ELEM_S] / 46.0 + 14.0 * rptr[IDX_ELEM_N] / 46.0);
    ab[IDX_OCNII] = ab[IDX_OCNII] * (14.0 * rptr[IDX_ELEM_N] / 42.0 + 16.0 * rptr[IDX_ELEM_O] / 42.0 + 12.0 * rptr[IDX_ELEM_C] / 42.0);
    ab[IDX_PC4HII] = ab[IDX_PC4HII] * (31.0 * rptr[IDX_ELEM_P] / 80.0 + 48.0 * rptr[IDX_ELEM_C] / 80.0 + 1.0 * rptr[IDX_ELEM_H] / 80.0);
    ab[IDX_PH3II] = ab[IDX_PH3II] * (31.0 * rptr[IDX_ELEM_P] / 34.0 + 3.0 * rptr[IDX_ELEM_H] / 34.0);
    ab[IDX_SiCH4II] = ab[IDX_SiCH4II] * (28.0 * rptr[IDX_ELEM_Si] / 44.0 + 12.0 * rptr[IDX_ELEM_C] / 44.0 + 4.0 * rptr[IDX_ELEM_H] / 44.0);
    ab[IDX_C10II] = ab[IDX_C10II] * (120.0 * rptr[IDX_ELEM_C] / 120.0);
    ab[IDX_C10H2II] = ab[IDX_C10H2II] * (120.0 * rptr[IDX_ELEM_C] / 122.0 + 2.0 * rptr[IDX_ELEM_H] / 122.0);
    ab[IDX_C2N2II] = ab[IDX_C2N2II] * (28.0 * rptr[IDX_ELEM_N] / 52.0 + 24.0 * rptr[IDX_ELEM_C] / 52.0);
    ab[IDX_CH3CNII] = ab[IDX_CH3CNII] * (14.0 * rptr[IDX_ELEM_N] / 41.0 + 24.0 * rptr[IDX_ELEM_C] / 41.0 + 3.0 * rptr[IDX_ELEM_H] / 41.0);
    ab[IDX_GC2H2I] = ab[IDX_GC2H2I] * (24.0 * rptr[IDX_ELEM_C] / 26.0 + 2.0 * rptr[IDX_ELEM_H] / 26.0);
    ab[IDX_GC5I] = ab[IDX_GC5I] * (60.0 * rptr[IDX_ELEM_C] / 60.0);
    ab[IDX_GC7HI] = ab[IDX_GC7HI] * (84.0 * rptr[IDX_ELEM_C] / 85.0 + 1.0 * rptr[IDX_ELEM_H] / 85.0);
    ab[IDX_GCH3COI] = ab[IDX_GCH3COI] * (16.0 * rptr[IDX_ELEM_O] / 43.0 + 24.0 * rptr[IDX_ELEM_C] / 43.0 + 3.0 * rptr[IDX_ELEM_H] / 43.0);
    ab[IDX_GHCOOCH3I] = ab[IDX_GHCOOCH3I] * (32.0 * rptr[IDX_ELEM_O] / 60.0 + 24.0 * rptr[IDX_ELEM_C] / 60.0 + 4.0 * rptr[IDX_ELEM_H] / 60.0);
    ab[IDX_HC4SII] = ab[IDX_HC4SII] * (32.0 * rptr[IDX_ELEM_S] / 81.0 + 48.0 * rptr[IDX_ELEM_C] / 81.0 + 1.0 * rptr[IDX_ELEM_H] / 81.0);
    ab[IDX_HC5NHII] = ab[IDX_HC5NHII] * (14.0 * rptr[IDX_ELEM_N] / 76.0 + 60.0 * rptr[IDX_ELEM_C] / 76.0 + 2.0 * rptr[IDX_ELEM_H] / 76.0);
    ab[IDX_SiC2H3II] = ab[IDX_SiC2H3II] * (28.0 * rptr[IDX_ELEM_Si] / 55.0 + 24.0 * rptr[IDX_ELEM_C] / 55.0 + 3.0 * rptr[IDX_ELEM_H] / 55.0);
    ab[IDX_SiC3HI] = ab[IDX_SiC3HI] * (28.0 * rptr[IDX_ELEM_Si] / 65.0 + 36.0 * rptr[IDX_ELEM_C] / 65.0 + 1.0 * rptr[IDX_ELEM_H] / 65.0);
    ab[IDX_SiCH3II] = ab[IDX_SiCH3II] * (28.0 * rptr[IDX_ELEM_Si] / 43.0 + 12.0 * rptr[IDX_ELEM_C] / 43.0 + 3.0 * rptr[IDX_ELEM_H] / 43.0);
    ab[IDX_SiNH2II] = ab[IDX_SiNH2II] * (28.0 * rptr[IDX_ELEM_Si] / 44.0 + 14.0 * rptr[IDX_ELEM_N] / 44.0 + 2.0 * rptr[IDX_ELEM_H] / 44.0);
    ab[IDX_C11II] = ab[IDX_C11II] * (132.0 * rptr[IDX_ELEM_C] / 132.0);
    ab[IDX_C3NII] = ab[IDX_C3NII] * (14.0 * rptr[IDX_ELEM_N] / 50.0 + 36.0 * rptr[IDX_ELEM_C] / 50.0);
    ab[IDX_C4H4II] = ab[IDX_C4H4II] * (48.0 * rptr[IDX_ELEM_C] / 52.0 + 4.0 * rptr[IDX_ELEM_H] / 52.0);
    ab[IDX_CH2CNII] = ab[IDX_CH2CNII] * (14.0 * rptr[IDX_ELEM_N] / 40.0 + 24.0 * rptr[IDX_ELEM_C] / 40.0 + 2.0 * rptr[IDX_ELEM_H] / 40.0);
    ab[IDX_CH3COCH3II] = ab[IDX_CH3COCH3II] * (16.0 * rptr[IDX_ELEM_O] / 58.0 + 36.0 * rptr[IDX_ELEM_C] / 58.0 + 6.0 * rptr[IDX_ELEM_H] / 58.0);
    ab[IDX_CPII] = ab[IDX_CPII] * (31.0 * rptr[IDX_ELEM_P] / 43.0 + 12.0 * rptr[IDX_ELEM_C] / 43.0);
    ab[IDX_GC6I] = ab[IDX_GC6I] * (72.0 * rptr[IDX_ELEM_C] / 72.0);
    ab[IDX_GC8I] = ab[IDX_GC8I] * (96.0 * rptr[IDX_ELEM_C] / 96.0);
    ab[IDX_GCH2OHI] = ab[IDX_GCH2OHI] * (16.0 * rptr[IDX_ELEM_O] / 31.0 + 12.0 * rptr[IDX_ELEM_C] / 31.0 + 3.0 * rptr[IDX_ELEM_H] / 31.0);
    ab[IDX_GCH3OHI] = ab[IDX_GCH3OHI] * (16.0 * rptr[IDX_ELEM_O] / 32.0 + 12.0 * rptr[IDX_ELEM_C] / 32.0 + 4.0 * rptr[IDX_ELEM_H] / 32.0);
    ab[IDX_H2S2II] = ab[IDX_H2S2II] * (64.0 * rptr[IDX_ELEM_S] / 66.0 + 2.0 * rptr[IDX_ELEM_H] / 66.0);
    ab[IDX_HC2PII] = ab[IDX_HC2PII] * (31.0 * rptr[IDX_ELEM_P] / 56.0 + 24.0 * rptr[IDX_ELEM_C] / 56.0 + 1.0 * rptr[IDX_ELEM_H] / 56.0);
    ab[IDX_HC5NII] = ab[IDX_HC5NII] * (14.0 * rptr[IDX_ELEM_N] / 75.0 + 60.0 * rptr[IDX_ELEM_C] / 75.0 + 1.0 * rptr[IDX_ELEM_H] / 75.0);
    ab[IDX_HFI] = ab[IDX_HFI] * (19.0 * rptr[IDX_ELEM_F] / 20.0 + 1.0 * rptr[IDX_ELEM_H] / 20.0);
    ab[IDX_HONCI] = ab[IDX_HONCI] * (14.0 * rptr[IDX_ELEM_N] / 43.0 + 16.0 * rptr[IDX_ELEM_O] / 43.0 + 12.0 * rptr[IDX_ELEM_C] / 43.0 + 1.0 * rptr[IDX_ELEM_H] / 43.0);
    ab[IDX_HSiSII] = ab[IDX_HSiSII] * (28.0 * rptr[IDX_ELEM_Si] / 61.0 + 32.0 * rptr[IDX_ELEM_S] / 61.0 + 1.0 * rptr[IDX_ELEM_H] / 61.0);
    ab[IDX_PCH2II] = ab[IDX_PCH2II] * (31.0 * rptr[IDX_ELEM_P] / 45.0 + 12.0 * rptr[IDX_ELEM_C] / 45.0 + 2.0 * rptr[IDX_ELEM_H] / 45.0);
    ab[IDX_SO2II] = ab[IDX_SO2II] * (32.0 * rptr[IDX_ELEM_S] / 64.0 + 32.0 * rptr[IDX_ELEM_O] / 64.0);
    ab[IDX_SiNII] = ab[IDX_SiNII] * (28.0 * rptr[IDX_ELEM_Si] / 42.0 + 14.0 * rptr[IDX_ELEM_N] / 42.0);
    ab[IDX_C9II] = ab[IDX_C9II] * (108.0 * rptr[IDX_ELEM_C] / 108.0);
    ab[IDX_CH2OHCOI] = ab[IDX_CH2OHCOI] * (32.0 * rptr[IDX_ELEM_O] / 59.0 + 24.0 * rptr[IDX_ELEM_C] / 59.0 + 3.0 * rptr[IDX_ELEM_H] / 59.0);
    ab[IDX_CH3C5NI] = ab[IDX_CH3C5NI] * (14.0 * rptr[IDX_ELEM_N] / 89.0 + 72.0 * rptr[IDX_ELEM_C] / 89.0 + 3.0 * rptr[IDX_ELEM_H] / 89.0);
    ab[IDX_CH3C7NI] = ab[IDX_CH3C7NI] * (14.0 * rptr[IDX_ELEM_N] / 113.0 + 96.0 * rptr[IDX_ELEM_C] / 113.0 + 3.0 * rptr[IDX_ELEM_H] / 113.0);
    ab[IDX_GC3HI] = ab[IDX_GC3HI] * (36.0 * rptr[IDX_ELEM_C] / 37.0 + 1.0 * rptr[IDX_ELEM_H] / 37.0);
    ab[IDX_GC4HI] = ab[IDX_GC4HI] * (48.0 * rptr[IDX_ELEM_C] / 49.0 + 1.0 * rptr[IDX_ELEM_H] / 49.0);
    ab[IDX_GC5HI] = ab[IDX_GC5HI] * (60.0 * rptr[IDX_ELEM_C] / 61.0 + 1.0 * rptr[IDX_ELEM_H] / 61.0);
    ab[IDX_GC6HI] = ab[IDX_GC6HI] * (72.0 * rptr[IDX_ELEM_C] / 73.0 + 1.0 * rptr[IDX_ELEM_H] / 73.0);
    ab[IDX_H2NCII] = ab[IDX_H2NCII] * (14.0 * rptr[IDX_ELEM_N] / 28.0 + 12.0 * rptr[IDX_ELEM_C] / 28.0 + 2.0 * rptr[IDX_ELEM_H] / 28.0);
    ab[IDX_H2SiOI] = ab[IDX_H2SiOI] * (28.0 * rptr[IDX_ELEM_Si] / 46.0 + 16.0 * rptr[IDX_ELEM_O] / 46.0 + 2.0 * rptr[IDX_ELEM_H] / 46.0);
    ab[IDX_H3CSII] = ab[IDX_H3CSII] * (32.0 * rptr[IDX_ELEM_S] / 47.0 + 12.0 * rptr[IDX_ELEM_C] / 47.0 + 3.0 * rptr[IDX_ELEM_H] / 47.0);
    ab[IDX_HC2OII] = ab[IDX_HC2OII] * (16.0 * rptr[IDX_ELEM_O] / 41.0 + 24.0 * rptr[IDX_ELEM_C] / 41.0 + 1.0 * rptr[IDX_ELEM_H] / 41.0);
    ab[IDX_HCPII] = ab[IDX_HCPII] * (31.0 * rptr[IDX_ELEM_P] / 44.0 + 12.0 * rptr[IDX_ELEM_C] / 44.0 + 1.0 * rptr[IDX_ELEM_H] / 44.0);
    ab[IDX_SiC4I] = ab[IDX_SiC4I] * (28.0 * rptr[IDX_ELEM_Si] / 76.0 + 48.0 * rptr[IDX_ELEM_C] / 76.0);
    ab[IDX_SiNCI] = ab[IDX_SiNCI] * (28.0 * rptr[IDX_ELEM_Si] / 54.0 + 14.0 * rptr[IDX_ELEM_N] / 54.0 + 12.0 * rptr[IDX_ELEM_C] / 54.0);
    ab[IDX_SiNCII] = ab[IDX_SiNCII] * (28.0 * rptr[IDX_ELEM_Si] / 54.0 + 14.0 * rptr[IDX_ELEM_N] / 54.0 + 12.0 * rptr[IDX_ELEM_C] / 54.0);
    ab[IDX_CNOI] = ab[IDX_CNOI] * (14.0 * rptr[IDX_ELEM_N] / 42.0 + 16.0 * rptr[IDX_ELEM_O] / 42.0 + 12.0 * rptr[IDX_ELEM_C] / 42.0);
    ab[IDX_GC2H3I] = ab[IDX_GC2H3I] * (24.0 * rptr[IDX_ELEM_C] / 27.0 + 3.0 * rptr[IDX_ELEM_H] / 27.0);
    ab[IDX_GCH3OI] = ab[IDX_GCH3OI] * (16.0 * rptr[IDX_ELEM_O] / 31.0 + 12.0 * rptr[IDX_ELEM_C] / 31.0 + 3.0 * rptr[IDX_ELEM_H] / 31.0);
    ab[IDX_GH2OI] = ab[IDX_GH2OI] * (16.0 * rptr[IDX_ELEM_O] / 18.0 + 2.0 * rptr[IDX_ELEM_H] / 18.0);
    ab[IDX_HC3SII] = ab[IDX_HC3SII] * (32.0 * rptr[IDX_ELEM_S] / 69.0 + 36.0 * rptr[IDX_ELEM_C] / 69.0 + 1.0 * rptr[IDX_ELEM_H] / 69.0);
    ab[IDX_HCOOH2II] = ab[IDX_HCOOH2II] * (32.0 * rptr[IDX_ELEM_O] / 47.0 + 12.0 * rptr[IDX_ELEM_C] / 47.0 + 3.0 * rptr[IDX_ELEM_H] / 47.0);
    ab[IDX_HNSiI] = ab[IDX_HNSiI] * (28.0 * rptr[IDX_ELEM_Si] / 43.0 + 14.0 * rptr[IDX_ELEM_N] / 43.0 + 1.0 * rptr[IDX_ELEM_H] / 43.0);
    ab[IDX_HPOII] = ab[IDX_HPOII] * (31.0 * rptr[IDX_ELEM_P] / 48.0 + 16.0 * rptr[IDX_ELEM_O] / 48.0 + 1.0 * rptr[IDX_ELEM_H] / 48.0);
    ab[IDX_HS2II] = ab[IDX_HS2II] * (64.0 * rptr[IDX_ELEM_S] / 65.0 + 1.0 * rptr[IDX_ELEM_H] / 65.0);
    ab[IDX_SiC2H2I] = ab[IDX_SiC2H2I] * (28.0 * rptr[IDX_ELEM_Si] / 54.0 + 24.0 * rptr[IDX_ELEM_C] / 54.0 + 2.0 * rptr[IDX_ELEM_H] / 54.0);
    ab[IDX_SiC3II] = ab[IDX_SiC3II] * (28.0 * rptr[IDX_ELEM_Si] / 64.0 + 36.0 * rptr[IDX_ELEM_C] / 64.0);
    ab[IDX_C2NHII] = ab[IDX_C2NHII] * (14.0 * rptr[IDX_ELEM_N] / 39.0 + 24.0 * rptr[IDX_ELEM_C] / 39.0 + 1.0 * rptr[IDX_ELEM_H] / 39.0);
    ab[IDX_C3H6II] = ab[IDX_C3H6II] * (36.0 * rptr[IDX_ELEM_C] / 42.0 + 6.0 * rptr[IDX_ELEM_H] / 42.0);
    ab[IDX_C6H7II] = ab[IDX_C6H7II] * (72.0 * rptr[IDX_ELEM_C] / 79.0 + 7.0 * rptr[IDX_ELEM_H] / 79.0);
    ab[IDX_C8H4II] = ab[IDX_C8H4II] * (96.0 * rptr[IDX_ELEM_C] / 100.0 + 4.0 * rptr[IDX_ELEM_H] / 100.0);
    ab[IDX_CH2OHCHOI] = ab[IDX_CH2OHCHOI] * (32.0 * rptr[IDX_ELEM_O] / 60.0 + 24.0 * rptr[IDX_ELEM_C] / 60.0 + 4.0 * rptr[IDX_ELEM_H] / 60.0);
    ab[IDX_ClI] = ab[IDX_ClI] * (35.0 * rptr[IDX_ELEM_Cl] / 35.0);
    ab[IDX_GC7I] = ab[IDX_GC7I] * (84.0 * rptr[IDX_ELEM_C] / 84.0);
    ab[IDX_GSiI] = ab[IDX_GSiI] * (28.0 * rptr[IDX_ELEM_Si] / 28.0);
    ab[IDX_HC3OII] = ab[IDX_HC3OII] * (16.0 * rptr[IDX_ELEM_O] / 53.0 + 36.0 * rptr[IDX_ELEM_C] / 53.0 + 1.0 * rptr[IDX_ELEM_H] / 53.0);
    ab[IDX_HCNOI] = ab[IDX_HCNOI] * (14.0 * rptr[IDX_ELEM_N] / 43.0 + 16.0 * rptr[IDX_ELEM_O] / 43.0 + 12.0 * rptr[IDX_ELEM_C] / 43.0 + 1.0 * rptr[IDX_ELEM_H] / 43.0);
    ab[IDX_SiC3HII] = ab[IDX_SiC3HII] * (28.0 * rptr[IDX_ELEM_Si] / 65.0 + 36.0 * rptr[IDX_ELEM_C] / 65.0 + 1.0 * rptr[IDX_ELEM_H] / 65.0);
    ab[IDX_SiCH3I] = ab[IDX_SiCH3I] * (28.0 * rptr[IDX_ELEM_Si] / 43.0 + 12.0 * rptr[IDX_ELEM_C] / 43.0 + 3.0 * rptr[IDX_ELEM_H] / 43.0);
    ab[IDX_C3H7II] = ab[IDX_C3H7II] * (36.0 * rptr[IDX_ELEM_C] / 43.0 + 7.0 * rptr[IDX_ELEM_H] / 43.0);
    ab[IDX_C4NII] = ab[IDX_C4NII] * (14.0 * rptr[IDX_ELEM_N] / 62.0 + 48.0 * rptr[IDX_ELEM_C] / 62.0);
    ab[IDX_C8II] = ab[IDX_C8II] * (96.0 * rptr[IDX_ELEM_C] / 96.0);
    ab[IDX_C9H4II] = ab[IDX_C9H4II] * (108.0 * rptr[IDX_ELEM_C] / 112.0 + 4.0 * rptr[IDX_ELEM_H] / 112.0);
    ab[IDX_CH2COII] = ab[IDX_CH2COII] * (16.0 * rptr[IDX_ELEM_O] / 42.0 + 24.0 * rptr[IDX_ELEM_C] / 42.0 + 2.0 * rptr[IDX_ELEM_H] / 42.0);
    ab[IDX_CH2PHI] = ab[IDX_CH2PHI] * (31.0 * rptr[IDX_ELEM_P] / 46.0 + 12.0 * rptr[IDX_ELEM_C] / 46.0 + 3.0 * rptr[IDX_ELEM_H] / 46.0);
    ab[IDX_CH3C3NI] = ab[IDX_CH3C3NI] * (14.0 * rptr[IDX_ELEM_N] / 65.0 + 48.0 * rptr[IDX_ELEM_C] / 65.0 + 3.0 * rptr[IDX_ELEM_H] / 65.0);
    ab[IDX_GC4I] = ab[IDX_GC4I] * (48.0 * rptr[IDX_ELEM_C] / 48.0);
    ab[IDX_GCH3CHOI] = ab[IDX_GCH3CHOI] * (16.0 * rptr[IDX_ELEM_O] / 44.0 + 24.0 * rptr[IDX_ELEM_C] / 44.0 + 4.0 * rptr[IDX_ELEM_H] / 44.0);
    ab[IDX_GH2COI] = ab[IDX_GH2COI] * (16.0 * rptr[IDX_ELEM_O] / 30.0 + 12.0 * rptr[IDX_ELEM_C] / 30.0 + 2.0 * rptr[IDX_ELEM_H] / 30.0);
    ab[IDX_HOCNI] = ab[IDX_HOCNI] * (14.0 * rptr[IDX_ELEM_N] / 43.0 + 16.0 * rptr[IDX_ELEM_O] / 43.0 + 12.0 * rptr[IDX_ELEM_C] / 43.0 + 1.0 * rptr[IDX_ELEM_H] / 43.0);
    ab[IDX_HPOI] = ab[IDX_HPOI] * (31.0 * rptr[IDX_ELEM_P] / 48.0 + 16.0 * rptr[IDX_ELEM_O] / 48.0 + 1.0 * rptr[IDX_ELEM_H] / 48.0);
    ab[IDX_HS2I] = ab[IDX_HS2I] * (64.0 * rptr[IDX_ELEM_S] / 65.0 + 1.0 * rptr[IDX_ELEM_H] / 65.0);
    ab[IDX_SiC2II] = ab[IDX_SiC2II] * (28.0 * rptr[IDX_ELEM_Si] / 52.0 + 24.0 * rptr[IDX_ELEM_C] / 52.0);
    ab[IDX_SiC2H2II] = ab[IDX_SiC2H2II] * (28.0 * rptr[IDX_ELEM_Si] / 54.0 + 24.0 * rptr[IDX_ELEM_C] / 54.0 + 2.0 * rptr[IDX_ELEM_H] / 54.0);
    ab[IDX_C6H4II] = ab[IDX_C6H4II] * (72.0 * rptr[IDX_ELEM_C] / 76.0 + 4.0 * rptr[IDX_ELEM_H] / 76.0);
    ab[IDX_CH3COI] = ab[IDX_CH3COI] * (16.0 * rptr[IDX_ELEM_O] / 43.0 + 24.0 * rptr[IDX_ELEM_C] / 43.0 + 3.0 * rptr[IDX_ELEM_H] / 43.0);
    ab[IDX_CH3COCH4II] = ab[IDX_CH3COCH4II] * (16.0 * rptr[IDX_ELEM_O] / 59.0 + 36.0 * rptr[IDX_ELEM_C] / 59.0 + 7.0 * rptr[IDX_ELEM_H] / 59.0);
    ab[IDX_CH3OHII] = ab[IDX_CH3OHII] * (16.0 * rptr[IDX_ELEM_O] / 32.0 + 12.0 * rptr[IDX_ELEM_C] / 32.0 + 4.0 * rptr[IDX_ELEM_H] / 32.0);
    ab[IDX_COOHI] = ab[IDX_COOHI] * (32.0 * rptr[IDX_ELEM_O] / 45.0 + 12.0 * rptr[IDX_ELEM_C] / 45.0 + 1.0 * rptr[IDX_ELEM_H] / 45.0);
    ab[IDX_GNOI] = ab[IDX_GNOI] * (14.0 * rptr[IDX_ELEM_N] / 30.0 + 16.0 * rptr[IDX_ELEM_O] / 30.0);
    ab[IDX_POII] = ab[IDX_POII] * (31.0 * rptr[IDX_ELEM_P] / 47.0 + 16.0 * rptr[IDX_ELEM_O] / 47.0);
    ab[IDX_C7H4II] = ab[IDX_C7H4II] * (84.0 * rptr[IDX_ELEM_C] / 88.0 + 4.0 * rptr[IDX_ELEM_H] / 88.0);
    ab[IDX_C7H5II] = ab[IDX_C7H5II] * (84.0 * rptr[IDX_ELEM_C] / 89.0 + 5.0 * rptr[IDX_ELEM_H] / 89.0);
    ab[IDX_CH3C6HI] = ab[IDX_CH3C6HI] * (84.0 * rptr[IDX_ELEM_C] / 88.0 + 4.0 * rptr[IDX_ELEM_H] / 88.0);
    ab[IDX_H2CSII] = ab[IDX_H2CSII] * (32.0 * rptr[IDX_ELEM_S] / 46.0 + 12.0 * rptr[IDX_ELEM_C] / 46.0 + 2.0 * rptr[IDX_ELEM_H] / 46.0);
    ab[IDX_HC9NI] = ab[IDX_HC9NI] * (14.0 * rptr[IDX_ELEM_N] / 123.0 + 108.0 * rptr[IDX_ELEM_C] / 123.0 + 1.0 * rptr[IDX_ELEM_H] / 123.0);
    ab[IDX_HCOOCH3I] = ab[IDX_HCOOCH3I] * (32.0 * rptr[IDX_ELEM_O] / 60.0 + 24.0 * rptr[IDX_ELEM_C] / 60.0 + 4.0 * rptr[IDX_ELEM_H] / 60.0);
    ab[IDX_NCCNHII] = ab[IDX_NCCNHII] * (28.0 * rptr[IDX_ELEM_N] / 53.0 + 24.0 * rptr[IDX_ELEM_C] / 53.0 + 1.0 * rptr[IDX_ELEM_H] / 53.0);
    ab[IDX_NO2I] = ab[IDX_NO2I] * (14.0 * rptr[IDX_ELEM_N] / 46.0 + 32.0 * rptr[IDX_ELEM_O] / 46.0);
    ab[IDX_PH2I] = ab[IDX_PH2I] * (31.0 * rptr[IDX_ELEM_P] / 33.0 + 2.0 * rptr[IDX_ELEM_H] / 33.0);
    ab[IDX_PNI] = ab[IDX_PNI] * (31.0 * rptr[IDX_ELEM_P] / 45.0 + 14.0 * rptr[IDX_ELEM_N] / 45.0);
    ab[IDX_SiC2HI] = ab[IDX_SiC2HI] * (28.0 * rptr[IDX_ELEM_Si] / 53.0 + 24.0 * rptr[IDX_ELEM_C] / 53.0 + 1.0 * rptr[IDX_ELEM_H] / 53.0);
    ab[IDX_SiC3I] = ab[IDX_SiC3I] * (28.0 * rptr[IDX_ELEM_Si] / 64.0 + 36.0 * rptr[IDX_ELEM_C] / 64.0);
    ab[IDX_C4PI] = ab[IDX_C4PI] * (31.0 * rptr[IDX_ELEM_P] / 79.0 + 48.0 * rptr[IDX_ELEM_C] / 79.0);
    ab[IDX_C7II] = ab[IDX_C7II] * (84.0 * rptr[IDX_ELEM_C] / 84.0);
    ab[IDX_CH2CNI] = ab[IDX_CH2CNI] * (14.0 * rptr[IDX_ELEM_N] / 40.0 + 24.0 * rptr[IDX_ELEM_C] / 40.0 + 2.0 * rptr[IDX_ELEM_H] / 40.0);
    ab[IDX_CH3C4HII] = ab[IDX_CH3C4HII] * (60.0 * rptr[IDX_ELEM_C] / 64.0 + 4.0 * rptr[IDX_ELEM_H] / 64.0);
    ab[IDX_GCH4I] = ab[IDX_GCH4I] * (12.0 * rptr[IDX_ELEM_C] / 16.0 + 4.0 * rptr[IDX_ELEM_H] / 16.0);
    ab[IDX_HCPI] = ab[IDX_HCPI] * (31.0 * rptr[IDX_ELEM_P] / 44.0 + 12.0 * rptr[IDX_ELEM_C] / 44.0 + 1.0 * rptr[IDX_ELEM_H] / 44.0);
    ab[IDX_HClI] = ab[IDX_HClI] * (35.0 * rptr[IDX_ELEM_Cl] / 36.0 + 1.0 * rptr[IDX_ELEM_H] / 36.0);
    ab[IDX_HNCOI] = ab[IDX_HNCOI] * (14.0 * rptr[IDX_ELEM_N] / 43.0 + 16.0 * rptr[IDX_ELEM_O] / 43.0 + 12.0 * rptr[IDX_ELEM_C] / 43.0 + 1.0 * rptr[IDX_ELEM_H] / 43.0);
    ab[IDX_C3PI] = ab[IDX_C3PI] * (31.0 * rptr[IDX_ELEM_P] / 67.0 + 36.0 * rptr[IDX_ELEM_C] / 67.0);
    ab[IDX_C9NI] = ab[IDX_C9NI] * (14.0 * rptr[IDX_ELEM_N] / 122.0 + 108.0 * rptr[IDX_ELEM_C] / 122.0);
    ab[IDX_CH3C4HI] = ab[IDX_CH3C4HI] * (60.0 * rptr[IDX_ELEM_C] / 64.0 + 4.0 * rptr[IDX_ELEM_H] / 64.0);
    ab[IDX_CH3COOHI] = ab[IDX_CH3COOHI] * (32.0 * rptr[IDX_ELEM_O] / 60.0 + 24.0 * rptr[IDX_ELEM_C] / 60.0 + 4.0 * rptr[IDX_ELEM_H] / 60.0);
    ab[IDX_CH3OCH3I] = ab[IDX_CH3OCH3I] * (16.0 * rptr[IDX_ELEM_O] / 46.0 + 24.0 * rptr[IDX_ELEM_C] / 46.0 + 6.0 * rptr[IDX_ELEM_H] / 46.0);
    ab[IDX_GC2HI] = ab[IDX_GC2HI] * (24.0 * rptr[IDX_ELEM_C] / 25.0 + 1.0 * rptr[IDX_ELEM_H] / 25.0);
    ab[IDX_GNH2I] = ab[IDX_GNH2I] * (14.0 * rptr[IDX_ELEM_N] / 16.0 + 2.0 * rptr[IDX_ELEM_H] / 16.0);
    ab[IDX_H2CSI] = ab[IDX_H2CSI] * (32.0 * rptr[IDX_ELEM_S] / 46.0 + 12.0 * rptr[IDX_ELEM_C] / 46.0 + 2.0 * rptr[IDX_ELEM_H] / 46.0);
    ab[IDX_PH2II] = ab[IDX_PH2II] * (31.0 * rptr[IDX_ELEM_P] / 33.0 + 2.0 * rptr[IDX_ELEM_H] / 33.0);
    ab[IDX_POI] = ab[IDX_POI] * (31.0 * rptr[IDX_ELEM_P] / 47.0 + 16.0 * rptr[IDX_ELEM_O] / 47.0);
    ab[IDX_S2I] = ab[IDX_S2I] * (64.0 * rptr[IDX_ELEM_S] / 64.0);
    ab[IDX_S2II] = ab[IDX_S2II] * (64.0 * rptr[IDX_ELEM_S] / 64.0);
    ab[IDX_SiCII] = ab[IDX_SiCII] * (28.0 * rptr[IDX_ELEM_Si] / 40.0 + 12.0 * rptr[IDX_ELEM_C] / 40.0);
    ab[IDX_C3SI] = ab[IDX_C3SI] * (32.0 * rptr[IDX_ELEM_S] / 68.0 + 36.0 * rptr[IDX_ELEM_C] / 68.0);
    ab[IDX_C6H5II] = ab[IDX_C6H5II] * (72.0 * rptr[IDX_ELEM_C] / 77.0 + 5.0 * rptr[IDX_ELEM_H] / 77.0);
    ab[IDX_CH2NHI] = ab[IDX_CH2NHI] * (14.0 * rptr[IDX_ELEM_N] / 29.0 + 12.0 * rptr[IDX_ELEM_C] / 29.0 + 3.0 * rptr[IDX_ELEM_H] / 29.0);
    ab[IDX_CH3OI] = ab[IDX_CH3OI] * (16.0 * rptr[IDX_ELEM_O] / 31.0 + 12.0 * rptr[IDX_ELEM_C] / 31.0 + 3.0 * rptr[IDX_ELEM_H] / 31.0);
    ab[IDX_GHCOI] = ab[IDX_GHCOI] * (16.0 * rptr[IDX_ELEM_O] / 29.0 + 12.0 * rptr[IDX_ELEM_C] / 29.0 + 1.0 * rptr[IDX_ELEM_H] / 29.0);
    ab[IDX_HC2PI] = ab[IDX_HC2PI] * (31.0 * rptr[IDX_ELEM_P] / 56.0 + 24.0 * rptr[IDX_ELEM_C] / 56.0 + 1.0 * rptr[IDX_ELEM_H] / 56.0);
    ab[IDX_HCSiI] = ab[IDX_HCSiI] * (28.0 * rptr[IDX_ELEM_Si] / 41.0 + 12.0 * rptr[IDX_ELEM_C] / 41.0 + 1.0 * rptr[IDX_ELEM_H] / 41.0);
    ab[IDX_SiCH2I] = ab[IDX_SiCH2I] * (28.0 * rptr[IDX_ELEM_Si] / 42.0 + 12.0 * rptr[IDX_ELEM_C] / 42.0 + 2.0 * rptr[IDX_ELEM_H] / 42.0);
    ab[IDX_C10HII] = ab[IDX_C10HII] * (120.0 * rptr[IDX_ELEM_C] / 121.0 + 1.0 * rptr[IDX_ELEM_H] / 121.0);
    ab[IDX_HCSiII] = ab[IDX_HCSiII] * (28.0 * rptr[IDX_ELEM_Si] / 41.0 + 12.0 * rptr[IDX_ELEM_C] / 41.0 + 1.0 * rptr[IDX_ELEM_H] / 41.0);
    ab[IDX_OCSII] = ab[IDX_OCSII] * (32.0 * rptr[IDX_ELEM_S] / 60.0 + 16.0 * rptr[IDX_ELEM_O] / 60.0 + 12.0 * rptr[IDX_ELEM_C] / 60.0);
    ab[IDX_SiC2I] = ab[IDX_SiC2I] * (28.0 * rptr[IDX_ELEM_Si] / 52.0 + 24.0 * rptr[IDX_ELEM_C] / 52.0);
    ab[IDX_C2H6II] = ab[IDX_C2H6II] * (24.0 * rptr[IDX_ELEM_C] / 30.0 + 6.0 * rptr[IDX_ELEM_H] / 30.0);
    ab[IDX_C2OI] = ab[IDX_C2OI] * (16.0 * rptr[IDX_ELEM_O] / 40.0 + 24.0 * rptr[IDX_ELEM_C] / 40.0);
    ab[IDX_C8HII] = ab[IDX_C8HII] * (96.0 * rptr[IDX_ELEM_C] / 97.0 + 1.0 * rptr[IDX_ELEM_H] / 97.0);
    ab[IDX_CH2OHI] = ab[IDX_CH2OHI] * (16.0 * rptr[IDX_ELEM_O] / 31.0 + 12.0 * rptr[IDX_ELEM_C] / 31.0 + 3.0 * rptr[IDX_ELEM_H] / 31.0);
    ab[IDX_SiNI] = ab[IDX_SiNI] * (28.0 * rptr[IDX_ELEM_Si] / 42.0 + 14.0 * rptr[IDX_ELEM_N] / 42.0);
    ab[IDX_C5H5II] = ab[IDX_C5H5II] * (60.0 * rptr[IDX_ELEM_C] / 65.0 + 5.0 * rptr[IDX_ELEM_H] / 65.0);
    ab[IDX_GNHI] = ab[IDX_GNHI] * (14.0 * rptr[IDX_ELEM_N] / 15.0 + 1.0 * rptr[IDX_ELEM_H] / 15.0);
    ab[IDX_GSI] = ab[IDX_GSI] * (32.0 * rptr[IDX_ELEM_S] / 32.0);
    ab[IDX_HCSI] = ab[IDX_HCSI] * (32.0 * rptr[IDX_ELEM_S] / 45.0 + 12.0 * rptr[IDX_ELEM_C] / 45.0 + 1.0 * rptr[IDX_ELEM_H] / 45.0);
    ab[IDX_NSI] = ab[IDX_NSI] * (32.0 * rptr[IDX_ELEM_S] / 46.0 + 14.0 * rptr[IDX_ELEM_N] / 46.0);
    ab[IDX_SiC2HII] = ab[IDX_SiC2HII] * (28.0 * rptr[IDX_ELEM_Si] / 53.0 + 24.0 * rptr[IDX_ELEM_C] / 53.0 + 1.0 * rptr[IDX_ELEM_H] / 53.0);
    ab[IDX_SiH2I] = ab[IDX_SiH2I] * (28.0 * rptr[IDX_ELEM_Si] / 30.0 + 2.0 * rptr[IDX_ELEM_H] / 30.0);
    ab[IDX_C3OI] = ab[IDX_C3OI] * (16.0 * rptr[IDX_ELEM_O] / 52.0 + 36.0 * rptr[IDX_ELEM_C] / 52.0);
    ab[IDX_C6HII] = ab[IDX_C6HII] * (72.0 * rptr[IDX_ELEM_C] / 73.0 + 1.0 * rptr[IDX_ELEM_H] / 73.0);
    ab[IDX_C9H2I] = ab[IDX_C9H2I] * (108.0 * rptr[IDX_ELEM_C] / 110.0 + 2.0 * rptr[IDX_ELEM_H] / 110.0);
    ab[IDX_CH2CHCNI] = ab[IDX_CH2CHCNI] * (14.0 * rptr[IDX_ELEM_N] / 53.0 + 36.0 * rptr[IDX_ELEM_C] / 53.0 + 3.0 * rptr[IDX_ELEM_H] / 53.0);
    ab[IDX_CH3CHOHII] = ab[IDX_CH3CHOHII] * (16.0 * rptr[IDX_ELEM_O] / 45.0 + 24.0 * rptr[IDX_ELEM_C] / 45.0 + 5.0 * rptr[IDX_ELEM_H] / 45.0);
    ab[IDX_CH3CNHII] = ab[IDX_CH3CNHII] * (14.0 * rptr[IDX_ELEM_N] / 42.0 + 24.0 * rptr[IDX_ELEM_C] / 42.0 + 4.0 * rptr[IDX_ELEM_H] / 42.0);
    ab[IDX_CH3COII] = ab[IDX_CH3COII] * (16.0 * rptr[IDX_ELEM_O] / 43.0 + 24.0 * rptr[IDX_ELEM_C] / 43.0 + 3.0 * rptr[IDX_ELEM_H] / 43.0);
    ab[IDX_HC7NI] = ab[IDX_HC7NI] * (14.0 * rptr[IDX_ELEM_N] / 99.0 + 84.0 * rptr[IDX_ELEM_C] / 99.0 + 1.0 * rptr[IDX_ELEM_H] / 99.0);
    ab[IDX_C6II] = ab[IDX_C6II] * (72.0 * rptr[IDX_ELEM_C] / 72.0);
    ab[IDX_C6H6I] = ab[IDX_C6H6I] * (72.0 * rptr[IDX_ELEM_C] / 78.0 + 6.0 * rptr[IDX_ELEM_H] / 78.0);
    ab[IDX_C7NI] = ab[IDX_C7NI] * (14.0 * rptr[IDX_ELEM_N] / 98.0 + 84.0 * rptr[IDX_ELEM_C] / 98.0);
    ab[IDX_HCOOHI] = ab[IDX_HCOOHI] * (32.0 * rptr[IDX_ELEM_O] / 46.0 + 12.0 * rptr[IDX_ELEM_C] / 46.0 + 2.0 * rptr[IDX_ELEM_H] / 46.0);
    ab[IDX_SiH2II] = ab[IDX_SiH2II] * (28.0 * rptr[IDX_ELEM_Si] / 30.0 + 2.0 * rptr[IDX_ELEM_H] / 30.0);
    ab[IDX_SiH3II] = ab[IDX_SiH3II] * (28.0 * rptr[IDX_ELEM_Si] / 31.0 + 3.0 * rptr[IDX_ELEM_H] / 31.0);
    ab[IDX_C10H2I] = ab[IDX_C10H2I] * (120.0 * rptr[IDX_ELEM_C] / 122.0 + 2.0 * rptr[IDX_ELEM_H] / 122.0);
    ab[IDX_C4II] = ab[IDX_C4II] * (48.0 * rptr[IDX_ELEM_C] / 48.0);
    ab[IDX_CCPI] = ab[IDX_CCPI] * (31.0 * rptr[IDX_ELEM_P] / 55.0 + 24.0 * rptr[IDX_ELEM_C] / 55.0);
    ab[IDX_CPI] = ab[IDX_CPI] * (31.0 * rptr[IDX_ELEM_P] / 43.0 + 12.0 * rptr[IDX_ELEM_C] / 43.0);
    ab[IDX_GC3I] = ab[IDX_GC3I] * (36.0 * rptr[IDX_ELEM_C] / 36.0);
    ab[IDX_O2HI] = ab[IDX_O2HI] * (32.0 * rptr[IDX_ELEM_O] / 33.0 + 1.0 * rptr[IDX_ELEM_H] / 33.0);
    ab[IDX_SO2I] = ab[IDX_SO2I] * (32.0 * rptr[IDX_ELEM_S] / 64.0 + 32.0 * rptr[IDX_ELEM_O] / 64.0);
    ab[IDX_SiCI] = ab[IDX_SiCI] * (28.0 * rptr[IDX_ELEM_Si] / 40.0 + 12.0 * rptr[IDX_ELEM_C] / 40.0);
    ab[IDX_C2H5OHI] = ab[IDX_C2H5OHI] * (16.0 * rptr[IDX_ELEM_O] / 46.0 + 24.0 * rptr[IDX_ELEM_C] / 46.0 + 6.0 * rptr[IDX_ELEM_H] / 46.0);
    ab[IDX_HNOI] = ab[IDX_HNOI] * (14.0 * rptr[IDX_ELEM_N] / 31.0 + 16.0 * rptr[IDX_ELEM_O] / 31.0 + 1.0 * rptr[IDX_ELEM_H] / 31.0);
    ab[IDX_N2OI] = ab[IDX_N2OI] * (28.0 * rptr[IDX_ELEM_N] / 44.0 + 16.0 * rptr[IDX_ELEM_O] / 44.0);
    ab[IDX_SiH3I] = ab[IDX_SiH3I] * (28.0 * rptr[IDX_ELEM_Si] / 31.0 + 3.0 * rptr[IDX_ELEM_H] / 31.0);
    ab[IDX_CH3OH2II] = ab[IDX_CH3OH2II] * (16.0 * rptr[IDX_ELEM_O] / 33.0 + 12.0 * rptr[IDX_ELEM_C] / 33.0 + 5.0 * rptr[IDX_ELEM_H] / 33.0);
    ab[IDX_C2NII] = ab[IDX_C2NII] * (14.0 * rptr[IDX_ELEM_N] / 38.0 + 24.0 * rptr[IDX_ELEM_C] / 38.0);
    ab[IDX_C5II] = ab[IDX_C5II] * (60.0 * rptr[IDX_ELEM_C] / 60.0);
    ab[IDX_C6H3II] = ab[IDX_C6H3II] * (72.0 * rptr[IDX_ELEM_C] / 75.0 + 3.0 * rptr[IDX_ELEM_H] / 75.0);
    ab[IDX_C8H3II] = ab[IDX_C8H3II] * (96.0 * rptr[IDX_ELEM_C] / 99.0 + 3.0 * rptr[IDX_ELEM_H] / 99.0);
    ab[IDX_C9HII] = ab[IDX_C9HII] * (108.0 * rptr[IDX_ELEM_C] / 109.0 + 1.0 * rptr[IDX_ELEM_H] / 109.0);
    ab[IDX_C9H3II] = ab[IDX_C9H3II] * (108.0 * rptr[IDX_ELEM_C] / 111.0 + 3.0 * rptr[IDX_ELEM_H] / 111.0);
    ab[IDX_CH2COI] = ab[IDX_CH2COI] * (16.0 * rptr[IDX_ELEM_O] / 42.0 + 24.0 * rptr[IDX_ELEM_C] / 42.0 + 2.0 * rptr[IDX_ELEM_H] / 42.0);
    ab[IDX_HNC3I] = ab[IDX_HNC3I] * (14.0 * rptr[IDX_ELEM_N] / 51.0 + 36.0 * rptr[IDX_ELEM_C] / 51.0 + 1.0 * rptr[IDX_ELEM_H] / 51.0);
    ab[IDX_OCNI] = ab[IDX_OCNI] * (14.0 * rptr[IDX_ELEM_N] / 42.0 + 16.0 * rptr[IDX_ELEM_O] / 42.0 + 12.0 * rptr[IDX_ELEM_C] / 42.0);
    ab[IDX_SiCH2II] = ab[IDX_SiCH2II] * (28.0 * rptr[IDX_ELEM_Si] / 42.0 + 12.0 * rptr[IDX_ELEM_C] / 42.0 + 2.0 * rptr[IDX_ELEM_H] / 42.0);
    ab[IDX_SiHII] = ab[IDX_SiHII] * (28.0 * rptr[IDX_ELEM_Si] / 29.0 + 1.0 * rptr[IDX_ELEM_H] / 29.0);
    ab[IDX_CH3COCH3I] = ab[IDX_CH3COCH3I] * (16.0 * rptr[IDX_ELEM_O] / 58.0 + 36.0 * rptr[IDX_ELEM_C] / 58.0 + 6.0 * rptr[IDX_ELEM_H] / 58.0);
    ab[IDX_HC3NII] = ab[IDX_HC3NII] * (14.0 * rptr[IDX_ELEM_N] / 51.0 + 36.0 * rptr[IDX_ELEM_C] / 51.0 + 1.0 * rptr[IDX_ELEM_H] / 51.0);
    ab[IDX_C8H2I] = ab[IDX_C8H2I] * (96.0 * rptr[IDX_ELEM_C] / 98.0 + 2.0 * rptr[IDX_ELEM_H] / 98.0);
    ab[IDX_CO2II] = ab[IDX_CO2II] * (32.0 * rptr[IDX_ELEM_O] / 44.0 + 12.0 * rptr[IDX_ELEM_C] / 44.0);
    ab[IDX_GOHI] = ab[IDX_GOHI] * (16.0 * rptr[IDX_ELEM_O] / 17.0 + 1.0 * rptr[IDX_ELEM_H] / 17.0);
    ab[IDX_HC5NI] = ab[IDX_HC5NI] * (14.0 * rptr[IDX_ELEM_N] / 75.0 + 60.0 * rptr[IDX_ELEM_C] / 75.0 + 1.0 * rptr[IDX_ELEM_H] / 75.0);
    ab[IDX_SiHI] = ab[IDX_SiHI] * (28.0 * rptr[IDX_ELEM_Si] / 29.0 + 1.0 * rptr[IDX_ELEM_H] / 29.0);
    ab[IDX_SiH4I] = ab[IDX_SiH4I] * (28.0 * rptr[IDX_ELEM_Si] / 32.0 + 4.0 * rptr[IDX_ELEM_H] / 32.0);
    ab[IDX_C7HII] = ab[IDX_C7HII] * (84.0 * rptr[IDX_ELEM_C] / 85.0 + 1.0 * rptr[IDX_ELEM_H] / 85.0);
    ab[IDX_C9H2II] = ab[IDX_C9H2II] * (108.0 * rptr[IDX_ELEM_C] / 110.0 + 2.0 * rptr[IDX_ELEM_H] / 110.0);
    ab[IDX_PHI] = ab[IDX_PHI] * (31.0 * rptr[IDX_ELEM_P] / 32.0 + 1.0 * rptr[IDX_ELEM_H] / 32.0);
    ab[IDX_C7H2I] = ab[IDX_C7H2I] * (84.0 * rptr[IDX_ELEM_C] / 86.0 + 2.0 * rptr[IDX_ELEM_H] / 86.0);
    ab[IDX_GCOI] = ab[IDX_GCOI] * (16.0 * rptr[IDX_ELEM_O] / 28.0 + 12.0 * rptr[IDX_ELEM_C] / 28.0);
    ab[IDX_HCSII] = ab[IDX_HCSII] * (32.0 * rptr[IDX_ELEM_S] / 45.0 + 12.0 * rptr[IDX_ELEM_C] / 45.0 + 1.0 * rptr[IDX_ELEM_H] / 45.0);
    ab[IDX_PHII] = ab[IDX_PHII] * (31.0 * rptr[IDX_ELEM_P] / 32.0 + 1.0 * rptr[IDX_ELEM_H] / 32.0);
    ab[IDX_C3H4II] = ab[IDX_C3H4II] * (36.0 * rptr[IDX_ELEM_C] / 40.0 + 4.0 * rptr[IDX_ELEM_H] / 40.0);
    ab[IDX_C8H2II] = ab[IDX_C8H2II] * (96.0 * rptr[IDX_ELEM_C] / 98.0 + 2.0 * rptr[IDX_ELEM_H] / 98.0);
    ab[IDX_GCH2I] = ab[IDX_GCH2I] * (12.0 * rptr[IDX_ELEM_C] / 14.0 + 2.0 * rptr[IDX_ELEM_H] / 14.0);
    ab[IDX_HCO2II] = ab[IDX_HCO2II] * (32.0 * rptr[IDX_ELEM_O] / 45.0 + 12.0 * rptr[IDX_ELEM_C] / 45.0 + 1.0 * rptr[IDX_ELEM_H] / 45.0);
    ab[IDX_C5H3II] = ab[IDX_C5H3II] * (60.0 * rptr[IDX_ELEM_C] / 63.0 + 3.0 * rptr[IDX_ELEM_H] / 63.0);
    ab[IDX_HC3NHII] = ab[IDX_HC3NHII] * (14.0 * rptr[IDX_ELEM_N] / 52.0 + 36.0 * rptr[IDX_ELEM_C] / 52.0 + 2.0 * rptr[IDX_ELEM_H] / 52.0);
    ab[IDX_C5HII] = ab[IDX_C5HII] * (60.0 * rptr[IDX_ELEM_C] / 61.0 + 1.0 * rptr[IDX_ELEM_H] / 61.0);
    ab[IDX_CH3CHOI] = ab[IDX_CH3CHOI] * (16.0 * rptr[IDX_ELEM_O] / 44.0 + 24.0 * rptr[IDX_ELEM_C] / 44.0 + 4.0 * rptr[IDX_ELEM_H] / 44.0);
    ab[IDX_CSII] = ab[IDX_CSII] * (32.0 * rptr[IDX_ELEM_S] / 44.0 + 12.0 * rptr[IDX_ELEM_C] / 44.0);
    ab[IDX_C3H5II] = ab[IDX_C3H5II] * (36.0 * rptr[IDX_ELEM_C] / 41.0 + 5.0 * rptr[IDX_ELEM_H] / 41.0);
    ab[IDX_C7H3II] = ab[IDX_C7H3II] * (84.0 * rptr[IDX_ELEM_C] / 87.0 + 3.0 * rptr[IDX_ELEM_H] / 87.0);
    ab[IDX_PII] = ab[IDX_PII] * (31.0 * rptr[IDX_ELEM_P] / 31.0);
    ab[IDX_CH3CNI] = ab[IDX_CH3CNI] * (14.0 * rptr[IDX_ELEM_N] / 41.0 + 24.0 * rptr[IDX_ELEM_C] / 41.0 + 3.0 * rptr[IDX_ELEM_H] / 41.0);
    ab[IDX_NCCNI] = ab[IDX_NCCNI] * (28.0 * rptr[IDX_ELEM_N] / 52.0 + 24.0 * rptr[IDX_ELEM_C] / 52.0);
    ab[IDX_C7H2II] = ab[IDX_C7H2II] * (84.0 * rptr[IDX_ELEM_C] / 86.0 + 2.0 * rptr[IDX_ELEM_H] / 86.0);
    ab[IDX_GC2I] = ab[IDX_GC2I] * (24.0 * rptr[IDX_ELEM_C] / 24.0);
    ab[IDX_C3II] = ab[IDX_C3II] * (36.0 * rptr[IDX_ELEM_C] / 36.0);
    ab[IDX_H3SII] = ab[IDX_H3SII] * (32.0 * rptr[IDX_ELEM_S] / 35.0 + 3.0 * rptr[IDX_ELEM_H] / 35.0);
    ab[IDX_O2HII] = ab[IDX_O2HII] * (32.0 * rptr[IDX_ELEM_O] / 33.0 + 1.0 * rptr[IDX_ELEM_H] / 33.0);
    ab[IDX_C6H2II] = ab[IDX_C6H2II] * (72.0 * rptr[IDX_ELEM_C] / 74.0 + 2.0 * rptr[IDX_ELEM_H] / 74.0);
    ab[IDX_CH4II] = ab[IDX_CH4II] * (12.0 * rptr[IDX_ELEM_C] / 16.0 + 4.0 * rptr[IDX_ELEM_H] / 16.0);
    ab[IDX_HNOII] = ab[IDX_HNOII] * (14.0 * rptr[IDX_ELEM_N] / 31.0 + 16.0 * rptr[IDX_ELEM_O] / 31.0 + 1.0 * rptr[IDX_ELEM_H] / 31.0);
    ab[IDX_H2CCCI] = ab[IDX_H2CCCI] * (36.0 * rptr[IDX_ELEM_C] / 38.0 + 2.0 * rptr[IDX_ELEM_H] / 38.0);
    ab[IDX_C6H2I] = ab[IDX_C6H2I] * (72.0 * rptr[IDX_ELEM_C] / 74.0 + 2.0 * rptr[IDX_ELEM_H] / 74.0);
    ab[IDX_GCNI] = ab[IDX_GCNI] * (14.0 * rptr[IDX_ELEM_N] / 26.0 + 12.0 * rptr[IDX_ELEM_C] / 26.0);
    ab[IDX_C2H5II] = ab[IDX_C2H5II] * (24.0 * rptr[IDX_ELEM_C] / 29.0 + 5.0 * rptr[IDX_ELEM_H] / 29.0);
    ab[IDX_C5H2I] = ab[IDX_C5H2I] * (60.0 * rptr[IDX_ELEM_C] / 62.0 + 2.0 * rptr[IDX_ELEM_H] / 62.0);
    ab[IDX_C5H2II] = ab[IDX_C5H2II] * (60.0 * rptr[IDX_ELEM_C] / 62.0 + 2.0 * rptr[IDX_ELEM_H] / 62.0);
    ab[IDX_CH2CCH2I] = ab[IDX_CH2CCH2I] * (36.0 * rptr[IDX_ELEM_C] / 40.0 + 4.0 * rptr[IDX_ELEM_H] / 40.0);
    ab[IDX_CNII] = ab[IDX_CNII] * (14.0 * rptr[IDX_ELEM_N] / 26.0 + 12.0 * rptr[IDX_ELEM_C] / 26.0);
    ab[IDX_C2H6I] = ab[IDX_C2H6I] * (24.0 * rptr[IDX_ELEM_C] / 30.0 + 6.0 * rptr[IDX_ELEM_H] / 30.0);
    ab[IDX_GNI] = ab[IDX_GNI] * (14.0 * rptr[IDX_ELEM_N] / 14.0);
    ab[IDX_C3H2I] = ab[IDX_C3H2I] * (36.0 * rptr[IDX_ELEM_C] / 38.0 + 2.0 * rptr[IDX_ELEM_H] / 38.0);
    ab[IDX_C2H5I] = ab[IDX_C2H5I] * (24.0 * rptr[IDX_ELEM_C] / 29.0 + 5.0 * rptr[IDX_ELEM_H] / 29.0);
    ab[IDX_GOI] = ab[IDX_GOI] * (16.0 * rptr[IDX_ELEM_O] / 16.0);
    ab[IDX_N2II] = ab[IDX_N2II] * (28.0 * rptr[IDX_ELEM_N] / 28.0);
    ab[IDX_C4SII] = ab[IDX_C4SII] * (32.0 * rptr[IDX_ELEM_S] / 80.0 + 48.0 * rptr[IDX_ELEM_C] / 80.0);
    ab[IDX_HCNII] = ab[IDX_HCNII] * (14.0 * rptr[IDX_ELEM_N] / 27.0 + 12.0 * rptr[IDX_ELEM_C] / 27.0 + 1.0 * rptr[IDX_ELEM_H] / 27.0);
    ab[IDX_CH3CHCH2I] = ab[IDX_CH3CHCH2I] * (36.0 * rptr[IDX_ELEM_C] / 42.0 + 6.0 * rptr[IDX_ELEM_H] / 42.0);
    ab[IDX_C2II] = ab[IDX_C2II] * (24.0 * rptr[IDX_ELEM_C] / 24.0);
    ab[IDX_C4HII] = ab[IDX_C4HII] * (48.0 * rptr[IDX_ELEM_C] / 49.0 + 1.0 * rptr[IDX_ELEM_H] / 49.0);
    ab[IDX_CNCII] = ab[IDX_CNCII] * (14.0 * rptr[IDX_ELEM_N] / 38.0 + 24.0 * rptr[IDX_ELEM_C] / 38.0);
    ab[IDX_HSII] = ab[IDX_HSII] * (32.0 * rptr[IDX_ELEM_S] / 33.0 + 1.0 * rptr[IDX_ELEM_H] / 33.0);
    ab[IDX_NHII] = ab[IDX_NHII] * (14.0 * rptr[IDX_ELEM_N] / 15.0 + 1.0 * rptr[IDX_ELEM_H] / 15.0);
    ab[IDX_O2M] = ab[IDX_O2M] * (32.0 * rptr[IDX_ELEM_O] / 32.0);
    ab[IDX_CHM] = ab[IDX_CHM] * (12.0 * rptr[IDX_ELEM_C] / 13.0 + 1.0 * rptr[IDX_ELEM_H] / 13.0);
    ab[IDX_H2II] = ab[IDX_H2II] * (2.0 * rptr[IDX_ELEM_H] / 2.0);
    ab[IDX_OCSI] = ab[IDX_OCSI] * (32.0 * rptr[IDX_ELEM_S] / 60.0 + 16.0 * rptr[IDX_ELEM_O] / 60.0 + 12.0 * rptr[IDX_ELEM_C] / 60.0);
    ab[IDX_GCH3I] = ab[IDX_GCH3I] * (12.0 * rptr[IDX_ELEM_C] / 15.0 + 3.0 * rptr[IDX_ELEM_H] / 15.0);
    ab[IDX_NH2II] = ab[IDX_NH2II] * (14.0 * rptr[IDX_ELEM_N] / 16.0 + 2.0 * rptr[IDX_ELEM_H] / 16.0);
    ab[IDX_PI] = ab[IDX_PI] * (31.0 * rptr[IDX_ELEM_P] / 31.0);
    ab[IDX_SM] = ab[IDX_SM] * (32.0 * rptr[IDX_ELEM_S] / 32.0);
    ab[IDX_SiSII] = ab[IDX_SiSII] * (28.0 * rptr[IDX_ELEM_Si] / 60.0 + 32.0 * rptr[IDX_ELEM_S] / 60.0);
    ab[IDX_C2HM] = ab[IDX_C2HM] * (24.0 * rptr[IDX_ELEM_C] / 25.0 + 1.0 * rptr[IDX_ELEM_H] / 25.0);
    ab[IDX_C4HM] = ab[IDX_C4HM] * (48.0 * rptr[IDX_ELEM_C] / 49.0 + 1.0 * rptr[IDX_ELEM_H] / 49.0);
    ab[IDX_COII] = ab[IDX_COII] * (16.0 * rptr[IDX_ELEM_O] / 28.0 + 12.0 * rptr[IDX_ELEM_C] / 28.0);
    ab[IDX_CH3CCHI] = ab[IDX_CH3CCHI] * (36.0 * rptr[IDX_ELEM_C] / 40.0 + 4.0 * rptr[IDX_ELEM_H] / 40.0);
    ab[IDX_CH5II] = ab[IDX_CH5II] * (12.0 * rptr[IDX_ELEM_C] / 17.0 + 5.0 * rptr[IDX_ELEM_H] / 17.0);
    ab[IDX_HC2SII] = ab[IDX_HC2SII] * (32.0 * rptr[IDX_ELEM_S] / 57.0 + 24.0 * rptr[IDX_ELEM_C] / 57.0 + 1.0 * rptr[IDX_ELEM_H] / 57.0);
    ab[IDX_OHM] = ab[IDX_OHM] * (16.0 * rptr[IDX_ELEM_O] / 17.0 + 1.0 * rptr[IDX_ELEM_H] / 17.0);
    ab[IDX_C10HM] = ab[IDX_C10HM] * (120.0 * rptr[IDX_ELEM_C] / 121.0 + 1.0 * rptr[IDX_ELEM_H] / 121.0);
    ab[IDX_C3HM] = ab[IDX_C3HM] * (36.0 * rptr[IDX_ELEM_C] / 37.0 + 1.0 * rptr[IDX_ELEM_H] / 37.0);
    ab[IDX_C4SI] = ab[IDX_C4SI] * (32.0 * rptr[IDX_ELEM_S] / 80.0 + 48.0 * rptr[IDX_ELEM_C] / 80.0);
    ab[IDX_C6HM] = ab[IDX_C6HM] * (72.0 * rptr[IDX_ELEM_C] / 73.0 + 1.0 * rptr[IDX_ELEM_H] / 73.0);
    ab[IDX_C8HM] = ab[IDX_C8HM] * (96.0 * rptr[IDX_ELEM_C] / 97.0 + 1.0 * rptr[IDX_ELEM_H] / 97.0);
    ab[IDX_C7HM] = ab[IDX_C7HM] * (84.0 * rptr[IDX_ELEM_C] / 85.0 + 1.0 * rptr[IDX_ELEM_H] / 85.0);
    ab[IDX_C9HM] = ab[IDX_C9HM] * (108.0 * rptr[IDX_ELEM_C] / 109.0 + 1.0 * rptr[IDX_ELEM_H] / 109.0);
    ab[IDX_C10M] = ab[IDX_C10M] * (120.0 * rptr[IDX_ELEM_C] / 120.0);
    ab[IDX_C5HM] = ab[IDX_C5HM] * (60.0 * rptr[IDX_ELEM_C] / 61.0 + 1.0 * rptr[IDX_ELEM_H] / 61.0);
    ab[IDX_C5NM] = ab[IDX_C5NM] * (14.0 * rptr[IDX_ELEM_N] / 74.0 + 60.0 * rptr[IDX_ELEM_C] / 74.0);
    ab[IDX_HSI] = ab[IDX_HSI] * (32.0 * rptr[IDX_ELEM_S] / 33.0 + 1.0 * rptr[IDX_ELEM_H] / 33.0);
    ab[IDX_CM] = ab[IDX_CM] * (12.0 * rptr[IDX_ELEM_C] / 12.0);
    ab[IDX_CSI] = ab[IDX_CSI] * (32.0 * rptr[IDX_ELEM_S] / 44.0 + 12.0 * rptr[IDX_ELEM_C] / 44.0);
    ab[IDX_OM] = ab[IDX_OM] * (16.0 * rptr[IDX_ELEM_O] / 16.0);
    ab[IDX_C9M] = ab[IDX_C9M] * (108.0 * rptr[IDX_ELEM_C] / 108.0);
    ab[IDX_CH3OHI] = ab[IDX_CH3OHI] * (16.0 * rptr[IDX_ELEM_O] / 32.0 + 12.0 * rptr[IDX_ELEM_C] / 32.0 + 4.0 * rptr[IDX_ELEM_H] / 32.0);
    ab[IDX_HM] = ab[IDX_HM] * (1.0 * rptr[IDX_ELEM_H] / 1.0);
    ab[IDX_HC3NI] = ab[IDX_HC3NI] * (14.0 * rptr[IDX_ELEM_N] / 51.0 + 36.0 * rptr[IDX_ELEM_C] / 51.0 + 1.0 * rptr[IDX_ELEM_H] / 51.0);
    ab[IDX_SiSI] = ab[IDX_SiSI] * (28.0 * rptr[IDX_ELEM_Si] / 60.0 + 32.0 * rptr[IDX_ELEM_S] / 60.0);
    ab[IDX_C3HII] = ab[IDX_C3HII] * (36.0 * rptr[IDX_ELEM_C] / 37.0 + 1.0 * rptr[IDX_ELEM_H] / 37.0);
    ab[IDX_C3H3II] = ab[IDX_C3H3II] * (36.0 * rptr[IDX_ELEM_C] / 39.0 + 3.0 * rptr[IDX_ELEM_H] / 39.0);
    ab[IDX_C8M] = ab[IDX_C8M] * (96.0 * rptr[IDX_ELEM_C] / 96.0);
    ab[IDX_C2M] = ab[IDX_C2M] * (24.0 * rptr[IDX_ELEM_C] / 24.0);
    ab[IDX_C2SI] = ab[IDX_C2SI] * (32.0 * rptr[IDX_ELEM_S] / 56.0 + 24.0 * rptr[IDX_ELEM_C] / 56.0);
    ab[IDX_C3M] = ab[IDX_C3M] * (36.0 * rptr[IDX_ELEM_C] / 36.0);
    ab[IDX_C4M] = ab[IDX_C4M] * (48.0 * rptr[IDX_ELEM_C] / 48.0);
    ab[IDX_C6M] = ab[IDX_C6M] * (72.0 * rptr[IDX_ELEM_C] / 72.0);
    ab[IDX_H3COII] = ab[IDX_H3COII] * (16.0 * rptr[IDX_ELEM_O] / 31.0 + 12.0 * rptr[IDX_ELEM_C] / 31.0 + 3.0 * rptr[IDX_ELEM_H] / 31.0);
    ab[IDX_C5M] = ab[IDX_C5M] * (60.0 * rptr[IDX_ELEM_C] / 60.0);
    ab[IDX_C7M] = ab[IDX_C7M] * (84.0 * rptr[IDX_ELEM_C] / 84.0);
    ab[IDX_SiOII] = ab[IDX_SiOII] * (28.0 * rptr[IDX_ELEM_Si] / 44.0 + 16.0 * rptr[IDX_ELEM_O] / 44.0);
    ab[IDX_C2NI] = ab[IDX_C2NI] * (14.0 * rptr[IDX_ELEM_N] / 38.0 + 24.0 * rptr[IDX_ELEM_C] / 38.0);
    ab[IDX_GCHI] = ab[IDX_GCHI] * (12.0 * rptr[IDX_ELEM_C] / 13.0 + 1.0 * rptr[IDX_ELEM_H] / 13.0);
    ab[IDX_SiOHII] = ab[IDX_SiOHII] * (28.0 * rptr[IDX_ELEM_Si] / 45.0 + 16.0 * rptr[IDX_ELEM_O] / 45.0 + 1.0 * rptr[IDX_ELEM_H] / 45.0);
    ab[IDX_NH2I] = ab[IDX_NH2I] * (14.0 * rptr[IDX_ELEM_N] / 16.0 + 2.0 * rptr[IDX_ELEM_H] / 16.0);
    ab[IDX_C4H3I] = ab[IDX_C4H3I] * (48.0 * rptr[IDX_ELEM_C] / 51.0 + 3.0 * rptr[IDX_ELEM_H] / 51.0);
    ab[IDX_C2HII] = ab[IDX_C2HII] * (24.0 * rptr[IDX_ELEM_C] / 25.0 + 1.0 * rptr[IDX_ELEM_H] / 25.0);
    ab[IDX_C3H2II] = ab[IDX_C3H2II] * (36.0 * rptr[IDX_ELEM_C] / 38.0 + 2.0 * rptr[IDX_ELEM_H] / 38.0);
    ab[IDX_C5NI] = ab[IDX_C5NI] * (14.0 * rptr[IDX_ELEM_N] / 74.0 + 60.0 * rptr[IDX_ELEM_C] / 74.0);
    ab[IDX_CH2II] = ab[IDX_CH2II] * (12.0 * rptr[IDX_ELEM_C] / 14.0 + 2.0 * rptr[IDX_ELEM_H] / 14.0);
    ab[IDX_OHII] = ab[IDX_OHII] * (16.0 * rptr[IDX_ELEM_O] / 17.0 + 1.0 * rptr[IDX_ELEM_H] / 17.0);
    ab[IDX_H2OII] = ab[IDX_H2OII] * (16.0 * rptr[IDX_ELEM_O] / 18.0 + 2.0 * rptr[IDX_ELEM_H] / 18.0);
    ab[IDX_C3NM] = ab[IDX_C3NM] * (14.0 * rptr[IDX_ELEM_N] / 50.0 + 36.0 * rptr[IDX_ELEM_C] / 50.0);
    ab[IDX_GH2I] = ab[IDX_GH2I] * (2.0 * rptr[IDX_ELEM_H] / 2.0);
    ab[IDX_C3NI] = ab[IDX_C3NI] * (14.0 * rptr[IDX_ELEM_N] / 50.0 + 36.0 * rptr[IDX_ELEM_C] / 50.0);
    ab[IDX_CHII] = ab[IDX_CHII] * (12.0 * rptr[IDX_ELEM_C] / 13.0 + 1.0 * rptr[IDX_ELEM_H] / 13.0);
    ab[IDX_CO2I] = ab[IDX_CO2I] * (32.0 * rptr[IDX_ELEM_O] / 44.0 + 12.0 * rptr[IDX_ELEM_C] / 44.0);
    ab[IDX_O2II] = ab[IDX_O2II] * (32.0 * rptr[IDX_ELEM_O] / 32.0);
    ab[IDX_C10HI] = ab[IDX_C10HI] * (120.0 * rptr[IDX_ELEM_C] / 121.0 + 1.0 * rptr[IDX_ELEM_H] / 121.0);
    ab[IDX_C9HI] = ab[IDX_C9HI] * (108.0 * rptr[IDX_ELEM_C] / 109.0 + 1.0 * rptr[IDX_ELEM_H] / 109.0);
    ab[IDX_SiOI] = ab[IDX_SiOI] * (28.0 * rptr[IDX_ELEM_Si] / 44.0 + 16.0 * rptr[IDX_ELEM_O] / 44.0);
    ab[IDX_FeII] = ab[IDX_FeII] * (56.0 * rptr[IDX_ELEM_Fe] / 56.0);
    ab[IDX_FeI] = ab[IDX_FeI] * (56.0 * rptr[IDX_ELEM_Fe] / 56.0);
    ab[IDX_MgII] = ab[IDX_MgII] * (24.0 * rptr[IDX_ELEM_Mg] / 24.0);
    ab[IDX_NaII] = ab[IDX_NaII] * (23.0 * rptr[IDX_ELEM_Na] / 23.0);
    ab[IDX_MgI] = ab[IDX_MgI] * (24.0 * rptr[IDX_ELEM_Mg] / 24.0);
    ab[IDX_NaI] = ab[IDX_NaI] * (23.0 * rptr[IDX_ELEM_Na] / 23.0);
    ab[IDX_C10I] = ab[IDX_C10I] * (120.0 * rptr[IDX_ELEM_C] / 120.0);
    ab[IDX_C8HI] = ab[IDX_C8HI] * (96.0 * rptr[IDX_ELEM_C] / 97.0 + 1.0 * rptr[IDX_ELEM_H] / 97.0);
    ab[IDX_C7I] = ab[IDX_C7I] * (84.0 * rptr[IDX_ELEM_C] / 84.0);
    ab[IDX_C9I] = ab[IDX_C9I] * (108.0 * rptr[IDX_ELEM_C] / 108.0);
    ab[IDX_C8I] = ab[IDX_C8I] * (96.0 * rptr[IDX_ELEM_C] / 96.0);
    ab[IDX_GCI] = ab[IDX_GCI] * (12.0 * rptr[IDX_ELEM_C] / 12.0);
    ab[IDX_NHI] = ab[IDX_NHI] * (14.0 * rptr[IDX_ELEM_N] / 15.0 + 1.0 * rptr[IDX_ELEM_H] / 15.0);
    ab[IDX_C5I] = ab[IDX_C5I] * (60.0 * rptr[IDX_ELEM_C] / 60.0);
    ab[IDX_C6I] = ab[IDX_C6I] * (72.0 * rptr[IDX_ELEM_C] / 72.0);
    ab[IDX_CNM] = ab[IDX_CNM] * (14.0 * rptr[IDX_ELEM_N] / 26.0 + 12.0 * rptr[IDX_ELEM_C] / 26.0);
    ab[IDX_C4I] = ab[IDX_C4I] * (48.0 * rptr[IDX_ELEM_C] / 48.0);
    ab[IDX_SOI] = ab[IDX_SOI] * (32.0 * rptr[IDX_ELEM_S] / 48.0 + 16.0 * rptr[IDX_ELEM_O] / 48.0);
    ab[IDX_C7HI] = ab[IDX_C7HI] * (84.0 * rptr[IDX_ELEM_C] / 85.0 + 1.0 * rptr[IDX_ELEM_H] / 85.0);
    ab[IDX_C2H4II] = ab[IDX_C2H4II] * (24.0 * rptr[IDX_ELEM_C] / 28.0 + 4.0 * rptr[IDX_ELEM_H] / 28.0);
    ab[IDX_C4H3II] = ab[IDX_C4H3II] * (48.0 * rptr[IDX_ELEM_C] / 51.0 + 3.0 * rptr[IDX_ELEM_H] / 51.0);
    ab[IDX_N2HII] = ab[IDX_N2HII] * (28.0 * rptr[IDX_ELEM_N] / 29.0 + 1.0 * rptr[IDX_ELEM_H] / 29.0);
    ab[IDX_CH2CCHI] = ab[IDX_CH2CCHI] * (36.0 * rptr[IDX_ELEM_C] / 39.0 + 3.0 * rptr[IDX_ELEM_H] / 39.0);
    ab[IDX_SOII] = ab[IDX_SOII] * (32.0 * rptr[IDX_ELEM_S] / 48.0 + 16.0 * rptr[IDX_ELEM_O] / 48.0);
    ab[IDX_CH2CCHII] = ab[IDX_CH2CCHII] * (36.0 * rptr[IDX_ELEM_C] / 39.0 + 3.0 * rptr[IDX_ELEM_H] / 39.0);
    ab[IDX_C6HI] = ab[IDX_C6HI] * (72.0 * rptr[IDX_ELEM_C] / 73.0 + 1.0 * rptr[IDX_ELEM_H] / 73.0);
    ab[IDX_C5HI] = ab[IDX_C5HI] * (60.0 * rptr[IDX_ELEM_C] / 61.0 + 1.0 * rptr[IDX_ELEM_H] / 61.0);
    ab[IDX_H2SII] = ab[IDX_H2SII] * (32.0 * rptr[IDX_ELEM_S] / 34.0 + 2.0 * rptr[IDX_ELEM_H] / 34.0);
    ab[IDX_NII] = ab[IDX_NII] * (14.0 * rptr[IDX_ELEM_N] / 14.0);
    ab[IDX_C3HI] = ab[IDX_C3HI] * (36.0 * rptr[IDX_ELEM_C] / 37.0 + 1.0 * rptr[IDX_ELEM_H] / 37.0);
    ab[IDX_H2COII] = ab[IDX_H2COII] * (16.0 * rptr[IDX_ELEM_O] / 30.0 + 12.0 * rptr[IDX_ELEM_C] / 30.0 + 2.0 * rptr[IDX_ELEM_H] / 30.0);
    ab[IDX_C3I] = ab[IDX_C3I] * (36.0 * rptr[IDX_ELEM_C] / 36.0);
    ab[IDX_C4HI] = ab[IDX_C4HI] * (48.0 * rptr[IDX_ELEM_C] / 49.0 + 1.0 * rptr[IDX_ELEM_H] / 49.0);
    ab[IDX_NOII] = ab[IDX_NOII] * (14.0 * rptr[IDX_ELEM_N] / 30.0 + 16.0 * rptr[IDX_ELEM_O] / 30.0);
    ab[IDX_OII] = ab[IDX_OII] * (16.0 * rptr[IDX_ELEM_O] / 16.0);
    ab[IDX_C4H2I] = ab[IDX_C4H2I] * (48.0 * rptr[IDX_ELEM_C] / 50.0 + 2.0 * rptr[IDX_ELEM_H] / 50.0);
    ab[IDX_C4H2II] = ab[IDX_C4H2II] * (48.0 * rptr[IDX_ELEM_C] / 50.0 + 2.0 * rptr[IDX_ELEM_H] / 50.0);
    ab[IDX_HCNHII] = ab[IDX_HCNHII] * (14.0 * rptr[IDX_ELEM_N] / 28.0 + 12.0 * rptr[IDX_ELEM_C] / 28.0 + 2.0 * rptr[IDX_ELEM_H] / 28.0);
    ab[IDX_CH2I] = ab[IDX_CH2I] * (12.0 * rptr[IDX_ELEM_C] / 14.0 + 2.0 * rptr[IDX_ELEM_H] / 14.0);
    ab[IDX_HNCI] = ab[IDX_HNCI] * (14.0 * rptr[IDX_ELEM_N] / 27.0 + 12.0 * rptr[IDX_ELEM_C] / 27.0 + 1.0 * rptr[IDX_ELEM_H] / 27.0);
    ab[IDX_NH3II] = ab[IDX_NH3II] * (14.0 * rptr[IDX_ELEM_N] / 17.0 + 3.0 * rptr[IDX_ELEM_H] / 17.0);
    ab[IDX_C2H4I] = ab[IDX_C2H4I] * (24.0 * rptr[IDX_ELEM_C] / 28.0 + 4.0 * rptr[IDX_ELEM_H] / 28.0);
    ab[IDX_C2H3I] = ab[IDX_C2H3I] * (24.0 * rptr[IDX_ELEM_C] / 27.0 + 3.0 * rptr[IDX_ELEM_H] / 27.0);
    ab[IDX_NH4II] = ab[IDX_NH4II] * (14.0 * rptr[IDX_ELEM_N] / 18.0 + 4.0 * rptr[IDX_ELEM_H] / 18.0);
    ab[IDX_N2I] = ab[IDX_N2I] * (28.0 * rptr[IDX_ELEM_N] / 28.0);
    ab[IDX_O2I] = ab[IDX_O2I] * (32.0 * rptr[IDX_ELEM_O] / 32.0);
    ab[IDX_SiII] = ab[IDX_SiII] * (28.0 * rptr[IDX_ELEM_Si] / 28.0);
    ab[IDX_H2SI] = ab[IDX_H2SI] * (32.0 * rptr[IDX_ELEM_S] / 34.0 + 2.0 * rptr[IDX_ELEM_H] / 34.0);
    ab[IDX_SII] = ab[IDX_SII] * (32.0 * rptr[IDX_ELEM_S] / 32.0);
    ab[IDX_SiI] = ab[IDX_SiI] * (28.0 * rptr[IDX_ELEM_Si] / 28.0);
    ab[IDX_C2H3II] = ab[IDX_C2H3II] * (24.0 * rptr[IDX_ELEM_C] / 27.0 + 3.0 * rptr[IDX_ELEM_H] / 27.0);
    ab[IDX_NOI] = ab[IDX_NOI] * (14.0 * rptr[IDX_ELEM_N] / 30.0 + 16.0 * rptr[IDX_ELEM_O] / 30.0);
    ab[IDX_HCOI] = ab[IDX_HCOI] * (16.0 * rptr[IDX_ELEM_O] / 29.0 + 12.0 * rptr[IDX_ELEM_C] / 29.0 + 1.0 * rptr[IDX_ELEM_H] / 29.0);
    ab[IDX_CH4I] = ab[IDX_CH4I] * (12.0 * rptr[IDX_ELEM_C] / 16.0 + 4.0 * rptr[IDX_ELEM_H] / 16.0);
    ab[IDX_C2HI] = ab[IDX_C2HI] * (24.0 * rptr[IDX_ELEM_C] / 25.0 + 1.0 * rptr[IDX_ELEM_H] / 25.0);
    ab[IDX_H2COI] = ab[IDX_H2COI] * (16.0 * rptr[IDX_ELEM_O] / 30.0 + 12.0 * rptr[IDX_ELEM_C] / 30.0 + 2.0 * rptr[IDX_ELEM_H] / 30.0);
    ab[IDX_HCNI] = ab[IDX_HCNI] * (14.0 * rptr[IDX_ELEM_N] / 27.0 + 12.0 * rptr[IDX_ELEM_C] / 27.0 + 1.0 * rptr[IDX_ELEM_H] / 27.0);
    ab[IDX_CHI] = ab[IDX_CHI] * (12.0 * rptr[IDX_ELEM_C] / 13.0 + 1.0 * rptr[IDX_ELEM_H] / 13.0);
    ab[IDX_SI] = ab[IDX_SI] * (32.0 * rptr[IDX_ELEM_S] / 32.0);
    ab[IDX_C2I] = ab[IDX_C2I] * (24.0 * rptr[IDX_ELEM_C] / 24.0);
    ab[IDX_OHI] = ab[IDX_OHI] * (16.0 * rptr[IDX_ELEM_O] / 17.0 + 1.0 * rptr[IDX_ELEM_H] / 17.0);
    ab[IDX_NH3I] = ab[IDX_NH3I] * (14.0 * rptr[IDX_ELEM_N] / 17.0 + 3.0 * rptr[IDX_ELEM_H] / 17.0);
    ab[IDX_C2H2II] = ab[IDX_C2H2II] * (24.0 * rptr[IDX_ELEM_C] / 26.0 + 2.0 * rptr[IDX_ELEM_H] / 26.0);
    ab[IDX_CH3II] = ab[IDX_CH3II] * (12.0 * rptr[IDX_ELEM_C] / 15.0 + 3.0 * rptr[IDX_ELEM_H] / 15.0);
    ab[IDX_C2H2I] = ab[IDX_C2H2I] * (24.0 * rptr[IDX_ELEM_C] / 26.0 + 2.0 * rptr[IDX_ELEM_H] / 26.0);
    ab[IDX_CNI] = ab[IDX_CNI] * (14.0 * rptr[IDX_ELEM_N] / 26.0 + 12.0 * rptr[IDX_ELEM_C] / 26.0);
    ab[IDX_GHI] = ab[IDX_GHI] * (1.0 * rptr[IDX_ELEM_H] / 1.0);
    ab[IDX_CH3I] = ab[IDX_CH3I] * (12.0 * rptr[IDX_ELEM_C] / 15.0 + 3.0 * rptr[IDX_ELEM_H] / 15.0);
    ab[IDX_NI] = ab[IDX_NI] * (14.0 * rptr[IDX_ELEM_N] / 14.0);
    ab[IDX_H3OII] = ab[IDX_H3OII] * (16.0 * rptr[IDX_ELEM_O] / 19.0 + 3.0 * rptr[IDX_ELEM_H] / 19.0);
    ab[IDX_OI] = ab[IDX_OI] * (16.0 * rptr[IDX_ELEM_O] / 16.0);
    ab[IDX_HeII] = ab[IDX_HeII] * (4.0 * rptr[IDX_ELEM_He] / 4.0);
    ab[IDX_HeI] = ab[IDX_HeI] * (4.0 * rptr[IDX_ELEM_He] / 4.0);
    ab[IDX_CII] = ab[IDX_CII] * (12.0 * rptr[IDX_ELEM_C] / 12.0);
    ab[IDX_H2OI] = ab[IDX_H2OI] * (16.0 * rptr[IDX_ELEM_O] / 18.0 + 2.0 * rptr[IDX_ELEM_H] / 18.0);
    ab[IDX_HII] = ab[IDX_HII] * (1.0 * rptr[IDX_ELEM_H] / 1.0);
    ab[IDX_CI] = ab[IDX_CI] * (12.0 * rptr[IDX_ELEM_C] / 12.0);
    ab[IDX_HCOII] = ab[IDX_HCOII] * (16.0 * rptr[IDX_ELEM_O] / 29.0 + 12.0 * rptr[IDX_ELEM_C] / 29.0 + 1.0 * rptr[IDX_ELEM_H] / 29.0);
    ab[IDX_H3II] = ab[IDX_H3II] * (3.0 * rptr[IDX_ELEM_H] / 3.0);
    ab[IDX_COI] = ab[IDX_COI] * (16.0 * rptr[IDX_ELEM_O] / 28.0 + 12.0 * rptr[IDX_ELEM_C] / 28.0);
    ab[IDX_GRAINM] = ab[IDX_GRAINM] * (0.0 * rptr[IDX_ELEM_GRAIN] / 0.0);
    ab[IDX_GRAIN0I] = ab[IDX_GRAIN0I] * (0.0 * rptr[IDX_ELEM_GRAIN] / 0.0);
    ab[IDX_H2I] = ab[IDX_H2I] * (2.0 * rptr[IDX_ELEM_H] / 2.0);
    ab[IDX_eM] = ab[IDX_eM] * (1.0);
    ab[IDX_HI] = ab[IDX_HI] * (1.0 * rptr[IDX_ELEM_H] / 1.0);
        // clang-format on

    return NAUNET_SUCCESS;
}