//===-- X86IntinsicsInfo.h - X86 Instrinsics ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the details for lowering X86 intrinsics
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_X86_X86INTRINSICSINFO_H
#define LLVM_LIB_TARGET_X86_X86INTRINSICSINFO_H

using namespace llvm;

enum IntrinsicType {
  GATHER, SCATTER, PREFETCH, RDSEED, RDRAND, RDPMC, RDTSC, XTEST, ADX,
  INTR_TYPE_1OP, INTR_TYPE_2OP, INTR_TYPE_3OP, VSHIFT,
  COMI
};

struct IntrinsicData {
  IntrinsicData(IntrinsicType IType, unsigned IOpc0, unsigned IOpc1)
    :Type(IType), Opc0(IOpc0), Opc1(IOpc1) {}
  IntrinsicType Type;
  unsigned      Opc0;
  unsigned      Opc1;
};

#define INTRINSIC_WITH_CHAIN(id, type, op0, op1) \
 IntrWithChainMap.insert(std::make_pair(Intrinsic::id, \
   IntrinsicData(type, op0, op1)))

std::map < unsigned, IntrinsicData> IntrWithChainMap;
void InitIntrinsicsWithChain() {
  INTRINSIC_WITH_CHAIN(x86_avx512_gather_qps_512, GATHER, X86::VGATHERQPSZrm, 0);
  INTRINSIC_WITH_CHAIN(x86_avx512_gather_qpd_512, GATHER, X86::VGATHERQPDZrm, 0);
  INTRINSIC_WITH_CHAIN(x86_avx512_gather_dps_512, GATHER, X86::VGATHERDPSZrm, 0);
  INTRINSIC_WITH_CHAIN(x86_avx512_gather_dpd_512, GATHER, X86::VGATHERDPDZrm, 0);

  INTRINSIC_WITH_CHAIN(x86_avx512_gather_qpi_512, GATHER, X86::VPGATHERQDZrm, 0);
  INTRINSIC_WITH_CHAIN(x86_avx512_gather_qpq_512, GATHER, X86::VPGATHERQQZrm, 0);
  INTRINSIC_WITH_CHAIN(x86_avx512_gather_dpi_512, GATHER, X86::VPGATHERDDZrm, 0);
  INTRINSIC_WITH_CHAIN(x86_avx512_gather_dpq_512, GATHER, X86::VPGATHERDQZrm, 0);

  INTRINSIC_WITH_CHAIN(x86_avx512_scatter_qps_512, SCATTER, X86::VSCATTERQPSZmr, 0);
  INTRINSIC_WITH_CHAIN(x86_avx512_scatter_qpd_512, SCATTER, X86::VSCATTERQPDZmr, 0);
  INTRINSIC_WITH_CHAIN(x86_avx512_scatter_dps_512, SCATTER, X86::VSCATTERDPSZmr, 0);
  INTRINSIC_WITH_CHAIN(x86_avx512_scatter_dpd_512, SCATTER, X86::VSCATTERDPDZmr, 0);

  INTRINSIC_WITH_CHAIN(x86_avx512_scatter_qpi_512, SCATTER, X86::VPSCATTERQDZmr, 0);
  INTRINSIC_WITH_CHAIN(x86_avx512_scatter_qpq_512, SCATTER, X86::VPSCATTERQQZmr, 0);
  INTRINSIC_WITH_CHAIN(x86_avx512_scatter_dpi_512, SCATTER, X86::VPSCATTERDDZmr, 0);
  INTRINSIC_WITH_CHAIN(x86_avx512_scatter_dpq_512, SCATTER, X86::VPSCATTERDQZmr, 0);

  INTRINSIC_WITH_CHAIN(x86_avx512_gatherpf_qps_512, PREFETCH, X86::VGATHERPF0QPSm, X86::VGATHERPF1QPSm);
  INTRINSIC_WITH_CHAIN(x86_avx512_gatherpf_qpd_512, PREFETCH, X86::VGATHERPF0QPDm, X86::VGATHERPF1QPDm);
  INTRINSIC_WITH_CHAIN(x86_avx512_gatherpf_dps_512, PREFETCH, X86::VGATHERPF0DPSm, X86::VGATHERPF1DPSm);
  INTRINSIC_WITH_CHAIN(x86_avx512_gatherpf_dpd_512, PREFETCH, X86::VGATHERPF0DPDm, X86::VGATHERPF1DPDm);

  INTRINSIC_WITH_CHAIN(x86_avx512_scatterpf_qps_512, PREFETCH, X86::VSCATTERPF0QPSm, X86::VSCATTERPF1QPSm);
  INTRINSIC_WITH_CHAIN(x86_avx512_scatterpf_qpd_512, PREFETCH, X86::VSCATTERPF0QPDm, X86::VSCATTERPF1QPDm);
  INTRINSIC_WITH_CHAIN(x86_avx512_scatterpf_dps_512, PREFETCH, X86::VSCATTERPF0DPSm, X86::VSCATTERPF1DPSm);
  INTRINSIC_WITH_CHAIN(x86_avx512_scatterpf_dpd_512, PREFETCH, X86::VSCATTERPF0DPDm, X86::VSCATTERPF1DPDm);

  INTRINSIC_WITH_CHAIN(x86_rdrand_16, RDRAND, X86ISD::RDRAND, 0);
  INTRINSIC_WITH_CHAIN(x86_rdrand_32, RDRAND, X86ISD::RDRAND, 0);
  INTRINSIC_WITH_CHAIN(x86_rdrand_64, RDRAND, X86ISD::RDRAND, 0);
  
  INTRINSIC_WITH_CHAIN(x86_rdseed_16, RDSEED, X86ISD::RDSEED, 0);
  INTRINSIC_WITH_CHAIN(x86_rdseed_32, RDSEED, X86ISD::RDSEED, 0);
  INTRINSIC_WITH_CHAIN(x86_rdseed_64, RDSEED, X86ISD::RDSEED, 0);
  INTRINSIC_WITH_CHAIN(x86_xtest,     XTEST,  X86ISD::XTEST,  0);
  INTRINSIC_WITH_CHAIN(x86_rdtsc,     RDTSC,  X86ISD::RDTSC_DAG, 0);
  INTRINSIC_WITH_CHAIN(x86_rdtscp,    RDTSC,  X86ISD::RDTSCP_DAG, 0);
  INTRINSIC_WITH_CHAIN(x86_rdpmc,     RDPMC,  X86ISD::RDPMC_DAG, 0);

  INTRINSIC_WITH_CHAIN(x86_addcarryx_u32, ADX, X86ISD::ADC, 0);
  INTRINSIC_WITH_CHAIN(x86_addcarryx_u64, ADX, X86ISD::ADC, 0);
  INTRINSIC_WITH_CHAIN(x86_addcarry_u32,  ADX, X86ISD::ADC, 0);
  INTRINSIC_WITH_CHAIN(x86_addcarry_u64,  ADX, X86ISD::ADC, 0);
  INTRINSIC_WITH_CHAIN(x86_subborrow_u32, ADX, X86ISD::SBB, 0);
  INTRINSIC_WITH_CHAIN(x86_subborrow_u64, ADX, X86ISD::SBB, 0);

}

const IntrinsicData* GetIntrinsicWithChain(unsigned IntNo) {
  std::map < unsigned, IntrinsicData>::const_iterator itr =
    IntrWithChainMap.find(IntNo);
  if (itr == IntrWithChainMap.end())
    return NULL;
  return &(itr->second);
}

#define INTRINSIC_WO_CHAIN(id, type, op0, op1) \
 IntrWithoutChainMap.insert(std::make_pair(Intrinsic::id, \
   IntrinsicData(type, op0, op1)))


std::map < unsigned, IntrinsicData> IntrWithoutChainMap;

void InitIntrinsicsWithoutChain() {
  INTRINSIC_WO_CHAIN(x86_sse_sqrt_ps,     INTR_TYPE_1OP, ISD::FSQRT, 0);
  INTRINSIC_WO_CHAIN(x86_sse2_sqrt_pd,    INTR_TYPE_1OP, ISD::FSQRT, 0);
  INTRINSIC_WO_CHAIN(x86_avx_sqrt_ps_256, INTR_TYPE_1OP, ISD::FSQRT, 0);
  INTRINSIC_WO_CHAIN(x86_avx_sqrt_pd_256, INTR_TYPE_1OP, ISD::FSQRT, 0);

  INTRINSIC_WO_CHAIN(x86_sse2_psubus_b,     INTR_TYPE_2OP, X86ISD::SUBUS, 0);
  INTRINSIC_WO_CHAIN(x86_sse2_psubus_w,     INTR_TYPE_2OP, X86ISD::SUBUS, 0);
  INTRINSIC_WO_CHAIN(x86_avx2_psubus_b,     INTR_TYPE_2OP, X86ISD::SUBUS, 0);
  INTRINSIC_WO_CHAIN(x86_avx2_psubus_w,     INTR_TYPE_2OP, X86ISD::SUBUS, 0);

  INTRINSIC_WO_CHAIN(x86_sse3_hadd_ps,      INTR_TYPE_2OP, X86ISD::FHADD, 0);
  INTRINSIC_WO_CHAIN(x86_sse3_hadd_pd,      INTR_TYPE_2OP, X86ISD::FHADD, 0);
  INTRINSIC_WO_CHAIN(x86_avx_hadd_ps_256,   INTR_TYPE_2OP, X86ISD::FHADD, 0);
  INTRINSIC_WO_CHAIN(x86_avx_hadd_pd_256,   INTR_TYPE_2OP, X86ISD::FHADD, 0);
  INTRINSIC_WO_CHAIN(x86_sse3_hsub_ps,      INTR_TYPE_2OP, X86ISD::FHSUB, 0);
  INTRINSIC_WO_CHAIN(x86_sse3_hsub_pd,      INTR_TYPE_2OP, X86ISD::FHSUB, 0);
  INTRINSIC_WO_CHAIN(x86_avx_hsub_ps_256,   INTR_TYPE_2OP, X86ISD::FHSUB, 0);
  INTRINSIC_WO_CHAIN(x86_avx_hsub_pd_256,   INTR_TYPE_2OP, X86ISD::FHSUB, 0);
  INTRINSIC_WO_CHAIN(x86_ssse3_phadd_w_128, INTR_TYPE_2OP, X86ISD::HADD, 0);
  INTRINSIC_WO_CHAIN(x86_ssse3_phadd_d_128, INTR_TYPE_2OP, X86ISD::HADD, 0);
  INTRINSIC_WO_CHAIN(x86_avx2_phadd_w,      INTR_TYPE_2OP, X86ISD::HADD, 0);
  INTRINSIC_WO_CHAIN(x86_avx2_phadd_d,      INTR_TYPE_2OP, X86ISD::HADD, 0);
  INTRINSIC_WO_CHAIN(x86_ssse3_phsub_w_128, INTR_TYPE_2OP, X86ISD::HSUB, 0);
  INTRINSIC_WO_CHAIN(x86_ssse3_phsub_d_128, INTR_TYPE_2OP, X86ISD::HSUB, 0);
  INTRINSIC_WO_CHAIN(x86_avx2_phsub_w,      INTR_TYPE_2OP, X86ISD::HSUB, 0);
  INTRINSIC_WO_CHAIN(x86_avx2_phsub_d,      INTR_TYPE_2OP, X86ISD::HSUB, 0);

  INTRINSIC_WO_CHAIN(x86_sse2_pmaxu_b,      INTR_TYPE_2OP, X86ISD::UMAX, 0);
  INTRINSIC_WO_CHAIN(x86_sse41_pmaxuw,      INTR_TYPE_2OP, X86ISD::UMAX, 0);
  INTRINSIC_WO_CHAIN(x86_sse41_pmaxud,      INTR_TYPE_2OP, X86ISD::UMAX, 0);
  INTRINSIC_WO_CHAIN(x86_avx2_pmaxu_b,      INTR_TYPE_2OP, X86ISD::UMAX, 0);
  INTRINSIC_WO_CHAIN(x86_avx2_pmaxu_w,      INTR_TYPE_2OP, X86ISD::UMAX, 0);
  INTRINSIC_WO_CHAIN(x86_avx2_pmaxu_d,      INTR_TYPE_2OP, X86ISD::UMAX, 0);
  INTRINSIC_WO_CHAIN(x86_sse2_pminu_b,      INTR_TYPE_2OP, X86ISD::UMIN, 0);
  INTRINSIC_WO_CHAIN(x86_sse41_pminuw,      INTR_TYPE_2OP, X86ISD::UMIN, 0);
  INTRINSIC_WO_CHAIN(x86_sse41_pminud,      INTR_TYPE_2OP, X86ISD::UMIN, 0);
  INTRINSIC_WO_CHAIN(x86_avx2_pminu_b,      INTR_TYPE_2OP, X86ISD::UMIN, 0);
  INTRINSIC_WO_CHAIN(x86_avx2_pminu_w,      INTR_TYPE_2OP, X86ISD::UMIN, 0);
  INTRINSIC_WO_CHAIN(x86_avx2_pminu_d,      INTR_TYPE_2OP, X86ISD::UMIN, 0);
  INTRINSIC_WO_CHAIN(x86_sse41_pmaxsb,      INTR_TYPE_2OP, X86ISD::SMAX, 0);
  INTRINSIC_WO_CHAIN(x86_sse2_pmaxs_w,      INTR_TYPE_2OP, X86ISD::SMAX, 0);
  INTRINSIC_WO_CHAIN(x86_sse41_pmaxsd,      INTR_TYPE_2OP, X86ISD::SMAX, 0);
  INTRINSIC_WO_CHAIN(x86_avx2_pmaxs_b,      INTR_TYPE_2OP, X86ISD::SMAX, 0);
  INTRINSIC_WO_CHAIN(x86_avx2_pmaxs_w,      INTR_TYPE_2OP, X86ISD::SMAX, 0);
  INTRINSIC_WO_CHAIN(x86_avx2_pmaxs_d,      INTR_TYPE_2OP, X86ISD::SMAX, 0);
  INTRINSIC_WO_CHAIN(x86_sse41_pminsb,      INTR_TYPE_2OP, X86ISD::SMIN, 0);
  INTRINSIC_WO_CHAIN(x86_sse2_pmins_w,      INTR_TYPE_2OP, X86ISD::SMIN, 0);
  INTRINSIC_WO_CHAIN(x86_sse41_pminsd,      INTR_TYPE_2OP, X86ISD::SMIN, 0);
  INTRINSIC_WO_CHAIN(x86_avx2_pmins_b,      INTR_TYPE_2OP, X86ISD::SMIN, 0);
  INTRINSIC_WO_CHAIN(x86_avx2_pmins_w,      INTR_TYPE_2OP, X86ISD::SMIN, 0);
  INTRINSIC_WO_CHAIN(x86_avx2_pmins_d,      INTR_TYPE_2OP, X86ISD::SMIN, 0);

  INTRINSIC_WO_CHAIN(x86_sse2_psll_w,      INTR_TYPE_2OP, X86ISD::VSHL, 0);
  INTRINSIC_WO_CHAIN(x86_sse2_psll_d,      INTR_TYPE_2OP, X86ISD::VSHL, 0);
  INTRINSIC_WO_CHAIN(x86_sse2_psll_q,      INTR_TYPE_2OP, X86ISD::VSHL, 0);
  INTRINSIC_WO_CHAIN(x86_avx2_psll_w,      INTR_TYPE_2OP, X86ISD::VSHL, 0);
  INTRINSIC_WO_CHAIN(x86_avx2_psll_d,      INTR_TYPE_2OP, X86ISD::VSHL, 0);
  INTRINSIC_WO_CHAIN(x86_avx2_psll_q,      INTR_TYPE_2OP, X86ISD::VSHL, 0);
  INTRINSIC_WO_CHAIN(x86_sse2_psrl_w,      INTR_TYPE_2OP, X86ISD::VSRL, 0);
  INTRINSIC_WO_CHAIN(x86_sse2_psrl_d,      INTR_TYPE_2OP, X86ISD::VSRL, 0);
  INTRINSIC_WO_CHAIN(x86_sse2_psrl_q,      INTR_TYPE_2OP, X86ISD::VSRL, 0);
  INTRINSIC_WO_CHAIN(x86_avx2_psrl_w,      INTR_TYPE_2OP, X86ISD::VSRL, 0);
  INTRINSIC_WO_CHAIN(x86_avx2_psrl_d,      INTR_TYPE_2OP, X86ISD::VSRL, 0);
  INTRINSIC_WO_CHAIN(x86_avx2_psrl_q,      INTR_TYPE_2OP, X86ISD::VSRL, 0);
  INTRINSIC_WO_CHAIN(x86_sse2_psra_w,      INTR_TYPE_2OP, X86ISD::VSRA, 0);
  INTRINSIC_WO_CHAIN(x86_sse2_psra_d,      INTR_TYPE_2OP, X86ISD::VSRA, 0);
  INTRINSIC_WO_CHAIN(x86_avx2_psra_w,      INTR_TYPE_2OP, X86ISD::VSRA, 0);
  INTRINSIC_WO_CHAIN(x86_avx2_psra_d,      INTR_TYPE_2OP, X86ISD::VSRA, 0);

  INTRINSIC_WO_CHAIN(x86_sse2_pslli_w,     VSHIFT, X86ISD::VSHLI, 0);
  INTRINSIC_WO_CHAIN(x86_sse2_pslli_d,     VSHIFT, X86ISD::VSHLI, 0);
  INTRINSIC_WO_CHAIN(x86_sse2_pslli_q,     VSHIFT, X86ISD::VSHLI, 0);
  INTRINSIC_WO_CHAIN(x86_avx2_pslli_w,     VSHIFT, X86ISD::VSHLI, 0);
  INTRINSIC_WO_CHAIN(x86_avx2_pslli_d,     VSHIFT, X86ISD::VSHLI, 0);
  INTRINSIC_WO_CHAIN(x86_avx2_pslli_q,     VSHIFT, X86ISD::VSHLI, 0);
  INTRINSIC_WO_CHAIN(x86_sse2_psrli_w,     VSHIFT, X86ISD::VSRLI, 0);
  INTRINSIC_WO_CHAIN(x86_sse2_psrli_d,     VSHIFT, X86ISD::VSRLI, 0);
  INTRINSIC_WO_CHAIN(x86_sse2_psrli_q,     VSHIFT, X86ISD::VSRLI, 0);
  INTRINSIC_WO_CHAIN(x86_avx2_psrli_w,     VSHIFT, X86ISD::VSRLI, 0);
  INTRINSIC_WO_CHAIN(x86_avx2_psrli_d,     VSHIFT, X86ISD::VSRLI, 0);
  INTRINSIC_WO_CHAIN(x86_avx2_psrli_q,     VSHIFT, X86ISD::VSRLI, 0);
  INTRINSIC_WO_CHAIN(x86_sse2_psrai_w,     VSHIFT, X86ISD::VSRAI, 0);
  INTRINSIC_WO_CHAIN(x86_sse2_psrai_d,     VSHIFT, X86ISD::VSRAI, 0);
  INTRINSIC_WO_CHAIN(x86_avx2_psrai_w,     VSHIFT, X86ISD::VSRAI, 0);
  INTRINSIC_WO_CHAIN(x86_avx2_psrai_d,     VSHIFT, X86ISD::VSRAI, 0);

  INTRINSIC_WO_CHAIN(x86_avx_vperm2f128_ps_256, INTR_TYPE_3OP, X86ISD::VPERM2X128, 0);
  INTRINSIC_WO_CHAIN(x86_avx_vperm2f128_pd_256, INTR_TYPE_3OP, X86ISD::VPERM2X128, 0);
  INTRINSIC_WO_CHAIN(x86_avx_vperm2f128_si_256, INTR_TYPE_3OP, X86ISD::VPERM2X128, 0);
  INTRINSIC_WO_CHAIN(x86_avx2_vperm2i128,       INTR_TYPE_3OP, X86ISD::VPERM2X128, 0);

  INTRINSIC_WO_CHAIN(x86_sse41_insertps,        INTR_TYPE_3OP, X86ISD::INSERTPS, 0);

  INTRINSIC_WO_CHAIN(x86_sse_comieq_ss,  COMI, X86ISD::COMI, ISD::SETEQ);
  INTRINSIC_WO_CHAIN(x86_sse2_comieq_sd, COMI, X86ISD::COMI, ISD::SETEQ);
  INTRINSIC_WO_CHAIN(x86_sse_comilt_ss,  COMI, X86ISD::COMI, ISD::SETLT);
  INTRINSIC_WO_CHAIN(x86_sse2_comilt_sd, COMI, X86ISD::COMI, ISD::SETLT);
  INTRINSIC_WO_CHAIN(x86_sse_comile_ss,  COMI, X86ISD::COMI, ISD::SETLE);
  INTRINSIC_WO_CHAIN(x86_sse2_comile_sd, COMI, X86ISD::COMI, ISD::SETLE);
  INTRINSIC_WO_CHAIN(x86_sse_comigt_ss,  COMI, X86ISD::COMI, ISD::SETGT);
  INTRINSIC_WO_CHAIN(x86_sse2_comigt_sd, COMI, X86ISD::COMI, ISD::SETGT);
  INTRINSIC_WO_CHAIN(x86_sse_comige_ss,  COMI, X86ISD::COMI, ISD::SETGE);
  INTRINSIC_WO_CHAIN(x86_sse2_comige_sd, COMI, X86ISD::COMI, ISD::SETGE);
  INTRINSIC_WO_CHAIN(x86_sse_comineq_ss, COMI, X86ISD::COMI, ISD::SETNE);
  INTRINSIC_WO_CHAIN(x86_sse2_comineq_sd,COMI, X86ISD::COMI, ISD::SETNE);

  INTRINSIC_WO_CHAIN(x86_sse_ucomieq_ss,  COMI, X86ISD::UCOMI, ISD::SETEQ);
  INTRINSIC_WO_CHAIN(x86_sse2_ucomieq_sd, COMI, X86ISD::UCOMI, ISD::SETEQ);
  INTRINSIC_WO_CHAIN(x86_sse_ucomilt_ss,  COMI, X86ISD::UCOMI, ISD::SETLT);
  INTRINSIC_WO_CHAIN(x86_sse2_ucomilt_sd, COMI, X86ISD::UCOMI, ISD::SETLT);
  INTRINSIC_WO_CHAIN(x86_sse_ucomile_ss,  COMI, X86ISD::UCOMI, ISD::SETLE);
  INTRINSIC_WO_CHAIN(x86_sse2_ucomile_sd, COMI, X86ISD::UCOMI, ISD::SETLE);
  INTRINSIC_WO_CHAIN(x86_sse_ucomigt_ss,  COMI, X86ISD::UCOMI, ISD::SETGT);
  INTRINSIC_WO_CHAIN(x86_sse2_ucomigt_sd, COMI, X86ISD::UCOMI, ISD::SETGT);
  INTRINSIC_WO_CHAIN(x86_sse_ucomige_ss,  COMI, X86ISD::UCOMI, ISD::SETGE);
  INTRINSIC_WO_CHAIN(x86_sse2_ucomige_sd, COMI, X86ISD::UCOMI, ISD::SETGE);
  INTRINSIC_WO_CHAIN(x86_sse_ucomineq_ss, COMI, X86ISD::UCOMI, ISD::SETNE);
  INTRINSIC_WO_CHAIN(x86_sse2_ucomineq_sd,COMI, X86ISD::UCOMI, ISD::SETNE);
}

const IntrinsicData* GetIntrinsicWithoutChain(unsigned IntNo) {
  std::map < unsigned, IntrinsicData>::const_iterator itr =
    IntrWithoutChainMap.find(IntNo);
  if (itr == IntrWithoutChainMap.end())
    return NULL;
  return &(itr->second);
}

// Initialize intrinsics data
void InitIntrinsicTables() {
  InitIntrinsicsWithChain();
  InitIntrinsicsWithoutChain();
}


#endif
