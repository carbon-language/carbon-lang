; RUN: llc -mtriple=arm-eabi -mattr=+v8.2a,+neon,+fullfp16 -float-abi=hard < %s | FileCheck %s

%struct.float16x4x2_t = type { [2 x <4 x half>] }
%struct.float16x8x2_t = type { [2 x <8 x half>] }

define dso_local <4 x half> @test_vabs_f16(<4 x half> %a) {
; CHECKLABEL: test_vabs_f16:
; CHECK:         vabs.f16 d0, d0
; CHECK-NEXT:    bx lr
entry:
  %vabs1.i = tail call <4 x half> @llvm.fabs.v4f16(<4 x half> %a)
  ret <4 x half> %vabs1.i
}

define dso_local <8 x half> @test_vabsq_f16(<8 x half> %a) {
; CHECKLABEL: test_vabsq_f16:
; CHECK:         vabs.f16 q0, q0
; CHECK-NEXT:    bx lr
entry:
  %vabs1.i = tail call <8 x half> @llvm.fabs.v8f16(<8 x half> %a)
  ret <8 x half> %vabs1.i
}

define dso_local <4 x i16> @test_vceqz_f16(<4 x half> %a) {
; CHECKLABEL: test_vceqz_f16:
; CHECK:         vceq.f16 d0, d0, #0
; CHECK-NEXT:    bx lr
entry:
  %0 = fcmp oeq <4 x half> %a, zeroinitializer
  %vceqz.i = sext <4 x i1> %0 to <4 x i16>
  ret <4 x i16> %vceqz.i
}

define dso_local <8 x i16> @test_vceqzq_f16(<8 x half> %a) {
; CHECKLABEL: test_vceqzq_f16:
; CHECK:         vceq.f16 q0, q0, #0
; CHECK-NEXT:    bx lr
entry:
  %0 = fcmp oeq <8 x half> %a, zeroinitializer
  %vceqz.i = sext <8 x i1> %0 to <8 x i16>
  ret <8 x i16> %vceqz.i
}

define dso_local <4 x i16> @test_vcgez_f16(<4 x half> %a) {
; CHECKLABEL: test_vcgez_f16:
; CHECK:         vcge.f16 d0, d0, #0
; CHECK-NEXT:    bx lr
entry:
  %0 = fcmp oge <4 x half> %a, zeroinitializer
  %vcgez.i = sext <4 x i1> %0 to <4 x i16>
  ret <4 x i16> %vcgez.i
}

define dso_local <8 x i16> @test_vcgezq_f16(<8 x half> %a) {
; CHECKLABEL: test_vcgezq_f16:
; CHECK:         vcge.f16 q0, q0, #0
; CHECK-NEXT:    bx lr
entry:
  %0 = fcmp oge <8 x half> %a, zeroinitializer
  %vcgez.i = sext <8 x i1> %0 to <8 x i16>
  ret <8 x i16> %vcgez.i
}

define dso_local <4 x i16> @test_vcgtz_f16(<4 x half> %a) {
; CHECKLABEL: test_vcgtz_f16:
; CHECK:         vcgt.f16 d0, d0, #0
; CHECK-NEXT:    bx lr
entry:
  %0 = fcmp ogt <4 x half> %a, zeroinitializer
  %vcgtz.i = sext <4 x i1> %0 to <4 x i16>
  ret <4 x i16> %vcgtz.i
}

define dso_local <8 x i16> @test_vcgtzq_f16(<8 x half> %a) {
; CHECKLABEL: test_vcgtzq_f16:
; CHECK:         vcgt.f16 q0, q0, #0
; CHECK-NEXT:    bx lr
entry:
  %0 = fcmp ogt <8 x half> %a, zeroinitializer
  %vcgtz.i = sext <8 x i1> %0 to <8 x i16>
  ret <8 x i16> %vcgtz.i
}

define dso_local <4 x i16> @test_vclez_f16(<4 x half> %a) {
; CHECKLABEL: test_vclez_f16:
; CHECK:         vcle.f16 d0, d0, #0
; CHECK-NEXT:    bx lr
entry:
  %0 = fcmp ole <4 x half> %a, zeroinitializer
  %vclez.i = sext <4 x i1> %0 to <4 x i16>
  ret <4 x i16> %vclez.i
}

define dso_local <8 x i16> @test_vclezq_f16(<8 x half> %a) {
; CHECKLABEL: test_vclezq_f16:
; CHECK:         vcle.f16 q0, q0, #0
; CHECK-NEXT:    bx lr
entry:
  %0 = fcmp ole <8 x half> %a, zeroinitializer
  %vclez.i = sext <8 x i1> %0 to <8 x i16>
  ret <8 x i16> %vclez.i
}

define dso_local <4 x i16> @test_vcltz_f16(<4 x half> %a) {
; CHECKLABEL: test_vcltz_f16:
; CHECK:         vclt.f16 d0, d0, #0
; CHECK-NEXT:    bx lr
entry:
  %0 = fcmp olt <4 x half> %a, zeroinitializer
  %vcltz.i = sext <4 x i1> %0 to <4 x i16>
  ret <4 x i16> %vcltz.i
}

define dso_local <8 x i16> @test_vcltzq_f16(<8 x half> %a) {
; CHECKLABEL: test_vcltzq_f16:
; CHECK:         vclt.f16 q0, q0, #0
; CHECK-NEXT:    bx lr
entry:
  %0 = fcmp olt <8 x half> %a, zeroinitializer
  %vcltz.i = sext <8 x i1> %0 to <8 x i16>
  ret <8 x i16> %vcltz.i
}

define dso_local <4 x half> @test_vcvt_f16_s16(<4 x i16> %a) {
; CHECK-LABEL: test_vcvt_f16_s16:
; CHECK:         vcvt.f16.s16 d0, d0
; CHECK-NEXT:    bx lr
entry:
  %vcvt.i = sitofp <4 x i16> %a to <4 x half>
  ret <4 x half> %vcvt.i
}

define dso_local <8 x half> @test_vcvtq_f16_s16(<8 x i16> %a) {
; CHECK-LABEL: test_vcvtq_f16_s16:
; CHECK:         vcvt.f16.s16 q0, q0
; CHECK-NEXT:    bx lr
entry:
  %vcvt.i = sitofp <8 x i16> %a to <8 x half>
  ret <8 x half> %vcvt.i
}

define dso_local <4 x half> @test_vcvt_f16_u16(<4 x i16> %a) {
; CHECK-LABEL: test_vcvt_f16_u16:
; CHECK:         vcvt.f16.u16 d0, d0
; CHECK-NEXT:    bx lr
entry:
  %vcvt.i = uitofp <4 x i16> %a to <4 x half>
  ret <4 x half> %vcvt.i
}

define dso_local <8 x half> @test_vcvtq_f16_u16(<8 x i16> %a) {
; CHECK-LABEL: test_vcvtq_f16_u16:
; CHECK:         vcvt.f16.u16 q0, q0
; CHECK-NEXT:    bx lr
entry:
  %vcvt.i = uitofp <8 x i16> %a to <8 x half>
  ret <8 x half> %vcvt.i
}

define dso_local <4 x i16> @test_vcvt_s16_f16(<4 x half> %a) {
; CHECK-LABEL: test_vcvt_s16_f16:
; CHECK:         vcvt.s16.f16 d0, d0
; CHECK-NEXT:    bx lr
entry:
  %vcvt.i = fptosi <4 x half> %a to <4 x i16>
  ret <4 x i16> %vcvt.i
}

define dso_local <8 x i16> @test_vcvtq_s16_f16(<8 x half> %a) {
; CHECK-LABEL: test_vcvtq_s16_f16:
; CHECK:         vcvt.s16.f16 q0, q0
; CHECK-NEXT:    bx lr
entry:
  %vcvt.i = fptosi <8 x half> %a to <8 x i16>
  ret <8 x i16> %vcvt.i
}

define dso_local <4 x i16> @test_vcvt_u16_f16(<4 x half> %a) {
; CHECK-LABEL: test_vcvt_u16_f16:
; CHECK:         vcvt.u16.f16 d0, d0
; CHECK-NEXT:    bx lr
entry:
  %vcvt.i = fptoui <4 x half> %a to <4 x i16>
  ret <4 x i16> %vcvt.i
}

define dso_local <8 x i16> @test_vcvtq_u16_f16(<8 x half> %a) {
; CHECK-LABEL: test_vcvtq_u16_f16:
; CHECK:         vcvt.u16.f16 q0, q0
; CHECK-NEXT:    bx lr
entry:
  %vcvt.i = fptoui <8 x half> %a to <8 x i16>
  ret <8 x i16> %vcvt.i
}

define dso_local <4 x i16> @test_vcvta_s16_f16(<4 x half> %a) {
; CHECK-LABEL: test_vcvta_s16_f16:
; CHECK:         vcvta.s16.f16 d0, d0
; CHECK-NEXT:    bx lr
entry:
  %vcvta_s16_v1.i = tail call <4 x i16> @llvm.arm.neon.vcvtas.v4i16.v4f16(<4 x half> %a)
  ret <4 x i16> %vcvta_s16_v1.i
}

define dso_local <4 x i16> @test_vcvta_u16_f16(<4 x half> %a) {
; CHECK-LABEL: test_vcvta_u16_f16:
; CHECK:         vcvta.u16.f16 d0, d0
; CHECK-NEXT:    bx lr
entry:
  %vcvta_u16_v1.i = tail call <4 x i16> @llvm.arm.neon.vcvtau.v4i16.v4f16(<4 x half> %a)
  ret <4 x i16> %vcvta_u16_v1.i
}

define dso_local <8 x i16> @test_vcvtaq_s16_f16(<8 x half> %a) {
; CHECK-LABEL: test_vcvtaq_s16_f16:
; CHECK:         vcvta.s16.f16 q0, q0
; CHECK-NEXT:    bx lr
entry:
  %vcvtaq_s16_v1.i = tail call <8 x i16> @llvm.arm.neon.vcvtas.v8i16.v8f16(<8 x half> %a)
  ret <8 x i16> %vcvtaq_s16_v1.i
}

define dso_local <4 x i16> @test_vcvtm_s16_f16(<4 x half> %a) {
; CHECK-LABEL: test_vcvtm_s16_f16:
; CHECK:         vcvtm.s16.f16 d0, d0
; CHECK-NEXT:    bx lr
entry:
  %vcvtm_s16_v1.i = tail call <4 x i16> @llvm.arm.neon.vcvtms.v4i16.v4f16(<4 x half> %a)
  ret <4 x i16> %vcvtm_s16_v1.i
}

define dso_local <8 x i16> @test_vcvtmq_s16_f16(<8 x half> %a) {
; CHECK-LABEL: test_vcvtmq_s16_f16:
; CHECK:         vcvtm.s16.f16 q0, q0
; CHECK-NEXT:    bx lr
entry:
  %vcvtmq_s16_v1.i = tail call <8 x i16> @llvm.arm.neon.vcvtms.v8i16.v8f16(<8 x half> %a)
  ret <8 x i16> %vcvtmq_s16_v1.i
}

define dso_local <4 x i16> @test_vcvtm_u16_f16(<4 x half> %a) {
; CHECK-LABEL: test_vcvtm_u16_f16:
; CHECK:         vcvtm.u16.f16 d0, d0
; CHECK-NEXT:    bx lr
entry:
  %vcvtm_u16_v1.i = tail call <4 x i16> @llvm.arm.neon.vcvtmu.v4i16.v4f16(<4 x half> %a)
  ret <4 x i16> %vcvtm_u16_v1.i
}

define dso_local <8 x i16> @test_vcvtmq_u16_f16(<8 x half> %a) {
; CHECK-LABEL: test_vcvtmq_u16_f16:
; CHECK:         vcvtm.u16.f16 q0, q0
; CHECK-NEXT:    bx lr
entry:
  %vcvtmq_u16_v1.i = tail call <8 x i16> @llvm.arm.neon.vcvtmu.v8i16.v8f16(<8 x half> %a)
  ret <8 x i16> %vcvtmq_u16_v1.i
}

define dso_local <4 x i16> @test_vcvtn_s16_f16(<4 x half> %a) {
; CHECK-LABEL: test_vcvtn_s16_f16:
; CHECK:         vcvtn.s16.f16 d0, d0
; CHECK-NEXT:    bx lr
entry:
  %vcvtn_s16_v1.i = tail call <4 x i16> @llvm.arm.neon.vcvtns.v4i16.v4f16(<4 x half> %a)
  ret <4 x i16> %vcvtn_s16_v1.i
}

define dso_local <8 x i16> @test_vcvtnq_s16_f16(<8 x half> %a) {
; CHECK-LABEL: test_vcvtnq_s16_f16:
; CHECK:         vcvtn.s16.f16 q0, q0
; CHECK-NEXT:    bx lr
entry:
  %vcvtnq_s16_v1.i = tail call <8 x i16> @llvm.arm.neon.vcvtns.v8i16.v8f16(<8 x half> %a)
  ret <8 x i16> %vcvtnq_s16_v1.i
}

define dso_local <4 x i16> @test_vcvtn_u16_f16(<4 x half> %a) {
; CHECK-LABEL: test_vcvtn_u16_f16:
; CHECK:         vcvtn.u16.f16 d0, d0
; CHECK-NEXT:    bx lr
entry:
  %vcvtn_u16_v1.i = tail call <4 x i16> @llvm.arm.neon.vcvtnu.v4i16.v4f16(<4 x half> %a)
  ret <4 x i16> %vcvtn_u16_v1.i
}

define dso_local <8 x i16> @test_vcvtnq_u16_f16(<8 x half> %a) {
; CHECK-LABEL: test_vcvtnq_u16_f16:
; CHECK:         vcvtn.u16.f16 q0, q0
; CHECK-NEXT:    bx lr
entry:
  %vcvtnq_u16_v1.i = tail call <8 x i16> @llvm.arm.neon.vcvtnu.v8i16.v8f16(<8 x half> %a)
  ret <8 x i16> %vcvtnq_u16_v1.i
}

define dso_local <4 x i16> @test_vcvtp_s16_f16(<4 x half> %a) {
; CHECK-LABEL: test_vcvtp_s16_f16:
; CHECK:         vcvtp.s16.f16 d0, d0
; CHECK-NEXT:    bx lr
entry:
  %vcvtp_s16_v1.i = tail call <4 x i16> @llvm.arm.neon.vcvtps.v4i16.v4f16(<4 x half> %a)
  ret <4 x i16> %vcvtp_s16_v1.i
}

define dso_local <8 x i16> @test_vcvtpq_s16_f16(<8 x half> %a) {
; CHECK-LABEL: test_vcvtpq_s16_f16:
; CHECK:         vcvtp.s16.f16 q0, q0
; CHECK-NEXT:    bx lr
entry:
  %vcvtpq_s16_v1.i = tail call <8 x i16> @llvm.arm.neon.vcvtps.v8i16.v8f16(<8 x half> %a)
  ret <8 x i16> %vcvtpq_s16_v1.i
}

define dso_local <4 x i16> @test_vcvtp_u16_f16(<4 x half> %a) {
; CHECK-LABEL: test_vcvtp_u16_f16:
; CHECK:         vcvtp.u16.f16 d0, d0
; CHECK-NEXT:    bx lr
entry:
  %vcvtp_u16_v1.i = tail call <4 x i16> @llvm.arm.neon.vcvtpu.v4i16.v4f16(<4 x half> %a)
  ret <4 x i16> %vcvtp_u16_v1.i
}

define dso_local <8 x i16> @test_vcvtpq_u16_f16(<8 x half> %a) {
; CHECK-LABEL: test_vcvtpq_u16_f16:
; CHECK:         vcvtp.u16.f16 q0, q0
; CHECK-NEXT:    bx lr
entry:
  %vcvtpq_u16_v1.i = tail call <8 x i16> @llvm.arm.neon.vcvtpu.v8i16.v8f16(<8 x half> %a)
  ret <8 x i16> %vcvtpq_u16_v1.i
}

define dso_local <4 x half> @test_vneg_f16(<4 x half> %a) {
; CHECKLABEL: test_vneg_f16:
; CHECK:         vneg.f16 d0, d0
; CHECK-NEXT:    bx lr
entry:
  %sub.i = fsub <4 x half> <half 0xH8000, half 0xH8000, half 0xH8000, half 0xH8000>, %a
  ret <4 x half> %sub.i
}

define dso_local <8 x half> @test_vnegq_f16(<8 x half> %a) {
; CHECKLABEL: test_vnegq_f16:
; CHECK:         vneg.f16 q0, q0
; CHECK-NEXT:    bx lr
entry:
  %sub.i = fsub <8 x half> <half 0xH8000, half 0xH8000, half 0xH8000, half 0xH8000, half 0xH8000, half 0xH8000, half 0xH8000, half 0xH8000>, %a
  ret <8 x half> %sub.i
}

define dso_local <4 x half> @test_vrecpe_f16(<4 x half> %a) {
; CHECKLABEL: test_vrecpe_f16:
; CHECK:         vrecpe.f16 d0, d0
; CHECK-NEXT:    bx lr
entry:
  %vrecpe_v1.i = tail call <4 x half> @llvm.arm.neon.vrecpe.v4f16(<4 x half> %a)
  ret <4 x half> %vrecpe_v1.i
}

define dso_local <8 x half> @test_vrecpeq_f16(<8 x half> %a) {
; CHECKLABEL: test_vrecpeq_f16:
; CHECK:         vrecpe.f16 q0, q0
; CHECK-NEXT:    bx lr
entry:
  %vrecpeq_v1.i = tail call <8 x half> @llvm.arm.neon.vrecpe.v8f16(<8 x half> %a)
  ret <8 x half> %vrecpeq_v1.i
}

define dso_local <4 x half> @test_vrnd_f16(<4 x half> %a) {
; CHECKLABEL: test_vrnd_f16:
; CHECK:         vrintz.f16 d0, d0
; CHECK-NEXT:    bx lr
entry:
  %vrnd_v1.i = tail call <4 x half> @llvm.arm.neon.vrintz.v4f16(<4 x half> %a)
  ret <4 x half> %vrnd_v1.i
}

define dso_local <8 x half> @test_vrndq_f16(<8 x half> %a) {
; CHECKLABEL: test_vrndq_f16:
; CHECK:         vrintz.f16 q0, q0
; CHECK-NEXT:    bx lr
entry:
  %vrndq_v1.i = tail call <8 x half> @llvm.arm.neon.vrintz.v8f16(<8 x half> %a)
  ret <8 x half> %vrndq_v1.i
}

define dso_local <4 x half> @test_vrnda_f16(<4 x half> %a) {
; CHECKLABEL: test_vrnda_f16:
; CHECK:         vrinta.f16 d0, d0
; CHECK-NEXT:    bx lr
entry:
  %vrnda_v1.i = tail call <4 x half> @llvm.arm.neon.vrinta.v4f16(<4 x half> %a)
  ret <4 x half> %vrnda_v1.i
}

define dso_local <8 x half> @test_vrndaq_f16(<8 x half> %a) {
; CHECKLABEL: test_vrndaq_f16:
; CHECK:         vrinta.f16 q0, q0
; CHECK-NEXT:    bx lr
entry:
  %vrndaq_v1.i = tail call <8 x half> @llvm.arm.neon.vrinta.v8f16(<8 x half> %a)
  ret <8 x half> %vrndaq_v1.i
}

define dso_local <4 x half> @test_vrndm_f16(<4 x half> %a) {
; CHECKLABEL: test_vrndm_f16:
; CHECK:         vrintm.f16 d0, d0
; CHECK-NEXT:    bx lr
entry:
  %vrndm_v1.i = tail call <4 x half> @llvm.arm.neon.vrintm.v4f16(<4 x half> %a)
  ret <4 x half> %vrndm_v1.i
}

define dso_local <8 x half> @test_vrndmq_f16(<8 x half> %a) {
; CHECKLABEL: test_vrndmq_f16:
; CHECK:         vrintm.f16 q0, q0
; CHECK-NEXT:    bx lr
entry:
  %vrndmq_v1.i = tail call <8 x half> @llvm.arm.neon.vrintm.v8f16(<8 x half> %a)
  ret <8 x half> %vrndmq_v1.i
}

define dso_local <4 x half> @test_vrndn_f16(<4 x half> %a) {
; CHECKLABEL: test_vrndn_f16:
; CHECK:         vrintn.f16 d0, d0
; CHECK-NEXT:    bx lr
entry:
  %vrndn_v1.i = tail call <4 x half> @llvm.arm.neon.vrintn.v4f16(<4 x half> %a)
  ret <4 x half> %vrndn_v1.i
}

define dso_local <8 x half> @test_vrndnq_f16(<8 x half> %a) {
; CHECKLABEL: test_vrndnq_f16:
; CHECK:         vrintn.f16 q0, q0
; CHECK-NEXT:    bx lr
entry:
  %vrndnq_v1.i = tail call <8 x half> @llvm.arm.neon.vrintn.v8f16(<8 x half> %a)
  ret <8 x half> %vrndnq_v1.i
}

define dso_local <4 x half> @test_vrndp_f16(<4 x half> %a) {
; CHECKLABEL: test_vrndp_f16:
; CHECK:         vrintp.f16 d0, d0
; CHECK-NEXT:    bx lr
entry:
  %vrndp_v1.i = tail call <4 x half> @llvm.arm.neon.vrintp.v4f16(<4 x half> %a)
  ret <4 x half> %vrndp_v1.i
}

define dso_local <8 x half> @test_vrndpq_f16(<8 x half> %a) {
; CHECKLABEL: test_vrndpq_f16:
; CHECK:         vrintp.f16 q0, q0
; CHECK-NEXT:    bx lr
entry:
  %vrndpq_v1.i = tail call <8 x half> @llvm.arm.neon.vrintp.v8f16(<8 x half> %a)
  ret <8 x half> %vrndpq_v1.i
}

define dso_local <4 x half> @test_vrndx_f16(<4 x half> %a) {
; CHECKLABEL: test_vrndx_f16:
; CHECK:         vrintx.f16 d0, d0
; CHECK-NEXT:    bx lr
entry:
  %vrndx_v1.i = tail call <4 x half> @llvm.arm.neon.vrintx.v4f16(<4 x half> %a)
  ret <4 x half> %vrndx_v1.i
}

define dso_local <8 x half> @test_vrndxq_f16(<8 x half> %a) {
; CHECKLABEL: test_vrndxq_f16:
; CHECK:         vrintx.f16 q0, q0
; CHECK-NEXT:    bx lr
entry:
  %vrndxq_v1.i = tail call <8 x half> @llvm.arm.neon.vrintx.v8f16(<8 x half> %a)
  ret <8 x half> %vrndxq_v1.i
}

define dso_local <4 x half> @test_vrsqrte_f16(<4 x half> %a) {
; CHECKLABEL: test_vrsqrte_f16:
; CHECK:         vrsqrte.f16 d0, d0
; CHECK-NEXT:    bx lr
entry:
  %vrsqrte_v1.i = tail call <4 x half> @llvm.arm.neon.vrsqrte.v4f16(<4 x half> %a)
  ret <4 x half> %vrsqrte_v1.i
}

define dso_local <8 x half> @test_vrsqrteq_f16(<8 x half> %a) {
; CHECKLABEL: test_vrsqrteq_f16:
; CHECK:         vrsqrte.f16 q0, q0
; CHECK-NEXT:    bx lr
entry:
  %vrsqrteq_v1.i = tail call <8 x half> @llvm.arm.neon.vrsqrte.v8f16(<8 x half> %a)
  ret <8 x half> %vrsqrteq_v1.i
}

define dso_local <4 x half> @test_vadd_f16(<4 x half> %a, <4 x half> %b) {
; CHECKLABEL: test_vadd_f16:
; CHECK:         vadd.f16 d0, d0, d1
; CHECK-NEXT:    bx lr
entry:
  %add.i = fadd <4 x half> %a, %b
  ret <4 x half> %add.i
}

define dso_local <8 x half> @test_vaddq_f16(<8 x half> %a, <8 x half> %b) {
; CHECKLABEL: test_vaddq_f16:
; CHECK:         vadd.f16 q0, q0, q1
; CHECK-NEXT:    bx lr
entry:
  %add.i = fadd <8 x half> %a, %b
  ret <8 x half> %add.i
}

define dso_local <4 x half> @test_vabd_f16(<4 x half> %a, <4 x half> %b) {
; CHECKLABEL: test_vabd_f16:
; CHECK:         vabd.f16 d0, d0, d1
; CHECK-NEXT:    bx lr
entry:
  %vabd_v2.i = tail call <4 x half> @llvm.arm.neon.vabds.v4f16(<4 x half> %a, <4 x half> %b)
  ret <4 x half> %vabd_v2.i
}

define dso_local <8 x half> @test_vabdq_f16(<8 x half> %a, <8 x half> %b) {
; CHECKLABEL: test_vabdq_f16:
; CHECK:         vabd.f16 q0, q0, q1
; CHECK-NEXT:    bx lr
entry:
  %vabdq_v2.i = tail call <8 x half> @llvm.arm.neon.vabds.v8f16(<8 x half> %a, <8 x half> %b)
  ret <8 x half> %vabdq_v2.i
}

define dso_local <4 x i16> @test_vcage_f16(<4 x half> %a, <4 x half> %b) {
; CHECKLABEL: test_vcage_f16:
; CHECK:         vacge.f16 d0, d0, d1
; CHECK-NEXT:    bx lr
entry:
  %vcage_v2.i = tail call <4 x i16> @llvm.arm.neon.vacge.v4i16.v4f16(<4 x half> %a, <4 x half> %b)
  ret <4 x i16> %vcage_v2.i
}

define dso_local <8 x i16> @test_vcageq_f16(<8 x half> %a, <8 x half> %b) {
; CHECKLABEL: test_vcageq_f16:
; CHECK:         vacge.f16 q0, q0, q1
; CHECK-NEXT:    bx lr
entry:
  %vcageq_v2.i = tail call <8 x i16> @llvm.arm.neon.vacge.v8i16.v8f16(<8 x half> %a, <8 x half> %b)
  ret <8 x i16> %vcageq_v2.i
}

define dso_local <4 x i16> @test_vcagt_f16(<4 x half> %a, <4 x half> %b) {
; CHECK-LABEL: test_vcagt_f16:
; CHECK:         vacgt.f16 d0, d0, d1
; CHECK-NEXT:    bx lr
entry:
  %vcagt_v2.i = tail call <4 x i16> @llvm.arm.neon.vacgt.v4i16.v4f16(<4 x half> %a, <4 x half> %b)
  ret <4 x i16> %vcagt_v2.i
}

define dso_local <8 x i16> @test_vcagtq_f16(<8 x half> %a, <8 x half> %b) {
; CHECK-LABEL: test_vcagtq_f16:
; CHECK:         vacgt.f16 q0, q0, q1
; CHECK-NEXT:    bx lr
entry:
  %vcagtq_v2.i = tail call <8 x i16> @llvm.arm.neon.vacgt.v8i16.v8f16(<8 x half> %a, <8 x half> %b)
  ret <8 x i16> %vcagtq_v2.i
}

define dso_local <4 x i16> @test_vcale_f16(<4 x half> %a, <4 x half> %b) {
; CHECKLABEL: test_vcale_f16:
; CHECK:         vacge.f16 d0, d1, d0
; CHECK-NEXT:    bx lr
entry:
  %vcale_v2.i = tail call <4 x i16> @llvm.arm.neon.vacge.v4i16.v4f16(<4 x half> %b, <4 x half> %a)
  ret <4 x i16> %vcale_v2.i
}

define dso_local <8 x i16> @test_vcaleq_f16(<8 x half> %a, <8 x half> %b) {
; CHECKLABEL: test_vcaleq_f16:
; CHECK:         vacge.f16 q0, q1, q0
; CHECK-NEXT:    bx lr
entry:
  %vcaleq_v2.i = tail call <8 x i16> @llvm.arm.neon.vacge.v8i16.v8f16(<8 x half> %b, <8 x half> %a)
  ret <8 x i16> %vcaleq_v2.i
}

define dso_local <4 x i16> @test_vceq_f16(<4 x half> %a, <4 x half> %b) {
; CHECKLABEL: test_vceq_f16:
; CHECK:         vceq.f16 d0, d0, d1
; CHECK-NEXT:    bx lr
entry:
  %cmp.i = fcmp oeq <4 x half> %a, %b
  %sext.i = sext <4 x i1> %cmp.i to <4 x i16>
  ret <4 x i16> %sext.i
}

define dso_local <8 x i16> @test_vceqq_f16(<8 x half> %a, <8 x half> %b) {
; CHECKLABEL: test_vceqq_f16:
; CHECK:         vceq.f16 q0, q0, q1
; CHECK-NEXT:    bx lr
entry:
  %cmp.i = fcmp oeq <8 x half> %a, %b
  %sext.i = sext <8 x i1> %cmp.i to <8 x i16>
  ret <8 x i16> %sext.i
}

define dso_local <4 x i16> @test_vcge_f16(<4 x half> %a, <4 x half> %b) {
; CHECKLABEL: test_vcge_f16:
; CHECK:         vcge.f16 d0, d0, d1
; CHECK-NEXT:    bx lr
entry:
  %cmp.i = fcmp oge <4 x half> %a, %b
  %sext.i = sext <4 x i1> %cmp.i to <4 x i16>
  ret <4 x i16> %sext.i
}

define dso_local <8 x i16> @test_vcgeq_f16(<8 x half> %a, <8 x half> %b) {
; CHECKLABEL: test_vcgeq_f16:
; CHECK:         vcge.f16 q0, q0, q1
; CHECK-NEXT:    bx lr
entry:
  %cmp.i = fcmp oge <8 x half> %a, %b
  %sext.i = sext <8 x i1> %cmp.i to <8 x i16>
  ret <8 x i16> %sext.i
}

define dso_local <4 x i16> @test_vcgt_f16(<4 x half> %a, <4 x half> %b) {
; CHECKLABEL: test_vcgt_f16:
; CHECK:         vcgt.f16 d0, d0, d1
; CHECK-NEXT:    bx lr
entry:
  %cmp.i = fcmp ogt <4 x half> %a, %b
  %sext.i = sext <4 x i1> %cmp.i to <4 x i16>
  ret <4 x i16> %sext.i
}

define dso_local <8 x i16> @test_vcgtq_f16(<8 x half> %a, <8 x half> %b) {
; CHECKLABEL: test_vcgtq_f16:
; CHECK:         vcgt.f16 q0, q0, q1
; CHECK-NEXT:    bx lr
entry:
  %cmp.i = fcmp ogt <8 x half> %a, %b
  %sext.i = sext <8 x i1> %cmp.i to <8 x i16>
  ret <8 x i16> %sext.i
}

define dso_local <4 x i16> @test_vcle_f16(<4 x half> %a, <4 x half> %b) {
; CHECKLABEL: test_vcle_f16:
; CHECK:         vcge.f16 d0, d1, d0
; CHECK-NEXT:    bx lr
entry:
  %cmp.i = fcmp ole <4 x half> %a, %b
  %sext.i = sext <4 x i1> %cmp.i to <4 x i16>
  ret <4 x i16> %sext.i
}

define dso_local <8 x i16> @test_vcleq_f16(<8 x half> %a, <8 x half> %b) {
; CHECKLABEL: test_vcleq_f16:
; CHECK:         vcge.f16 q0, q1, q0
; CHECK-NEXT:    bx lr
entry:
  %cmp.i = fcmp ole <8 x half> %a, %b
  %sext.i = sext <8 x i1> %cmp.i to <8 x i16>
  ret <8 x i16> %sext.i
}

define dso_local <4 x i16> @test_vclt_f16(<4 x half> %a, <4 x half> %b) {
; CHECKLABEL: test_vclt_f16:
; CHECK:         vcgt.f16 d0, d1, d0
; CHECK-NEXT:    bx lr
entry:
  %cmp.i = fcmp olt <4 x half> %a, %b
  %sext.i = sext <4 x i1> %cmp.i to <4 x i16>
  ret <4 x i16> %sext.i
}

define dso_local <8 x i16> @test_vcltq_f16(<8 x half> %a, <8 x half> %b) {
; CHECKLABEL: test_vcltq_f16:
; CHECK:         vcgt.f16 q0, q1, q0
; CHECK-NEXT:    bx lr
entry:
  %cmp.i = fcmp olt <8 x half> %a, %b
  %sext.i = sext <8 x i1> %cmp.i to <8 x i16>
  ret <8 x i16> %sext.i
}

define dso_local <4 x half> @test_vcvt_n_f16_s16(<4 x i16> %a) {
; CHECKLABEL: test_vcvt_n_f16_s16:
; CHECK:         vcvt.f16.s16 d0, d0, #2
; CHECK-NEXT:    bx lr
entry:
  %vcvt_n1 = tail call <4 x half> @llvm.arm.neon.vcvtfxs2fp.v4f16.v4i16(<4 x i16> %a, i32 2)
  ret <4 x half> %vcvt_n1
}

declare <4 x half> @llvm.arm.neon.vcvtfxs2fp.v4f16.v4i16(<4 x i16>, i32) #2

define dso_local <8 x half> @test_vcvtq_n_f16_s16(<8 x i16> %a) {
; CHECKLABEL: test_vcvtq_n_f16_s16:
; CHECK:         vcvt.f16.s16 q0, q0, #2
; CHECK-NEXT:    bx lr
entry:
  %vcvt_n1 = tail call <8 x half> @llvm.arm.neon.vcvtfxs2fp.v8f16.v8i16(<8 x i16> %a, i32 2)
  ret <8 x half> %vcvt_n1
}

declare <8 x half> @llvm.arm.neon.vcvtfxs2fp.v8f16.v8i16(<8 x i16>, i32) #2

define dso_local <4 x half> @test_vcvt_n_f16_u16(<4 x i16> %a) {
; CHECKLABEL: test_vcvt_n_f16_u16:
; CHECK:         vcvt.f16.u16 d0, d0, #2
; CHECK-NEXT:    bx lr
entry:
  %vcvt_n1 = tail call <4 x half> @llvm.arm.neon.vcvtfxu2fp.v4f16.v4i16(<4 x i16> %a, i32 2)
  ret <4 x half> %vcvt_n1
}

declare <4 x half> @llvm.arm.neon.vcvtfxu2fp.v4f16.v4i16(<4 x i16>, i32) #2

define dso_local <8 x half> @test_vcvtq_n_f16_u16(<8 x i16> %a) {
; CHECKLABEL: test_vcvtq_n_f16_u16:
; CHECK:         vcvt.f16.u16 q0, q0, #2
; CHECK-NEXT:    bx lr
entry:
  %vcvt_n1 = tail call <8 x half> @llvm.arm.neon.vcvtfxu2fp.v8f16.v8i16(<8 x i16> %a, i32 2)
  ret <8 x half> %vcvt_n1
}

declare <8 x half> @llvm.arm.neon.vcvtfxu2fp.v8f16.v8i16(<8 x i16>, i32) #2

define dso_local <4 x i16> @test_vcvt_n_s16_f16(<4 x half> %a) {
; CHECKLABEL: test_vcvt_n_s16_f16:
; CHECK:         vcvt.s16.f16 d0, d0, #2
; CHECK-NEXT:    bx lr
entry:
  %vcvt_n1 = tail call <4 x i16> @llvm.arm.neon.vcvtfp2fxs.v4i16.v4f16(<4 x half> %a, i32 2)
  ret <4 x i16> %vcvt_n1
}

declare <4 x i16> @llvm.arm.neon.vcvtfp2fxs.v4i16.v4f16(<4 x half>, i32) #2

define dso_local <8 x i16> @test_vcvtq_n_s16_f16(<8 x half> %a) {
; CHECKLABEL: test_vcvtq_n_s16_f16:
; CHECK:         vcvt.s16.f16 q0, q0, #2
; CHECK-NEXT:    bx lr
entry:
  %vcvt_n1 = tail call <8 x i16> @llvm.arm.neon.vcvtfp2fxs.v8i16.v8f16(<8 x half> %a, i32 2)
  ret <8 x i16> %vcvt_n1
}

declare <8 x i16> @llvm.arm.neon.vcvtfp2fxs.v8i16.v8f16(<8 x half>, i32) #2

define dso_local <4 x i16> @test_vcvt_n_u16_f16(<4 x half> %a) {
; CHECKLABEL: test_vcvt_n_u16_f16:
; CHECK:         vcvt.u16.f16 d0, d0, #2
; CHECK-NEXT:    bx lr
entry:
  %vcvt_n1 = tail call <4 x i16> @llvm.arm.neon.vcvtfp2fxu.v4i16.v4f16(<4 x half> %a, i32 2)
  ret <4 x i16> %vcvt_n1
}

declare <4 x i16> @llvm.arm.neon.vcvtfp2fxu.v4i16.v4f16(<4 x half>, i32) #2

define dso_local <8 x i16> @test_vcvtq_n_u16_f16(<8 x half> %a) {
; CHECKLABEL: test_vcvtq_n_u16_f16:
; CHECK:         vcvt.u16.f16 q0, q0, #2
; CHECK-NEXT:    bx lr
entry:
  %vcvt_n1 = tail call <8 x i16> @llvm.arm.neon.vcvtfp2fxu.v8i16.v8f16(<8 x half> %a, i32 2)
  ret <8 x i16> %vcvt_n1
}

declare <8 x i16> @llvm.arm.neon.vcvtfp2fxu.v8i16.v8f16(<8 x half>, i32) #2

define dso_local <4 x half> @test_vmax_f16(<4 x half> %a, <4 x half> %b) {
; CHECKLABEL: test_vmax_f16:
; CHECK:         vmax.f16 d0, d0, d1
; CHECK-NEXT:    bx lr
entry:
  %vmax_v2.i = tail call <4 x half> @llvm.arm.neon.vmaxs.v4f16(<4 x half> %a, <4 x half> %b)
  ret <4 x half> %vmax_v2.i
}

define dso_local <8 x half> @test_vmaxq_f16(<8 x half> %a, <8 x half> %b) {
; CHECKLABEL: test_vmaxq_f16:
; CHECK:         vmax.f16 q0, q0, q1
; CHECK-NEXT:    bx lr
entry:
  %vmaxq_v2.i = tail call <8 x half> @llvm.arm.neon.vmaxs.v8f16(<8 x half> %a, <8 x half> %b)
  ret <8 x half> %vmaxq_v2.i
}

define dso_local <4 x half> @test_vmaxnm_f16(<4 x half> %a, <4 x half> %b) {
; CHECK-LABEL: test_vmaxnm_f16:
; CHECK:         vmaxnm.f16 d0, d0, d1
; CHECK-NEXT:    bx lr
entry:
  %vmaxnm_v2.i = tail call <4 x half> @llvm.arm.neon.vmaxnm.v4f16(<4 x half> %a, <4 x half> %b)
  ret <4 x half> %vmaxnm_v2.i
}

define dso_local <8 x half> @test_vmaxnmq_f16(<8 x half> %a, <8 x half> %b) {
; CHECK-LABEL: test_vmaxnmq_f16:
; CHECK:         vmaxnm.f16 q0, q0, q1
; CHECK-NEXT:    bx lr
entry:
  %vmaxnmq_v2.i = tail call <8 x half> @llvm.arm.neon.vmaxnm.v8f16(<8 x half> %a, <8 x half> %b)
  ret <8 x half> %vmaxnmq_v2.i
}

define dso_local <4 x half> @test_vmin_f16(<4 x half> %a, <4 x half> %b) {
; CHECK-LABEL: test_vmin_f16:
; CHECK:         vmin.f16 d0, d0, d1
; CHECK-NEXT:    bx lr
entry:
  %vmin_v2.i = tail call <4 x half> @llvm.arm.neon.vmins.v4f16(<4 x half> %a, <4 x half> %b)
  ret <4 x half> %vmin_v2.i
}

define dso_local <8 x half> @test_vminq_f16(<8 x half> %a, <8 x half> %b) {
; CHECK-LABEL: test_vminq_f16:
; CHECK:         vmin.f16 q0, q0, q1
; CHECK-NEXT:    bx lr
entry:
  %vminq_v2.i = tail call <8 x half> @llvm.arm.neon.vmins.v8f16(<8 x half> %a, <8 x half> %b)
  ret <8 x half> %vminq_v2.i
}

define dso_local <4 x half> @test_vminnm_f16(<4 x half> %a, <4 x half> %b) {
; CHECK-LABEL: test_vminnm_f16:
; CHECK:         vminnm.f16 d0, d0, d1
; CHECK-NEXT:    bx lr
entry:
  %vminnm_v2.i = tail call <4 x half> @llvm.arm.neon.vminnm.v4f16(<4 x half> %a, <4 x half> %b)
  ret <4 x half> %vminnm_v2.i
}

define dso_local <8 x half> @test_vminnmq_f16(<8 x half> %a, <8 x half> %b) {
; CHECK-LABEL: test_vminnmq_f16:
; CHECK:         vminnm.f16 q0, q0, q1
; CHECK-NEXT:    bx lr
entry:
  %vminnmq_v2.i = tail call <8 x half> @llvm.arm.neon.vminnm.v8f16(<8 x half> %a, <8 x half> %b)
  ret <8 x half> %vminnmq_v2.i
}

define dso_local <4 x half> @test_vmul_f16(<4 x half> %a, <4 x half> %b) {
; CHECKLABEL: test_vmul_f16:
; CHECK:         vmul.f16 d0, d0, d1
; CHECK-NEXT:    bx lr
entry:
  %mul.i = fmul <4 x half> %a, %b
  ret <4 x half> %mul.i
}

define dso_local <8 x half> @test_vmulq_f16(<8 x half> %a, <8 x half> %b) {
; CHECKLABEL: test_vmulq_f16:
; CHECK:         vmul.f16 q0, q0, q1
; CHECK-NEXT:    bx lr
entry:
  %mul.i = fmul <8 x half> %a, %b
  ret <8 x half> %mul.i
}

define dso_local <4 x half> @test_vpadd_f16(<4 x half> %a, <4 x half> %b) {
; CHECKLABEL: test_vpadd_f16:
; CHECK:         vpadd.f16 d0, d0, d1
; CHECK-NEXT:    bx lr
entry:
  %vpadd_v2.i = tail call <4 x half> @llvm.arm.neon.vpadd.v4f16(<4 x half> %a, <4 x half> %b)
  ret <4 x half> %vpadd_v2.i
}

define dso_local <4 x half> @test_vpmax_f16(<4 x half> %a, <4 x half> %b) {
; CHECKLABEL: test_vpmax_f16:
; CHECK:         vpmax.f16 d0, d0, d1
; CHECK-NEXT:    bx lr
entry:
  %vpmax_v2.i = tail call <4 x half> @llvm.arm.neon.vpmaxs.v4f16(<4 x half> %a, <4 x half> %b)
  ret <4 x half> %vpmax_v2.i
}

define dso_local <4 x half> @test_vpmin_f16(<4 x half> %a, <4 x half> %b) {
; CHECKLABEL: test_vpmin_f16:
; CHECK:         vpmin.f16 d0, d0, d1
; CHECK-NEXT:    bx lr
entry:
  %vpmin_v2.i = tail call <4 x half> @llvm.arm.neon.vpmins.v4f16(<4 x half> %a, <4 x half> %b)
  ret <4 x half> %vpmin_v2.i
}

define dso_local <4 x half> @test_vrecps_f16(<4 x half> %a, <4 x half> %b) {
; CHECKLABEL: test_vrecps_f16:
; CHECK:         vrecps.f16 d0, d0, d1
; CHECK-NEXT:    bx lr
entry:
  %vrecps_v2.i = tail call <4 x half> @llvm.arm.neon.vrecps.v4f16(<4 x half> %a, <4 x half> %b)
  ret <4 x half> %vrecps_v2.i
}

define dso_local <8 x half> @test_vrecpsq_f16(<8 x half> %a, <8 x half> %b) {
; CHECKLABEL: test_vrecpsq_f16:
; CHECK:         vrecps.f16 q0, q0, q1
; CHECK-NEXT:    bx lr
entry:
  %vrecpsq_v2.i = tail call <8 x half> @llvm.arm.neon.vrecps.v8f16(<8 x half> %a, <8 x half> %b)
  ret <8 x half> %vrecpsq_v2.i
}

define dso_local <4 x half> @test_vrsqrts_f16(<4 x half> %a, <4 x half> %b) {
; CHECKLABEL: test_vrsqrts_f16:
; CHECK:         vrsqrts.f16 d0, d0, d1
; CHECK-NEXT:    bx lr
entry:
  %vrsqrts_v2.i = tail call <4 x half> @llvm.arm.neon.vrsqrts.v4f16(<4 x half> %a, <4 x half> %b)
  ret <4 x half> %vrsqrts_v2.i
}

define dso_local <8 x half> @test_vrsqrtsq_f16(<8 x half> %a, <8 x half> %b) {
; CHECKLABEL: test_vrsqrtsq_f16:
; CHECK:         vrsqrts.f16 q0, q0, q1
; CHECK-NEXT:    bx lr
entry:
  %vrsqrtsq_v2.i = tail call <8 x half> @llvm.arm.neon.vrsqrts.v8f16(<8 x half> %a, <8 x half> %b)
  ret <8 x half> %vrsqrtsq_v2.i
}

define dso_local <4 x half> @test_vsub_f16(<4 x half> %a, <4 x half> %b) {
; CHECKLABEL: test_vsub_f16:
; CHECK:         vsub.f16 d0, d0, d1
; CHECK-NEXT:    bx lr
entry:
  %sub.i = fsub <4 x half> %a, %b
  ret <4 x half> %sub.i
}

define dso_local <8 x half> @test_vsubq_f16(<8 x half> %a, <8 x half> %b) {
; CHECKLABEL: test_vsubq_f16:
; CHECK:         vsub.f16 q0, q0, q1
; CHECK-NEXT:    bx lr
entry:
  %sub.i = fsub <8 x half> %a, %b
  ret <8 x half> %sub.i
}

define dso_local <4 x half> @test_vfma_f16(<4 x half> %a, <4 x half> %b, <4 x half> %c) {
; CHECK-LABEL: test_vfma_f16:
; CHECK:         vfma.f16 d0, d1, d2
; CHECK-NEXT:    bx lr
entry:
  %0 = tail call <4 x half> @llvm.fma.v4f16(<4 x half> %b, <4 x half> %c, <4 x half> %a)
  ret <4 x half> %0
}

define dso_local <8 x half> @test_vfmaq_f16(<8 x half> %a, <8 x half> %b, <8 x half> %c) {
; CHECK-LABEL: test_vfmaq_f16:
; CHECK:         vfma.f16 q0, q1, q2
; CHECK-NEXT:    bx lr
entry:
  %0 = tail call <8 x half> @llvm.fma.v8f16(<8 x half> %b, <8 x half> %c, <8 x half> %a)
  ret <8 x half> %0
}

define dso_local <4 x half> @test_vfms_f16(<4 x half> %a, <4 x half> %b, <4 x half> %c) {
; CHECK-LABEL: test_vfms_f16:
; CHECK:         vneg.f16 [[D16:d[0-9]+]], d1
; CHECK-NEXT:    vfma.f16 d0, [[D16]], d2
; CHECK-NEXT:    bx lr
entry:
  %sub.i = fsub <4 x half> <half 0xH8000, half 0xH8000, half 0xH8000, half 0xH8000>, %b
  %0 = tail call <4 x half> @llvm.fma.v4f16(<4 x half> %sub.i, <4 x half> %c, <4 x half> %a)
  ret <4 x half> %0
}

define dso_local <8 x half> @test_vfmsq_f16(<8 x half> %a, <8 x half> %b, <8 x half> %c) {
; CHECK-LABEL: test_vfmsq_f16:
; CHECK:         vneg.f16 [[Q8:q[0-9]+]], q1
; CHECK-NEXT:    vfma.f16 q0, [[Q8]], q2
; CHECK-NEXT:    bx lr
entry:
  %sub.i = fsub <8 x half> <half 0xH8000, half 0xH8000, half 0xH8000, half 0xH8000, half 0xH8000, half 0xH8000, half 0xH8000, half 0xH8000>, %b
  %0 = tail call <8 x half> @llvm.fma.v8f16(<8 x half> %sub.i, <8 x half> %c, <8 x half> %a)
  ret <8 x half> %0
}

define dso_local <4 x half> @test_vmul_lane_f16(<4 x half> %a, <4 x half> %b) {
; CHECK-LABEL: test_vmul_lane_f16:
; CHECK:         vmul.f16 d0, d0, d1[3]
; CHECK-NEXT:    bx lr
entry:
  %shuffle = shufflevector <4 x half> %b, <4 x half> undef, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mul = fmul <4 x half> %shuffle, %a
  ret <4 x half> %mul
}

define dso_local <8 x half> @test_vmulq_lane_f16(<8 x half> %a, <4 x half> %b) {
; CHECK-LABEL: test_vmulq_lane_f16:
; CHECK:         vmul.f16 q0, q0, d2[3]
; CHECK-NEXT:    bx lr
entry:
  %shuffle = shufflevector <4 x half> %b, <4 x half> undef, <8 x i32> <i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3>
  %mul = fmul <8 x half> %shuffle, %a
  ret <8 x half> %mul
}

define dso_local <4 x half> @test_vmul_n_f16(<4 x half> %a, float %b.coerce) {
; CHECK-LABEL: test_vmul_n_f16:
; CHECK:         vmul.f16 d0, d0, d1[0]
; CHECK-NEXT:    bx lr
entry:
  %0 = bitcast float %b.coerce to i32
  %tmp.0.extract.trunc = trunc i32 %0 to i16
  %1 = bitcast i16 %tmp.0.extract.trunc to half
  %vecinit = insertelement <4 x half> undef, half %1, i32 0
  %vecinit4 = shufflevector <4 x half> %vecinit, <4 x half> undef, <4 x i32> zeroinitializer
  %mul = fmul <4 x half> %vecinit4, %a
  ret <4 x half> %mul
}

define dso_local <8 x half> @test_vmulq_n_f16(<8 x half> %a, float %b.coerce) {
; CHECK-LABEL: test_vmulq_n_f16:
; CHECK:         vmul.f16 q0, q0, d2[0]
; CHECK-NEXT:    bx lr
entry:
  %0 = bitcast float %b.coerce to i32
  %tmp.0.extract.trunc = trunc i32 %0 to i16
  %1 = bitcast i16 %tmp.0.extract.trunc to half
  %vecinit = insertelement <8 x half> undef, half %1, i32 0
  %vecinit8 = shufflevector <8 x half> %vecinit, <8 x half> undef, <8 x i32> zeroinitializer
  %mul = fmul <8 x half> %vecinit8, %a
  ret <8 x half> %mul
}

define dso_local <4 x half> @test_vbsl_f16(<4 x i16> %a, <4 x half> %b, <4 x half> %c) {
; CHECKLABEL: test_vbsl_f16:
; CHECK:         vbsl d0, d1, d2
; CHECK-NEXT:    bx lr
entry:
  %0 = bitcast <4 x i16> %a to <8 x i8>
  %1 = bitcast <4 x half> %b to <8 x i8>
  %2 = bitcast <4 x half> %c to <8 x i8>
  %vbsl_v.i = tail call <8 x i8> @llvm.arm.neon.vbsl.v8i8(<8 x i8> %0, <8 x i8> %1, <8 x i8> %2)
  %3 = bitcast <8 x i8> %vbsl_v.i to <4 x half>
  ret <4 x half> %3
}

define dso_local <8 x half> @test_vbslq_f16(<8 x i16> %a, <8 x half> %b, <8 x half> %c) {
; CHECKLABEL: test_vbslq_f16:
; CHECK:         vbsl q0, q1, q2
; CHECK-NEXT:    bx lr
entry:
  %0 = bitcast <8 x i16> %a to <16 x i8>
  %1 = bitcast <8 x half> %b to <16 x i8>
  %2 = bitcast <8 x half> %c to <16 x i8>
  %vbslq_v.i = tail call <16 x i8> @llvm.arm.neon.vbsl.v16i8(<16 x i8> %0, <16 x i8> %1, <16 x i8> %2)
  %3 = bitcast <16 x i8> %vbslq_v.i to <8 x half>
  ret <8 x half> %3
}

define dso_local %struct.float16x4x2_t @test_vzip_f16(<4 x half> %a, <4 x half> %b) {
; CHECK-LABEL: test_vzip_f16:
; CHECK:         vzip.16 d0, d1
; CHECK-NEXT:    bx lr
entry:
  %vzip.i = shufflevector <4 x half> %a, <4 x half> %b, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
  %vzip1.i = shufflevector <4 x half> %a, <4 x half> %b, <4 x i32> <i32 2, i32 6, i32 3, i32 7>
  %.fca.0.0.insert = insertvalue %struct.float16x4x2_t undef, <4 x half> %vzip.i, 0, 0
  %.fca.0.1.insert = insertvalue %struct.float16x4x2_t %.fca.0.0.insert, <4 x half> %vzip1.i, 0, 1
  ret %struct.float16x4x2_t %.fca.0.1.insert
}

define dso_local %struct.float16x8x2_t @test_vzipq_f16(<8 x half> %a, <8 x half> %b) {
; CHECK-LABEL: test_vzipq_f16:
; CHECK:         vzip.16 q0, q1
; CHECK-NEXT:    bx lr
entry:
  %vzip.i = shufflevector <8 x half> %a, <8 x half> %b, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
  %vzip1.i = shufflevector <8 x half> %a, <8 x half> %b, <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
  %.fca.0.0.insert = insertvalue %struct.float16x8x2_t undef, <8 x half> %vzip.i, 0, 0
  %.fca.0.1.insert = insertvalue %struct.float16x8x2_t %.fca.0.0.insert, <8 x half> %vzip1.i, 0, 1
  ret %struct.float16x8x2_t %.fca.0.1.insert
}

define dso_local %struct.float16x4x2_t @test_vuzp_f16(<4 x half> %a, <4 x half> %b) {
; CHECK-LABEL: test_vuzp_f16:
; CHECK:         vuzp.16 d0, d1
; CHECK-NEXT:    bx lr
entry:
  %vuzp.i = shufflevector <4 x half> %a, <4 x half> %b, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %vuzp1.i = shufflevector <4 x half> %a, <4 x half> %b, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %.fca.0.0.insert = insertvalue %struct.float16x4x2_t undef, <4 x half> %vuzp.i, 0, 0
  %.fca.0.1.insert = insertvalue %struct.float16x4x2_t %.fca.0.0.insert, <4 x half> %vuzp1.i, 0, 1
  ret %struct.float16x4x2_t %.fca.0.1.insert
}

define dso_local %struct.float16x8x2_t @test_vuzpq_f16(<8 x half> %a, <8 x half> %b) {
; CHECK-LABEL: test_vuzpq_f16:
; CHECK:         vuzp.16 q0, q1
; CHECK-NEXT:    bx lr
entry:
  %vuzp.i = shufflevector <8 x half> %a, <8 x half> %b, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
  %vuzp1.i = shufflevector <8 x half> %a, <8 x half> %b, <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
  %.fca.0.0.insert = insertvalue %struct.float16x8x2_t undef, <8 x half> %vuzp.i, 0, 0
  %.fca.0.1.insert = insertvalue %struct.float16x8x2_t %.fca.0.0.insert, <8 x half> %vuzp1.i, 0, 1
  ret %struct.float16x8x2_t %.fca.0.1.insert
}

define dso_local %struct.float16x4x2_t @test_vtrn_f16(<4 x half> %a, <4 x half> %b) {
; CHECK-LABEL: test_vtrn_f16:
; CHECK:         vtrn.16 d0, d1
; CHECK-NEXT:    bx lr
entry:
  %vtrn.i = shufflevector <4 x half> %a, <4 x half> %b, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
  %vtrn1.i = shufflevector <4 x half> %a, <4 x half> %b, <4 x i32> <i32 1, i32 5, i32 3, i32 7>
  %.fca.0.0.insert = insertvalue %struct.float16x4x2_t undef, <4 x half> %vtrn.i, 0, 0
  %.fca.0.1.insert = insertvalue %struct.float16x4x2_t %.fca.0.0.insert, <4 x half> %vtrn1.i, 0, 1
  ret %struct.float16x4x2_t %.fca.0.1.insert
}

define dso_local %struct.float16x8x2_t @test_vtrnq_f16(<8 x half> %a, <8 x half> %b) {
; CHECK-LABEL: test_vtrnq_f16:
; CHECK:         vtrn.16 q0, q1
; CHECK-NEXT:    bx lr
entry:
  %vtrn.i = shufflevector <8 x half> %a, <8 x half> %b, <8 x i32> <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
  %vtrn1.i = shufflevector <8 x half> %a, <8 x half> %b, <8 x i32> <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
  %.fca.0.0.insert = insertvalue %struct.float16x8x2_t undef, <8 x half> %vtrn.i, 0, 0
  %.fca.0.1.insert = insertvalue %struct.float16x8x2_t %.fca.0.0.insert, <8 x half> %vtrn1.i, 0, 1
  ret %struct.float16x8x2_t %.fca.0.1.insert
}

define dso_local <4 x half> @test_vmov_n_f16(float %a.coerce) {
; CHECK-LABEL: test_vmov_n_f16:
; CHECK:         vdup.16 d0, d0[0]
; CHECK-NEXT:    bx lr
entry:
  %0 = bitcast float %a.coerce to i32
  %tmp.0.extract.trunc = trunc i32 %0 to i16
  %1 = bitcast i16 %tmp.0.extract.trunc to half
  %vecinit = insertelement <4 x half> undef, half %1, i32 0
  %vecinit4 = shufflevector <4 x half> %vecinit, <4 x half> undef, <4 x i32> zeroinitializer
  ret <4 x half> %vecinit4
}

define dso_local <8 x half> @test_vmovq_n_f16(float %a.coerce) {
; CHECK-LABEL: test_vmovq_n_f16:
; CHECK:         vdup.16 q0, d0[0]
; CHECK-NEXT:    bx lr
entry:
  %0 = bitcast float %a.coerce to i32
  %tmp.0.extract.trunc = trunc i32 %0 to i16
  %1 = bitcast i16 %tmp.0.extract.trunc to half
  %vecinit = insertelement <8 x half> undef, half %1, i32 0
  %vecinit8 = shufflevector <8 x half> %vecinit, <8 x half> undef, <8 x i32> zeroinitializer
  ret <8 x half> %vecinit8
}

define dso_local <4 x half> @test_vdup_n_f16(float %a.coerce) {
; CHECK-LABEL: test_vdup_n_f16:
; CHECK:         vdup.16 d0, d0[0]
; CHECK-NEXT:    bx lr
entry:
  %0 = bitcast float %a.coerce to i32
  %tmp.0.extract.trunc = trunc i32 %0 to i16
  %1 = bitcast i16 %tmp.0.extract.trunc to half
  %vecinit = insertelement <4 x half> undef, half %1, i32 0
  %vecinit4 = shufflevector <4 x half> %vecinit, <4 x half> undef, <4 x i32> zeroinitializer
  ret <4 x half> %vecinit4
}

define dso_local <8 x half> @test_vdupq_n_f16(float %a.coerce) {
; CHECK-LABEL: test_vdupq_n_f16:
; CHECK:        vdup.16 q0, d0[0]
; CHECK-NEXT:    bx lr
entry:
  %0 = bitcast float %a.coerce to i32
  %tmp.0.extract.trunc = trunc i32 %0 to i16
  %1 = bitcast i16 %tmp.0.extract.trunc to half
  %vecinit = insertelement <8 x half> undef, half %1, i32 0
  %vecinit8 = shufflevector <8 x half> %vecinit, <8 x half> undef, <8 x i32> zeroinitializer
  ret <8 x half> %vecinit8
}

define dso_local <4 x half> @test_vdup_lane_f16(<4 x half> %a) {
; CHECK-LABEL: test_vdup_lane_f16:
; CHECK:         vdup.32 d0, d0[3]
; CHECK-NEXT:    bx lr
entry:
  %shuffle = shufflevector <4 x half> %a, <4 x half> undef, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  ret <4 x half> %shuffle
}

define dso_local <8 x half> @test_vdupq_lane_f16(<4 x half> %a) {
; CHECK-LABEL: test_vdupq_lane_f16:
; CHECK:         vdup.16 q0, d0[3]
; CHECK-NEXT:    bx lr
entry:
  %shuffle = shufflevector <4 x half> %a, <4 x half> undef, <8 x i32> <i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3>
  ret <8 x half> %shuffle
}

define dso_local <4 x half> @test_vext_f16(<4 x half> %a, <4 x half> %b) {
; CHECK-LABEL: test_vext_f16:
; CHECK:         vext.16 d0, d0, d1, #2
; CHECK-NEXT:    bx lr
entry:
  %vext = shufflevector <4 x half> %a, <4 x half> %b, <4 x i32> <i32 2, i32 3, i32 4, i32 5>
  ret <4 x half> %vext
}

define dso_local <8 x half> @test_vextq_f16(<8 x half> %a, <8 x half> %b) {
; CHECK-LABEL: test_vextq_f16:
; CHECK:         vext.16 q0, q0, q1, #5
; CHECK-NEXT:    bx lr
entry:
  %vext = shufflevector <8 x half> %a, <8 x half> %b, <8 x i32> <i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12>
  ret <8 x half> %vext
}

define dso_local <4 x half> @test_vrev64_f16(<4 x half> %a) {
entry:
  %shuffle.i = shufflevector <4 x half> %a, <4 x half> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  ret <4 x half> %shuffle.i
}

define dso_local <8 x half> @test_vrev64q_f16(<8 x half> %a) {
entry:
  %shuffle.i = shufflevector <8 x half> %a, <8 x half> undef, <8 x i32> <i32 3, i32 2, i32 1, i32 0, i32 7, i32 6, i32 5, i32 4>
  ret <8 x half> %shuffle.i
}

define <4 x half> @test_vld_dup1_4xhalf(half* %b) {
; CHECK-LABEL: test_vld_dup1_4xhalf:
; CHECK:       vld1.16 {d0[]}, [r0:16]
; CHECK-NEXT:  bx      lr

entry:
  %b1 = load half, half* %b, align 2
  %vecinit = insertelement <4 x half> undef, half %b1, i32 0
  %vecinit2 = insertelement <4 x half> %vecinit, half %b1, i32 1
  %vecinit3 = insertelement <4 x half> %vecinit2, half %b1, i32 2
  %vecinit4 = insertelement <4 x half> %vecinit3, half %b1, i32 3
  ret <4 x half> %vecinit4
}

define <8 x half> @test_vld_dup1_8xhalf(half* %b) local_unnamed_addr {
; CHECK-LABEL: test_vld_dup1_8xhalf:
; CHECK:       vld1.16 {d0[], d1[]}, [r0:16]
; CHECK-NEXT:  bx      lr

entry:
  %b1 = load half, half* %b, align 2
  %vecinit = insertelement <8 x half> undef, half %b1, i32 0
  %vecinit8 = shufflevector <8 x half> %vecinit, <8 x half> undef, <8 x i32> zeroinitializer
  ret <8 x half> %vecinit8
}

define <8 x half> @test_shufflevector8xhalf(<4 x half> %a) {
; CHECK-LABEL: test_shufflevector8xhalf:
; CHECK:       vmov.f64        d1, d0
; CHECK-NEXT:  bx      lr

entry:
  %r = shufflevector <4 x half> %a, <4 x half> %a, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  ret <8 x half> %r
}

declare <4 x half> @llvm.fabs.v4f16(<4 x half>)
declare <8 x half> @llvm.fabs.v8f16(<8 x half>)
declare <4 x i16> @llvm.arm.neon.vcvtas.v4i16.v4f16(<4 x half>)
declare <4 x i16> @llvm.arm.neon.vcvtau.v4i16.v4f16(<4 x half>)
declare <8 x i16> @llvm.arm.neon.vcvtas.v8i16.v8f16(<8 x half>)
declare <4 x i16> @llvm.arm.neon.vcvtms.v4i16.v4f16(<4 x half>)
declare <8 x i16> @llvm.arm.neon.vcvtms.v8i16.v8f16(<8 x half>)
declare <4 x i16> @llvm.arm.neon.vcvtmu.v4i16.v4f16(<4 x half>)
declare <8 x i16> @llvm.arm.neon.vcvtmu.v8i16.v8f16(<8 x half>)
declare <4 x i16> @llvm.arm.neon.vcvtns.v4i16.v4f16(<4 x half>)
declare <8 x i16> @llvm.arm.neon.vcvtns.v8i16.v8f16(<8 x half>)
declare <4 x i16> @llvm.arm.neon.vcvtnu.v4i16.v4f16(<4 x half>)
declare <8 x i16> @llvm.arm.neon.vcvtnu.v8i16.v8f16(<8 x half>)
declare <4 x i16> @llvm.arm.neon.vcvtps.v4i16.v4f16(<4 x half>)
declare <8 x i16> @llvm.arm.neon.vcvtps.v8i16.v8f16(<8 x half>)
declare <4 x i16> @llvm.arm.neon.vcvtpu.v4i16.v4f16(<4 x half>)
declare <8 x i16> @llvm.arm.neon.vcvtpu.v8i16.v8f16(<8 x half>)
declare <4 x half> @llvm.arm.neon.vrecpe.v4f16(<4 x half>)
declare <8 x half> @llvm.arm.neon.vrecpe.v8f16(<8 x half>)
declare <4 x half> @llvm.arm.neon.vrintz.v4f16(<4 x half>)
declare <8 x half> @llvm.arm.neon.vrintz.v8f16(<8 x half>)
declare <4 x half> @llvm.arm.neon.vrinta.v4f16(<4 x half>)
declare <8 x half> @llvm.arm.neon.vrinta.v8f16(<8 x half>)
declare <4 x half> @llvm.arm.neon.vrintm.v4f16(<4 x half>)
declare <8 x half> @llvm.arm.neon.vrintm.v8f16(<8 x half>)
declare <4 x half> @llvm.arm.neon.vrintn.v4f16(<4 x half>)
declare <8 x half> @llvm.arm.neon.vrintn.v8f16(<8 x half>)
declare <4 x half> @llvm.arm.neon.vrintp.v4f16(<4 x half>)
declare <8 x half> @llvm.arm.neon.vrintp.v8f16(<8 x half>)
declare <4 x half> @llvm.arm.neon.vrintx.v4f16(<4 x half>)
declare <8 x half> @llvm.arm.neon.vrintx.v8f16(<8 x half>)
declare <4 x half> @llvm.arm.neon.vrsqrte.v4f16(<4 x half>)
declare <8 x half> @llvm.arm.neon.vrsqrte.v8f16(<8 x half>)
declare <4 x half> @llvm.arm.neon.vabds.v4f16(<4 x half>, <4 x half>)
declare <8 x half> @llvm.arm.neon.vabds.v8f16(<8 x half>, <8 x half>)
declare <4 x i16> @llvm.arm.neon.vacge.v4i16.v4f16(<4 x half>, <4 x half>)
declare <8 x i16> @llvm.arm.neon.vacge.v8i16.v8f16(<8 x half>, <8 x half>)
declare <4 x i16> @llvm.arm.neon.vacgt.v4i16.v4f16(<4 x half>, <4 x half>)
declare <8 x i16> @llvm.arm.neon.vacgt.v8i16.v8f16(<8 x half>, <8 x half>)
declare <4 x half> @llvm.arm.neon.vmaxs.v4f16(<4 x half>, <4 x half>)
declare <8 x half> @llvm.arm.neon.vmaxs.v8f16(<8 x half>, <8 x half>)
declare <4 x half> @llvm.arm.neon.vmaxnm.v4f16(<4 x half>, <4 x half>)
declare <8 x half> @llvm.arm.neon.vmaxnm.v8f16(<8 x half>, <8 x half>)
declare <4 x half> @llvm.arm.neon.vmins.v4f16(<4 x half>, <4 x half>)
declare <8 x half> @llvm.arm.neon.vmins.v8f16(<8 x half>, <8 x half>)
declare <4 x half> @llvm.arm.neon.vminnm.v4f16(<4 x half>, <4 x half>)
declare <8 x half> @llvm.arm.neon.vminnm.v8f16(<8 x half>, <8 x half>)
declare <4 x half> @llvm.arm.neon.vpadd.v4f16(<4 x half>, <4 x half>)
declare <4 x half> @llvm.arm.neon.vpmaxs.v4f16(<4 x half>, <4 x half>)
declare <4 x half> @llvm.arm.neon.vpmins.v4f16(<4 x half>, <4 x half>)
declare <4 x half> @llvm.arm.neon.vrecps.v4f16(<4 x half>, <4 x half>)
declare <8 x half> @llvm.arm.neon.vrecps.v8f16(<8 x half>, <8 x half>)
declare <4 x half> @llvm.arm.neon.vrsqrts.v4f16(<4 x half>, <4 x half>)
declare <8 x half> @llvm.arm.neon.vrsqrts.v8f16(<8 x half>, <8 x half>)
declare <4 x half> @llvm.fma.v4f16(<4 x half>, <4 x half>, <4 x half>)
declare <8 x half> @llvm.fma.v8f16(<8 x half>, <8 x half>, <8 x half>)
declare <8 x i8> @llvm.arm.neon.vbsl.v8i8(<8 x i8>, <8 x i8>, <8 x i8>)
declare <16 x i8> @llvm.arm.neon.vbsl.v16i8(<16 x i8>, <16 x i8>, <16 x i8>)
declare { <8 x half>, <8 x half> } @llvm.arm.neon.vld2lane.v8f16.p0i8(i8*, <8 x half>, <8 x half>, i32, i32)
declare { <4 x half>, <4 x half> } @llvm.arm.neon.vld2lane.v4f16.p0i8(i8*, <4 x half>, <4 x half>, i32, i32)
declare { <8 x half>, <8 x half>, <8 x half> } @llvm.arm.neon.vld3lane.v8f16.p0i8(i8*, <8 x half>, <8 x half>, <8 x half>, i32, i32)
declare { <4 x half>, <4 x half>, <4 x half> } @llvm.arm.neon.vld3lane.v4f16.p0i8(i8*, <4 x half>, <4 x half>, <4 x half>, i32, i32)
declare { <8 x half>, <8 x half>, <8 x half>, <8 x half> } @llvm.arm.neon.vld4lane.v8f16.p0i8(i8*, <8 x half>, <8 x half>, <8 x half>, <8 x half>, i32, i32)
declare { <4 x half>, <4 x half>, <4 x half>, <4 x half> } @llvm.arm.neon.vld4lane.v4f16.p0i8(i8*, <4 x half>, <4 x half>, <4 x half>, <4 x half>, i32, i32)
declare void @llvm.arm.neon.vst2lane.p0i8.v8f16(i8*, <8 x half>, <8 x half>, i32, i32)
declare void @llvm.arm.neon.vst2lane.p0i8.v4f16(i8*, <4 x half>, <4 x half>, i32, i32)
declare void @llvm.arm.neon.vst3lane.p0i8.v8f16(i8*, <8 x half>, <8 x half>, <8 x half>, i32, i32)
declare void @llvm.arm.neon.vst3lane.p0i8.v4f16(i8*, <4 x half>, <4 x half>, <4 x half>, i32, i32)
declare void @llvm.arm.neon.vst4lane.p0i8.v8f16(i8*, <8 x half>, <8 x half>, <8 x half>, <8 x half>, i32, i32)
declare void @llvm.arm.neon.vst4lane.p0i8.v4f16(i8*, <4 x half>, <4 x half>, <4 x half>, <4 x half>, i32, i32)

define { <8 x half>, <8 x half> } @test_vld2q_lane_f16(i8*, <8 x half>, <8 x half>) {
; CHECK-LABEL: test_vld2q_lane_f16:
; CHECK:    vld2.16 {d1[3], d3[3]}, [r0]
; CHECK-NEXT:    bx lr
entry:
  %3 = tail call { <8 x half>, <8 x half> } @llvm.arm.neon.vld2lane.v8f16.p0i8(i8* %0, <8 x half> %1, <8 x half> %2, i32 7, i32 2)
  ret { <8 x half>, <8 x half> } %3
}

define { <4 x half>, <4 x half> } @test_vld2_lane_f16(i8*, <4 x half>, <4 x half>) {
; CHECK-LABEL: test_vld2_lane_f16:
; CHECK:       vld2.16 {d0[3], d1[3]}, [r0]
; CHECK-NEXT:  bx lr
entry:
  %3 = tail call { <4 x half>, <4 x half> } @llvm.arm.neon.vld2lane.v4f16.p0i8(i8* %0, <4 x half> %1, <4 x half> %2, i32 3, i32 2)
  ret { <4 x half>, <4 x half> } %3
}

define { <8 x half>, <8 x half>, <8 x half> } @test_vld3q_lane_f16(i8*, <8 x half>, <8 x half>, <8 x half>) {
; CHECK-LABEL: test_vld3q_lane_f16:
; CHECK:       vld3.16 {d1[3], d3[3], d5[3]}, [r0]
; CHECK-NEXT:  bx lr
entry:
  %4 = tail call { <8 x half>, <8 x half>, <8 x half> } @llvm.arm.neon.vld3lane.v8f16.p0i8(i8* %0, <8 x half> %1, <8 x half> %2, <8 x half> %3, i32 7, i32 2)
  ret { <8 x half>, <8 x half>, <8 x half> } %4
}

define { <4 x half>, <4 x half>, <4 x half> } @test_vld3_lane_f16(i8*, <4 x half>, <4 x half>, <4 x half>) {
; CHECK-LABEL: test_vld3_lane_f16:
; CHECK:       vld3.16 {d0[3], d1[3], d2[3]}, [r0]
; CHECK-NEXT:  bx lr
entry:
  %4 = tail call { <4 x half>, <4 x half>, <4 x half> } @llvm.arm.neon.vld3lane.v4f16.p0i8(i8* %0, <4 x half> %1, <4 x half> %2, <4 x half> %3, i32 3, i32 2)
  ret { <4 x half>, <4 x half>, <4 x half> } %4
}
define { <8 x half>, <8 x half>, <8 x half>, <8 x half> } @test_vld4lane_v8f16_p0i8(i8*, <8 x half>, <8 x half>, <8 x half>, <8 x half>) {
; CHECK-LABEL: test_vld4lane_v8f16_p0i8:
; CHECK:       vld4.16 {d1[3], d3[3], d5[3], d7[3]}, [r0]
; CHECK-NEXT:  bx lr
entry:
  %5 = tail call { <8 x half>, <8 x half>, <8 x half>, <8 x half> } @llvm.arm.neon.vld4lane.v8f16.p0i8(i8* %0, <8 x half> %1, <8 x half> %2, <8 x half> %3, <8 x half> %4, i32 7, i32 2)
  ret { <8 x half>, <8 x half>, <8 x half>, <8 x half> } %5
}
define { <4 x half>, <4 x half>, <4 x half>, <4 x half> } @test_vld4lane_v4f16_p0i8(i8*, <4 x half>, <4 x half>, <4 x half>, <4 x half>) {
; CHECK-LABEL: test_vld4lane_v4f16_p0i8:
; CHECK:       vld4.16 {d0[3], d1[3], d2[3], d3[3]}, [r0]
; CHECK-NEXT:  bx lr
entry:
 %5 = tail call { <4 x half>, <4 x half>, <4 x half>, <4 x half> } @llvm.arm.neon.vld4lane.v4f16.p0i8(i8* %0, <4 x half> %1, <4 x half> %2, <4 x half> %3, <4 x half> %4, i32 3, i32 2)
 ret { <4 x half>, <4 x half>, <4 x half>, <4 x half> } %5
}
define void @test_vst2lane_p0i8_v8f16(i8*, <8 x half>, <8 x half>) {
; CHECK-LABEL: test_vst2lane_p0i8_v8f16:
; CHECK:       vst2.16 {d0[0], d2[0]}, [r0]
; CHECK-NEXT:  bx lr
entry:
  tail call void @llvm.arm.neon.vst2lane.p0i8.v8f16(i8* %0, <8 x half> %1, <8 x half> %2, i32 0, i32 1)
  ret void
}
define void @test_vst2lane_p0i8_v4f16(i8*, <4 x half>, <4 x half>) {
; CHECK-LABEL: test_vst2lane_p0i8_v4f16:
; CHECK:       vst2.16 {d0[0], d1[0]}, [r0:32]
; CHECK-NEXT:  bx lr
entry:
  tail call void @llvm.arm.neon.vst2lane.p0i8.v4f16(i8* %0, <4 x half> %1, <4 x half> %2, i32 0, i32 0)
  ret void
}
define void @test_vst3lane_p0i8_v8f16(i8*, <8 x half>, <8 x half>, <8 x half>) {
; CHECK-LABEL: test_vst3lane_p0i8_v8f16:
; CHECK:       vst3.16 {d0[0], d2[0], d4[0]}, [r0]
; CHECK-NEXT:  bx lr
entry:
  tail call void @llvm.arm.neon.vst3lane.p0i8.v8f16(i8* %0, <8 x half> %1, <8 x half> %2, <8 x half> %3, i32 0, i32 0)
  ret void
}
define void @test_vst3lane_p0i8_v4f16(i8*, <4 x half>, <4 x half>, <4 x half>) {
; CHECK-LABEL: test_vst3lane_p0i8_v4f16:
; CHECK:       vst3.16 {d0[0], d1[0], d2[0]}, [r0]
; CHECK-NEXT:  bx lr
entry:
  tail call void @llvm.arm.neon.vst3lane.p0i8.v4f16(i8* %0, <4 x half> %1, <4 x half> %2, <4 x half> %3, i32 0, i32 0)
  ret void
}
define void @test_vst4lane_p0i8_v8f16(i8*, <8 x half>, <8 x half>, <8 x half>, <8 x half>) {
; CHECK-LABEL: test_vst4lane_p0i8_v8f16:
; CHECK:       vst4.16 {d0[0], d2[0], d4[0], d6[0]}, [r0:64]
; CHECK-NEXT:  bx lr
entry:
  tail call void @llvm.arm.neon.vst4lane.p0i8.v8f16(i8* %0, <8 x half> %1, <8 x half> %2, <8 x half> %3, <8 x half> %4, i32 0, i32 0)
  ret void
}
define void @test_vst4lane_p0i8_v4f16(i8*, <4 x half>, <4 x half>, <4 x half>, <4 x half>) {
; CHECK-LABEL: test_vst4lane_p0i8_v4f16:
; CHECK:       vst4.16 {d0[0], d1[0], d2[0], d3[0]}, [r0:64]
; CHECK-NEXT:  bx lr
entry:
  tail call void @llvm.arm.neon.vst4lane.p0i8.v4f16(i8* %0, <4 x half> %1, <4 x half> %2, <4 x half> %3, <4 x half> %4, i32 0, i32 0)
  ret void
}
