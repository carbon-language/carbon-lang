; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=knl --show-mc-encoding| FileCheck %s

declare i32 @llvm.x86.avx512.kortestz.w(i16, i16) nounwind readnone
; CHECK-LABEL: test_kortestz
; CHECK: kortestw
; CHECK: sete
define i32 @test_kortestz(i16 %a0, i16 %a1) {
  %res = call i32 @llvm.x86.avx512.kortestz.w(i16 %a0, i16 %a1) 
  ret i32 %res
}

declare i32 @llvm.x86.avx512.kortestc.w(i16, i16) nounwind readnone
; CHECK-LABEL: test_kortestc
; CHECK: kortestw
; CHECK: sbbl
define i32 @test_kortestc(i16 %a0, i16 %a1) {
  %res = call i32 @llvm.x86.avx512.kortestc.w(i16 %a0, i16 %a1) 
  ret i32 %res
}

declare i16 @llvm.x86.avx512.kand.w(i16, i16) nounwind readnone
; CHECK-LABEL: test_kand
; CHECK: kandw
; CHECK: kandw
define i16 @test_kand(i16 %a0, i16 %a1) {
  %t1 = call i16 @llvm.x86.avx512.kand.w(i16 %a0, i16 8)
  %t2 = call i16 @llvm.x86.avx512.kand.w(i16 %t1, i16 %a1)
  ret i16 %t2
}

declare i16 @llvm.x86.avx512.knot.w(i16) nounwind readnone
; CHECK-LABEL: test_knot
; CHECK: knotw
define i16 @test_knot(i16 %a0) {
  %res = call i16 @llvm.x86.avx512.knot.w(i16 %a0)
  ret i16 %res
}

declare i16 @llvm.x86.avx512.kunpck.bw(i16, i16) nounwind readnone

; CHECK-LABEL: unpckbw_test
; CHECK: kunpckbw
; CHECK:ret
define i16 @unpckbw_test(i16 %a0, i16 %a1) {
  %res = call i16 @llvm.x86.avx512.kunpck.bw(i16 %a0, i16 %a1)
  ret i16 %res
}

define <16 x float> @test_rcp_ps_512(<16 x float> %a0) {
  ; CHECK: vrcp14ps {{.*}}encoding: [0x62,0xf2,0x7d,0x48,0x4c,0xc0]
  %res = call <16 x float> @llvm.x86.avx512.rcp14.ps.512(<16 x float> %a0, <16 x float> zeroinitializer, i16 -1) ; <<16 x float>> [#uses=1]
  ret <16 x float> %res
}
declare <16 x float> @llvm.x86.avx512.rcp14.ps.512(<16 x float>, <16 x float>, i16) nounwind readnone

define <8 x double> @test_rcp_pd_512(<8 x double> %a0) {
  ; CHECK: vrcp14pd {{.*}}encoding: [0x62,0xf2,0xfd,0x48,0x4c,0xc0]
  %res = call <8 x double> @llvm.x86.avx512.rcp14.pd.512(<8 x double> %a0, <8 x double> zeroinitializer, i8 -1) ; <<8 x double>> [#uses=1]
  ret <8 x double> %res
}
declare <8 x double> @llvm.x86.avx512.rcp14.pd.512(<8 x double>, <8 x double>, i8) nounwind readnone

define <16 x float> @test_rcp28_ps_512(<16 x float> %a0) {
  ; CHECK: vrcp28ps {sae}, {{.*}}encoding: [0x62,0xf2,0x7d,0x18,0xca,0xc0]
  %res = call <16 x float> @llvm.x86.avx512.rcp28.ps(<16 x float> %a0, <16 x float> zeroinitializer, i16 -1, i32 8) ; <<16 x float>> [#uses=1]
  ret <16 x float> %res
}
declare <16 x float> @llvm.x86.avx512.rcp28.ps(<16 x float>, <16 x float>, i16, i32) nounwind readnone

define <8 x double> @test_rcp28_pd_512(<8 x double> %a0) {
  ; CHECK: vrcp28pd {sae}, {{.*}}encoding: [0x62,0xf2,0xfd,0x18,0xca,0xc0]
  %res = call <8 x double> @llvm.x86.avx512.rcp28.pd(<8 x double> %a0, <8 x double> zeroinitializer, i8 -1, i32 8) ; <<8 x double>> [#uses=1]
  ret <8 x double> %res
}
declare <8 x double> @llvm.x86.avx512.rcp28.pd(<8 x double>, <8 x double>, i8, i32) nounwind readnone

declare <8 x double> @llvm.x86.avx512.mask.rndscale.pd.512(<8 x double>, i32, <8 x double>, i8, i32)

define <8 x double> @test7(<8 x double> %a) {
; CHECK: vrndscalepd {{.*}}encoding: [0x62,0xf3,0xfd,0x48,0x09,0xc0,0x0b]
  %res = call <8 x double> @llvm.x86.avx512.mask.rndscale.pd.512(<8 x double> %a, i32 11, <8 x double> %a, i8 -1, i32 4)
  ret <8 x double>%res
}

declare <16 x float> @llvm.x86.avx512.mask.rndscale.ps.512(<16 x float>, i32, <16 x float>, i16, i32)

define <16 x float> @test8(<16 x float> %a) {
; CHECK: vrndscaleps {{.*}}encoding: [0x62,0xf3,0x7d,0x48,0x08,0xc0,0x0b]
  %res = call <16 x float> @llvm.x86.avx512.mask.rndscale.ps.512(<16 x float> %a, i32 11, <16 x float> %a, i16 -1, i32 4)
  ret <16 x float>%res
}

define <16 x float> @test_rsqrt_ps_512(<16 x float> %a0) {
  ; CHECK: vrsqrt14ps {{.*}}encoding: [0x62,0xf2,0x7d,0x48,0x4e,0xc0]
  %res = call <16 x float> @llvm.x86.avx512.rsqrt14.ps.512(<16 x float> %a0, <16 x float> zeroinitializer, i16 -1) ; <<16 x float>> [#uses=1]
  ret <16 x float> %res
}
declare <16 x float> @llvm.x86.avx512.rsqrt14.ps.512(<16 x float>, <16 x float>, i16) nounwind readnone

define <16 x float> @test_rsqrt28_ps_512(<16 x float> %a0) {
  ; CHECK: vrsqrt28ps {sae}, {{.*}}encoding: [0x62,0xf2,0x7d,0x18,0xcc,0xc0]
  %res = call <16 x float> @llvm.x86.avx512.rsqrt28.ps(<16 x float> %a0, <16 x float> zeroinitializer, i16 -1, i32 8) ; <<16 x float>> [#uses=1]
  ret <16 x float> %res
}
declare <16 x float> @llvm.x86.avx512.rsqrt28.ps(<16 x float>, <16 x float>, i16, i32) nounwind readnone

define <4 x float> @test_rsqrt14_ss(<4 x float> %a0) {
  ; CHECK: vrsqrt14ss {{.*}}encoding: [0x62,0xf2,0x7d,0x08,0x4f,0xc0]
  %res = call <4 x float> @llvm.x86.avx512.rsqrt14.ss(<4 x float> %a0, <4 x float> %a0, <4 x float> zeroinitializer, i8 -1) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.avx512.rsqrt14.ss(<4 x float>, <4 x float>, <4 x float>, i8) nounwind readnone

define <4 x float> @test_rsqrt28_ss(<4 x float> %a0) {
  ; CHECK: vrsqrt28ss {sae}, {{.*}}encoding: [0x62,0xf2,0x7d,0x18,0xcd,0xc0]
  %res = call <4 x float> @llvm.x86.avx512.rsqrt28.ss(<4 x float> %a0, <4 x float> %a0, <4 x float> zeroinitializer, i8 -1, i32 8) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.avx512.rsqrt28.ss(<4 x float>, <4 x float>, <4 x float>, i8, i32) nounwind readnone

define <4 x float> @test_rcp14_ss(<4 x float> %a0) {
  ; CHECK: vrcp14ss {{.*}}encoding: [0x62,0xf2,0x7d,0x08,0x4d,0xc0]
  %res = call <4 x float> @llvm.x86.avx512.rcp14.ss(<4 x float> %a0, <4 x float> %a0, <4 x float> zeroinitializer, i8 -1) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.avx512.rcp14.ss(<4 x float>, <4 x float>, <4 x float>, i8) nounwind readnone

define <4 x float> @test_rcp28_ss(<4 x float> %a0) {
  ; CHECK: vrcp28ss {sae}, {{.*}}encoding: [0x62,0xf2,0x7d,0x18,0xcb,0xc0]
  %res = call <4 x float> @llvm.x86.avx512.rcp28.ss(<4 x float> %a0, <4 x float> %a0, <4 x float> zeroinitializer, i8 -1, i32 8) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.avx512.rcp28.ss(<4 x float>, <4 x float>, <4 x float>, i8, i32) nounwind readnone

define <8 x double> @test_sqrt_pd_512(<8 x double> %a0) {
  ; CHECK: vsqrtpd
  %res = call <8 x double> @llvm.x86.avx512.sqrt.pd.512(<8 x double> %a0,  <8 x double> zeroinitializer, i8 -1, i32 4) ; <<8 x double>> [#uses=1]
  ret <8 x double> %res
}
declare <8 x double> @llvm.x86.avx512.sqrt.pd.512(<8 x double>, <8 x double>, i8, i32) nounwind readnone

define <16 x float> @test_sqrt_ps_512(<16 x float> %a0) {
  ; CHECK: vsqrtps
  %res = call <16 x float> @llvm.x86.avx512.sqrt.ps.512(<16 x float> %a0, <16 x float> zeroinitializer, i16 -1, i32 4) ; <<16 x float>> [#uses=1]
  ret <16 x float> %res
}
declare <16 x float> @llvm.x86.avx512.sqrt.ps.512(<16 x float>, <16 x float>, i16, i32) nounwind readnone

define <4 x float> @test_sqrt_ss(<4 x float> %a0, <4 x float> %a1) {
  ; CHECK: vsqrtss {{.*}}encoding: [0x62
  %res = call <4 x float> @llvm.x86.avx512.sqrt.ss(<4 x float> %a0, <4 x float> %a1) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.avx512.sqrt.ss(<4 x float>, <4 x float>) nounwind readnone

define <2 x double> @test_sqrt_sd(<2 x double> %a0, <2 x double> %a1) {
  ; CHECK: vsqrtsd {{.*}}encoding: [0x62
  %res = call <2 x double> @llvm.x86.avx512.sqrt.sd(<2 x double> %a0, <2 x double> %a1) ; <<2 x double>> [#uses=1]
  ret <2 x double> %res
}
declare <2 x double> @llvm.x86.avx512.sqrt.sd(<2 x double>, <2 x double>) nounwind readnone

define i64 @test_x86_sse2_cvtsd2si64(<2 x double> %a0) {
  ; CHECK: vcvtsd2si {{.*}}encoding: [0x62
  %res = call i64 @llvm.x86.sse2.cvtsd2si64(<2 x double> %a0) ; <i64> [#uses=1]
  ret i64 %res
}
declare i64 @llvm.x86.sse2.cvtsd2si64(<2 x double>) nounwind readnone

define <2 x double> @test_x86_sse2_cvtsi642sd(<2 x double> %a0, i64 %a1) {
  ; CHECK: vcvtsi2sdq {{.*}}encoding: [0x62
  %res = call <2 x double> @llvm.x86.sse2.cvtsi642sd(<2 x double> %a0, i64 %a1) ; <<2 x double>> [#uses=1]
  ret <2 x double> %res
}
declare <2 x double> @llvm.x86.sse2.cvtsi642sd(<2 x double>, i64) nounwind readnone

define <2 x double> @test_x86_avx512_cvtusi642sd(<2 x double> %a0, i64 %a1) {
  ; CHECK: vcvtusi2sdq {{.*}}encoding: [0x62
  %res = call <2 x double> @llvm.x86.avx512.cvtusi642sd(<2 x double> %a0, i64 %a1) ; <<2 x double>> [#uses=1]
  ret <2 x double> %res
}
declare <2 x double> @llvm.x86.avx512.cvtusi642sd(<2 x double>, i64) nounwind readnone

define i64 @test_x86_sse2_cvttsd2si64(<2 x double> %a0) {
  ; CHECK: vcvttsd2si {{.*}}encoding: [0x62
  %res = call i64 @llvm.x86.sse2.cvttsd2si64(<2 x double> %a0) ; <i64> [#uses=1]
  ret i64 %res
}
declare i64 @llvm.x86.sse2.cvttsd2si64(<2 x double>) nounwind readnone


define i64 @test_x86_sse_cvtss2si64(<4 x float> %a0) {
  ; CHECK: vcvtss2si {{.*}}encoding: [0x62
  %res = call i64 @llvm.x86.sse.cvtss2si64(<4 x float> %a0) ; <i64> [#uses=1]
  ret i64 %res
}
declare i64 @llvm.x86.sse.cvtss2si64(<4 x float>) nounwind readnone


define <4 x float> @test_x86_sse_cvtsi642ss(<4 x float> %a0, i64 %a1) {
  ; CHECK: vcvtsi2ssq {{.*}}encoding: [0x62
  %res = call <4 x float> @llvm.x86.sse.cvtsi642ss(<4 x float> %a0, i64 %a1) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.sse.cvtsi642ss(<4 x float>, i64) nounwind readnone


define i64 @test_x86_sse_cvttss2si64(<4 x float> %a0) {
  ; CHECK: vcvttss2si {{.*}}encoding: [0x62
  %res = call i64 @llvm.x86.sse.cvttss2si64(<4 x float> %a0) ; <i64> [#uses=1]
  ret i64 %res
}
declare i64 @llvm.x86.sse.cvttss2si64(<4 x float>) nounwind readnone

define i64 @test_x86_avx512_cvtsd2usi64(<2 x double> %a0) {
  ; CHECK: vcvtsd2usi {{.*}}encoding: [0x62
  %res = call i64 @llvm.x86.avx512.cvtsd2usi64(<2 x double> %a0) ; <i64> [#uses=1]
  ret i64 %res
}
declare i64 @llvm.x86.avx512.cvtsd2usi64(<2 x double>) nounwind readnone

define <16 x float> @test_x86_vcvtph2ps_512(<16 x i16> %a0) {
  ; CHECK: vcvtph2ps  %ymm0, %zmm0    ## encoding: [0x62,0xf2,0x7d,0x48,0x13,0xc0]
  %res = call <16 x float> @llvm.x86.avx512.mask.vcvtph2ps.512(<16 x i16> %a0, <16 x float> zeroinitializer, i16 -1, i32 4)
  ret <16 x float> %res
}
declare <16 x float> @llvm.x86.avx512.mask.vcvtph2ps.512(<16 x i16>, <16 x float>, i16, i32) nounwind readonly


define <16 x i16> @test_x86_vcvtps2ph_256(<16 x float> %a0) {
  ; CHECK: vcvtps2ph $2, %zmm0, %ymm0  ## encoding: [0x62,0xf3,0x7d,0x48,0x1d,0xc0,0x02]
  %res = call <16 x i16> @llvm.x86.avx512.mask.vcvtps2ph.512(<16 x float> %a0, i32 2, <16 x i16> zeroinitializer, i16 -1)
  ret <16 x i16> %res
}

declare <16 x i16> @llvm.x86.avx512.mask.vcvtps2ph.512(<16 x float>, i32, <16 x i16>, i16) nounwind readonly

define <16 x float> @test_x86_vbroadcast_ss_512(i8* %a0) {
  ; CHECK: vbroadcastss
  %res = call <16 x float> @llvm.x86.avx512.vbroadcast.ss.512(i8* %a0) ; <<16 x float>> [#uses=1]
  ret <16 x float> %res
}
declare <16 x float> @llvm.x86.avx512.vbroadcast.ss.512(i8*) nounwind readonly

define <8 x double> @test_x86_vbroadcast_sd_512(i8* %a0) {
  ; CHECK: vbroadcastsd
  %res = call <8 x double> @llvm.x86.avx512.vbroadcast.sd.512(i8* %a0) ; <<8 x double>> [#uses=1]
  ret <8 x double> %res
}
declare <8 x double> @llvm.x86.avx512.vbroadcast.sd.512(i8*) nounwind readonly

define <16 x float> @test_x86_vbroadcast_ss_ps_512(<4 x float> %a0) {
  ; CHECK: vbroadcastss
  %res = call <16 x float> @llvm.x86.avx512.vbroadcast.ss.ps.512(<4 x float> %a0) ; <<16 x float>> [#uses=1]
  ret <16 x float> %res
}
declare <16 x float> @llvm.x86.avx512.vbroadcast.ss.ps.512(<4 x float>) nounwind readonly

define <8 x double> @test_x86_vbroadcast_sd_pd_512(<2 x double> %a0) {
  ; CHECK: vbroadcastsd
  %res = call <8 x double> @llvm.x86.avx512.vbroadcast.sd.pd.512(<2 x double> %a0) ; <<8 x double>> [#uses=1]
  ret <8 x double> %res
}
declare <8 x double> @llvm.x86.avx512.vbroadcast.sd.pd.512(<2 x double>) nounwind readonly

define <16 x i32> @test_x86_pbroadcastd_512(<4 x i32>  %a0) {
  ; CHECK: vpbroadcastd
  %res = call <16 x i32> @llvm.x86.avx512.pbroadcastd.512(<4 x i32> %a0) ; <<16 x i32>> [#uses=1]
  ret <16 x i32> %res
}
declare <16 x i32> @llvm.x86.avx512.pbroadcastd.512(<4 x i32>) nounwind readonly

define <16 x i32> @test_x86_pbroadcastd_i32_512(i32  %a0) {
  ; CHECK: vpbroadcastd
  %res = call <16 x i32> @llvm.x86.avx512.pbroadcastd.i32.512(i32 %a0) ; <<16 x i32>> [#uses=1]
  ret <16 x i32> %res
}
declare <16 x i32> @llvm.x86.avx512.pbroadcastd.i32.512(i32) nounwind readonly

define <8 x i64> @test_x86_pbroadcastq_512(<2 x i64> %a0) {
  ; CHECK: vpbroadcastq
  %res = call <8 x i64> @llvm.x86.avx512.pbroadcastq.512(<2 x i64> %a0) ; <<8 x i64>> [#uses=1]
  ret <8 x i64> %res
}
declare <8 x i64> @llvm.x86.avx512.pbroadcastq.512(<2 x i64>) nounwind readonly

define <8 x i64> @test_x86_pbroadcastq_i64_512(i64 %a0) {
  ; CHECK: vpbroadcastq
  %res = call <8 x i64> @llvm.x86.avx512.pbroadcastq.i64.512(i64 %a0) ; <<8 x i64>> [#uses=1]
  ret <8 x i64> %res
}
declare <8 x i64> @llvm.x86.avx512.pbroadcastq.i64.512(i64) nounwind readonly

define <16 x i32> @test_conflict_d(<16 x i32> %a) {
  ; CHECK: movw $-1, %ax
  ; CHECK: vpxor
  ; CHECK: vpconflictd
  %res = call <16 x i32> @llvm.x86.avx512.mask.conflict.d.512(<16 x i32> %a, <16 x i32> zeroinitializer, i16 -1)
  ret <16 x i32> %res
}

declare <16 x i32> @llvm.x86.avx512.mask.conflict.d.512(<16 x i32>, <16 x i32>, i16) nounwind readonly

define <8 x i64> @test_conflict_q(<8 x i64> %a) {
  ; CHECK: movb $-1, %al
  ; CHECK: vpxor
  ; CHECK: vpconflictq
  %res = call <8 x i64> @llvm.x86.avx512.mask.conflict.q.512(<8 x i64> %a, <8 x i64> zeroinitializer, i8 -1)
  ret <8 x i64> %res
}

declare <8 x i64> @llvm.x86.avx512.mask.conflict.q.512(<8 x i64>, <8 x i64>, i8) nounwind readonly

define <16 x i32> @test_maskz_conflict_d(<16 x i32> %a, i16 %mask) {
  ; CHECK: vpconflictd 
  %res = call <16 x i32> @llvm.x86.avx512.mask.conflict.d.512(<16 x i32> %a, <16 x i32> zeroinitializer, i16 %mask)
  ret <16 x i32> %res
}

define <8 x i64> @test_mask_conflict_q(<8 x i64> %a, <8 x i64> %b, i8 %mask) {
  ; CHECK: vpconflictq
  %res = call <8 x i64> @llvm.x86.avx512.mask.conflict.q.512(<8 x i64> %a, <8 x i64> %b, i8 %mask)
  ret <8 x i64> %res
}

define <16 x i32> @test_lzcnt_d(<16 x i32> %a) {
  ; CHECK: movw $-1, %ax
  ; CHECK: vpxor
  ; CHECK: vplzcntd
  %res = call <16 x i32> @llvm.x86.avx512.mask.lzcnt.d.512(<16 x i32> %a, <16 x i32> zeroinitializer, i16 -1)
  ret <16 x i32> %res
}

declare <16 x i32> @llvm.x86.avx512.mask.lzcnt.d.512(<16 x i32>, <16 x i32>, i16) nounwind readonly

define <8 x i64> @test_lzcnt_q(<8 x i64> %a) {
  ; CHECK: movb $-1, %al
  ; CHECK: vpxor
  ; CHECK: vplzcntq
  %res = call <8 x i64> @llvm.x86.avx512.mask.lzcnt.q.512(<8 x i64> %a, <8 x i64> zeroinitializer, i8 -1)
  ret <8 x i64> %res
}

declare <8 x i64> @llvm.x86.avx512.mask.lzcnt.q.512(<8 x i64>, <8 x i64>, i8) nounwind readonly


define <16 x i32> @test_mask_lzcnt_d(<16 x i32> %a, <16 x i32> %b, i16 %mask) {
  ; CHECK: vplzcntd
  %res = call <16 x i32> @llvm.x86.avx512.mask.lzcnt.d.512(<16 x i32> %a, <16 x i32> %b, i16 %mask)
  ret <16 x i32> %res
}

define <8 x i64> @test_mask_lzcnt_q(<8 x i64> %a, <8 x i64> %b, i8 %mask) {
  ; CHECK: vplzcntq
  %res = call <8 x i64> @llvm.x86.avx512.mask.lzcnt.q.512(<8 x i64> %a, <8 x i64> %b, i8 %mask)
  ret <8 x i64> %res
}

define <16 x i32> @test_ctlz_d(<16 x i32> %a) {
  ; CHECK-LABEL: test_ctlz_d
  ; CHECK: vplzcntd
  %res = call <16 x i32> @llvm.ctlz.v16i32(<16 x i32> %a, i1 false)
  ret <16 x i32> %res
}

declare <16 x i32> @llvm.ctlz.v16i32(<16 x i32>, i1) nounwind readonly

define <8 x i64> @test_ctlz_q(<8 x i64> %a) {
  ; CHECK-LABEL: test_ctlz_q
  ; CHECK: vplzcntq
  %res = call <8 x i64> @llvm.ctlz.v8i64(<8 x i64> %a, i1 false)
  ret <8 x i64> %res
}

declare <8 x i64> @llvm.ctlz.v8i64(<8 x i64>, i1) nounwind readonly

define <16 x float> @test_x86_mask_blend_ps_512(i16 %a0, <16 x float> %a1, <16 x float> %a2) {
  ; CHECK: vblendmps
  %res = call <16 x float> @llvm.x86.avx512.mask.blend.ps.512(<16 x float> %a1, <16 x float> %a2, i16 %a0) ; <<16 x float>> [#uses=1]
  ret <16 x float> %res
}

declare <16 x float> @llvm.x86.avx512.mask.blend.ps.512(<16 x float>, <16 x float>, i16) nounwind readonly

define <8 x double> @test_x86_mask_blend_pd_512(i8 %a0, <8 x double> %a1, <8 x double> %a2) {
  ; CHECK: vblendmpd
  %res = call <8 x double> @llvm.x86.avx512.mask.blend.pd.512(<8 x double> %a1, <8 x double> %a2, i8 %a0) ; <<8 x double>> [#uses=1]
  ret <8 x double> %res
}

define <8 x double> @test_x86_mask_blend_pd_512_memop(<8 x double> %a, <8 x double>* %ptr, i8 %mask) {
  ; CHECK-LABEL: test_x86_mask_blend_pd_512_memop
  ; CHECK: vblendmpd (%
  %b = load <8 x double>* %ptr
  %res = call <8 x double> @llvm.x86.avx512.mask.blend.pd.512(<8 x double> %a, <8 x double> %b, i8 %mask) ; <<8 x double>> [#uses=1]
  ret <8 x double> %res
}
declare <8 x double> @llvm.x86.avx512.mask.blend.pd.512(<8 x double>, <8 x double>, i8) nounwind readonly

define <16 x i32> @test_x86_mask_blend_d_512(i16 %a0, <16 x i32> %a1, <16 x i32> %a2) {
  ; CHECK: vpblendmd
  %res = call <16 x i32> @llvm.x86.avx512.mask.blend.d.512(<16 x i32> %a1, <16 x i32> %a2, i16 %a0) ; <<16 x i32>> [#uses=1]
  ret <16 x i32> %res
}
declare <16 x i32> @llvm.x86.avx512.mask.blend.d.512(<16 x i32>, <16 x i32>, i16) nounwind readonly

define <8 x i64> @test_x86_mask_blend_q_512(i8 %a0, <8 x i64> %a1, <8 x i64> %a2) {
  ; CHECK: vpblendmq
  %res = call <8 x i64> @llvm.x86.avx512.mask.blend.q.512(<8 x i64> %a1, <8 x i64> %a2, i8 %a0) ; <<8 x i64>> [#uses=1]
  ret <8 x i64> %res
}
declare <8 x i64> @llvm.x86.avx512.mask.blend.q.512(<8 x i64>, <8 x i64>, i8) nounwind readonly

 define <8 x i32> @test_cvtpd2udq(<8 x double> %a) {
 ;CHECK: vcvtpd2udq {ru-sae}{{.*}}encoding: [0x62,0xf1,0xfc,0x58,0x79,0xc0]
  %res = call <8 x i32> @llvm.x86.avx512.mask.cvtpd2udq.512(<8 x double> %a, <8 x i32>zeroinitializer, i8 -1, i32 2)
  ret <8 x i32>%res
 }
 declare <8 x i32> @llvm.x86.avx512.mask.cvtpd2udq.512(<8 x double>, <8 x i32>, i8, i32)
 
 define <16 x i32> @test_cvtps2udq(<16 x float> %a) {
 ;CHECK: vcvtps2udq {rd-sae}{{.*}}encoding: [0x62,0xf1,0x7c,0x38,0x79,0xc0]
  %res = call <16 x i32> @llvm.x86.avx512.mask.cvtps2udq.512(<16 x float> %a, <16 x i32>zeroinitializer, i16 -1, i32 1)
  ret <16 x i32>%res
 }
 declare <16 x i32> @llvm.x86.avx512.mask.cvtps2udq.512(<16 x float>, <16 x i32>, i16, i32)

 define i16 @test_cmpps(<16 x float> %a, <16 x float> %b) {
 ;CHECK: vcmpleps {sae}{{.*}}encoding: [0x62,0xf1,0x7c,0x18,0xc2,0xc1,0x02]
   %res = call i16 @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> %a, <16 x float> %b, i32 2, i16 -1, i32 8)
   ret i16 %res
 }
 declare i16 @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> , <16 x float> , i32, i16, i32)

 define i8 @test_cmppd(<8 x double> %a, <8 x double> %b) {
 ;CHECK: vcmpneqpd %zmm{{.*}}encoding: [0x62,0xf1,0xfd,0x48,0xc2,0xc1,0x04]
   %res = call i8 @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> %a, <8 x double> %b, i32 4, i8 -1, i32 4)
   ret i8 %res
 }
 declare i8 @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> , <8 x double> , i32, i8, i32)

 ; cvt intrinsics
 define <16 x float> @test_cvtdq2ps(<16 x i32> %a) {
 ;CHECK: vcvtdq2ps {rd-sae}{{.*}}encoding: [0x62,0xf1,0x7c,0x38,0x5b,0xc0]
  %res = call <16 x float> @llvm.x86.avx512.mask.cvtdq2ps.512(<16 x i32> %a, <16 x float>zeroinitializer, i16 -1, i32 1)
  ret <16 x float>%res
 }
 declare <16 x float> @llvm.x86.avx512.mask.cvtdq2ps.512(<16 x i32>, <16 x float>, i16, i32)

 define <16 x float> @test_cvtudq2ps(<16 x i32> %a) {
 ;CHECK: vcvtudq2ps {rd-sae}{{.*}}encoding: [0x62,0xf1,0x7f,0x38,0x7a,0xc0]
  %res = call <16 x float> @llvm.x86.avx512.mask.cvtudq2ps.512(<16 x i32> %a, <16 x float>zeroinitializer, i16 -1, i32 1)
  ret <16 x float>%res
 }
 declare <16 x float> @llvm.x86.avx512.mask.cvtudq2ps.512(<16 x i32>, <16 x float>, i16, i32)

 define <8 x double> @test_cvtdq2pd(<8 x i32> %a) {
 ;CHECK: vcvtdq2pd {{.*}}encoding: [0x62,0xf1,0x7e,0x48,0xe6,0xc0]
  %res = call <8 x double> @llvm.x86.avx512.mask.cvtdq2pd.512(<8 x i32> %a, <8 x double>zeroinitializer, i8 -1)
  ret <8 x double>%res
 }
 declare <8 x double> @llvm.x86.avx512.mask.cvtdq2pd.512(<8 x i32>, <8 x double>, i8)

 define <8 x double> @test_cvtudq2pd(<8 x i32> %a) {
 ;CHECK: vcvtudq2pd {{.*}}encoding: [0x62,0xf1,0x7e,0x48,0x7a,0xc0]
  %res = call <8 x double> @llvm.x86.avx512.mask.cvtudq2pd.512(<8 x i32> %a, <8 x double>zeroinitializer, i8 -1)
  ret <8 x double>%res
 }
 declare <8 x double> @llvm.x86.avx512.mask.cvtudq2pd.512(<8 x i32>, <8 x double>, i8)

 ; fp min - max
define <16 x float> @test_vmaxps(<16 x float> %a0, <16 x float> %a1) {
  ; CHECK: vmaxps
  %res = call <16 x float> @llvm.x86.avx512.mask.max.ps.512(<16 x float> %a0, <16 x float> %a1,
                    <16 x float>zeroinitializer, i16 -1, i32 4)
  ret <16 x float> %res
}
declare <16 x float> @llvm.x86.avx512.mask.max.ps.512(<16 x float>, <16 x float>,
                    <16 x float>, i16, i32)

define <8 x double> @test_vmaxpd(<8 x double> %a0, <8 x double> %a1) {
  ; CHECK: vmaxpd
  %res = call <8 x double> @llvm.x86.avx512.mask.max.pd.512(<8 x double> %a0, <8 x double> %a1,
                    <8 x double>zeroinitializer, i8 -1, i32 4)
  ret <8 x double> %res
}
declare <8 x double> @llvm.x86.avx512.mask.max.pd.512(<8 x double>, <8 x double>,
                    <8 x double>, i8, i32)

define <16 x float> @test_vminps(<16 x float> %a0, <16 x float> %a1) {
  ; CHECK: vminps
  %res = call <16 x float> @llvm.x86.avx512.mask.min.ps.512(<16 x float> %a0, <16 x float> %a1,
                    <16 x float>zeroinitializer, i16 -1, i32 4)
  ret <16 x float> %res
}
declare <16 x float> @llvm.x86.avx512.mask.min.ps.512(<16 x float>, <16 x float>,
                    <16 x float>, i16, i32)

define <8 x double> @test_vminpd(<8 x double> %a0, <8 x double> %a1) {
  ; CHECK: vminpd
  %res = call <8 x double> @llvm.x86.avx512.mask.min.pd.512(<8 x double> %a0, <8 x double> %a1,
                    <8 x double>zeroinitializer, i8 -1, i32 4)
  ret <8 x double> %res
}
declare <8 x double> @llvm.x86.avx512.mask.min.pd.512(<8 x double>, <8 x double>,
                    <8 x double>, i8, i32)

 define <8 x float> @test_cvtpd2ps(<8 x double> %a) {
 ;CHECK: vcvtpd2ps {rd-sae}{{.*}}encoding: [0x62,0xf1,0xfd,0x38,0x5a,0xc0]
  %res = call <8 x float> @llvm.x86.avx512.mask.cvtpd2ps.512(<8 x double> %a, <8 x float>zeroinitializer, i8 -1, i32 1)
  ret <8 x float>%res
 }
 declare <8 x float> @llvm.x86.avx512.mask.cvtpd2ps.512(<8 x double>, <8 x float>, i8, i32)

 define <16 x i32> @test_pabsd(<16 x i32> %a) {
 ;CHECK: vpabsd {{.*}}encoding: [0x62,0xf2,0x7d,0x48,0x1e,0xc0]
 %res = call <16 x i32> @llvm.x86.avx512.mask.pabs.d.512(<16 x i32> %a, <16 x i32>zeroinitializer, i16 -1)
 ret < 16 x i32> %res
 }
 declare <16 x i32> @llvm.x86.avx512.mask.pabs.d.512(<16 x i32>, <16 x i32>, i16)

 define <8 x i64> @test_pabsq(<8 x i64> %a) {
 ;CHECK: vpabsq {{.*}}encoding: [0x62,0xf2,0xfd,0x48,0x1f,0xc0]
 %res = call <8 x i64> @llvm.x86.avx512.mask.pabs.q.512(<8 x i64> %a, <8 x i64>zeroinitializer, i8 -1)
 ret <8 x i64> %res
 }
 declare <8 x i64> @llvm.x86.avx512.mask.pabs.q.512(<8 x i64>, <8 x i64>, i8)

define <8 x i64> @test_vpmaxq(<8 x i64> %a0, <8 x i64> %a1) {
  ; CHECK: vpmaxsq {{.*}}encoding: [0x62,0xf2,0xfd,0x48,0x3d,0xc1]
  %res = call <8 x i64> @llvm.x86.avx512.mask.pmaxs.q.512(<8 x i64> %a0, <8 x i64> %a1,
                    <8 x i64>zeroinitializer, i8 -1)
  ret <8 x i64> %res
}
declare <8 x i64> @llvm.x86.avx512.mask.pmaxs.q.512(<8 x i64>, <8 x i64>, <8 x i64>, i8)

define <16 x i32> @test_vpminud(<16 x i32> %a0, <16 x i32> %a1) {
  ; CHECK: vpminud {{.*}}encoding: [0x62,0xf2,0x7d,0x48,0x3b,0xc1]
  %res = call <16 x i32> @llvm.x86.avx512.mask.pminu.d.512(<16 x i32> %a0, <16 x i32> %a1,
                    <16 x i32>zeroinitializer, i16 -1)
  ret <16 x i32> %res
}
declare <16 x i32> @llvm.x86.avx512.mask.pminu.d.512(<16 x i32>, <16 x i32>, <16 x i32>, i16)

define <16 x i32> @test_vpmaxsd(<16 x i32> %a0, <16 x i32> %a1) {
  ; CHECK: vpmaxsd {{.*}}encoding: [0x62,0xf2,0x7d,0x48,0x3d,0xc1]
  %res = call <16 x i32> @llvm.x86.avx512.mask.pmaxs.d.512(<16 x i32> %a0, <16 x i32> %a1,
                    <16 x i32>zeroinitializer, i16 -1)
  ret <16 x i32> %res
}
declare <16 x i32> @llvm.x86.avx512.mask.pmaxs.d.512(<16 x i32>, <16 x i32>, <16 x i32>, i16)

define <8 x i64> @test_vpmuludq(<16 x i32> %a0, <16 x i32> %a1) {
  ; CHECK: vpmuludq {{.*}}encoding: [0x62,0xf1,0xfd,0x48,0xf4,0xc1]
  %res = call <8 x i64> @llvm.x86.avx512.mask.pmulu.dq.512(<16 x i32> %a0, <16 x i32> %a1,
                    <8 x i64>zeroinitializer, i8 -1)
  ret <8 x i64> %res
}
declare <8 x i64> @llvm.x86.avx512.mask.pmulu.dq.512(<16 x i32>, <16 x i32>, <8 x i64>, i8)

define i8 @test_vptestmq(<8 x i64> %a0, <8 x i64> %a1) {
  ; CHECK: vptestmq {{.*}}encoding: [0x62,0xf2,0xfd,0x48,0x27,0xc1]
  %res = call i8 @llvm.x86.avx512.mask.ptestm.q.512(<8 x i64> %a0, <8 x i64> %a1, i8 -1)
  ret i8 %res
}
declare i8 @llvm.x86.avx512.mask.ptestm.q.512(<8 x i64>, <8 x i64>, i8)

define i16 @test_vptestmd(<16 x i32> %a0, <16 x i32> %a1) {
  ; CHECK: vptestmd {{.*}}encoding: [0x62,0xf2,0x7d,0x48,0x27,0xc1]
  %res = call i16 @llvm.x86.avx512.mask.ptestm.d.512(<16 x i32> %a0, <16 x i32> %a1, i16 -1)
  ret i16 %res
}
declare i16 @llvm.x86.avx512.mask.ptestm.d.512(<16 x i32>, <16 x i32>, i16)

define void @test_store1(<16 x float> %data, i8* %ptr, i16 %mask) {
; CHECK: vmovups {{.*}}encoding: [0x62,0xf1,0x7c,0x49,0x11,0x07]
  call void @llvm.x86.avx512.mask.storeu.ps.512(i8* %ptr, <16 x float> %data, i16 %mask)
  ret void
}

declare void @llvm.x86.avx512.mask.storeu.ps.512(i8*, <16 x float>, i16 )

define void @test_store2(<8 x double> %data, i8* %ptr, i8 %mask) {
; CHECK: vmovupd {{.*}}encoding: [0x62,0xf1,0xfd,0x49,0x11,0x07]
  call void @llvm.x86.avx512.mask.storeu.pd.512(i8* %ptr, <8 x double> %data, i8 %mask)
  ret void
}

declare void @llvm.x86.avx512.mask.storeu.pd.512(i8*, <8 x double>, i8 )

define <16 x float> @test_vpermt2ps(<16 x float>%x, <16 x float>%y, <16 x i32>%perm) {
; CHECK: vpermt2ps {{.*}}encoding: [0x62,0xf2,0x6d,0x48,0x7f,0xc1]
  %res = call <16 x float> @llvm.x86.avx512.mask.vpermt.ps.512(<16 x i32>%perm, <16 x float>%x, <16 x float>%y, i16 -1)
  ret <16 x float> %res
}

define <16 x float> @test_vpermt2ps_mask(<16 x float>%x, <16 x float>%y, <16 x i32>%perm, i16 %mask) {
; CHECK-LABEL: test_vpermt2ps_mask:
; CHECK: vpermt2ps %zmm1, %zmm2, %zmm0 {%k1} ## encoding: [0x62,0xf2,0x6d,0x49,0x7f,0xc1]
  %res = call <16 x float> @llvm.x86.avx512.mask.vpermt.ps.512(<16 x i32>%perm, <16 x float>%x, <16 x float>%y, i16 %mask)
  ret <16 x float> %res
}

declare <16 x float> @llvm.x86.avx512.mask.vpermt.ps.512(<16 x i32>, <16 x float>, <16 x float>, i16)

define <8 x i64> @test_vmovntdqa(i8 *%x) {
; CHECK-LABEL: test_vmovntdqa:
; CHECK: vmovntdqa (%rdi), %zmm0 ## encoding: [0x62,0xf2,0x7d,0x48,0x2a,0x07]
  %res = call <8 x i64> @llvm.x86.avx512.movntdqa(i8* %x)
  ret <8 x i64> %res
}

declare <8 x i64> @llvm.x86.avx512.movntdqa(i8*)

define <8 x i64> @test_valign_q(<8 x i64> %a, <8 x i64> %b) {
; CHECK-LABEL: test_valign_q:
; CHECK: valignq $2, %zmm1, %zmm0, %zmm0
  %res = call <8 x i64> @llvm.x86.avx512.mask.valign.q.512(<8 x i64> %a, <8 x i64> %b, i8 2, <8 x i64> zeroinitializer, i8 -1)
  ret <8 x i64> %res
}

define <8 x i64> @test_mask_valign_q(<8 x i64> %a, <8 x i64> %b, <8 x i64> %src, i8 %mask) {
; CHECK-LABEL: test_mask_valign_q:
; CHECK: valignq $2, %zmm1, %zmm0, %zmm2 {%k1}
  %res = call <8 x i64> @llvm.x86.avx512.mask.valign.q.512(<8 x i64> %a, <8 x i64> %b, i8 2, <8 x i64> %src, i8 %mask)
  ret <8 x i64> %res
}

declare <8 x i64> @llvm.x86.avx512.mask.valign.q.512(<8 x i64>, <8 x i64>, i8, <8 x i64>, i8)

define <16 x i32> @test_maskz_valign_d(<16 x i32> %a, <16 x i32> %b, i16 %mask) {
; CHECK-LABEL: test_maskz_valign_d:
; CHECK: valignd $5, %zmm1, %zmm0, %zmm0 {%k1} {z} ## encoding: [0x62,0xf3,0x7d,0xc9,0x03,0xc1,0x05]
  %res = call <16 x i32> @llvm.x86.avx512.mask.valign.d.512(<16 x i32> %a, <16 x i32> %b, i8 5, <16 x i32> zeroinitializer, i16 %mask)
  ret <16 x i32> %res
}

declare <16 x i32> @llvm.x86.avx512.mask.valign.d.512(<16 x i32>, <16 x i32>, i8, <16 x i32>, i16)

define void @test_mask_store_ss(i8* %ptr, <4 x float> %data, i8 %mask) {
 ; CHECK-LABEL: test_mask_store_ss
 ; CHECK: vmovss %xmm0, (%rdi) {%k1}     ## encoding: [0x62,0xf1,0x7e,0x09,0x11,0x07]
 call void @llvm.x86.avx512.mask.store.ss(i8* %ptr, <4 x float> %data, i8 %mask)
 ret void
}

declare void @llvm.x86.avx512.mask.store.ss(i8*, <4 x float>, i8 )

define i16 @test_pcmpeq_d(<16 x i32> %a, <16 x i32> %b) {
; CHECK-LABEL: test_pcmpeq_d
; CHECK: vpcmpeqd %zmm1, %zmm0, %k0 ##
  %res = call i16 @llvm.x86.avx512.mask.pcmpeq.d.512(<16 x i32> %a, <16 x i32> %b, i16 -1)
  ret i16 %res
}

define i16 @test_mask_pcmpeq_d(<16 x i32> %a, <16 x i32> %b, i16 %mask) {
; CHECK-LABEL: test_mask_pcmpeq_d
; CHECK: vpcmpeqd %zmm1, %zmm0, %k0 {%k1} ##
  %res = call i16 @llvm.x86.avx512.mask.pcmpeq.d.512(<16 x i32> %a, <16 x i32> %b, i16 %mask)
  ret i16 %res
}

declare i16 @llvm.x86.avx512.mask.pcmpeq.d.512(<16 x i32>, <16 x i32>, i16)

define i8 @test_pcmpeq_q(<8 x i64> %a, <8 x i64> %b) {
; CHECK-LABEL: test_pcmpeq_q
; CHECK: vpcmpeqq %zmm1, %zmm0, %k0 ##
  %res = call i8 @llvm.x86.avx512.mask.pcmpeq.q.512(<8 x i64> %a, <8 x i64> %b, i8 -1)
  ret i8 %res
}

define i8 @test_mask_pcmpeq_q(<8 x i64> %a, <8 x i64> %b, i8 %mask) {
; CHECK-LABEL: test_mask_pcmpeq_q
; CHECK: vpcmpeqq %zmm1, %zmm0, %k0 {%k1} ##
  %res = call i8 @llvm.x86.avx512.mask.pcmpeq.q.512(<8 x i64> %a, <8 x i64> %b, i8 %mask)
  ret i8 %res
}

declare i8 @llvm.x86.avx512.mask.pcmpeq.q.512(<8 x i64>, <8 x i64>, i8)

define i16 @test_pcmpgt_d(<16 x i32> %a, <16 x i32> %b) {
; CHECK-LABEL: test_pcmpgt_d
; CHECK: vpcmpgtd %zmm1, %zmm0, %k0 ##
  %res = call i16 @llvm.x86.avx512.mask.pcmpgt.d.512(<16 x i32> %a, <16 x i32> %b, i16 -1)
  ret i16 %res
}

define i16 @test_mask_pcmpgt_d(<16 x i32> %a, <16 x i32> %b, i16 %mask) {
; CHECK-LABEL: test_mask_pcmpgt_d
; CHECK: vpcmpgtd %zmm1, %zmm0, %k0 {%k1} ##
  %res = call i16 @llvm.x86.avx512.mask.pcmpgt.d.512(<16 x i32> %a, <16 x i32> %b, i16 %mask)
  ret i16 %res
}

declare i16 @llvm.x86.avx512.mask.pcmpgt.d.512(<16 x i32>, <16 x i32>, i16)

define i8 @test_pcmpgt_q(<8 x i64> %a, <8 x i64> %b) {
; CHECK-LABEL: test_pcmpgt_q
; CHECK: vpcmpgtq %zmm1, %zmm0, %k0 ##
  %res = call i8 @llvm.x86.avx512.mask.pcmpgt.q.512(<8 x i64> %a, <8 x i64> %b, i8 -1)
  ret i8 %res
}

define i8 @test_mask_pcmpgt_q(<8 x i64> %a, <8 x i64> %b, i8 %mask) {
; CHECK-LABEL: test_mask_pcmpgt_q
; CHECK: vpcmpgtq %zmm1, %zmm0, %k0 {%k1} ##
  %res = call i8 @llvm.x86.avx512.mask.pcmpgt.q.512(<8 x i64> %a, <8 x i64> %b, i8 %mask)
  ret i8 %res
}

declare i8 @llvm.x86.avx512.mask.pcmpgt.q.512(<8 x i64>, <8 x i64>, i8)

define <8 x i16> @test_cmp_d_512(<16 x i32> %a0, <16 x i32> %a1) {
; CHECK_LABEL: test_cmp_d_512
; CHECK: vpcmpeqd %zmm1, %zmm0, %k0 ##
  %res0 = call i16 @llvm.x86.avx512.mask.cmp.d.512(<16 x i32> %a0, <16 x i32> %a1, i32 0, i16 -1)
  %vec0 = insertelement <8 x i16> undef, i16 %res0, i32 0
; CHECK: vpcmpltd %zmm1, %zmm0, %k0 ##
  %res1 = call i16 @llvm.x86.avx512.mask.cmp.d.512(<16 x i32> %a0, <16 x i32> %a1, i32 1, i16 -1)
  %vec1 = insertelement <8 x i16> %vec0, i16 %res1, i32 1
; CHECK: vpcmpled %zmm1, %zmm0, %k0 ##
  %res2 = call i16 @llvm.x86.avx512.mask.cmp.d.512(<16 x i32> %a0, <16 x i32> %a1, i32 2, i16 -1)
  %vec2 = insertelement <8 x i16> %vec1, i16 %res2, i32 2
; CHECK: vpcmpunordd %zmm1, %zmm0, %k0 ##
  %res3 = call i16 @llvm.x86.avx512.mask.cmp.d.512(<16 x i32> %a0, <16 x i32> %a1, i32 3, i16 -1)
  %vec3 = insertelement <8 x i16> %vec2, i16 %res3, i32 3
; CHECK: vpcmpneqd %zmm1, %zmm0, %k0 ##
  %res4 = call i16 @llvm.x86.avx512.mask.cmp.d.512(<16 x i32> %a0, <16 x i32> %a1, i32 4, i16 -1)
  %vec4 = insertelement <8 x i16> %vec3, i16 %res4, i32 4
; CHECK: vpcmpnltd %zmm1, %zmm0, %k0 ##
  %res5 = call i16 @llvm.x86.avx512.mask.cmp.d.512(<16 x i32> %a0, <16 x i32> %a1, i32 5, i16 -1)
  %vec5 = insertelement <8 x i16> %vec4, i16 %res5, i32 5
; CHECK: vpcmpnled %zmm1, %zmm0, %k0 ##
  %res6 = call i16 @llvm.x86.avx512.mask.cmp.d.512(<16 x i32> %a0, <16 x i32> %a1, i32 6, i16 -1)
  %vec6 = insertelement <8 x i16> %vec5, i16 %res6, i32 6
; CHECK: vpcmpordd %zmm1, %zmm0, %k0 ##
  %res7 = call i16 @llvm.x86.avx512.mask.cmp.d.512(<16 x i32> %a0, <16 x i32> %a1, i32 7, i16 -1)
  %vec7 = insertelement <8 x i16> %vec6, i16 %res7, i32 7
  ret <8 x i16> %vec7
}

define <8 x i16> @test_mask_cmp_d_512(<16 x i32> %a0, <16 x i32> %a1, i16 %mask) {
; CHECK_LABEL: test_mask_cmp_d_512
; CHECK: vpcmpeqd %zmm1, %zmm0, %k0 {%k1} ##
  %res0 = call i16 @llvm.x86.avx512.mask.cmp.d.512(<16 x i32> %a0, <16 x i32> %a1, i32 0, i16 %mask)
  %vec0 = insertelement <8 x i16> undef, i16 %res0, i32 0
; CHECK: vpcmpltd %zmm1, %zmm0, %k0 {%k1} ##
  %res1 = call i16 @llvm.x86.avx512.mask.cmp.d.512(<16 x i32> %a0, <16 x i32> %a1, i32 1, i16 %mask)
  %vec1 = insertelement <8 x i16> %vec0, i16 %res1, i32 1
; CHECK: vpcmpled %zmm1, %zmm0, %k0 {%k1} ##
  %res2 = call i16 @llvm.x86.avx512.mask.cmp.d.512(<16 x i32> %a0, <16 x i32> %a1, i32 2, i16 %mask)
  %vec2 = insertelement <8 x i16> %vec1, i16 %res2, i32 2
; CHECK: vpcmpunordd %zmm1, %zmm0, %k0 {%k1} ##
  %res3 = call i16 @llvm.x86.avx512.mask.cmp.d.512(<16 x i32> %a0, <16 x i32> %a1, i32 3, i16 %mask)
  %vec3 = insertelement <8 x i16> %vec2, i16 %res3, i32 3
; CHECK: vpcmpneqd %zmm1, %zmm0, %k0 {%k1} ##
  %res4 = call i16 @llvm.x86.avx512.mask.cmp.d.512(<16 x i32> %a0, <16 x i32> %a1, i32 4, i16 %mask)
  %vec4 = insertelement <8 x i16> %vec3, i16 %res4, i32 4
; CHECK: vpcmpnltd %zmm1, %zmm0, %k0 {%k1} ##
  %res5 = call i16 @llvm.x86.avx512.mask.cmp.d.512(<16 x i32> %a0, <16 x i32> %a1, i32 5, i16 %mask)
  %vec5 = insertelement <8 x i16> %vec4, i16 %res5, i32 5
; CHECK: vpcmpnled %zmm1, %zmm0, %k0 {%k1} ##
  %res6 = call i16 @llvm.x86.avx512.mask.cmp.d.512(<16 x i32> %a0, <16 x i32> %a1, i32 6, i16 %mask)
  %vec6 = insertelement <8 x i16> %vec5, i16 %res6, i32 6
; CHECK: vpcmpordd %zmm1, %zmm0, %k0 {%k1} ##
  %res7 = call i16 @llvm.x86.avx512.mask.cmp.d.512(<16 x i32> %a0, <16 x i32> %a1, i32 7, i16 %mask)
  %vec7 = insertelement <8 x i16> %vec6, i16 %res7, i32 7
  ret <8 x i16> %vec7
}

declare i16 @llvm.x86.avx512.mask.cmp.d.512(<16 x i32>, <16 x i32>, i32, i16) nounwind readnone

define <8 x i16> @test_ucmp_d_512(<16 x i32> %a0, <16 x i32> %a1) {
; CHECK_LABEL: test_ucmp_d_512
; CHECK: vpcmpequd %zmm1, %zmm0, %k0 ##
  %res0 = call i16 @llvm.x86.avx512.mask.ucmp.d.512(<16 x i32> %a0, <16 x i32> %a1, i32 0, i16 -1)
  %vec0 = insertelement <8 x i16> undef, i16 %res0, i32 0
; CHECK: vpcmpltud %zmm1, %zmm0, %k0 ##
  %res1 = call i16 @llvm.x86.avx512.mask.ucmp.d.512(<16 x i32> %a0, <16 x i32> %a1, i32 1, i16 -1)
  %vec1 = insertelement <8 x i16> %vec0, i16 %res1, i32 1
; CHECK: vpcmpleud %zmm1, %zmm0, %k0 ##
  %res2 = call i16 @llvm.x86.avx512.mask.ucmp.d.512(<16 x i32> %a0, <16 x i32> %a1, i32 2, i16 -1)
  %vec2 = insertelement <8 x i16> %vec1, i16 %res2, i32 2
; CHECK: vpcmpunordud %zmm1, %zmm0, %k0 ##
  %res3 = call i16 @llvm.x86.avx512.mask.ucmp.d.512(<16 x i32> %a0, <16 x i32> %a1, i32 3, i16 -1)
  %vec3 = insertelement <8 x i16> %vec2, i16 %res3, i32 3
; CHECK: vpcmpnequd %zmm1, %zmm0, %k0 ##
  %res4 = call i16 @llvm.x86.avx512.mask.ucmp.d.512(<16 x i32> %a0, <16 x i32> %a1, i32 4, i16 -1)
  %vec4 = insertelement <8 x i16> %vec3, i16 %res4, i32 4
; CHECK: vpcmpnltud %zmm1, %zmm0, %k0 ##
  %res5 = call i16 @llvm.x86.avx512.mask.ucmp.d.512(<16 x i32> %a0, <16 x i32> %a1, i32 5, i16 -1)
  %vec5 = insertelement <8 x i16> %vec4, i16 %res5, i32 5
; CHECK: vpcmpnleud %zmm1, %zmm0, %k0 ##
  %res6 = call i16 @llvm.x86.avx512.mask.ucmp.d.512(<16 x i32> %a0, <16 x i32> %a1, i32 6, i16 -1)
  %vec6 = insertelement <8 x i16> %vec5, i16 %res6, i32 6
; CHECK: vpcmpordud %zmm1, %zmm0, %k0 ##
  %res7 = call i16 @llvm.x86.avx512.mask.ucmp.d.512(<16 x i32> %a0, <16 x i32> %a1, i32 7, i16 -1)
  %vec7 = insertelement <8 x i16> %vec6, i16 %res7, i32 7
  ret <8 x i16> %vec7
}

define <8 x i16> @test_mask_ucmp_d_512(<16 x i32> %a0, <16 x i32> %a1, i16 %mask) {
; CHECK_LABEL: test_mask_ucmp_d_512
; CHECK: vpcmpequd %zmm1, %zmm0, %k0 {%k1} ##
  %res0 = call i16 @llvm.x86.avx512.mask.ucmp.d.512(<16 x i32> %a0, <16 x i32> %a1, i32 0, i16 %mask)
  %vec0 = insertelement <8 x i16> undef, i16 %res0, i32 0
; CHECK: vpcmpltud %zmm1, %zmm0, %k0 {%k1} ##
  %res1 = call i16 @llvm.x86.avx512.mask.ucmp.d.512(<16 x i32> %a0, <16 x i32> %a1, i32 1, i16 %mask)
  %vec1 = insertelement <8 x i16> %vec0, i16 %res1, i32 1
; CHECK: vpcmpleud %zmm1, %zmm0, %k0 {%k1} ##
  %res2 = call i16 @llvm.x86.avx512.mask.ucmp.d.512(<16 x i32> %a0, <16 x i32> %a1, i32 2, i16 %mask)
  %vec2 = insertelement <8 x i16> %vec1, i16 %res2, i32 2
; CHECK: vpcmpunordud %zmm1, %zmm0, %k0 {%k1} ##
  %res3 = call i16 @llvm.x86.avx512.mask.ucmp.d.512(<16 x i32> %a0, <16 x i32> %a1, i32 3, i16 %mask)
  %vec3 = insertelement <8 x i16> %vec2, i16 %res3, i32 3
; CHECK: vpcmpnequd %zmm1, %zmm0, %k0 {%k1} ##
  %res4 = call i16 @llvm.x86.avx512.mask.ucmp.d.512(<16 x i32> %a0, <16 x i32> %a1, i32 4, i16 %mask)
  %vec4 = insertelement <8 x i16> %vec3, i16 %res4, i32 4
; CHECK: vpcmpnltud %zmm1, %zmm0, %k0 {%k1} ##
  %res5 = call i16 @llvm.x86.avx512.mask.ucmp.d.512(<16 x i32> %a0, <16 x i32> %a1, i32 5, i16 %mask)
  %vec5 = insertelement <8 x i16> %vec4, i16 %res5, i32 5
; CHECK: vpcmpnleud %zmm1, %zmm0, %k0 {%k1} ##
  %res6 = call i16 @llvm.x86.avx512.mask.ucmp.d.512(<16 x i32> %a0, <16 x i32> %a1, i32 6, i16 %mask)
  %vec6 = insertelement <8 x i16> %vec5, i16 %res6, i32 6
; CHECK: vpcmpordud %zmm1, %zmm0, %k0 {%k1} ##
  %res7 = call i16 @llvm.x86.avx512.mask.ucmp.d.512(<16 x i32> %a0, <16 x i32> %a1, i32 7, i16 %mask)
  %vec7 = insertelement <8 x i16> %vec6, i16 %res7, i32 7
  ret <8 x i16> %vec7
}

declare i16 @llvm.x86.avx512.mask.ucmp.d.512(<16 x i32>, <16 x i32>, i32, i16) nounwind readnone

define <8 x i8> @test_cmp_q_512(<8 x i64> %a0, <8 x i64> %a1) {
; CHECK_LABEL: test_cmp_q_512
; CHECK: vpcmpeqq %zmm1, %zmm0, %k0 ##
  %res0 = call i8 @llvm.x86.avx512.mask.cmp.q.512(<8 x i64> %a0, <8 x i64> %a1, i32 0, i8 -1)
  %vec0 = insertelement <8 x i8> undef, i8 %res0, i32 0
; CHECK: vpcmpltq %zmm1, %zmm0, %k0 ##
  %res1 = call i8 @llvm.x86.avx512.mask.cmp.q.512(<8 x i64> %a0, <8 x i64> %a1, i32 1, i8 -1)
  %vec1 = insertelement <8 x i8> %vec0, i8 %res1, i32 1
; CHECK: vpcmpleq %zmm1, %zmm0, %k0 ##
  %res2 = call i8 @llvm.x86.avx512.mask.cmp.q.512(<8 x i64> %a0, <8 x i64> %a1, i32 2, i8 -1)
  %vec2 = insertelement <8 x i8> %vec1, i8 %res2, i32 2
; CHECK: vpcmpunordq %zmm1, %zmm0, %k0 ##
  %res3 = call i8 @llvm.x86.avx512.mask.cmp.q.512(<8 x i64> %a0, <8 x i64> %a1, i32 3, i8 -1)
  %vec3 = insertelement <8 x i8> %vec2, i8 %res3, i32 3
; CHECK: vpcmpneqq %zmm1, %zmm0, %k0 ##
  %res4 = call i8 @llvm.x86.avx512.mask.cmp.q.512(<8 x i64> %a0, <8 x i64> %a1, i32 4, i8 -1)
  %vec4 = insertelement <8 x i8> %vec3, i8 %res4, i32 4
; CHECK: vpcmpnltq %zmm1, %zmm0, %k0 ##
  %res5 = call i8 @llvm.x86.avx512.mask.cmp.q.512(<8 x i64> %a0, <8 x i64> %a1, i32 5, i8 -1)
  %vec5 = insertelement <8 x i8> %vec4, i8 %res5, i32 5
; CHECK: vpcmpnleq %zmm1, %zmm0, %k0 ##
  %res6 = call i8 @llvm.x86.avx512.mask.cmp.q.512(<8 x i64> %a0, <8 x i64> %a1, i32 6, i8 -1)
  %vec6 = insertelement <8 x i8> %vec5, i8 %res6, i32 6
; CHECK: vpcmpordq %zmm1, %zmm0, %k0 ##
  %res7 = call i8 @llvm.x86.avx512.mask.cmp.q.512(<8 x i64> %a0, <8 x i64> %a1, i32 7, i8 -1)
  %vec7 = insertelement <8 x i8> %vec6, i8 %res7, i32 7
  ret <8 x i8> %vec7
}

define <8 x i8> @test_mask_cmp_q_512(<8 x i64> %a0, <8 x i64> %a1, i8 %mask) {
; CHECK_LABEL: test_mask_cmp_q_512
; CHECK: vpcmpeqq %zmm1, %zmm0, %k0 {%k1} ##
  %res0 = call i8 @llvm.x86.avx512.mask.cmp.q.512(<8 x i64> %a0, <8 x i64> %a1, i32 0, i8 %mask)
  %vec0 = insertelement <8 x i8> undef, i8 %res0, i32 0
; CHECK: vpcmpltq %zmm1, %zmm0, %k0 {%k1} ##
  %res1 = call i8 @llvm.x86.avx512.mask.cmp.q.512(<8 x i64> %a0, <8 x i64> %a1, i32 1, i8 %mask)
  %vec1 = insertelement <8 x i8> %vec0, i8 %res1, i32 1
; CHECK: vpcmpleq %zmm1, %zmm0, %k0 {%k1} ##
  %res2 = call i8 @llvm.x86.avx512.mask.cmp.q.512(<8 x i64> %a0, <8 x i64> %a1, i32 2, i8 %mask)
  %vec2 = insertelement <8 x i8> %vec1, i8 %res2, i32 2
; CHECK: vpcmpunordq %zmm1, %zmm0, %k0 {%k1} ##
  %res3 = call i8 @llvm.x86.avx512.mask.cmp.q.512(<8 x i64> %a0, <8 x i64> %a1, i32 3, i8 %mask)
  %vec3 = insertelement <8 x i8> %vec2, i8 %res3, i32 3
; CHECK: vpcmpneqq %zmm1, %zmm0, %k0 {%k1} ##
  %res4 = call i8 @llvm.x86.avx512.mask.cmp.q.512(<8 x i64> %a0, <8 x i64> %a1, i32 4, i8 %mask)
  %vec4 = insertelement <8 x i8> %vec3, i8 %res4, i32 4
; CHECK: vpcmpnltq %zmm1, %zmm0, %k0 {%k1} ##
  %res5 = call i8 @llvm.x86.avx512.mask.cmp.q.512(<8 x i64> %a0, <8 x i64> %a1, i32 5, i8 %mask)
  %vec5 = insertelement <8 x i8> %vec4, i8 %res5, i32 5
; CHECK: vpcmpnleq %zmm1, %zmm0, %k0 {%k1} ##
  %res6 = call i8 @llvm.x86.avx512.mask.cmp.q.512(<8 x i64> %a0, <8 x i64> %a1, i32 6, i8 %mask)
  %vec6 = insertelement <8 x i8> %vec5, i8 %res6, i32 6
; CHECK: vpcmpordq %zmm1, %zmm0, %k0 {%k1} ##
  %res7 = call i8 @llvm.x86.avx512.mask.cmp.q.512(<8 x i64> %a0, <8 x i64> %a1, i32 7, i8 %mask)
  %vec7 = insertelement <8 x i8> %vec6, i8 %res7, i32 7
  ret <8 x i8> %vec7
}

declare i8 @llvm.x86.avx512.mask.cmp.q.512(<8 x i64>, <8 x i64>, i32, i8) nounwind readnone

define <8 x i8> @test_ucmp_q_512(<8 x i64> %a0, <8 x i64> %a1) {
; CHECK_LABEL: test_ucmp_q_512
; CHECK: vpcmpequq %zmm1, %zmm0, %k0 ##
  %res0 = call i8 @llvm.x86.avx512.mask.ucmp.q.512(<8 x i64> %a0, <8 x i64> %a1, i32 0, i8 -1)
  %vec0 = insertelement <8 x i8> undef, i8 %res0, i32 0
; CHECK: vpcmpltuq %zmm1, %zmm0, %k0 ##
  %res1 = call i8 @llvm.x86.avx512.mask.ucmp.q.512(<8 x i64> %a0, <8 x i64> %a1, i32 1, i8 -1)
  %vec1 = insertelement <8 x i8> %vec0, i8 %res1, i32 1
; CHECK: vpcmpleuq %zmm1, %zmm0, %k0 ##
  %res2 = call i8 @llvm.x86.avx512.mask.ucmp.q.512(<8 x i64> %a0, <8 x i64> %a1, i32 2, i8 -1)
  %vec2 = insertelement <8 x i8> %vec1, i8 %res2, i32 2
; CHECK: vpcmpunorduq %zmm1, %zmm0, %k0 ##
  %res3 = call i8 @llvm.x86.avx512.mask.ucmp.q.512(<8 x i64> %a0, <8 x i64> %a1, i32 3, i8 -1)
  %vec3 = insertelement <8 x i8> %vec2, i8 %res3, i32 3
; CHECK: vpcmpnequq %zmm1, %zmm0, %k0 ##
  %res4 = call i8 @llvm.x86.avx512.mask.ucmp.q.512(<8 x i64> %a0, <8 x i64> %a1, i32 4, i8 -1)
  %vec4 = insertelement <8 x i8> %vec3, i8 %res4, i32 4
; CHECK: vpcmpnltuq %zmm1, %zmm0, %k0 ##
  %res5 = call i8 @llvm.x86.avx512.mask.ucmp.q.512(<8 x i64> %a0, <8 x i64> %a1, i32 5, i8 -1)
  %vec5 = insertelement <8 x i8> %vec4, i8 %res5, i32 5
; CHECK: vpcmpnleuq %zmm1, %zmm0, %k0 ##
  %res6 = call i8 @llvm.x86.avx512.mask.ucmp.q.512(<8 x i64> %a0, <8 x i64> %a1, i32 6, i8 -1)
  %vec6 = insertelement <8 x i8> %vec5, i8 %res6, i32 6
; CHECK: vpcmporduq %zmm1, %zmm0, %k0 ##
  %res7 = call i8 @llvm.x86.avx512.mask.ucmp.q.512(<8 x i64> %a0, <8 x i64> %a1, i32 7, i8 -1)
  %vec7 = insertelement <8 x i8> %vec6, i8 %res7, i32 7
  ret <8 x i8> %vec7
}

define <8 x i8> @test_mask_ucmp_q_512(<8 x i64> %a0, <8 x i64> %a1, i8 %mask) {
; CHECK_LABEL: test_mask_ucmp_q_512
; CHECK: vpcmpequq %zmm1, %zmm0, %k0 {%k1} ##
  %res0 = call i8 @llvm.x86.avx512.mask.ucmp.q.512(<8 x i64> %a0, <8 x i64> %a1, i32 0, i8 %mask)
  %vec0 = insertelement <8 x i8> undef, i8 %res0, i32 0
; CHECK: vpcmpltuq %zmm1, %zmm0, %k0 {%k1} ##
  %res1 = call i8 @llvm.x86.avx512.mask.ucmp.q.512(<8 x i64> %a0, <8 x i64> %a1, i32 1, i8 %mask)
  %vec1 = insertelement <8 x i8> %vec0, i8 %res1, i32 1
; CHECK: vpcmpleuq %zmm1, %zmm0, %k0 {%k1} ##
  %res2 = call i8 @llvm.x86.avx512.mask.ucmp.q.512(<8 x i64> %a0, <8 x i64> %a1, i32 2, i8 %mask)
  %vec2 = insertelement <8 x i8> %vec1, i8 %res2, i32 2
; CHECK: vpcmpunorduq %zmm1, %zmm0, %k0 {%k1} ##
  %res3 = call i8 @llvm.x86.avx512.mask.ucmp.q.512(<8 x i64> %a0, <8 x i64> %a1, i32 3, i8 %mask)
  %vec3 = insertelement <8 x i8> %vec2, i8 %res3, i32 3
; CHECK: vpcmpnequq %zmm1, %zmm0, %k0 {%k1} ##
  %res4 = call i8 @llvm.x86.avx512.mask.ucmp.q.512(<8 x i64> %a0, <8 x i64> %a1, i32 4, i8 %mask)
  %vec4 = insertelement <8 x i8> %vec3, i8 %res4, i32 4
; CHECK: vpcmpnltuq %zmm1, %zmm0, %k0 {%k1} ##
  %res5 = call i8 @llvm.x86.avx512.mask.ucmp.q.512(<8 x i64> %a0, <8 x i64> %a1, i32 5, i8 %mask)
  %vec5 = insertelement <8 x i8> %vec4, i8 %res5, i32 5
; CHECK: vpcmpnleuq %zmm1, %zmm0, %k0 {%k1} ##
  %res6 = call i8 @llvm.x86.avx512.mask.ucmp.q.512(<8 x i64> %a0, <8 x i64> %a1, i32 6, i8 %mask)
  %vec6 = insertelement <8 x i8> %vec5, i8 %res6, i32 6
; CHECK: vpcmporduq %zmm1, %zmm0, %k0 {%k1} ##
  %res7 = call i8 @llvm.x86.avx512.mask.ucmp.q.512(<8 x i64> %a0, <8 x i64> %a1, i32 7, i8 %mask)
  %vec7 = insertelement <8 x i8> %vec6, i8 %res7, i32 7
  ret <8 x i8> %vec7
}

declare i8 @llvm.x86.avx512.mask.ucmp.q.512(<8 x i64>, <8 x i64>, i32, i8) nounwind readnone

define <4 x float> @test_mask_vextractf32x4(<4 x float> %b, <16 x float> %a, i8 %mask) {
; CHECK-LABEL: test_mask_vextractf32x4:
; CHECK: vextractf32x4 $2, %zmm1, %xmm0 {%k1}
  %res = call <4 x float> @llvm.x86.avx512.mask.vextractf32x4.512(<16 x float> %a, i8 2, <4 x float> %b, i8 %mask)
  ret <4 x float> %res
}

declare <4 x float> @llvm.x86.avx512.mask.vextractf32x4.512(<16 x float>, i8, <4 x float>, i8)

define <4 x i64> @test_mask_vextracti64x4(<4 x i64> %b, <8 x i64> %a, i8 %mask) {
; CHECK-LABEL: test_mask_vextracti64x4:
; CHECK: vextracti64x4 $2, %zmm1, %ymm0 {%k1}
  %res = call <4 x i64> @llvm.x86.avx512.mask.vextracti64x4.512(<8 x i64> %a, i8 2, <4 x i64> %b, i8 %mask)
  ret <4 x i64> %res
}

declare <4 x i64> @llvm.x86.avx512.mask.vextracti64x4.512(<8 x i64>, i8, <4 x i64>, i8)

define <4 x i32> @test_maskz_vextracti32x4(<16 x i32> %a, i8 %mask) {
; CHECK-LABEL: test_maskz_vextracti32x4:
; CHECK: vextracti32x4 $2, %zmm0, %xmm0 {%k1} {z}
  %res = call <4 x i32> @llvm.x86.avx512.mask.vextracti32x4.512(<16 x i32> %a, i8 2, <4 x i32> zeroinitializer, i8 %mask)
  ret <4 x i32> %res
}

declare <4 x i32> @llvm.x86.avx512.mask.vextracti32x4.512(<16 x i32>, i8, <4 x i32>, i8)

define <4 x double> @test_vextractf64x4(<8 x double> %a) {
; CHECK-LABEL: test_vextractf64x4:
; CHECK: vextractf64x4 $2, %zmm0, %ymm0 ##
  %res = call <4 x double> @llvm.x86.avx512.mask.vextractf64x4.512(<8 x double> %a, i8 2, <4 x double> zeroinitializer, i8 -1)
  ret <4 x double> %res
}

declare <4 x double> @llvm.x86.avx512.mask.vextractf64x4.512(<8 x double>, i8, <4 x double>, i8)
