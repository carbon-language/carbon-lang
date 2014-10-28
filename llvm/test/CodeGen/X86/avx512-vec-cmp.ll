; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=knl | FileCheck %s

define <16 x float> @test1(<16 x float> %x, <16 x float> %y) nounwind {
; CHECK-LABEL: test1:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vcmpleps %zmm1, %zmm0, %k1
; CHECK-NEXT:    vmovaps %zmm0, %zmm1 {%k1}
; CHECK-NEXT:    vmovaps %zmm1, %zmm0
; CHECK-NEXT:    retq
  %mask = fcmp ole <16 x float> %x, %y
  %max = select <16 x i1> %mask, <16 x float> %x, <16 x float> %y
  ret <16 x float> %max
}

define <8 x double> @test2(<8 x double> %x, <8 x double> %y) nounwind {
; CHECK-LABEL: test2:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vcmplepd %zmm1, %zmm0, %k1
; CHECK-NEXT:    vmovapd %zmm0, %zmm1 {%k1}
; CHECK-NEXT:    vmovaps %zmm1, %zmm0
; CHECK-NEXT:    retq
  %mask = fcmp ole <8 x double> %x, %y
  %max = select <8 x i1> %mask, <8 x double> %x, <8 x double> %y
  ret <8 x double> %max
}

define <16 x i32> @test3(<16 x i32> %x, <16 x i32> %x1, <16 x i32>* %yp) nounwind {
; CHECK-LABEL: test3:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vpcmpeqd (%rdi), %zmm0, %k1
; CHECK-NEXT:    vmovdqa32 %zmm0, %zmm1 {%k1}
; CHECK-NEXT:    vmovaps %zmm1, %zmm0
; CHECK-NEXT:    retq
  %y = load <16 x i32>* %yp, align 4
  %mask = icmp eq <16 x i32> %x, %y
  %max = select <16 x i1> %mask, <16 x i32> %x, <16 x i32> %x1
  ret <16 x i32> %max
}

define <16 x i32> @test4_unsigned(<16 x i32> %x, <16 x i32> %y) nounwind {
; CHECK-LABEL: test4_unsigned:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vpcmpnltud %zmm1, %zmm0, %k1
; CHECK-NEXT:    vmovdqa32 %zmm0, %zmm1 {%k1}
; CHECK-NEXT:    vmovaps %zmm1, %zmm0
; CHECK-NEXT:    retq
  %mask = icmp uge <16 x i32> %x, %y
  %max = select <16 x i1> %mask, <16 x i32> %x, <16 x i32> %y
  ret <16 x i32> %max
}

define <8 x i64> @test5(<8 x i64> %x, <8 x i64> %y) nounwind {
; CHECK-LABEL: test5:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vpcmpeqq %zmm1, %zmm0, %k1
; CHECK-NEXT:    vmovdqa64 %zmm0, %zmm1 {%k1}
; CHECK-NEXT:    vmovaps %zmm1, %zmm0
; CHECK-NEXT:    retq
  %mask = icmp eq <8 x i64> %x, %y
  %max = select <8 x i1> %mask, <8 x i64> %x, <8 x i64> %y
  ret <8 x i64> %max
}

define <8 x i64> @test6_unsigned(<8 x i64> %x, <8 x i64> %y) nounwind {
; CHECK-LABEL: test6_unsigned:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vpcmpnleuq %zmm1, %zmm0, %k1
; CHECK-NEXT:    vmovdqa64 %zmm0, %zmm1 {%k1}
; CHECK-NEXT:    vmovaps %zmm1, %zmm0
; CHECK-NEXT:    retq
  %mask = icmp ugt <8 x i64> %x, %y
  %max = select <8 x i1> %mask, <8 x i64> %x, <8 x i64> %y
  ret <8 x i64> %max
}

define <4 x float> @test7(<4 x float> %a, <4 x float> %b) {
; CHECK-LABEL: test7:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vxorps %xmm2, %xmm2, %xmm2
; CHECK-NEXT:    vcmpltps %xmm2, %xmm0, %xmm2
; CHECK-NEXT:    vblendvps %xmm2, %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %mask = fcmp olt <4 x float> %a, zeroinitializer
  %c = select <4 x i1>%mask, <4 x float>%a, <4 x float>%b
  ret <4 x float>%c
}

define <2 x double> @test8(<2 x double> %a, <2 x double> %b) {
; CHECK-LABEL: test8:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vxorpd %xmm2, %xmm2, %xmm2
; CHECK-NEXT:    vcmpltpd %xmm2, %xmm0, %xmm2
; CHECK-NEXT:    vblendvpd %xmm2, %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %mask = fcmp olt <2 x double> %a, zeroinitializer
  %c = select <2 x i1>%mask, <2 x double>%a, <2 x double>%b
  ret <2 x double>%c
}

define <8 x i32> @test9(<8 x i32> %x, <8 x i32> %y) nounwind {
; CHECK-LABEL: test9:
; CHECK:       ## BB#0:
; CHECK-NEXT:      ## kill: YMM1<def> YMM1<kill> ZMM1<def>
; CHECK-NEXT:      ## kill: YMM0<def> YMM0<kill> ZMM0<def>
; CHECK-NEXT:    vpcmpeqd %zmm1, %zmm0, %k1
; CHECK-NEXT:    vpblendmd %zmm0, %zmm1, %zmm0 {%k1}
; CHECK-NEXT:      ## kill: YMM0<def> YMM0<kill> ZMM0<kill>
; CHECK-NEXT:    retq
  %mask = icmp eq <8 x i32> %x, %y
  %max = select <8 x i1> %mask, <8 x i32> %x, <8 x i32> %y
  ret <8 x i32> %max
}

define <8 x float> @test10(<8 x float> %x, <8 x float> %y) nounwind {
; CHECK-LABEL: test10:
; CHECK:       ## BB#0:
; CHECK-NEXT:      ## kill: YMM1<def> YMM1<kill> ZMM1<def>
; CHECK-NEXT:      ## kill: YMM0<def> YMM0<kill> ZMM0<def>
; CHECK-NEXT:    vcmpeqps %zmm1, %zmm0, %k1
; CHECK-NEXT:    vblendmps %zmm0, %zmm1, %zmm0 {%k1}
; CHECK-NEXT:      ## kill: YMM0<def> YMM0<kill> ZMM0<kill>
; CHECK-NEXT:    retq
  %mask = fcmp oeq <8 x float> %x, %y
  %max = select <8 x i1> %mask, <8 x float> %x, <8 x float> %y
  ret <8 x float> %max
}

define <8 x i32> @test11_unsigned(<8 x i32> %x, <8 x i32> %y) nounwind {
; CHECK-LABEL: test11_unsigned:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vpmaxud %ymm1, %ymm0, %ymm0
; CHECK-NEXT:    retq
  %mask = icmp ugt <8 x i32> %x, %y
  %max = select <8 x i1> %mask, <8 x i32> %x, <8 x i32> %y
  ret <8 x i32> %max
}


define i16 @test12(<16 x i64> %a, <16 x i64> %b) nounwind {
; CHECK-LABEL: test12:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vpcmpeqq %zmm2, %zmm0, %k0
; CHECK-NEXT:    vpcmpeqq %zmm3, %zmm1, %k1
; CHECK-NEXT:    kunpckbw %k0, %k1, %k0
; CHECK-NEXT:    kmovw %k0, %eax
; CHECK-NEXT:      ## kill: AX<def> AX<kill> EAX<kill>
; CHECK-NEXT:    retq
  %res = icmp eq <16 x i64> %a, %b
  %res1 = bitcast <16 x i1> %res to i16
  ret i16 %res1
}

define <16 x i32> @test13(<16 x float>%a, <16 x float>%b)
; CHECK-LABEL: test13:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vcmpeqps %zmm1, %zmm0, %k1
; CHECK-NEXT:    vpbroadcastd {{.*}}(%rip), %zmm0 {%k1} {z}
; CHECK-NEXT:    retq
{
  %cmpvector_i = fcmp oeq <16 x float> %a, %b
  %conv = zext <16 x i1> %cmpvector_i to <16 x i32>
  ret <16 x i32> %conv
}

define <16 x i32> @test14(<16 x i32>%a, <16 x i32>%b) {
; CHECK-LABEL: test14:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vpsubd %zmm1, %zmm0, %zmm1
; CHECK-NEXT:    vpcmpgtd %zmm0, %zmm1, %k0
; CHECK-NEXT:    knotw %k0, %k0
; CHECK-NEXT:    knotw %k0, %k1
; CHECK-NEXT:    vmovdqu32 %zmm1, %zmm0 {%k1} {z}
; CHECK-NEXT:    retq
  %sub_r = sub <16 x i32> %a, %b
  %cmp.i2.i = icmp sgt <16 x i32> %sub_r, %a
  %sext.i3.i = sext <16 x i1> %cmp.i2.i to <16 x i32>
  %mask = icmp eq <16 x i32> %sext.i3.i, zeroinitializer
  %res = select <16 x i1> %mask, <16 x i32> zeroinitializer, <16 x i32> %sub_r
  ret <16 x i32>%res
}

define <8 x i64> @test15(<8 x i64>%a, <8 x i64>%b) {
; CHECK-LABEL: test15:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vpsubq %zmm1, %zmm0, %zmm1
; CHECK-NEXT:    vpcmpgtq %zmm0, %zmm1, %k0
; CHECK-NEXT:    knotw %k0, %k0
; CHECK-NEXT:    knotw %k0, %k1
; CHECK-NEXT:    vmovdqu64 %zmm1, %zmm0 {%k1} {z}
; CHECK-NEXT:    retq
  %sub_r = sub <8 x i64> %a, %b
  %cmp.i2.i = icmp sgt <8 x i64> %sub_r, %a
  %sext.i3.i = sext <8 x i1> %cmp.i2.i to <8 x i64>
  %mask = icmp eq <8 x i64> %sext.i3.i, zeroinitializer
  %res = select <8 x i1> %mask, <8 x i64> zeroinitializer, <8 x i64> %sub_r
  ret <8 x i64>%res
}

define <16 x i32> @test16(<16 x i32> %x, <16 x i32> %y) nounwind {
; CHECK-LABEL: test16:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vpcmpled %zmm0, %zmm1, %k1
; CHECK-NEXT:    vmovdqa32 %zmm0, %zmm1 {%k1}
; CHECK-NEXT:    vmovaps %zmm1, %zmm0
; CHECK-NEXT:    retq
  %mask = icmp sge <16 x i32> %x, %y
  %max = select <16 x i1> %mask, <16 x i32> %x, <16 x i32> %y
  ret <16 x i32> %max
}

define <16 x i32> @test17(<16 x i32> %x, <16 x i32> %x1, <16 x i32>* %y.ptr) nounwind {
; CHECK-LABEL: test17:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vpcmpgtd (%rdi), %zmm0, %k1
; CHECK-NEXT:    vmovdqa32 %zmm0, %zmm1 {%k1}
; CHECK-NEXT:    vmovaps %zmm1, %zmm0
; CHECK-NEXT:    retq
  %y = load <16 x i32>* %y.ptr, align 4
  %mask = icmp sgt <16 x i32> %x, %y
  %max = select <16 x i1> %mask, <16 x i32> %x, <16 x i32> %x1
  ret <16 x i32> %max
}

define <16 x i32> @test18(<16 x i32> %x, <16 x i32> %x1, <16 x i32>* %y.ptr) nounwind {
; CHECK-LABEL: test18:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vpcmpled (%rdi), %zmm0, %k1
; CHECK-NEXT:    vmovdqa32 %zmm0, %zmm1 {%k1}
; CHECK-NEXT:    vmovaps %zmm1, %zmm0
; CHECK-NEXT:    retq
  %y = load <16 x i32>* %y.ptr, align 4
  %mask = icmp sle <16 x i32> %x, %y
  %max = select <16 x i1> %mask, <16 x i32> %x, <16 x i32> %x1
  ret <16 x i32> %max
}

define <16 x i32> @test19(<16 x i32> %x, <16 x i32> %x1, <16 x i32>* %y.ptr) nounwind {
; CHECK-LABEL: test19:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vpcmpleud (%rdi), %zmm0, %k1
; CHECK-NEXT:    vmovdqa32 %zmm0, %zmm1 {%k1}
; CHECK-NEXT:    vmovaps %zmm1, %zmm0
; CHECK-NEXT:    retq
  %y = load <16 x i32>* %y.ptr, align 4
  %mask = icmp ule <16 x i32> %x, %y
  %max = select <16 x i1> %mask, <16 x i32> %x, <16 x i32> %x1
  ret <16 x i32> %max
}

define <16 x i32> @test20(<16 x i32> %x, <16 x i32> %y, <16 x i32> %x1, <16 x i32> %y1) nounwind {
; CHECK-LABEL: test20:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vpcmpeqd %zmm1, %zmm0, %k1
; CHECK-NEXT:    vpcmpeqd %zmm3, %zmm2, %k1 {%k1}
; CHECK-NEXT:    vmovdqa32 %zmm0, %zmm1 {%k1}
; CHECK-NEXT:    vmovaps %zmm1, %zmm0
; CHECK-NEXT:    retq
  %mask1 = icmp eq <16 x i32> %x1, %y1
  %mask0 = icmp eq <16 x i32> %x, %y
  %mask = select <16 x i1> %mask0, <16 x i1> %mask1, <16 x i1> zeroinitializer
  %max = select <16 x i1> %mask, <16 x i32> %x, <16 x i32> %y
  ret <16 x i32> %max
}

define <8 x i64> @test21(<8 x i64> %x, <8 x i64> %y, <8 x i64> %x1, <8 x i64> %y1) nounwind {
; CHECK-LABEL: test21:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vpcmpleq %zmm1, %zmm0, %k1
; CHECK-NEXT:    vpcmpleq %zmm2, %zmm3, %k1 {%k1}
; CHECK-NEXT:    vmovdqa64 %zmm0, %zmm2 {%k1}
; CHECK-NEXT:    vmovaps %zmm2, %zmm0
; CHECK-NEXT:    retq
  %mask1 = icmp sge <8 x i64> %x1, %y1
  %mask0 = icmp sle <8 x i64> %x, %y
  %mask = select <8 x i1> %mask0, <8 x i1> %mask1, <8 x i1> zeroinitializer
  %max = select <8 x i1> %mask, <8 x i64> %x, <8 x i64> %x1
  ret <8 x i64> %max
}

define <8 x i64> @test22(<8 x i64> %x, <8 x i64>* %y.ptr, <8 x i64> %x1, <8 x i64> %y1) nounwind {
; CHECK-LABEL: test22:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vpcmpgtq %zmm2, %zmm1, %k1
; CHECK-NEXT:    vpcmpgtq (%rdi), %zmm0, %k1 {%k1}
; CHECK-NEXT:    vmovdqa64 %zmm0, %zmm1 {%k1}
; CHECK-NEXT:    vmovaps %zmm1, %zmm0
; CHECK-NEXT:    retq
  %mask1 = icmp sgt <8 x i64> %x1, %y1
  %y = load <8 x i64>* %y.ptr, align 4
  %mask0 = icmp sgt <8 x i64> %x, %y
  %mask = select <8 x i1> %mask0, <8 x i1> %mask1, <8 x i1> zeroinitializer
  %max = select <8 x i1> %mask, <8 x i64> %x, <8 x i64> %x1
  ret <8 x i64> %max
}

define <16 x i32> @test23(<16 x i32> %x, <16 x i32>* %y.ptr, <16 x i32> %x1, <16 x i32> %y1) nounwind {
; CHECK-LABEL: test23:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vpcmpled %zmm1, %zmm2, %k1
; CHECK-NEXT:    vpcmpleud (%rdi), %zmm0, %k1 {%k1}
; CHECK-NEXT:    vmovdqa32 %zmm0, %zmm1 {%k1}
; CHECK-NEXT:    vmovaps %zmm1, %zmm0
; CHECK-NEXT:    retq
  %mask1 = icmp sge <16 x i32> %x1, %y1
  %y = load <16 x i32>* %y.ptr, align 4
  %mask0 = icmp ule <16 x i32> %x, %y
  %mask = select <16 x i1> %mask0, <16 x i1> %mask1, <16 x i1> zeroinitializer
  %max = select <16 x i1> %mask, <16 x i32> %x, <16 x i32> %x1
  ret <16 x i32> %max
}

define <8 x i64> @test24(<8 x i64> %x, <8 x i64> %x1, i64* %yb.ptr) nounwind {
; CHECK-LABEL: test24:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vpcmpeqq (%rdi){1to8}, %zmm0, %k1
; CHECK-NEXT:    vmovdqa64 %zmm0, %zmm1 {%k1}
; CHECK-NEXT:    vmovaps %zmm1, %zmm0
; CHECK-NEXT:    retq
  %yb = load i64* %yb.ptr, align 4
  %y.0 = insertelement <8 x i64> undef, i64 %yb, i32 0
  %y = shufflevector <8 x i64> %y.0, <8 x i64> undef, <8 x i32> zeroinitializer
  %mask = icmp eq <8 x i64> %x, %y
  %max = select <8 x i1> %mask, <8 x i64> %x, <8 x i64> %x1
  ret <8 x i64> %max
}

define <16 x i32> @test25(<16 x i32> %x, i32* %yb.ptr, <16 x i32> %x1) nounwind {
; CHECK-LABEL: test25:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vpcmpled (%rdi){1to16}, %zmm0, %k1
; CHECK-NEXT:    vmovdqa32 %zmm0, %zmm1 {%k1}
; CHECK-NEXT:    vmovaps %zmm1, %zmm0
; CHECK-NEXT:    retq
  %yb = load i32* %yb.ptr, align 4
  %y.0 = insertelement <16 x i32> undef, i32 %yb, i32 0
  %y = shufflevector <16 x i32> %y.0, <16 x i32> undef, <16 x i32> zeroinitializer
  %mask = icmp sle <16 x i32> %x, %y
  %max = select <16 x i1> %mask, <16 x i32> %x, <16 x i32> %x1
  ret <16 x i32> %max
}

define <16 x i32> @test26(<16 x i32> %x, i32* %yb.ptr, <16 x i32> %x1, <16 x i32> %y1) nounwind {
; CHECK-LABEL: test26:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vpcmpled %zmm1, %zmm2, %k1
; CHECK-NEXT:    vpcmpgtd (%rdi){1to16}, %zmm0, %k1 {%k1}
; CHECK-NEXT:    vmovdqa32 %zmm0, %zmm1 {%k1}
; CHECK-NEXT:    vmovaps %zmm1, %zmm0
; CHECK-NEXT:    retq
  %mask1 = icmp sge <16 x i32> %x1, %y1
  %yb = load i32* %yb.ptr, align 4
  %y.0 = insertelement <16 x i32> undef, i32 %yb, i32 0
  %y = shufflevector <16 x i32> %y.0, <16 x i32> undef, <16 x i32> zeroinitializer
  %mask0 = icmp sgt <16 x i32> %x, %y
  %mask = select <16 x i1> %mask0, <16 x i1> %mask1, <16 x i1> zeroinitializer
  %max = select <16 x i1> %mask, <16 x i32> %x, <16 x i32> %x1
  ret <16 x i32> %max
}

define <8 x i64> @test27(<8 x i64> %x, i64* %yb.ptr, <8 x i64> %x1, <8 x i64> %y1) nounwind {
; CHECK-LABEL: test27:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vpcmpleq        %zmm1, %zmm2, %k1
; CHECK-NEXT:    vpcmpleq        (%rdi){1to8}, %zmm0, %k1 {%k1}
; CHECK-NEXT:    vmovdqa64 %zmm0, %zmm1 {%k1}
; CHECK-NEXT:    vmovaps %zmm1, %zmm0
; CHECK-NEXT:    retq
  %mask1 = icmp sge <8 x i64> %x1, %y1
  %yb = load i64* %yb.ptr, align 4
  %y.0 = insertelement <8 x i64> undef, i64 %yb, i32 0
  %y = shufflevector <8 x i64> %y.0, <8 x i64> undef, <8 x i32> zeroinitializer
  %mask0 = icmp sle <8 x i64> %x, %y
  %mask = select <8 x i1> %mask0, <8 x i1> %mask1, <8 x i1> zeroinitializer
  %max = select <8 x i1> %mask, <8 x i64> %x, <8 x i64> %x1
  ret <8 x i64> %max
}
