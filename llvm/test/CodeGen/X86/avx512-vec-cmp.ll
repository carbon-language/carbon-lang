; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=knl | FileCheck %s

; CHECK-LABEL: test1
; CHECK: vcmpleps
; CHECK: vmovaps
; CHECK: ret
define <16 x float> @test1(<16 x float> %x, <16 x float> %y) nounwind {
  %mask = fcmp ole <16 x float> %x, %y
  %max = select <16 x i1> %mask, <16 x float> %x, <16 x float> %y
  ret <16 x float> %max
}

; CHECK-LABEL: test2
; CHECK: vcmplepd
; CHECK: vmovapd
; CHECK: ret
define <8 x double> @test2(<8 x double> %x, <8 x double> %y) nounwind {
  %mask = fcmp ole <8 x double> %x, %y
  %max = select <8 x i1> %mask, <8 x double> %x, <8 x double> %y
  ret <8 x double> %max
}

; CHECK-LABEL: test3
; CHECK: vpcmpeqd  (%rdi)
; CHECK: vmovdqa32
; CHECK: ret
define <16 x i32> @test3(<16 x i32> %x, <16 x i32> %x1, <16 x i32>* %yp) nounwind {
  %y = load <16 x i32>* %yp, align 4
  %mask = icmp eq <16 x i32> %x, %y
  %max = select <16 x i1> %mask, <16 x i32> %x, <16 x i32> %x1
  ret <16 x i32> %max
}

; CHECK-LABEL: @test4_unsigned
; CHECK: vpcmpnltud
; CHECK: vmovdqa32
; CHECK: ret
define <16 x i32> @test4_unsigned(<16 x i32> %x, <16 x i32> %y) nounwind {
  %mask = icmp uge <16 x i32> %x, %y
  %max = select <16 x i1> %mask, <16 x i32> %x, <16 x i32> %y
  ret <16 x i32> %max
}

; CHECK-LABEL: test5
; CHECK: vpcmpeqq {{.*}}%k1
; CHECK: vmovdqa64 {{.*}}%k1
; CHECK: ret
define <8 x i64> @test5(<8 x i64> %x, <8 x i64> %y) nounwind {
  %mask = icmp eq <8 x i64> %x, %y
  %max = select <8 x i1> %mask, <8 x i64> %x, <8 x i64> %y
  ret <8 x i64> %max
}

; CHECK-LABEL: test6_unsigned
; CHECK: vpcmpnleuq {{.*}}%k1
; CHECK: vmovdqa64 {{.*}}%k1
; CHECK: ret
define <8 x i64> @test6_unsigned(<8 x i64> %x, <8 x i64> %y) nounwind {
  %mask = icmp ugt <8 x i64> %x, %y
  %max = select <8 x i1> %mask, <8 x i64> %x, <8 x i64> %y
  ret <8 x i64> %max
}

; CHECK-LABEL: test7
; CHECK: xor
; CHECK: vcmpltps
; CHECK: vblendvps
; CHECK: ret
define <4 x float> @test7(<4 x float> %a, <4 x float> %b) {
  %mask = fcmp olt <4 x float> %a, zeroinitializer
  %c = select <4 x i1>%mask, <4 x float>%a, <4 x float>%b
  ret <4 x float>%c
}

; CHECK-LABEL: test8
; CHECK: xor
; CHECK: vcmpltpd
; CHECK: vblendvpd
; CHECK: ret
define <2 x double> @test8(<2 x double> %a, <2 x double> %b) {
  %mask = fcmp olt <2 x double> %a, zeroinitializer
  %c = select <2 x i1>%mask, <2 x double>%a, <2 x double>%b
  ret <2 x double>%c
}

; CHECK-LABEL: test9
; CHECK: vpcmpeqd
; CHECK: vpblendmd
; CHECK: ret
define <8 x i32> @test9(<8 x i32> %x, <8 x i32> %y) nounwind {
  %mask = icmp eq <8 x i32> %x, %y
  %max = select <8 x i1> %mask, <8 x i32> %x, <8 x i32> %y
  ret <8 x i32> %max
}

; CHECK-LABEL: test10
; CHECK: vcmpeqps
; CHECK: vblendmps
; CHECK: ret
define <8 x float> @test10(<8 x float> %x, <8 x float> %y) nounwind {
  %mask = fcmp oeq <8 x float> %x, %y
  %max = select <8 x i1> %mask, <8 x float> %x, <8 x float> %y
  ret <8 x float> %max
}

; CHECK-LABEL: test11_unsigned
; CHECK: vpmaxud
; CHECK: ret
define <8 x i32> @test11_unsigned(<8 x i32> %x, <8 x i32> %y) nounwind {
  %mask = icmp ugt <8 x i32> %x, %y
  %max = select <8 x i1> %mask, <8 x i32> %x, <8 x i32> %y
  ret <8 x i32> %max
}

; CHECK-LABEL: test12
; CHECK: vpcmpeqq        %zmm2, %zmm0, [[LO:%k[0-7]]]
; CHECK: vpcmpeqq        %zmm3, %zmm1, [[HI:%k[0-7]]]
; CHECK: kunpckbw        [[LO]], [[HI]], {{%k[0-7]}}

define i16 @test12(<16 x i64> %a, <16 x i64> %b) nounwind {
  %res = icmp eq <16 x i64> %a, %b
  %res1 = bitcast <16 x i1> %res to i16
  ret i16 %res1
}

; CHECK-LABEL: test13
; CHECK: vcmpeqps        %zmm
; CHECK: vpbroadcastd
; CHECK: ret
define <16 x i32> @test13(<16 x float>%a, <16 x float>%b)
{
  %cmpvector_i = fcmp oeq <16 x float> %a, %b
  %conv = zext <16 x i1> %cmpvector_i to <16 x i32>
  ret <16 x i32> %conv
}

; CHECK-LABEL: test14
; CHECK: vpcmp
; CHECK-NOT: vpcmp
; CHECK: vmovdqu32 {{.*}}{%k1} {z}
; CHECK: ret
define <16 x i32> @test14(<16 x i32>%a, <16 x i32>%b) {
  %sub_r = sub <16 x i32> %a, %b
  %cmp.i2.i = icmp sgt <16 x i32> %sub_r, %a
  %sext.i3.i = sext <16 x i1> %cmp.i2.i to <16 x i32>
  %mask = icmp eq <16 x i32> %sext.i3.i, zeroinitializer
  %res = select <16 x i1> %mask, <16 x i32> zeroinitializer, <16 x i32> %sub_r
  ret <16 x i32>%res
}

; CHECK-LABEL: test15
; CHECK: vpcmpgtq
; CHECK-NOT: vpcmp
; CHECK: vmovdqu64 {{.*}}{%k1} {z}
; CHECK: ret
define <8 x i64> @test15(<8 x i64>%a, <8 x i64>%b) {
  %sub_r = sub <8 x i64> %a, %b
  %cmp.i2.i = icmp sgt <8 x i64> %sub_r, %a
  %sext.i3.i = sext <8 x i1> %cmp.i2.i to <8 x i64>
  %mask = icmp eq <8 x i64> %sext.i3.i, zeroinitializer
  %res = select <8 x i1> %mask, <8 x i64> zeroinitializer, <8 x i64> %sub_r
  ret <8 x i64>%res
}

