; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=skx | FileCheck %s

; CHECK-LABEL: test256_1
; CHECK: vpcmpeqb {{.*%k[0-7]}}
; CHECK: vmovdqu8 {{.*}}%k1
; CHECK: ret
define <32 x i8> @test256_1(<32 x i8> %x, <32 x i8> %y) nounwind {
  %mask = icmp eq <32 x i8> %x, %y
  %max = select <32 x i1> %mask, <32 x i8> %x, <32 x i8> %y
  ret <32 x i8> %max
}

; CHECK-LABEL: test256_2
; CHECK: vpcmpgtb {{.*%k[0-7]}}
; CHECK: vmovdqu8 {{.*}}%k1
; CHECK: ret
define <32 x i8> @test256_2(<32 x i8> %x, <32 x i8> %y, <32 x i8> %x1) nounwind {
  %mask = icmp sgt <32 x i8> %x, %y
  %max = select <32 x i1> %mask, <32 x i8> %x, <32 x i8> %x1
  ret <32 x i8> %max
}

; CHECK-LABEL: @test256_3
; CHECK: vpcmplew {{.*%k[0-7]}}
; CHECK: vmovdqu16
; CHECK: ret
define <16 x i16> @test256_3(<16 x i16> %x, <16 x i16> %y, <16 x i16> %x1) nounwind {
  %mask = icmp sge <16 x i16> %x, %y
  %max = select <16 x i1> %mask, <16 x i16> %x1, <16 x i16> %y
  ret <16 x i16> %max
}

; CHECK-LABEL: test256_4
; CHECK: vpcmpnleub {{.*%k[0-7]}}
; CHECK: vmovdqu8 {{.*}}%k1
; CHECK: ret
define <32 x i8> @test256_4(<32 x i8> %x, <32 x i8> %y, <32 x i8> %x1) nounwind {
  %mask = icmp ugt <32 x i8> %x, %y
  %max = select <32 x i1> %mask, <32 x i8> %x, <32 x i8> %x1
  ret <32 x i8> %max
}

; CHECK-LABEL: test256_5
; CHECK: vpcmpeqw  (%rdi){{.*%k[0-7]}}
; CHECK: vmovdqu16
; CHECK: ret
define <16 x i16> @test256_5(<16 x i16> %x, <16 x i16> %x1, <16 x i16>* %yp) nounwind {
  %y = load <16 x i16>* %yp, align 4
  %mask = icmp eq <16 x i16> %x, %y
  %max = select <16 x i1> %mask, <16 x i16> %x, <16 x i16> %x1
  ret <16 x i16> %max
}

; CHECK-LABEL: @test256_6
; CHECK: vpcmpgtw (%rdi){{.*%k[0-7]}}
; CHECK: vmovdqu16
; CHECK: ret
define <16 x i16> @test256_6(<16 x i16> %x, <16 x i16> %x1, <16 x i16>* %y.ptr) nounwind {
  %y = load <16 x i16>* %y.ptr, align 4
  %mask = icmp sgt <16 x i16> %x, %y
  %max = select <16 x i1> %mask, <16 x i16> %x, <16 x i16> %x1
  ret <16 x i16> %max
}

; CHECK-LABEL: @test256_7
; CHECK: vpcmplew (%rdi){{.*%k[0-7]}}
; CHECK: vmovdqu16
; CHECK: ret
define <16 x i16> @test256_7(<16 x i16> %x, <16 x i16> %x1, <16 x i16>* %y.ptr) nounwind {
  %y = load <16 x i16>* %y.ptr, align 4
  %mask = icmp sle <16 x i16> %x, %y
  %max = select <16 x i1> %mask, <16 x i16> %x, <16 x i16> %x1
  ret <16 x i16> %max
}

; CHECK-LABEL: @test256_8
; CHECK: vpcmpleuw (%rdi){{.*%k[0-7]}}
; CHECK: vmovdqu16
; CHECK: ret
define <16 x i16> @test256_8(<16 x i16> %x, <16 x i16> %x1, <16 x i16>* %y.ptr) nounwind {
  %y = load <16 x i16>* %y.ptr, align 4
  %mask = icmp ule <16 x i16> %x, %y
  %max = select <16 x i1> %mask, <16 x i16> %x, <16 x i16> %x1
  ret <16 x i16> %max
}

; CHECK-LABEL: @test256_9
; CHECK: vpcmpeqw %ymm{{.*{%k[1-7]}}}
; CHECK: vmovdqu16
; CHECK: ret
define <16 x i16> @test256_9(<16 x i16> %x, <16 x i16> %y, <16 x i16> %x1, <16 x i16> %y1) nounwind {
  %mask1 = icmp eq <16 x i16> %x1, %y1
  %mask0 = icmp eq <16 x i16> %x, %y
  %mask = select <16 x i1> %mask0, <16 x i1> %mask1, <16 x i1> zeroinitializer
  %max = select <16 x i1> %mask, <16 x i16> %x, <16 x i16> %y
  ret <16 x i16> %max
}

; CHECK-LABEL: @test256_10
; CHECK: vpcmpleb %ymm{{.*{%k[1-7]}}}
; CHECK: vmovdqu8
; CHECK: ret
define <32 x i8> @test256_10(<32 x i8> %x, <32 x i8> %y, <32 x i8> %x1, <32 x i8> %y1) nounwind {
  %mask1 = icmp sge <32 x i8> %x1, %y1
  %mask0 = icmp sle <32 x i8> %x, %y
  %mask = select <32 x i1> %mask0, <32 x i1> %mask1, <32 x i1> zeroinitializer
  %max = select <32 x i1> %mask, <32 x i8> %x, <32 x i8> %x1
  ret <32 x i8> %max
}

; CHECK-LABEL: @test256_11
; CHECK: vpcmpgtb (%rdi){{.*{%k[1-7]}}}
; CHECK: vmovdqu8
; CHECK: ret
define <32 x i8> @test256_11(<32 x i8> %x, <32 x i8>* %y.ptr, <32 x i8> %x1, <32 x i8> %y1) nounwind {
  %mask1 = icmp sgt <32 x i8> %x1, %y1
  %y = load <32 x i8>* %y.ptr, align 4
  %mask0 = icmp sgt <32 x i8> %x, %y
  %mask = select <32 x i1> %mask0, <32 x i1> %mask1, <32 x i1> zeroinitializer
  %max = select <32 x i1> %mask, <32 x i8> %x, <32 x i8> %x1
  ret <32 x i8> %max
}

; CHECK-LABEL: @test256_12
; CHECK: vpcmpleuw (%rdi){{.*{%k[1-7]}}}
; CHECK: vmovdqu16
; CHECK: ret
define <16 x i16> @test256_12(<16 x i16> %x, <16 x i16>* %y.ptr, <16 x i16> %x1, <16 x i16> %y1) nounwind {
  %mask1 = icmp sge <16 x i16> %x1, %y1
  %y = load <16 x i16>* %y.ptr, align 4
  %mask0 = icmp ule <16 x i16> %x, %y
  %mask = select <16 x i1> %mask0, <16 x i1> %mask1, <16 x i1> zeroinitializer
  %max = select <16 x i1> %mask, <16 x i16> %x, <16 x i16> %x1
  ret <16 x i16> %max
}

; CHECK-LABEL: test128_1
; CHECK: vpcmpeqb {{.*%k[0-7]}}
; CHECK: vmovdqu8 {{.*}}%k1
; CHECK: ret
define <16 x i8> @test128_1(<16 x i8> %x, <16 x i8> %y) nounwind {
  %mask = icmp eq <16 x i8> %x, %y
  %max = select <16 x i1> %mask, <16 x i8> %x, <16 x i8> %y
  ret <16 x i8> %max
}

; CHECK-LABEL: test128_2
; CHECK: vpcmpgtb {{.*%k[0-7]}}
; CHECK: vmovdqu8 {{.*}}%k1
; CHECK: ret
define <16 x i8> @test128_2(<16 x i8> %x, <16 x i8> %y, <16 x i8> %x1) nounwind {
  %mask = icmp sgt <16 x i8> %x, %y
  %max = select <16 x i1> %mask, <16 x i8> %x, <16 x i8> %x1
  ret <16 x i8> %max
}

; CHECK-LABEL: @test128_3
; CHECK: vpcmplew {{.*%k[0-7]}}
; CHECK: vmovdqu16
; CHECK: ret
define <8 x i16> @test128_3(<8 x i16> %x, <8 x i16> %y, <8 x i16> %x1) nounwind {
  %mask = icmp sge <8 x i16> %x, %y
  %max = select <8 x i1> %mask, <8 x i16> %x1, <8 x i16> %y
  ret <8 x i16> %max
}

; CHECK-LABEL: test128_4
; CHECK: vpcmpnleub {{.*%k[0-7]}}
; CHECK: vmovdqu8 {{.*}}%k1
; CHECK: ret
define <16 x i8> @test128_4(<16 x i8> %x, <16 x i8> %y, <16 x i8> %x1) nounwind {
  %mask = icmp ugt <16 x i8> %x, %y
  %max = select <16 x i1> %mask, <16 x i8> %x, <16 x i8> %x1
  ret <16 x i8> %max
}

; CHECK-LABEL: test128_5
; CHECK: vpcmpeqw  (%rdi){{.*%k[0-7]}}
; CHECK: vmovdqu16
; CHECK: ret
define <8 x i16> @test128_5(<8 x i16> %x, <8 x i16> %x1, <8 x i16>* %yp) nounwind {
  %y = load <8 x i16>* %yp, align 4
  %mask = icmp eq <8 x i16> %x, %y
  %max = select <8 x i1> %mask, <8 x i16> %x, <8 x i16> %x1
  ret <8 x i16> %max
}

; CHECK-LABEL: @test128_6
; CHECK: vpcmpgtw (%rdi){{.*%k[0-7]}}
; CHECK: vmovdqu16
; CHECK: ret
define <8 x i16> @test128_6(<8 x i16> %x, <8 x i16> %x1, <8 x i16>* %y.ptr) nounwind {
  %y = load <8 x i16>* %y.ptr, align 4
  %mask = icmp sgt <8 x i16> %x, %y
  %max = select <8 x i1> %mask, <8 x i16> %x, <8 x i16> %x1
  ret <8 x i16> %max
}

; CHECK-LABEL: @test128_7
; CHECK: vpcmplew (%rdi){{.*%k[0-7]}}
; CHECK: vmovdqu16
; CHECK: ret
define <8 x i16> @test128_7(<8 x i16> %x, <8 x i16> %x1, <8 x i16>* %y.ptr) nounwind {
  %y = load <8 x i16>* %y.ptr, align 4
  %mask = icmp sle <8 x i16> %x, %y
  %max = select <8 x i1> %mask, <8 x i16> %x, <8 x i16> %x1
  ret <8 x i16> %max
}

; CHECK-LABEL: @test128_8
; CHECK: vpcmpleuw (%rdi){{.*%k[0-7]}}
; CHECK: vmovdqu16
; CHECK: ret
define <8 x i16> @test128_8(<8 x i16> %x, <8 x i16> %x1, <8 x i16>* %y.ptr) nounwind {
  %y = load <8 x i16>* %y.ptr, align 4
  %mask = icmp ule <8 x i16> %x, %y
  %max = select <8 x i1> %mask, <8 x i16> %x, <8 x i16> %x1
  ret <8 x i16> %max
}

; CHECK-LABEL: @test128_9
; CHECK: vpcmpeqw %xmm{{.*{%k[1-7]}}}
; CHECK: vmovdqu16
; CHECK: ret
define <8 x i16> @test128_9(<8 x i16> %x, <8 x i16> %y, <8 x i16> %x1, <8 x i16> %y1) nounwind {
  %mask1 = icmp eq <8 x i16> %x1, %y1
  %mask0 = icmp eq <8 x i16> %x, %y
  %mask = select <8 x i1> %mask0, <8 x i1> %mask1, <8 x i1> zeroinitializer
  %max = select <8 x i1> %mask, <8 x i16> %x, <8 x i16> %y
  ret <8 x i16> %max
}

; CHECK-LABEL: @test128_10
; CHECK: vpcmpleb %xmm{{.*{%k[1-7]}}}
; CHECK: vmovdqu8
; CHECK: ret
define <16 x i8> @test128_10(<16 x i8> %x, <16 x i8> %y, <16 x i8> %x1, <16 x i8> %y1) nounwind {
  %mask1 = icmp sge <16 x i8> %x1, %y1
  %mask0 = icmp sle <16 x i8> %x, %y
  %mask = select <16 x i1> %mask0, <16 x i1> %mask1, <16 x i1> zeroinitializer
  %max = select <16 x i1> %mask, <16 x i8> %x, <16 x i8> %x1
  ret <16 x i8> %max
}

; CHECK-LABEL: @test128_11
; CHECK: vpcmpgtb (%rdi){{.*{%k[1-7]}}}
; CHECK: vmovdqu8
; CHECK: ret
define <16 x i8> @test128_11(<16 x i8> %x, <16 x i8>* %y.ptr, <16 x i8> %x1, <16 x i8> %y1) nounwind {
  %mask1 = icmp sgt <16 x i8> %x1, %y1
  %y = load <16 x i8>* %y.ptr, align 4
  %mask0 = icmp sgt <16 x i8> %x, %y
  %mask = select <16 x i1> %mask0, <16 x i1> %mask1, <16 x i1> zeroinitializer
  %max = select <16 x i1> %mask, <16 x i8> %x, <16 x i8> %x1
  ret <16 x i8> %max
}

; CHECK-LABEL: @test128_12
; CHECK: vpcmpleuw (%rdi){{.*{%k[1-7]}}}
; CHECK: vmovdqu16
; CHECK: ret
define <8 x i16> @test128_12(<8 x i16> %x, <8 x i16>* %y.ptr, <8 x i16> %x1, <8 x i16> %y1) nounwind {
  %mask1 = icmp sge <8 x i16> %x1, %y1
  %y = load <8 x i16>* %y.ptr, align 4
  %mask0 = icmp ule <8 x i16> %x, %y
  %mask = select <8 x i1> %mask0, <8 x i1> %mask1, <8 x i1> zeroinitializer
  %max = select <8 x i1> %mask, <8 x i16> %x, <8 x i16> %x1
  ret <8 x i16> %max
}
