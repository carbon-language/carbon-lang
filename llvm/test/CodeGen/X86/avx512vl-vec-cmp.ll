; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=skx | FileCheck %s

; CHECK-LABEL: test256_1
; CHECK: vpcmpeqq {{.*%k[0-7]}}
; CHECK: vmovdqa64 {{.*}}%k1
; CHECK: ret
define <4 x i64> @test256_1(<4 x i64> %x, <4 x i64> %y) nounwind {
  %mask = icmp eq <4 x i64> %x, %y
  %max = select <4 x i1> %mask, <4 x i64> %x, <4 x i64> %y
  ret <4 x i64> %max
}

; CHECK-LABEL: test256_2
; CHECK: vpcmpgtq {{.*%k[0-7]}}
; CHECK: vmovdqa64 {{.*}}%k1
; CHECK: ret
define <4 x i64> @test256_2(<4 x i64> %x, <4 x i64> %y, <4 x i64> %x1) nounwind {
  %mask = icmp sgt <4 x i64> %x, %y
  %max = select <4 x i1> %mask, <4 x i64> %x1, <4 x i64> %y
  ret <4 x i64> %max
}

; CHECK-LABEL: @test256_3
; CHECK: vpcmpled {{.*%k[0-7]}}
; CHECK: vmovdqa32
; CHECK: ret
define <8 x i32> @test256_3(<8 x i32> %x, <8 x i32> %y, <8 x i32> %x1) nounwind {
  %mask = icmp sge <8 x i32> %x, %y
  %max = select <8 x i1> %mask, <8 x i32> %x1, <8 x i32> %y
  ret <8 x i32> %max
}

; CHECK-LABEL: test256_4
; CHECK: vpcmpnleuq {{.*%k[0-7]}}
; CHECK: vmovdqa64 {{.*}}%k1
; CHECK: ret
define <4 x i64> @test256_4(<4 x i64> %x, <4 x i64> %y, <4 x i64> %x1) nounwind {
  %mask = icmp ugt <4 x i64> %x, %y
  %max = select <4 x i1> %mask, <4 x i64> %x1, <4 x i64> %y
  ret <4 x i64> %max
}

; CHECK-LABEL: test256_5
; CHECK: vpcmpeqd  (%rdi){{.*%k[0-7]}}
; CHECK: vmovdqa32
; CHECK: ret
define <8 x i32> @test256_5(<8 x i32> %x, <8 x i32> %x1, <8 x i32>* %yp) nounwind {
  %y = load <8 x i32>, <8 x i32>* %yp, align 4
  %mask = icmp eq <8 x i32> %x, %y
  %max = select <8 x i1> %mask, <8 x i32> %x, <8 x i32> %x1
  ret <8 x i32> %max
}

; CHECK-LABEL: @test256_6
; CHECK: vpcmpgtd (%rdi){{.*%k[0-7]}}
; CHECK: vmovdqa32
; CHECK: ret
define <8 x i32> @test256_6(<8 x i32> %x, <8 x i32> %x1, <8 x i32>* %y.ptr) nounwind {
  %y = load <8 x i32>, <8 x i32>* %y.ptr, align 4
  %mask = icmp sgt <8 x i32> %x, %y
  %max = select <8 x i1> %mask, <8 x i32> %x, <8 x i32> %x1
  ret <8 x i32> %max
}

; CHECK-LABEL: @test256_7
; CHECK: vpcmpled (%rdi){{.*%k[0-7]}}
; CHECK: vmovdqa32
; CHECK: ret
define <8 x i32> @test256_7(<8 x i32> %x, <8 x i32> %x1, <8 x i32>* %y.ptr) nounwind {
  %y = load <8 x i32>, <8 x i32>* %y.ptr, align 4
  %mask = icmp sle <8 x i32> %x, %y
  %max = select <8 x i1> %mask, <8 x i32> %x, <8 x i32> %x1
  ret <8 x i32> %max
}

; CHECK-LABEL: @test256_8
; CHECK: vpcmpleud (%rdi){{.*%k[0-7]}}
; CHECK: vmovdqa32
; CHECK: ret
define <8 x i32> @test256_8(<8 x i32> %x, <8 x i32> %x1, <8 x i32>* %y.ptr) nounwind {
  %y = load <8 x i32>, <8 x i32>* %y.ptr, align 4
  %mask = icmp ule <8 x i32> %x, %y
  %max = select <8 x i1> %mask, <8 x i32> %x, <8 x i32> %x1
  ret <8 x i32> %max
}

; CHECK-LABEL: @test256_9
; CHECK: vpcmpeqd %ymm{{.*{%k[1-7]}}}
; CHECK: vmovdqa32
; CHECK: ret
define <8 x i32> @test256_9(<8 x i32> %x, <8 x i32> %y, <8 x i32> %x1, <8 x i32> %y1) nounwind {
  %mask1 = icmp eq <8 x i32> %x1, %y1
  %mask0 = icmp eq <8 x i32> %x, %y
  %mask = select <8 x i1> %mask0, <8 x i1> %mask1, <8 x i1> zeroinitializer
  %max = select <8 x i1> %mask, <8 x i32> %x, <8 x i32> %y
  ret <8 x i32> %max
}

; CHECK-LABEL: @test256_10
; CHECK: vpcmpleq %ymm{{.*{%k[1-7]}}}
; CHECK: vmovdqa64
; CHECK: ret
define <4 x i64> @test256_10(<4 x i64> %x, <4 x i64> %y, <4 x i64> %x1, <4 x i64> %y1) nounwind {
  %mask1 = icmp sge <4 x i64> %x1, %y1
  %mask0 = icmp sle <4 x i64> %x, %y
  %mask = select <4 x i1> %mask0, <4 x i1> %mask1, <4 x i1> zeroinitializer
  %max = select <4 x i1> %mask, <4 x i64> %x, <4 x i64> %x1
  ret <4 x i64> %max
}

; CHECK-LABEL: @test256_11
; CHECK: vpcmpgtq (%rdi){{.*{%k[1-7]}}}
; CHECK: vmovdqa64
; CHECK: ret
define <4 x i64> @test256_11(<4 x i64> %x, <4 x i64>* %y.ptr, <4 x i64> %x1, <4 x i64> %y1) nounwind {
  %mask1 = icmp sgt <4 x i64> %x1, %y1
  %y = load <4 x i64>, <4 x i64>* %y.ptr, align 4
  %mask0 = icmp sgt <4 x i64> %x, %y
  %mask = select <4 x i1> %mask0, <4 x i1> %mask1, <4 x i1> zeroinitializer
  %max = select <4 x i1> %mask, <4 x i64> %x, <4 x i64> %x1
  ret <4 x i64> %max
}

; CHECK-LABEL: @test256_12
; CHECK: vpcmpleud (%rdi){{.*{%k[1-7]}}}
; CHECK: vmovdqa32
; CHECK: ret
define <8 x i32> @test256_12(<8 x i32> %x, <8 x i32>* %y.ptr, <8 x i32> %x1, <8 x i32> %y1) nounwind {
  %mask1 = icmp sge <8 x i32> %x1, %y1
  %y = load <8 x i32>, <8 x i32>* %y.ptr, align 4
  %mask0 = icmp ule <8 x i32> %x, %y
  %mask = select <8 x i1> %mask0, <8 x i1> %mask1, <8 x i1> zeroinitializer
  %max = select <8 x i1> %mask, <8 x i32> %x, <8 x i32> %x1
  ret <8 x i32> %max
}

; CHECK-LABEL: test256_13
; CHECK: vpcmpeqq  (%rdi){1to4}, %ymm
; CHECK: vmovdqa64
; CHECK: ret
define <4 x i64> @test256_13(<4 x i64> %x, <4 x i64> %x1, i64* %yb.ptr) nounwind {
  %yb = load i64, i64* %yb.ptr, align 4
  %y.0 = insertelement <4 x i64> undef, i64 %yb, i32 0
  %y = shufflevector <4 x i64> %y.0, <4 x i64> undef, <4 x i32> zeroinitializer
  %mask = icmp eq <4 x i64> %x, %y
  %max = select <4 x i1> %mask, <4 x i64> %x, <4 x i64> %x1
  ret <4 x i64> %max
}

; CHECK-LABEL: test256_14
; CHECK: vpcmpled  (%rdi){1to8}, %ymm
; CHECK: vmovdqa32
; CHECK: ret
define <8 x i32> @test256_14(<8 x i32> %x, i32* %yb.ptr, <8 x i32> %x1) nounwind {
  %yb = load i32, i32* %yb.ptr, align 4
  %y.0 = insertelement <8 x i32> undef, i32 %yb, i32 0
  %y = shufflevector <8 x i32> %y.0, <8 x i32> undef, <8 x i32> zeroinitializer
  %mask = icmp sle <8 x i32> %x, %y
  %max = select <8 x i1> %mask, <8 x i32> %x, <8 x i32> %x1
  ret <8 x i32> %max
}

; CHECK-LABEL: test256_15
; CHECK: vpcmpgtd  (%rdi){1to8}, %ymm{{.*{%k[1-7]}}}
; CHECK: vmovdqa32
; CHECK: ret
define <8 x i32> @test256_15(<8 x i32> %x, i32* %yb.ptr, <8 x i32> %x1, <8 x i32> %y1) nounwind {
  %mask1 = icmp sge <8 x i32> %x1, %y1
  %yb = load i32, i32* %yb.ptr, align 4
  %y.0 = insertelement <8 x i32> undef, i32 %yb, i32 0
  %y = shufflevector <8 x i32> %y.0, <8 x i32> undef, <8 x i32> zeroinitializer
  %mask0 = icmp sgt <8 x i32> %x, %y
  %mask = select <8 x i1> %mask0, <8 x i1> %mask1, <8 x i1> zeroinitializer
  %max = select <8 x i1> %mask, <8 x i32> %x, <8 x i32> %x1
  ret <8 x i32> %max
}

; CHECK-LABEL: test256_16
; CHECK: vpcmpgtq  (%rdi){1to4}, %ymm{{.*{%k[1-7]}}}
; CHECK: vmovdqa64
; CHECK: ret
define <4 x i64> @test256_16(<4 x i64> %x, i64* %yb.ptr, <4 x i64> %x1, <4 x i64> %y1) nounwind {
  %mask1 = icmp sge <4 x i64> %x1, %y1
  %yb = load i64, i64* %yb.ptr, align 4
  %y.0 = insertelement <4 x i64> undef, i64 %yb, i32 0
  %y = shufflevector <4 x i64> %y.0, <4 x i64> undef, <4 x i32> zeroinitializer
  %mask0 = icmp sgt <4 x i64> %x, %y
  %mask = select <4 x i1> %mask0, <4 x i1> %mask1, <4 x i1> zeroinitializer
  %max = select <4 x i1> %mask, <4 x i64> %x, <4 x i64> %x1
  ret <4 x i64> %max
}

; CHECK-LABEL: test128_1
; CHECK: vpcmpeqq {{.*%k[0-7]}}
; CHECK: vmovdqa64 {{.*}}%k1
; CHECK: ret
define <2 x i64> @test128_1(<2 x i64> %x, <2 x i64> %y) nounwind {
  %mask = icmp eq <2 x i64> %x, %y
  %max = select <2 x i1> %mask, <2 x i64> %x, <2 x i64> %y
  ret <2 x i64> %max
}

; CHECK-LABEL: test128_2
; CHECK: vpcmpgtq {{.*%k[0-7]}}
; CHECK: vmovdqa64 {{.*}}%k1
; CHECK: ret
define <2 x i64> @test128_2(<2 x i64> %x, <2 x i64> %y, <2 x i64> %x1) nounwind {
  %mask = icmp sgt <2 x i64> %x, %y
  %max = select <2 x i1> %mask, <2 x i64> %x1, <2 x i64> %y
  ret <2 x i64> %max
}

; CHECK-LABEL: @test128_3
; CHECK: vpcmpled {{.*%k[0-7]}}
; CHECK: vmovdqa32
; CHECK: ret
define <4 x i32> @test128_3(<4 x i32> %x, <4 x i32> %y, <4 x i32> %x1) nounwind {
  %mask = icmp sge <4 x i32> %x, %y
  %max = select <4 x i1> %mask, <4 x i32> %x1, <4 x i32> %y
  ret <4 x i32> %max
}

; CHECK-LABEL: test128_4
; CHECK: vpcmpnleuq {{.*%k[0-7]}}
; CHECK: vmovdqa64 {{.*}}%k1
; CHECK: ret
define <2 x i64> @test128_4(<2 x i64> %x, <2 x i64> %y, <2 x i64> %x1) nounwind {
  %mask = icmp ugt <2 x i64> %x, %y
  %max = select <2 x i1> %mask, <2 x i64> %x1, <2 x i64> %y
  ret <2 x i64> %max
}

; CHECK-LABEL: test128_5
; CHECK: vpcmpeqd  (%rdi){{.*%k[0-7]}}
; CHECK: vmovdqa32
; CHECK: ret
define <4 x i32> @test128_5(<4 x i32> %x, <4 x i32> %x1, <4 x i32>* %yp) nounwind {
  %y = load <4 x i32>, <4 x i32>* %yp, align 4
  %mask = icmp eq <4 x i32> %x, %y
  %max = select <4 x i1> %mask, <4 x i32> %x, <4 x i32> %x1
  ret <4 x i32> %max
}

; CHECK-LABEL: @test128_6
; CHECK: vpcmpgtd (%rdi){{.*%k[0-7]}}
; CHECK: vmovdqa32
; CHECK: ret
define <4 x i32> @test128_6(<4 x i32> %x, <4 x i32> %x1, <4 x i32>* %y.ptr) nounwind {
  %y = load <4 x i32>, <4 x i32>* %y.ptr, align 4
  %mask = icmp sgt <4 x i32> %x, %y
  %max = select <4 x i1> %mask, <4 x i32> %x, <4 x i32> %x1
  ret <4 x i32> %max
}

; CHECK-LABEL: @test128_7
; CHECK: vpcmpled (%rdi){{.*%k[0-7]}}
; CHECK: vmovdqa32
; CHECK: ret
define <4 x i32> @test128_7(<4 x i32> %x, <4 x i32> %x1, <4 x i32>* %y.ptr) nounwind {
  %y = load <4 x i32>, <4 x i32>* %y.ptr, align 4
  %mask = icmp sle <4 x i32> %x, %y
  %max = select <4 x i1> %mask, <4 x i32> %x, <4 x i32> %x1
  ret <4 x i32> %max
}

; CHECK-LABEL: @test128_8
; CHECK: vpcmpleud (%rdi){{.*%k[0-7]}}
; CHECK: vmovdqa32
; CHECK: ret
define <4 x i32> @test128_8(<4 x i32> %x, <4 x i32> %x1, <4 x i32>* %y.ptr) nounwind {
  %y = load <4 x i32>, <4 x i32>* %y.ptr, align 4
  %mask = icmp ule <4 x i32> %x, %y
  %max = select <4 x i1> %mask, <4 x i32> %x, <4 x i32> %x1
  ret <4 x i32> %max
}

; CHECK-LABEL: @test128_9
; CHECK: vpcmpeqd %xmm{{.*{%k[1-7]}}}
; CHECK: vmovdqa32
; CHECK: ret
define <4 x i32> @test128_9(<4 x i32> %x, <4 x i32> %y, <4 x i32> %x1, <4 x i32> %y1) nounwind {
  %mask1 = icmp eq <4 x i32> %x1, %y1
  %mask0 = icmp eq <4 x i32> %x, %y
  %mask = select <4 x i1> %mask0, <4 x i1> %mask1, <4 x i1> zeroinitializer
  %max = select <4 x i1> %mask, <4 x i32> %x, <4 x i32> %y
  ret <4 x i32> %max
}

; CHECK-LABEL: @test128_10
; CHECK: vpcmpleq %xmm{{.*{%k[1-7]}}}
; CHECK: vmovdqa64
; CHECK: ret
define <2 x i64> @test128_10(<2 x i64> %x, <2 x i64> %y, <2 x i64> %x1, <2 x i64> %y1) nounwind {
  %mask1 = icmp sge <2 x i64> %x1, %y1
  %mask0 = icmp sle <2 x i64> %x, %y
  %mask = select <2 x i1> %mask0, <2 x i1> %mask1, <2 x i1> zeroinitializer
  %max = select <2 x i1> %mask, <2 x i64> %x, <2 x i64> %x1
  ret <2 x i64> %max
}

; CHECK-LABEL: @test128_11
; CHECK: vpcmpgtq (%rdi){{.*{%k[1-7]}}}
; CHECK: vmovdqa64
; CHECK: ret
define <2 x i64> @test128_11(<2 x i64> %x, <2 x i64>* %y.ptr, <2 x i64> %x1, <2 x i64> %y1) nounwind {
  %mask1 = icmp sgt <2 x i64> %x1, %y1
  %y = load <2 x i64>, <2 x i64>* %y.ptr, align 4
  %mask0 = icmp sgt <2 x i64> %x, %y
  %mask = select <2 x i1> %mask0, <2 x i1> %mask1, <2 x i1> zeroinitializer
  %max = select <2 x i1> %mask, <2 x i64> %x, <2 x i64> %x1
  ret <2 x i64> %max
}

; CHECK-LABEL: @test128_12
; CHECK: vpcmpleud (%rdi){{.*{%k[1-7]}}}
; CHECK: vmovdqa32
; CHECK: ret
define <4 x i32> @test128_12(<4 x i32> %x, <4 x i32>* %y.ptr, <4 x i32> %x1, <4 x i32> %y1) nounwind {
  %mask1 = icmp sge <4 x i32> %x1, %y1
  %y = load <4 x i32>, <4 x i32>* %y.ptr, align 4
  %mask0 = icmp ule <4 x i32> %x, %y
  %mask = select <4 x i1> %mask0, <4 x i1> %mask1, <4 x i1> zeroinitializer
  %max = select <4 x i1> %mask, <4 x i32> %x, <4 x i32> %x1
  ret <4 x i32> %max
}

; CHECK-LABEL: test128_13
; CHECK: vpcmpeqq  (%rdi){1to2}, %xmm
; CHECK: vmovdqa64
; CHECK: ret
define <2 x i64> @test128_13(<2 x i64> %x, <2 x i64> %x1, i64* %yb.ptr) nounwind {
  %yb = load i64, i64* %yb.ptr, align 4
  %y.0 = insertelement <2 x i64> undef, i64 %yb, i32 0
  %y = insertelement <2 x i64> %y.0, i64 %yb, i32 1
  %mask = icmp eq <2 x i64> %x, %y
  %max = select <2 x i1> %mask, <2 x i64> %x, <2 x i64> %x1
  ret <2 x i64> %max
}

; CHECK-LABEL: test128_14
; CHECK: vpcmpled  (%rdi){1to4}, %xmm
; CHECK: vmovdqa32
; CHECK: ret
define <4 x i32> @test128_14(<4 x i32> %x, i32* %yb.ptr, <4 x i32> %x1) nounwind {
  %yb = load i32, i32* %yb.ptr, align 4
  %y.0 = insertelement <4 x i32> undef, i32 %yb, i32 0
  %y = shufflevector <4 x i32> %y.0, <4 x i32> undef, <4 x i32> zeroinitializer
  %mask = icmp sle <4 x i32> %x, %y
  %max = select <4 x i1> %mask, <4 x i32> %x, <4 x i32> %x1
  ret <4 x i32> %max
}

; CHECK-LABEL: test128_15
; CHECK: vpcmpgtd  (%rdi){1to4}, %xmm{{.*{%k[1-7]}}}
; CHECK: vmovdqa32
; CHECK: ret
define <4 x i32> @test128_15(<4 x i32> %x, i32* %yb.ptr, <4 x i32> %x1, <4 x i32> %y1) nounwind {
  %mask1 = icmp sge <4 x i32> %x1, %y1
  %yb = load i32, i32* %yb.ptr, align 4
  %y.0 = insertelement <4 x i32> undef, i32 %yb, i32 0
  %y = shufflevector <4 x i32> %y.0, <4 x i32> undef, <4 x i32> zeroinitializer
  %mask0 = icmp sgt <4 x i32> %x, %y
  %mask = select <4 x i1> %mask0, <4 x i1> %mask1, <4 x i1> zeroinitializer
  %max = select <4 x i1> %mask, <4 x i32> %x, <4 x i32> %x1
  ret <4 x i32> %max
}

; CHECK-LABEL: test128_16
; CHECK: vpcmpgtq  (%rdi){1to2}, %xmm{{.*{%k[1-7]}}}
; CHECK: vmovdqa64
; CHECK: ret
define <2 x i64> @test128_16(<2 x i64> %x, i64* %yb.ptr, <2 x i64> %x1, <2 x i64> %y1) nounwind {
  %mask1 = icmp sge <2 x i64> %x1, %y1
  %yb = load i64, i64* %yb.ptr, align 4
  %y.0 = insertelement <2 x i64> undef, i64 %yb, i32 0
  %y = insertelement <2 x i64> %y.0, i64 %yb, i32 1
  %mask0 = icmp sgt <2 x i64> %x, %y
  %mask = select <2 x i1> %mask0, <2 x i1> %mask1, <2 x i1> zeroinitializer
  %max = select <2 x i1> %mask, <2 x i64> %x, <2 x i64> %x1
  ret <2 x i64> %max
}
