; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=knl -mattr=+avx512vl| FileCheck %s

; 256-bit

; CHECK-LABEL: vpaddq256_test
; CHECK: vpaddq %ymm{{.*}}
; CHECK: ret
define <4 x i64> @vpaddq256_test(<4 x i64> %i, <4 x i64> %j) nounwind readnone {
  %x = add <4 x i64> %i, %j
  ret <4 x i64> %x
}

; CHECK-LABEL: vpaddq256_fold_test
; CHECK: vpaddq (%rdi), %ymm{{.*}}
; CHECK: ret
define <4 x i64> @vpaddq256_fold_test(<4 x i64> %i, <4 x i64>* %j) nounwind {
  %tmp = load <4 x i64>* %j, align 4
  %x = add <4 x i64> %i, %tmp
  ret <4 x i64> %x
}

; CHECK-LABEL: vpaddq256_broadcast_test
; CHECK: vpaddq LCP{{.*}}(%rip){1to4}, %ymm{{.*}}
; CHECK: ret
define <4 x i64> @vpaddq256_broadcast_test(<4 x i64> %i) nounwind {
  %x = add <4 x i64> %i, <i64 1, i64 1, i64 1, i64 1>
  ret <4 x i64> %x
}

; CHECK-LABEL: vpaddq256_broadcast2_test
; CHECK: vpaddq (%rdi){1to4}, %ymm{{.*}}
; CHECK: ret
define <4 x i64> @vpaddq256_broadcast2_test(<4 x i64> %i, i64* %j.ptr) nounwind {
  %j = load i64* %j.ptr
  %j.0 = insertelement <4 x i64> undef, i64 %j, i32 0
  %j.v = shufflevector <4 x i64> %j.0, <4 x i64> undef, <4 x i32> zeroinitializer
  %x = add <4 x i64> %i, %j.v
  ret <4 x i64> %x
}

; CHECK-LABEL: vpaddd256_test
; CHECK: vpaddd %ymm{{.*}}
; CHECK: ret
define <8 x i32> @vpaddd256_test(<8 x i32> %i, <8 x i32> %j) nounwind readnone {
  %x = add <8 x i32> %i, %j
  ret <8 x i32> %x
}

; CHECK-LABEL: vpaddd256_fold_test
; CHECK: vpaddd (%rdi), %ymm{{.*}}
; CHECK: ret
define <8 x i32> @vpaddd256_fold_test(<8 x i32> %i, <8 x i32>* %j) nounwind {
  %tmp = load <8 x i32>* %j, align 4
  %x = add <8 x i32> %i, %tmp
  ret <8 x i32> %x
}

; CHECK-LABEL: vpaddd256_broadcast_test
; CHECK: vpaddd LCP{{.*}}(%rip){1to8}, %ymm{{.*}}
; CHECK: ret
define <8 x i32> @vpaddd256_broadcast_test(<8 x i32> %i) nounwind {
  %x = add <8 x i32> %i, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  ret <8 x i32> %x
}

; CHECK-LABEL: vpaddd256_mask_test
; CHECK: vpaddd %ymm{{.*%k[1-7].*}}
; CHECK: ret
define <8 x i32> @vpaddd256_mask_test(<8 x i32> %i, <8 x i32> %j, <8 x i32> %mask1) nounwind readnone {
  %mask = icmp ne <8 x i32> %mask1, zeroinitializer
  %x = add <8 x i32> %i, %j
  %r = select <8 x i1> %mask, <8 x i32> %x, <8 x i32> %i
  ret <8 x i32> %r
}

; CHECK-LABEL: vpaddd256_maskz_test
; CHECK: vpaddd %ymm{{.*{%k[1-7]} {z}.*}}
; CHECK: ret
define <8 x i32> @vpaddd256_maskz_test(<8 x i32> %i, <8 x i32> %j, <8 x i32> %mask1) nounwind readnone {
  %mask = icmp ne <8 x i32> %mask1, zeroinitializer
  %x = add <8 x i32> %i, %j
  %r = select <8 x i1> %mask, <8 x i32> %x, <8 x i32> zeroinitializer
  ret <8 x i32> %r
}

; CHECK-LABEL: vpaddd256_mask_fold_test
; CHECK: vpaddd (%rdi), %ymm{{.*%k[1-7]}}
; CHECK: ret
define <8 x i32> @vpaddd256_mask_fold_test(<8 x i32> %i, <8 x i32>* %j.ptr, <8 x i32> %mask1) nounwind readnone {
  %mask = icmp ne <8 x i32> %mask1, zeroinitializer
  %j = load <8 x i32>* %j.ptr
  %x = add <8 x i32> %i, %j
  %r = select <8 x i1> %mask, <8 x i32> %x, <8 x i32> %i
  ret <8 x i32> %r
}

; CHECK-LABEL: vpaddd256_mask_broadcast_test
; CHECK: vpaddd LCP{{.*}}(%rip){1to8}, %ymm{{.*{%k[1-7]}}}
; CHECK: ret
define <8 x i32> @vpaddd256_mask_broadcast_test(<8 x i32> %i, <8 x i32> %mask1) nounwind readnone {
  %mask = icmp ne <8 x i32> %mask1, zeroinitializer
  %x = add <8 x i32> %i, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %r = select <8 x i1> %mask, <8 x i32> %x, <8 x i32> %i
  ret <8 x i32> %r
}

; CHECK-LABEL: vpaddd256_maskz_fold_test
; CHECK: vpaddd (%rdi), %ymm{{.*{%k[1-7]} {z}}}
; CHECK: ret
define <8 x i32> @vpaddd256_maskz_fold_test(<8 x i32> %i, <8 x i32>* %j.ptr, <8 x i32> %mask1) nounwind readnone {
  %mask = icmp ne <8 x i32> %mask1, zeroinitializer
  %j = load <8 x i32>* %j.ptr
  %x = add <8 x i32> %i, %j
  %r = select <8 x i1> %mask, <8 x i32> %x, <8 x i32> zeroinitializer
  ret <8 x i32> %r
}

; CHECK-LABEL: vpaddd256_maskz_broadcast_test
; CHECK: vpaddd LCP{{.*}}(%rip){1to8}, %ymm{{.*{%k[1-7]} {z}}}
; CHECK: ret
define <8 x i32> @vpaddd256_maskz_broadcast_test(<8 x i32> %i, <8 x i32> %mask1) nounwind readnone {
  %mask = icmp ne <8 x i32> %mask1, zeroinitializer
  %x = add <8 x i32> %i, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %r = select <8 x i1> %mask, <8 x i32> %x, <8 x i32> zeroinitializer
  ret <8 x i32> %r
}

; CHECK-LABEL: vpsubq256_test
; CHECK: vpsubq %ymm{{.*}}
; CHECK: ret
define <4 x i64> @vpsubq256_test(<4 x i64> %i, <4 x i64> %j) nounwind readnone {
  %x = sub <4 x i64> %i, %j
  ret <4 x i64> %x
}

; CHECK-LABEL: vpsubd256_test
; CHECK: vpsubd %ymm{{.*}}
; CHECK: ret
define <8 x i32> @vpsubd256_test(<8 x i32> %i, <8 x i32> %j) nounwind readnone {
  %x = sub <8 x i32> %i, %j
  ret <8 x i32> %x
}

; CHECK-LABEL: vpmulld256_test
; CHECK: vpmulld %ymm{{.*}}
; CHECK: ret
define <8 x i32> @vpmulld256_test(<8 x i32> %i, <8 x i32> %j) {
  %x = mul <8 x i32> %i, %j
  ret <8 x i32> %x
}

; 128-bit

; CHECK-LABEL: vpaddq128_test
; CHECK: vpaddq %xmm{{.*}}
; CHECK: ret
define <2 x i64> @vpaddq128_test(<2 x i64> %i, <2 x i64> %j) nounwind readnone {
  %x = add <2 x i64> %i, %j
  ret <2 x i64> %x
}

; CHECK-LABEL: vpaddq128_fold_test
; CHECK: vpaddq (%rdi), %xmm{{.*}}
; CHECK: ret
define <2 x i64> @vpaddq128_fold_test(<2 x i64> %i, <2 x i64>* %j) nounwind {
  %tmp = load <2 x i64>* %j, align 4
  %x = add <2 x i64> %i, %tmp
  ret <2 x i64> %x
}

; CHECK-LABEL: vpaddq128_broadcast2_test
; CHECK: vpaddq (%rdi){1to2}, %xmm{{.*}}
; CHECK: ret
define <2 x i64> @vpaddq128_broadcast2_test(<2 x i64> %i, i64* %j) nounwind {
  %tmp = load i64* %j
  %j.0 = insertelement <2 x i64> undef, i64 %tmp, i32 0
  %j.1 = insertelement <2 x i64> %j.0, i64 %tmp, i32 1
  %x = add <2 x i64> %i, %j.1
  ret <2 x i64> %x
}

; CHECK-LABEL: vpaddd128_test
; CHECK: vpaddd %xmm{{.*}}
; CHECK: ret
define <4 x i32> @vpaddd128_test(<4 x i32> %i, <4 x i32> %j) nounwind readnone {
  %x = add <4 x i32> %i, %j
  ret <4 x i32> %x
}

; CHECK-LABEL: vpaddd128_fold_test
; CHECK: vpaddd (%rdi), %xmm{{.*}}
; CHECK: ret
define <4 x i32> @vpaddd128_fold_test(<4 x i32> %i, <4 x i32>* %j) nounwind {
  %tmp = load <4 x i32>* %j, align 4
  %x = add <4 x i32> %i, %tmp
  ret <4 x i32> %x
}

; CHECK-LABEL: vpaddd128_broadcast_test
; CHECK: vpaddd LCP{{.*}}(%rip){1to4}, %xmm{{.*}}
; CHECK: ret
define <4 x i32> @vpaddd128_broadcast_test(<4 x i32> %i) nounwind {
  %x = add <4 x i32> %i, <i32 1, i32 1, i32 1, i32 1>
  ret <4 x i32> %x
}

; CHECK-LABEL: vpaddd128_mask_test
; CHECK: vpaddd %xmm{{.*%k[1-7].*}}
; CHECK: ret
define <4 x i32> @vpaddd128_mask_test(<4 x i32> %i, <4 x i32> %j, <4 x i32> %mask1) nounwind readnone {
  %mask = icmp ne <4 x i32> %mask1, zeroinitializer
  %x = add <4 x i32> %i, %j
  %r = select <4 x i1> %mask, <4 x i32> %x, <4 x i32> %i
  ret <4 x i32> %r
}

; CHECK-LABEL: vpaddd128_maskz_test
; CHECK: vpaddd %xmm{{.*{%k[1-7]} {z}.*}}
; CHECK: ret
define <4 x i32> @vpaddd128_maskz_test(<4 x i32> %i, <4 x i32> %j, <4 x i32> %mask1) nounwind readnone {
  %mask = icmp ne <4 x i32> %mask1, zeroinitializer
  %x = add <4 x i32> %i, %j
  %r = select <4 x i1> %mask, <4 x i32> %x, <4 x i32> zeroinitializer
  ret <4 x i32> %r
}

; CHECK-LABEL: vpaddd128_mask_fold_test
; CHECK: vpaddd (%rdi), %xmm{{.*%k[1-7]}}
; CHECK: ret
define <4 x i32> @vpaddd128_mask_fold_test(<4 x i32> %i, <4 x i32>* %j.ptr, <4 x i32> %mask1) nounwind readnone {
  %mask = icmp ne <4 x i32> %mask1, zeroinitializer
  %j = load <4 x i32>* %j.ptr
  %x = add <4 x i32> %i, %j
  %r = select <4 x i1> %mask, <4 x i32> %x, <4 x i32> %i
  ret <4 x i32> %r
}

; CHECK-LABEL: vpaddd128_mask_broadcast_test
; CHECK: vpaddd LCP{{.*}}(%rip){1to4}, %xmm{{.*{%k[1-7]}}}
; CHECK: ret
define <4 x i32> @vpaddd128_mask_broadcast_test(<4 x i32> %i, <4 x i32> %mask1) nounwind readnone {
  %mask = icmp ne <4 x i32> %mask1, zeroinitializer
  %x = add <4 x i32> %i, <i32 1, i32 1, i32 1, i32 1>
  %r = select <4 x i1> %mask, <4 x i32> %x, <4 x i32> %i
  ret <4 x i32> %r
}

; CHECK-LABEL: vpaddd128_maskz_fold_test
; CHECK: vpaddd (%rdi), %xmm{{.*{%k[1-7]} {z}}}
; CHECK: ret
define <4 x i32> @vpaddd128_maskz_fold_test(<4 x i32> %i, <4 x i32>* %j.ptr, <4 x i32> %mask1) nounwind readnone {
  %mask = icmp ne <4 x i32> %mask1, zeroinitializer
  %j = load <4 x i32>* %j.ptr
  %x = add <4 x i32> %i, %j
  %r = select <4 x i1> %mask, <4 x i32> %x, <4 x i32> zeroinitializer
  ret <4 x i32> %r
}

; CHECK-LABEL: vpaddd128_maskz_broadcast_test
; CHECK: vpaddd LCP{{.*}}(%rip){1to4}, %xmm{{.*{%k[1-7]} {z}}}
; CHECK: ret
define <4 x i32> @vpaddd128_maskz_broadcast_test(<4 x i32> %i, <4 x i32> %mask1) nounwind readnone {
  %mask = icmp ne <4 x i32> %mask1, zeroinitializer
  %x = add <4 x i32> %i, <i32 1, i32 1, i32 1, i32 1>
  %r = select <4 x i1> %mask, <4 x i32> %x, <4 x i32> zeroinitializer
  ret <4 x i32> %r
}

; CHECK-LABEL: vpsubq128_test
; CHECK: vpsubq %xmm{{.*}}
; CHECK: ret
define <2 x i64> @vpsubq128_test(<2 x i64> %i, <2 x i64> %j) nounwind readnone {
  %x = sub <2 x i64> %i, %j
  ret <2 x i64> %x
}

; CHECK-LABEL: vpsubd128_test
; CHECK: vpsubd %xmm{{.*}}
; CHECK: ret
define <4 x i32> @vpsubd128_test(<4 x i32> %i, <4 x i32> %j) nounwind readnone {
  %x = sub <4 x i32> %i, %j
  ret <4 x i32> %x
}

; CHECK-LABEL: vpmulld128_test
; CHECK: vpmulld %xmm{{.*}}
; CHECK: ret
define <4 x i32> @vpmulld128_test(<4 x i32> %i, <4 x i32> %j) {
  %x = mul <4 x i32> %i, %j
  ret <4 x i32> %x
}
