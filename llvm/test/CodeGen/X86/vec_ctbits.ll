; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+sse2 | FileCheck %s

declare <2 x i64> @llvm.cttz.v2i64(<2 x i64>, i1)
declare <2 x i64> @llvm.ctlz.v2i64(<2 x i64>, i1)
declare <2 x i64> @llvm.ctpop.v2i64(<2 x i64>)

define <2 x i64> @footz(<2 x i64> %a) nounwind {
  %c = call <2 x i64> @llvm.cttz.v2i64(<2 x i64> %a, i1 true)
  ret <2 x i64> %c

; CHECK-LABEL: footz
; CHECK: bsfq
; CHECK: bsfq
}
define <2 x i64> @foolz(<2 x i64> %a) nounwind {
  %c = call <2 x i64> @llvm.ctlz.v2i64(<2 x i64> %a, i1 true)
  ret <2 x i64> %c

; CHECK-LABEL: foolz
; CHECK: bsrq
; CHECK: xorq $63
; CHECK: bsrq
; CHECK: xorq $63
}

define <2 x i64> @foopop(<2 x i64> %a) nounwind {
  %c = call <2 x i64> @llvm.ctpop.v2i64(<2 x i64> %a)
  ret <2 x i64> %c
}

declare <2 x i32> @llvm.cttz.v2i32(<2 x i32>, i1)
declare <2 x i32> @llvm.ctlz.v2i32(<2 x i32>, i1)
declare <2 x i32> @llvm.ctpop.v2i32(<2 x i32>)

define <2 x i32> @promtz(<2 x i32> %a) nounwind {
  %c = call <2 x i32> @llvm.cttz.v2i32(<2 x i32> %a, i1 false)
  ret <2 x i32> %c

; CHECK: .quad 4294967296
; CHECK: .quad 4294967296
; CHECK-LABEL: promtz
; CHECK: bsfq
; CHECK: cmov
; CHECK: bsfq
; CHECK: cmov
}
define <2 x i32> @promlz(<2 x i32> %a) nounwind {
  %c = call <2 x i32> @llvm.ctlz.v2i32(<2 x i32> %a, i1 false)
  ret <2 x i32> %c

; CHECK: .quad 4294967295
; CHECK: .quad 4294967295
; CHECK: .quad 32
; CHECK: .quad 32
; CHECK-LABEL: promlz
; CHECK: pand
; CHECK: bsrq
; CHECK: xorq $63
; CHECK: bsrq
; CHECK: xorq $63
; CHECK: psub
}

define <2 x i32> @prompop(<2 x i32> %a) nounwind {
  %c = call <2 x i32> @llvm.ctpop.v2i32(<2 x i32> %a)
  ret <2 x i32> %c
}
