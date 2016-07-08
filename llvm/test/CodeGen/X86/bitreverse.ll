; RUN: llc < %s -mtriple=i686-unknown | FileCheck %s

; These tests just check that the plumbing is in place for @llvm.bitreverse. The
; actual output is massive at the moment as llvm.bitreverse is not yet legal.

declare <2 x i16> @llvm.bitreverse.v2i16(<2 x i16>) readnone

define <2 x i16> @test_bitreverse_v2i16(<2 x i16> %a) {
; CHECK-LABEL: test_bitreverse_v2i16:
; CHECK: shll
  %b = call <2 x i16> @llvm.bitreverse.v2i16(<2 x i16> %a)
  ret <2 x i16> %b
}

declare i24 @llvm.bitreverse.i24(i24) readnone

define i24 @test_bitreverse_i24(i24 %a) {
; CHECK-LABEL: test_bitreverse_i24:
; CHECK: shll
  %b = call i24 @llvm.bitreverse.i24(i24 %a)
  ret i24 %b
}

declare i8 @llvm.bitreverse.i8(i8) readnone

define i8 @test_bitreverse_i8(i8 %a) {
; CHECK-LABEL: test_bitreverse_i8:
; CHECK: shlb
  %b = call i8 @llvm.bitreverse.i8(i8 %a)
  ret i8 %b
}

declare i4 @llvm.bitreverse.i4(i4) readnone

define i4 @test_bitreverse_i4(i4 %a) {
; CHECK-LABEL: test_bitreverse_i4:
; CHECK: shlb
  %b = call i4 @llvm.bitreverse.i4(i4 %a)
  ret i4 %b
}

; These tests check that bitreverse(constant) calls are folded

define <2 x i16> @fold_v2i16() {
; CHECK-LABEL: fold_v2i16:
; CHECK:       # BB#0:
; CHECK-NEXT:    movw $-4096, %ax # imm = 0xF000
; CHECK-NEXT:    movw $240, %dx
; CHECK-NEXT:    retl
  %b = call <2 x i16> @llvm.bitreverse.v2i16(<2 x i16> <i16 15, i16 3840>)
  ret <2 x i16> %b
}

define i24 @fold_i24() {
; CHECK-LABEL: fold_i24:
; CHECK:       # BB#0:
; CHECK-NEXT:    movl $2048, %eax
; CHECK-NEXT:    retl
  %b = call i24 @llvm.bitreverse.i24(i24 4096)
  ret i24 %b
}

define i8 @fold_i8() {
; CHECK-LABEL: fold_i8:
; CHECK:       # BB#0:
; CHECK-NEXT:    movb $-16, %al
; CHECK-NEXT:    retl
  %b = call i8 @llvm.bitreverse.i8(i8 15)
  ret i8 %b
}

define i4 @fold_i4() {
; CHECK-LABEL: fold_i4:
; CHECK:       # BB#0:
; CHECK-NEXT:    movb $1, %al
; CHECK-NEXT:    retl
  %b = call i4 @llvm.bitreverse.i4(i4 8)
  ret i4 %b
}

; These tests check that bitreverse(bitreverse()) calls are removed

define i8 @identity_i8(i8 %a) {
; CHECK-LABEL: identity_i8:
; CHECK:       # BB#0:
; CHECK-NEXT:    movb {{[0-9]+}}(%esp), %al
; CHECK-NEXT:    retl
  %b = call i8 @llvm.bitreverse.i8(i8 %a)
  %c = call i8 @llvm.bitreverse.i8(i8 %b)
  ret i8 %c
}

define <2 x i16> @identity_v2i16(<2 x i16> %a) {
; CHECK-LABEL: identity_v2i16:
; CHECK:       # BB#0:
; CHECK-NEXT:    movzwl {{[0-9]+}}(%esp), %eax
; CHECK-NEXT:    movzwl {{[0-9]+}}(%esp), %edx
; CHECK-NEXT:    retl
  %b = call <2 x i16> @llvm.bitreverse.v2i16(<2 x i16> %a)
  %c = call <2 x i16> @llvm.bitreverse.v2i16(<2 x i16> %b)
  ret <2 x i16> %c
}

; These tests check that bitreverse(undef) calls are removed

define i8 @undef_i8() {
; CHECK-LABEL: undef_i8:
; CHECK:       # BB#0:
; CHECK-NEXT:    retl
  %b = call i8 @llvm.bitreverse.i8(i8 undef)
  ret i8 %b
}

define <2 x i16> @undef_v2i16() {
; CHECK-LABEL: undef_v2i16:
; CHECK:       # BB#0:
; CHECK-NEXT:    retl
  %b = call <2 x i16> @llvm.bitreverse.v2i16(<2 x i16> undef)
  ret <2 x i16> %b
}
