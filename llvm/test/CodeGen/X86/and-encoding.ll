; RUN: llc -show-mc-encoding < %s | FileCheck %s

; Test that the direct object emission selects the and variant with 8 bit
; immediate.
; We used to get this wrong when using direct object emission, but not when
; reading assembly.


target triple = "x86_64-pc-linux"

define void @f1() {
; CHECK-LABEL: f1:
; CHECK: andq    $-32, %rsp              # encoding: [0x48,0x83,0xe4,0xe0]
  %foo = alloca i8, align 32
  ret void
}

define void @f2(i1 *%x, i16 *%y) {
; CHECK-LABEL: f2:
; CHECK: andl	$1, %eax                # encoding: [0x83,0xe0,0x01]
  %a = load i1, i1* %x
  %b = zext i1 %a to i16
  store i16 %b, i16* %y
  ret void
}

define i32 @f3(i1 *%x) {
; CHECK-LABEL: f3:
; CHECK: andl	$1, %eax                # encoding: [0x83,0xe0,0x01]
  %a = load i1, i1* %x
  %b = zext i1 %a to i32
  ret i32 %b
}

define i64 @f4(i1 *%x) {
; CHECK-LABEL: f4:
; CHECK: andl	$1, %eax                # encoding: [0x83,0xe0,0x01]
  %a = load i1, i1* %x
  %b = zext i1 %a to i64
  ret i64 %b
}
