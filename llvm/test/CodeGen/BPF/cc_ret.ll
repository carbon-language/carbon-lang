; RUN: llc < %s -march=bpfel | FileCheck %s

define void @test() #0 {
entry:
; CHECK: test:

; CHECK: call f_i16
; CHECK: sth 0(r1), r0
  %0 = call i16 @f_i16()
  store volatile i16 %0, i16* @g_i16

; CHECK: call f_i32
; CHECK: stw 0(r1), r0
  %1 = call i32 @f_i32()
  store volatile i32 %1, i32* @g_i32

; CHECK: call f_i64
; CHECK: std 0(r1), r0
  %2 = call i64 @f_i64()
  store volatile i64 %2, i64* @g_i64

  ret void
}

@g_i16 = common global i16 0, align 2
@g_i32 = common global i32 0, align 2
@g_i64 = common global i64 0, align 2

define i16 @f_i16() #0 {
; CHECK: f_i16:
; CHECK: mov r0, 1
; CHECK: ret
  ret i16 1
}

define i32 @f_i32() #0 {
; CHECK: f_i32:
; CHECK: mov r0, 16909060
; CHECK: ret
  ret i32 16909060
}

define i64 @f_i64() #0 {
; CHECK: f_i64:
; CHECK: ld_64 r0, 72623859790382856
; CHECK: ret
  ret i64 72623859790382856
}
