; RUN: llc -mtriple=aarch64-none-linux-gnu -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=aarch64-none-linux-gnu -mattr=-fp-armv8 -verify-machineinstrs < %s | FileCheck --check-prefix=CHECK-NOFP %s

declare void @use_addr(i8*)

define void @test_simple_alloca(i64 %n) {
; CHECK-LABEL: test_simple_alloca:

  %buf = alloca i8, i64 %n
  ; Make sure we align the stack change to 16 bytes:
; CHECK-DAG: add [[SPDELTA:x[0-9]+]], x0, #15
; CHECK-DAG: and x0, [[SPDELTA]], #0xfffffffffffffff0

  ; Make sure we change SP. It would be surprising if anything but x0 were used
  ; for the final sp, but it could be if it was then moved into x0.
; CHECK-DAG: mov [[TMP:x[0-9]+]], sp
; CHECK-DAG: sub x0, [[TMP]], [[SPDELTA]]
; CHECK: mov sp, x0

  call void @use_addr(i8* %buf)
; CHECK: bl use_addr

  ret void
  ; Make sure epilogue restores sp from fp
; CHECK: sub sp, x29, #16
; CHECK: ldp x29, x30, [sp, #16]
; CHECK: add sp, sp, #32
; CHECK: ret
}

declare void @use_addr_loc(i8*, i64*)

define i64 @test_alloca_with_local(i64 %n) {
; CHECK-LABEL: test_alloca_with_local:
; CHECK: sub sp, sp, #32
; CHECK: stp x29, x30, [sp, #16]

  %loc = alloca i64
  %buf = alloca i8, i64 %n
  ; Make sure we align the stack change to 16 bytes:
; CHECK-DAG: add [[SPDELTA:x[0-9]+]], x0, #15
; CHECK-DAG: and x0, [[SPDELTA]], #0xfffffffffffffff0

  ; Make sure we change SP. It would be surprising if anything but x0 were used
  ; for the final sp, but it could be if it was then moved into x0.
; CHECK-DAG: mov [[TMP:x[0-9]+]], sp
; CHECK-DAG: sub x0, [[TMP]], [[SPDELTA]]
; CHECK: mov sp, x0

  ; Obviously suboptimal code here, but it to get &local in x1
; CHECK: sub [[TMP:x[0-9]+]], x29, [[LOC_FROM_FP:#[0-9]+]]
; CHECK: add x1, [[TMP]], #0

  call void @use_addr_loc(i8* %buf, i64* %loc)
; CHECK: bl use_addr

  %val = load i64* %loc
; CHECK: sub x[[TMP:[0-9]+]], x29, [[LOC_FROM_FP]]
; CHECK: ldr x0, [x[[TMP]]]

  ret i64 %val
  ; Make sure epilogue restores sp from fp
; CHECK: sub sp, x29, #16
; CHECK: ldp x29, x30, [sp, #16]
; CHECK: add sp, sp, #32
; CHECK: ret
}

define void @test_variadic_alloca(i64 %n, ...) {
; CHECK: test_variadic_alloca:

; CHECK: sub     sp, sp, #208
; CHECK: stp     x29, x30, [sp, #192]
; CHECK: add     x29, sp, #192
; CHECK: sub     [[TMP:x[0-9]+]], x29, #192
; CHECK: add     x8, [[TMP]], #0
; CHECK-FP: str     q7, [x8, #112]
; [...]
; CHECK-FP: str     q1, [x8, #16]

; CHECK-NOFP: sub     sp, sp, #80
; CHECK-NOFP: stp     x29, x30, [sp, #64]
; CHECK-NOFP: add     x29, sp, #64
; CHECK-NOFP: sub     [[TMP:x[0-9]+]], x29, #64
; CHECK-NOFP: add     x8, [[TMP]], #0

  %addr = alloca i8, i64 %n

  call void @use_addr(i8* %addr)
; CHECK: bl use_addr

  ret void
; CHECK: sub sp, x29, #192
; CHECK: ldp x29, x30, [sp, #192]
; CHECK: add sp, sp, #208

; CHECK-NOFP: sub sp, x29, #64
; CHECK-NOFP: ldp x29, x30, [sp, #64]
; CHECK-NOFP: add sp, sp, #80
}

define void @test_alloca_large_frame(i64 %n) {
; CHECK-LABEL: test_alloca_large_frame:

; CHECK: sub sp, sp, #496
; CHECK: stp x29, x30, [sp, #480]
; CHECK: add x29, sp, #480
; CHECK: sub sp, sp, #48
; CHECK: sub sp, sp, #1953, lsl #12

  %addr1 = alloca i8, i64 %n
  %addr2 = alloca i64, i64 1000000

  call void @use_addr_loc(i8* %addr1, i64* %addr2)

  ret void
; CHECK: sub sp, x29, #480
; CHECK: ldp x29, x30, [sp, #480]
; CHECK: add sp, sp, #496
}

declare i8* @llvm.stacksave()
declare void @llvm.stackrestore(i8*)

define void @test_scoped_alloca(i64 %n) {
; CHECK-LABEL: test_scoped_alloca:
; CHECK: sub sp, sp, #32

  %sp = call i8* @llvm.stacksave()
; CHECK: mov [[SAVED_SP:x[0-9]+]], sp
; CHECK: mov [[OLDSP:x[0-9]+]], sp

  %addr = alloca i8, i64 %n
; CHECK: and [[SPDELTA:x[0-9]+]], {{x[0-9]+}}, #0xfffffffffffffff0
; CHECK-DAG: sub [[NEWSP:x[0-9]+]], [[OLDSP]], [[SPDELTA]]
; CHECK: mov sp, [[NEWSP]]

  call void @use_addr(i8* %addr)
; CHECK: bl use_addr

  call void @llvm.stackrestore(i8* %sp)
; CHECK: mov sp, [[SAVED_SP]]

  ret void
}
