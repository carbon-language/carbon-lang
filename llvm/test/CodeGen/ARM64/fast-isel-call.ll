; RUN: llc < %s -O0 -fast-isel-abort -mtriple=arm64-apple-darwin | FileCheck %s
; RUN: llc < %s -O0 -fast-isel-abort -mtriple=arm64_be-linux-gnu | FileCheck %s --check-prefix=CHECK-BE

define void @call0() nounwind {
entry:
  ret void
}

define void @foo0() nounwind {
entry:
; CHECK: foo0
; CHECK: bl _call0
  call void @call0()
  ret void
}

define i32 @call1(i32 %a) nounwind {
entry:
  %a.addr = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  %tmp = load i32* %a.addr, align 4
  ret i32 %tmp
}

define i32 @foo1(i32 %a) nounwind {
entry:
; CHECK: foo1
; CHECK: stur w0, [x29, #-4]
; CHECK-NEXT: ldur w0, [x29, #-4]
; CHECK-NEXT: bl _call1
  %a.addr = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  %tmp = load i32* %a.addr, align 4
  %call = call i32 @call1(i32 %tmp)
  ret i32 %call
}

define i32 @sext_(i8 %a, i16 %b) nounwind {
entry:
; CHECK: @sext_
; CHECK: sxtb w0, w0
; CHECK: sxth w1, w1
; CHECK: bl _foo_sext_
  call void @foo_sext_(i8 signext %a, i16 signext %b)
  ret i32 0
}

declare void @foo_sext_(i8 %a, i16 %b)

define i32 @zext_(i8 %a, i16 %b) nounwind {
entry:
; CHECK: @zext_
; CHECK: uxtb w0, w0
; CHECK: uxth w1, w1
  call void @foo_zext_(i8 zeroext %a, i16 zeroext %b)
  ret i32 0
}

declare void @foo_zext_(i8 %a, i16 %b)

define i32 @t1(i32 %argc, i8** nocapture %argv) {
entry:
; CHECK: @t1
; The last parameter will be passed on stack via i8.
; CHECK: strb w{{[0-9]+}}, [sp]
; CHECK-NEXT: bl _bar
  %call = call i32 @bar(i8 zeroext 0, i8 zeroext -8, i8 zeroext -69, i8 zeroext 28, i8 zeroext 40, i8 zeroext -70, i8 zeroext 28, i8 zeroext 39, i8 zeroext -41)
  ret i32 0
}

declare i32 @bar(i8 zeroext, i8 zeroext, i8 zeroext, i8 zeroext, i8 zeroext, i8 zeroext, i8 zeroext, i8 zeroext, i8 zeroext)

; Test materialization of integers.  Target-independent selector handles this.
define i32 @t2() {
entry:
; CHECK: @t2
; CHECK: movz x0, #0
; CHECK: orr w1, wzr, #0xfffffff8
; CHECK: orr w[[REG:[0-9]+]], wzr, #0x3ff
; CHECK: orr w[[REG2:[0-9]+]], wzr, #0x2
; CHECK: movz w[[REG3:[0-9]+]], #0
; CHECK: orr w[[REG4:[0-9]+]], wzr, #0x1
; CHECK: uxth w2, w[[REG]]
; CHECK: sxtb w3, w[[REG2]]
; CHECK: and w4, w[[REG3]], #0x1
; CHECK: and w5, w[[REG4]], #0x1
; CHECK: bl	_func2
  %call = call i32 @func2(i64 zeroext 0, i32 signext -8, i16 zeroext 1023, i8 signext -254, i1 zeroext 0, i1 zeroext 1)
  ret i32 0
}

declare i32 @func2(i64 zeroext, i32 signext, i16 zeroext, i8 signext, i1 zeroext, i1 zeroext)

declare void @callee_b0f(i8 %bp10, i8 %bp11, i8 %bp12, i8 %bp13, i8 %bp14, i8 %bp15, i8 %bp17, i8 %bp18, i8 %bp19)
define void @caller_b1f() {
entry:
  ; CHECK-BE: strb w{{.*}}, [sp, #7]
  call void @callee_b0f(i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 42)
  ret void
}
