; RUN: llc -relocation-model=static -verify-machineinstrs < %s -mcpu=pwr7 | FileCheck %s
; RUN: llc -relocation-model=static -verify-machineinstrs < %s -code-model=small -mcpu=pwr7 | FileCheck %s -check-prefix=SCM

target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"


define dso_local void @foo() nounwind noinline {
  ret void
}

define weak void @foo_weak() nounwind {
  ret void
}

; Calls to local function does not require the TOC restore 'nop'
define dso_local void @test_direct() nounwind readnone {
; CHECK-LABEL: test_direct:
  tail call void @foo() nounwind
; Because of tail call optimization, it can be 'b' instruction.
; CHECK: [[BR:b[l]?]] foo
; CHECK-NOT: nop
  ret void
}

; Calls to weak function requires a TOC restore 'nop' with all code models
; because the definition that gets choosen at link time may come from a
; different compilation unit that was compiled with PC Relative and has no TOC.
define dso_local void @test_weak() nounwind readnone {
  tail call void @foo_weak() nounwind
; CHECK-LABEL: test_weak:
; CHECK:       bl foo_weak
; CHECK-NEXT:  nop

; SCM-LABEL: test_weak:
; SCM:       bl foo_weak
; SCM-NEXT:  nop
  ret void
}

; Indirect calls requires a full stub creation
define dso_local void @test_indirect(void ()* nocapture %fp) nounwind {
; CHECK-LABEL: test_indirect:
  tail call void %fp() nounwind
; CHECK: ld [[FP:[0-9]+]], 0(3)
; CHECK: ld 11, 16(3)
; CHECK: ld 2, 8(3)
; CHECK-NEXT: mtctr [[FP]]
; CHECK-NEXT: bctrl
; CHECK-NEXT: ld 2, 40(1)
  ret void
}

; Absolute values must use the regular indirect call sequence
; The main purpose of this test is to ensure that BLA is not
; used on 64-bit SVR4 (as e.g. on Darwin).
define dso_local void @test_abs() nounwind {
; CHECK-LABEL: test_abs:
  tail call void inttoptr (i64 1024 to void ()*)() nounwind
; CHECK: ld [[FP:[0-9]+]], 1024(0)
; CHECK: ld 11, 1040(0)
; CHECK: ld 2, 1032(0)
; CHECK-NEXT: mtctr [[FP]]
; CHECK-NEXT: bctrl
; CHECK-NEXT: ld 2, 40(1)
  ret void
}

declare double @sin(double) nounwind

; External functions call should also have a 'nop'
define double @test_external(double %x) nounwind {
; CHECK-LABEL: test_external:
  %call = tail call double @sin(double %x) nounwind
; CHECK: bl sin
; CHECK-NEXT: nop
  ret double %call
}

; The 'ld 2, 40(1)' really must always come directly after the bctrl to make
; the unwinding code in libgcc happy.
@g = external global void ()*
declare void @h(i64)
define dso_local void @test_indir_toc_reload(i64 %x) {
  %1 = load void ()*, void ()** @g
  call void %1()
  call void @h(i64 %x)
  ret void

; CHECK-LABEL: @test_indir_toc_reload
; CHECK: bctrl
; CHECK-NEXT: ld 2, 40(1)
; CHECK: blr
}

