; RUN: not --crash llc < %s -mtriple=powerpc64-unknown-linux-gnu 2>&1 | FileCheck %s
; RUN: not --crash llc < %s -mtriple=powerpc64-unknown-linux-gnu 2>&1 | FileCheck %s

define i64 @get_reg() nounwind {
entry:
; FIXME: Include an allocatable-specific error message
; CHECK: Invalid register name global variable
        %reg = call i64 @llvm.read_register.i64(metadata !0)
  ret i64 %reg
}

declare i64 @llvm.read_register.i64(metadata) nounwind

!0 = !{!"r2\00"}
