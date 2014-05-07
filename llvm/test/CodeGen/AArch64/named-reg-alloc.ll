; RUN: not llc < %s -mtriple=aarch64-linux-gnueabi 2>&1 | FileCheck %s
; arm64 has separate copy of this test

define i32 @get_stack() nounwind {
entry:
; FIXME: Include an allocatable-specific error message
; CHECK: Invalid register name global variable
	%sp = call i32 @llvm.read_register.i32(metadata !0)
  ret i32 %sp
}

declare i32 @llvm.read_register.i32(metadata) nounwind

!0 = metadata !{metadata !"x5\00"}
