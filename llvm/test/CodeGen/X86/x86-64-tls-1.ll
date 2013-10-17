; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu | FileCheck %s
@tm_nest_level = internal thread_local global i32 0
define i64 @z() nounwind {
; FIXME: The codegen here is primitive at best and could be much better.
; The add and the moves can be folded together.
; CHECK-DAG: movq    $tm_nest_level@TPOFF, %rcx
; CHECK-DAG: movq    %fs:0, %rax
; CHECK: addl    %ecx, %eax
  ret i64 and (i64 ptrtoint (i32* @tm_nest_level to i64), i64 100)
}
