; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu | FileCheck %s
@tm_nest_level = internal thread_local global i32 0
define i64 @z() nounwind {
; CHECK: movl    $tm_nest_level@TPOFF, %ecx
  ret i64 and (i64 ptrtoint (i32* @tm_nest_level to i64), i64 100)
}
