; RUN: llc -march=mips64el -filetype=obj -mcpu=mips64r2 %s -o - | llvm-objdump -disassemble -triple mips64el - | FileCheck %s

; Sign extend from 32 to 64 was creating nonsense opcodes

; CHECK: sll ${{[0-9]+}}, ${{[0-9]+}}, 0

; ModuleID = '../sext.c'
;target datalayout = "e-p:64:64:64-i1:8:8-i8:8:32-i16:16:32-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v64:64:64-n32"
;target triple = "mips64el-unknown-linux"

define i64 @foo(i32 %ival) nounwind readnone {
entry:
  %conv = sext i32 %ival to i64
  ret i64 %conv
}
