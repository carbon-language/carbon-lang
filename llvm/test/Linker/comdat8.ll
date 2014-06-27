; RUN: not llvm-link %s %p/Inputs/comdat5.ll -S -o - 2>&1 | FileCheck %s
target datalayout = "e-m:w-p:32:32-i64:64-f80:32-n8:16:32-S32"
target triple = "i686-pc-windows-msvc"

$"\01??_7S@@6B@" = comdat largest
define void @some_name() {
  ret void
}
@"\01??_7S@@6B@" = alias i8* inttoptr (i32 ptrtoint (void ()* @some_name to i32) to i8*)
; CHECK: COMDAT key involves incomputable alias size.
