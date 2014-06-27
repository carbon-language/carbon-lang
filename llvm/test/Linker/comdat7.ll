; RUN: not llvm-link %s %p/Inputs/comdat5.ll -S -o - 2>&1 | FileCheck %s
target datalayout = "e-m:w-p:32:32-i64:64-f80:32-n8:16:32-S32"
target triple = "i686-pc-windows-msvc"

$"\01??_7S@@6B@" = comdat largest
define void @"\01??_7S@@6B@"() {
  ret void
}
; CHECK: GlobalVariable required for data dependent selection!
