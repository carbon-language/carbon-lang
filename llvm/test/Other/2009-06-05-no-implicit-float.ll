
; RUN: llvm-as < %s | opt -verify | llvm-dis | grep noimplicitfloat
define void @f() noimplicitfloat {
}
