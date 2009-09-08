
; RUN: opt %s -verify | llvm-dis | grep noimplicitfloat
define void @f() noimplicitfloat {
}
