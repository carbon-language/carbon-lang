
; RUN: opt < %s -verify -S | grep noimplicitfloat
define void @f() noimplicitfloat {
}
