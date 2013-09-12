; RUN: llvm-as < %s | llvm-dis | FileCheck %s

define <2 x i32> @main() {
  ret <2 x i32> select (<2 x i1> <i1 false, i1 undef>, <2 x i32> zeroinitializer, <2 x i32> <i32 0, i32 undef>)
}

; CHECK: define <2 x i32> @main() {
; CHECK:   ret <2 x i32> select (<2 x i1> <i1 false, i1 undef>, <2 x i32> zeroinitializer, <2 x i32> <i32 0, i32 undef>)
; CHECK: }
