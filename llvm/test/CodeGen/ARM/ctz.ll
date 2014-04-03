; RUN: llc -mtriple=arm-eabi -mattr=+v6t2 %s -o - | FileCheck %s

declare i32 @llvm.cttz.i32(i32, i1)

define i32 @f1(i32 %a) {
; CHECK-LABEL: f1:
; CHECK: rbit
; CHECK: clz
  %tmp = call i32 @llvm.cttz.i32( i32 %a, i1 true )
  ret i32 %tmp
}
