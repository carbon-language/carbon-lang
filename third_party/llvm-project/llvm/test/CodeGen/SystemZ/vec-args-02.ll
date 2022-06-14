; Test the handling of unnamed vector arguments.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s -check-prefix=CHECK-VEC
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s -check-prefix=CHECK-STACK

; This routine is called with two named vector argument (passed
; in %v24 and %v26) and two unnamed vector arguments (passed
; in the double-wide stack slots at 160 and 176).
declare void @bar(<4 x i32>, <4 x i32>, ...)

define void @foo() {
; CHECK-VEC-LABEL: foo:
; CHECK-VEC-DAG: vrepif %v24, 1
; CHECK-VEC-DAG: vrepif %v26, 2
; CHECK-VEC: brasl %r14, bar@PLT
;
; CHECK-STACK-LABEL: foo:
; CHECK-STACK: aghi %r15, -192
; CHECK-STACK-DAG: vrepif [[REG1:%v[0-9]+]], 3
; CHECK-STACK-DAG: vst [[REG1]], 160(%r15)
; CHECK-STACK-DAG: vrepif [[REG2:%v[0-9]+]], 4
; CHECK-STACK-DAG: vst [[REG2]], 176(%r15)
; CHECK-STACK: brasl %r14, bar@PLT

  call void (<4 x i32>, <4 x i32>, ...) @bar
              (<4 x i32> <i32 1, i32 1, i32 1, i32 1>,
               <4 x i32> <i32 2, i32 2, i32 2, i32 2>,
               <4 x i32> <i32 3, i32 3, i32 3, i32 3>,
               <4 x i32> <i32 4, i32 4, i32 4, i32 4>)
  ret void
}
