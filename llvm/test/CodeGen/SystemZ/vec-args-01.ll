; Test the handling of named vector arguments.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s -check-prefix=CHECK-VEC
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s -check-prefix=CHECK-STACK

; This routine has 6 integer arguments, which fill up r2-r5 and
; the stack slot at offset 160, and 10 vector arguments, which
; fill up v24-v31 and the two double-wide stack slots at 168
; and 184.
declare void @bar(i64, i64, i64, i64, i64, i64,
                  <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>,
                  <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>,
                  <4 x i32>, <4 x i32>)

define void @foo() {
; CHECK-VEC-LABEL: foo:
; CHECK-VEC-DAG: vrepif %v24, 1
; CHECK-VEC-DAG: vrepif %v26, 2
; CHECK-VEC-DAG: vrepif %v28, 3
; CHECK-VEC-DAG: vrepif %v30, 4
; CHECK-VEC-DAG: vrepif %v25, 5
; CHECK-VEC-DAG: vrepif %v27, 6
; CHECK-VEC-DAG: vrepif %v29, 7
; CHECK-VEC-DAG: vrepif %v31, 8
; CHECK-VEC: brasl %r14, bar@PLT
;
; CHECK-STACK-LABEL: foo:
; CHECK-STACK: aghi %r15, -200
; CHECK-STACK-DAG: mvghi 160(%r15), 6
; CHECK-STACK-DAG: vrepif [[REG1:%v[0-9]+]], 9
; CHECK-STACK-DAG: vst [[REG1]], 168(%r15)
; CHECK-STACK-DAG: vrepif [[REG2:%v[0-9]+]], 10
; CHECK-STACK-DAG: vst [[REG2]], 184(%r15)
; CHECK-STACK: brasl %r14, bar@PLT

  call void @bar (i64 1, i64 2, i64 3, i64 4, i64 5, i64 6,
                  <4 x i32> <i32 1, i32 1, i32 1, i32 1>,
                  <4 x i32> <i32 2, i32 2, i32 2, i32 2>,
                  <4 x i32> <i32 3, i32 3, i32 3, i32 3>,
                  <4 x i32> <i32 4, i32 4, i32 4, i32 4>,
                  <4 x i32> <i32 5, i32 5, i32 5, i32 5>,
                  <4 x i32> <i32 6, i32 6, i32 6, i32 6>,
                  <4 x i32> <i32 7, i32 7, i32 7, i32 7>,
                  <4 x i32> <i32 8, i32 8, i32 8, i32 8>,
                  <4 x i32> <i32 9, i32 9, i32 9, i32 9>,
                  <4 x i32> <i32 10, i32 10, i32 10, i32 10>)
  ret void
}
