; Test the handling of unnamed short vector arguments.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s -check-prefix=CHECK-VEC
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s -check-prefix=CHECK-STACK

; This routine is called with two named vector argument (passed
; in %v24 and %v26) and two unnamed vector arguments (passed
; in the single-wide stack slots at 160 and 168).
declare void @bar(<4 x i8>, <4 x i8>, ...)

define void @foo() {
; CHECK-VEC-LABEL: foo:
; CHECK-VEC-DAG: vrepib %v24, 1
; CHECK-VEC-DAG: vrepib %v26, 2
; CHECK-VEC: brasl %r14, bar@PLT
;
; CHECK-STACK-LABEL: foo:
; CHECK-STACK: aghi %r15, -176
; CHECK-STACK-DAG: llihf [[REG1:%r[0-9]+]], 50529027
; CHECK-STACK-DAG: stg [[REG1]], 160(%r15)
; CHECK-STACK-DAG: llihf [[REG2:%r[0-9]+]], 67372036
; CHECK-STACK-DAG: stg [[REG2]], 168(%r15)
; CHECK-STACK: brasl %r14, bar@PLT

  call void (<4 x i8>, <4 x i8>, ...) @bar
              (<4 x i8> <i8 1, i8 1, i8 1, i8 1>,
               <4 x i8> <i8 2, i8 2, i8 2, i8 2>,
               <4 x i8> <i8 3, i8 3, i8 3, i8 3>,
               <4 x i8> <i8 4, i8 4, i8 4, i8 4>)
  ret void
}

