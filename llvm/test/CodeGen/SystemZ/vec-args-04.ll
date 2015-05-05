; Test the handling of named short vector arguments.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s -check-prefix=CHECK-VEC
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s -check-prefix=CHECK-STACK

; This routine has 12 vector arguments, which fill up %v24-%v31
; and the four single-wide stack slots starting at 160.
declare void @bar(<1 x i8>, <2 x i8>, <4 x i8>, <8 x i8>,
                  <1 x i8>, <2 x i8>, <4 x i8>, <8 x i8>,
                  <1 x i8>, <2 x i8>, <4 x i8>, <8 x i8>)

define void @foo() {
; CHECK-VEC-LABEL: foo:
; CHECK-VEC-DAG: vrepib %v24, 1
; CHECK-VEC-DAG: vrepib %v26, 2
; CHECK-VEC-DAG: vrepib %v28, 3
; CHECK-VEC-DAG: vrepib %v30, 4
; CHECK-VEC-DAG: vrepib %v25, 5
; CHECK-VEC-DAG: vrepib %v27, 6
; CHECK-VEC-DAG: vrepib %v29, 7
; CHECK-VEC-DAG: vrepib %v31, 8
; CHECK-VEC: brasl %r14, bar@PLT
;
; CHECK-STACK-LABEL: foo:
; CHECK-STACK: aghi %r15, -192
; CHECK-STACK-DAG: llihh [[REG1:%r[0-9]+]], 2304
; CHECK-STACK-DAG: stg [[REG1]], 160(%r15)
; CHECK-STACK-DAG: llihh [[REG2:%r[0-9]+]], 2570
; CHECK-STACK-DAG: stg [[REG2]], 168(%r15)
; CHECK-STACK-DAG: llihf [[REG3:%r[0-9]+]], 185273099
; CHECK-STACK-DAG: stg [[REG3]], 176(%r15)
; CHECK-STACK-DAG: llihf [[REG4:%r[0-9]+]], 202116108
; CHECK-STACK-DAG: oilf [[REG4]], 202116108
; CHECK-STACK-DAG: stg [[REG4]], 176(%r15)
; CHECK-STACK: brasl %r14, bar@PLT

  call void @bar (<1 x i8> <i8 1>,
                  <2 x i8> <i8 2, i8 2>,
                  <4 x i8> <i8 3, i8 3, i8 3, i8 3>,
                  <8 x i8> <i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4>,
                  <1 x i8> <i8 5>,
                  <2 x i8> <i8 6, i8 6>,
                  <4 x i8> <i8 7, i8 7, i8 7, i8 7>,
                  <8 x i8> <i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8>,
                  <1 x i8> <i8 9>,
                  <2 x i8> <i8 10, i8 10>,
                  <4 x i8> <i8 11, i8 11, i8 11, i8 11>,
                  <8 x i8> <i8 12, i8 12, i8 12, i8 12, i8 12, i8 12, i8 12, i8 12>)
  ret void
}
