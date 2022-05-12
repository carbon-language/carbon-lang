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

; CHECK-STACK: .LCPI0_0:
; CHECK-STACK:	.quad	795741901033570304      # 0xb0b0b0b00000000
; CHECK-STACK:	.quad	868082074056920076      # 0xc0c0c0c0c0c0c0c
; CHECK-STACK: .LCPI0_1:
; CHECK-STACK:	.quad	648518346341351424      # 0x900000000000000
; CHECK-STACK:	.quad	723390690146385920      # 0xa0a000000000000

; CHECK-STACK-LABEL: foo:
; CHECK-STACK: aghi %r15, -192

; CHECK-STACK-DAG: larl [[REG1:%r[0-9]+]], .LCPI0_0
; CHECK-STACK-DAG: vl [[VREG0:%v[0-9]+]], 0([[REG1]])
; CHECK-STACK-DAG: vst [[VREG0]], 176(%r15)

; CHECK-STACK-DAG: larl [[REG2:%r[0-9]+]], .LCPI0_1
; CHECK-STACK-DAG: vl [[VREG1:%v[0-9]+]], 0([[REG2]])
; CHECK-STACK-DAG: vst [[VREG1]], 160(%r15)

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
