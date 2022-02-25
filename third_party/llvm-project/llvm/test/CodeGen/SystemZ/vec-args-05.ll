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
; CHECK-STACK: .LCPI0_0:
; CHECK-STACK: .quad	217020518463700992      # 0x303030300000000
; CHECK-STACK: .quad	289360691284934656      # 0x404040400000000
; CHECK-STACK-LABEL: foo:
; CHECK-STACK: aghi %r15, -176
; CHECK-STACK-DAG: larl [[REG1:%r[0-9]+]], .LCPI0_0
; CHECK-STACK-DAG: vl [[VREG:%v[0-9]+]], 0([[REG1]])
; CHECK-STACK-DAG: vst [[VREG]], 160(%r15)
; CHECK-STACK: brasl %r14, bar@PLT

  call void (<4 x i8>, <4 x i8>, ...) @bar
              (<4 x i8> <i8 1, i8 1, i8 1, i8 1>,
               <4 x i8> <i8 2, i8 2, i8 2, i8 2>,
               <4 x i8> <i8 3, i8 3, i8 3, i8 3>,
               <4 x i8> <i8 4, i8 4, i8 4, i8 4>)
  ret void
}

