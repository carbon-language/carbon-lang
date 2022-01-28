# RUN: llvm-as %S/Inputs/remangle1.ll -o %t.remangle1.bc
# RUN: llvm-as %S/Inputs/remangle2.ll -o %t.remangle2.bc
# RUN: llvm-link %t.remangle1.bc %t.remangle2.bc -o %t.remangle.linked.bc
# RUN: llvm-dis %t.remangle.linked.bc -o - | FileCheck %s

; CHECK-DAG: %fum.1 = type { %aab.0, i8, [7 x i8] }
; CHECK-DAG: %aab.0 = type { %aba }
; CHECK-DAG: %aba = type { [8 x i8] }
; CHECK-DAG: %fum.1.2 = type { %abb, i8, [7 x i8] }
; CHECK-DAG: %abb = type { %abc }
; CHECK-DAG: %abc = type { [4 x i8] }

; CHECK-LABEL: define void @foo1(%fum.1** %a, %fum.1.2** %b) {
; CHECK-NEXT:   %b.copy = call %fum.1.2** @llvm.ssa.copy.p0p0s_fum.1.2s(%fum.1.2** %b)
; CHECK-NEXT:   %a.copy = call %fum.1** @llvm.ssa.copy.p0p0s_fum.1s(%fum.1** %a)
; CHECK-NEXT:  ret void

; CHECK: declare %fum.1.2** @llvm.ssa.copy.p0p0s_fum.1.2s(%fum.1.2** returned)

; CHECK: declare %fum.1** @llvm.ssa.copy.p0p0s_fum.1s(%fum.1** returned)

; CHECK-LABEL: define void @foo2(%fum.1.2** %b, %fum.1** %a) {
; CHECK-NEXT:   %a.copy = call %fum.1** @llvm.ssa.copy.p0p0s_fum.1s(%fum.1** %a)
; CHECK-NEXT:  %b.copy = call %fum.1.2** @llvm.ssa.copy.p0p0s_fum.1.2s(%fum.1.2** %b)
; CHECK-NEXT:  ret void


