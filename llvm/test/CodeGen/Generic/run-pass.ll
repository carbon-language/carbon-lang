; RUN: llc < %s -debug-pass=Structure -run-pass=gc-lowering -o /dev/null 2>&1 | FileCheck %s

; CHECK: -gc-lowering
; CHECK: FunctionPass Manager
; CHECK-NEXT: Lower Garbage Collection Instructions
; CHECK-NEXT: Machine Function Analysis
; CHECK-NEXT: MIR Printing Pass
