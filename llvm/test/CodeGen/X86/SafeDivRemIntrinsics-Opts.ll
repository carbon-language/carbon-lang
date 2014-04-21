; RUN: llc < %s | FileCheck %s
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

%divovf32 = type { i32, i1 }

declare %divovf32 @llvm.safe.sdiv.i32(i32, i32) nounwind readnone
declare %divovf32 @llvm.safe.udiv.i32(i32, i32) nounwind readnone

; CHECK-LABEL: sdiv32_results_unused
; CHECK: entry
; CHECK-NEXT: ret
define void @sdiv32_results_unused(i32 %x, i32 %y) {
entry:
  %divr = call %divovf32 @llvm.safe.sdiv.i32(i32 %x, i32 %y)
  ret void
}

; CHECK-LABEL: sdiv32_div_result_unused
; CHECK-NOT: idiv
define i1 @sdiv32_div_result_unused(i32 %x, i32 %y) {
entry:
  %divr = call %divovf32 @llvm.safe.sdiv.i32(i32 %x, i32 %y)
  %bit = extractvalue %divovf32 %divr, 1
  ret i1 %bit
}

; CHECK-LABEL: sdiv32_flag_result_unused
; CHECK: idiv
define i32 @sdiv32_flag_result_unused(i32 %x, i32 %y) {
entry:
  %divr = call %divovf32 @llvm.safe.sdiv.i32(i32 %x, i32 %y)
  %div = extractvalue %divovf32 %divr, 0
  ret i32 %div
}

; CHECK-LABEL: sdiv32_result_returned
; CHECK: idiv
define %divovf32 @sdiv32_result_returned(i32 %x, i32 %y) {
entry:
  %divr = call %divovf32 @llvm.safe.sdiv.i32(i32 %x, i32 %y)
  ret %divovf32 %divr
}

; CHECK-LABEL: sdiv32_trap_relinked
; CHECK: %div.div{{min|z}}
define i32 @sdiv32_trap_relinked(i32 %x, i32 %y) {
entry:
  %divr = call %divovf32 @llvm.safe.sdiv.i32(i32 %x, i32 %y)
  %div = extractvalue %divovf32 %divr, 0
  %bit = extractvalue %divovf32 %divr, 1
  br i1 %bit, label %trap.bb, label %ok.bb
trap.bb:
  ret i32 7
ok.bb:
  ret i32 %div
}

; CHECK-LABEL: udiv32_results_unused
; CHECK: entry
; CHECK-NEXT: ret
define void @udiv32_results_unused(i32 %x, i32 %y) {
entry:
  %divr = call %divovf32 @llvm.safe.udiv.i32(i32 %x, i32 %y)
  ret void
}

; CHECK-LABEL: udiv32_div_result_unused
; CHECK-NOT: udiv{{[	 ]}}
define i1 @udiv32_div_result_unused(i32 %x, i32 %y) {
entry:
  %divr = call %divovf32 @llvm.safe.udiv.i32(i32 %x, i32 %y)
  %bit = extractvalue %divovf32 %divr, 1
  ret i1 %bit
}

; CHECK-LABEL: udiv32_flag_result_unused
; CHECK-NOT: cb
; CHECK: {{[ 	]}}div
define i32 @udiv32_flag_result_unused(i32 %x, i32 %y) {
entry:
  %divr = call %divovf32 @llvm.safe.udiv.i32(i32 %x, i32 %y)
  %div = extractvalue %divovf32 %divr, 0
  ret i32 %div
}

; CHECK-LABEL: udiv32_result_returned
; CHECK: {{[ 	]}}div
define %divovf32 @udiv32_result_returned(i32 %x, i32 %y) {
entry:
  %divr = call %divovf32 @llvm.safe.udiv.i32(i32 %x, i32 %y)
  ret %divovf32 %divr
}

; CHECK-LABEL: udiv32_trap_relinked
; CHECK: %div.divz 
define i32 @udiv32_trap_relinked(i32 %x, i32 %y) {
entry:
  %divr = call %divovf32 @llvm.safe.udiv.i32(i32 %x, i32 %y)
  %div = extractvalue %divovf32 %divr, 0
  %bit = extractvalue %divovf32 %divr, 1
  br i1 %bit, label %trap.bb, label %ok.bb
trap.bb:
  ret i32 7
ok.bb:
  ret i32 %div
}

!llvm.ident = !{!0}

!0 = metadata !{metadata !"clang version 3.5.0 "}
