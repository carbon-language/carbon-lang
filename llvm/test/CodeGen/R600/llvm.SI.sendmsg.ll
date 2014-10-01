;RUN: llc < %s -march=r600 -mcpu=verde -verify-machineinstrs | FileCheck %s

; CHECK-LABEL: {{^}}main:
; CHECK: S_SENDMSG Gs(emit stream 0)
; CHECK: S_SENDMSG Gs(cut stream 1)
; CHECK: S_SENDMSG Gs(emit-cut stream 2)
; CHECK: S_SENDMSG Gs_done(nop)

define void @main() {
main_body:
  call void @llvm.SI.sendmsg(i32 34, i32 0);
  call void @llvm.SI.sendmsg(i32 274, i32 0);
  call void @llvm.SI.sendmsg(i32 562, i32 0);
  call void @llvm.SI.sendmsg(i32 3, i32 0);
  ret void
}

; Function Attrs: nounwind
declare void @llvm.SI.sendmsg(i32, i32) #0

attributes #0 = { nounwind }
