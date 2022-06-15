; RUN: llc %s --filetype=asm -o - | FileCheck %s
target triple = "dxil-unknown-unknown"

; CHECK: Function Attrs: nounwind readnone
; Function Attrs: norecurse nounwind readnone willreturn
define dso_local float @fma(float %0, float %1, float %2) local_unnamed_addr #0 {
  %4 = fmul float %0, %1
  %5 = fadd float %4, %2
  ret float %5
}

; CHECK: Function Attrs: nounwind readnone
; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

; CHECK: attributes #0 = { nounwind readnone }
; CHECK-NOT attributes #

attributes #0 = { norecurse nounwind readnone willreturn }
attributes #1 = { nofree nosync nounwind readnone speculatable willreturn }
