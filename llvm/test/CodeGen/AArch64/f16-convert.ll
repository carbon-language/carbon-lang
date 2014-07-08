; RUN: llc < %s -mtriple=arm64-apple-ios -asm-verbose=false | FileCheck %s

define float @load0(i16* nocapture readonly %a) nounwind {
; CHECK-LABEL: load0:
; CHECK-NEXT: ldr [[HREG:h[0-9]+]], [x0]
; CHECK-NEXT: fcvt s0, [[HREG]]
; CHECK-NEXT: ret

  %tmp = load i16* %a, align 2
  %tmp1 = tail call float @llvm.convert.from.fp16(i16 %tmp)
  ret float %tmp1
}

define double @load1(i16* nocapture readonly %a) nounwind {
; CHECK-LABEL: load1:
; CHECK-NEXT: ldr [[HREG:h[0-9]+]], [x0]
; CHECK-NEXT: fcvt d0, [[HREG]]
; CHECK-NEXT: ret

  %tmp = load i16* %a, align 2
  %tmp1 = tail call float @llvm.convert.from.fp16(i16 %tmp)
  %conv = fpext float %tmp1 to double
  ret double %conv
}

define float @load2(i16* nocapture readonly %a, i32 %i) nounwind {
; CHECK-LABEL: load2:
; CHECK-NEXT: ldr [[HREG:h[0-9]+]], [x0, w1, sxtw #1]
; CHECK-NEXT: fcvt s0, [[HREG]]
; CHECK-NEXT: ret

  %idxprom = sext i32 %i to i64
  %arrayidx = getelementptr inbounds i16* %a, i64 %idxprom
  %tmp = load i16* %arrayidx, align 2
  %tmp1 = tail call float @llvm.convert.from.fp16(i16 %tmp)
  ret float %tmp1
}

define double @load3(i16* nocapture readonly %a, i32 %i) nounwind {
; CHECK-LABEL: load3:
; CHECK-NEXT: ldr [[HREG:h[0-9]+]], [x0, w1, sxtw #1]
; CHECK-NEXT: fcvt d0, [[HREG]]
; CHECK-NEXT: ret

  %idxprom = sext i32 %i to i64
  %arrayidx = getelementptr inbounds i16* %a, i64 %idxprom
  %tmp = load i16* %arrayidx, align 2
  %tmp1 = tail call float @llvm.convert.from.fp16(i16 %tmp)
  %conv = fpext float %tmp1 to double
  ret double %conv
}

define float @load4(i16* nocapture readonly %a, i64 %i) nounwind {
; CHECK-LABEL: load4:
; CHECK-NEXT: ldr [[HREG:h[0-9]+]], [x0, x1, lsl #1]
; CHECK-NEXT: fcvt s0, [[HREG]]
; CHECK-NEXT: ret

  %arrayidx = getelementptr inbounds i16* %a, i64 %i
  %tmp = load i16* %arrayidx, align 2
  %tmp1 = tail call float @llvm.convert.from.fp16(i16 %tmp)
  ret float %tmp1
}

define double @load5(i16* nocapture readonly %a, i64 %i) nounwind {
; CHECK-LABEL: load5:
; CHECK-NEXT: ldr [[HREG:h[0-9]+]], [x0, x1, lsl #1]
; CHECK-NEXT: fcvt d0, [[HREG]]
; CHECK-NEXT: ret

  %arrayidx = getelementptr inbounds i16* %a, i64 %i
  %tmp = load i16* %arrayidx, align 2
  %tmp1 = tail call float @llvm.convert.from.fp16(i16 %tmp)
  %conv = fpext float %tmp1 to double
  ret double %conv
}

define float @load6(i16* nocapture readonly %a) nounwind {
; CHECK-LABEL: load6:
; CHECK-NEXT: ldr [[HREG:h[0-9]+]], [x0, #20]
; CHECK-NEXT: fcvt s0, [[HREG]]
; CHECK-NEXT: ret

  %arrayidx = getelementptr inbounds i16* %a, i64 10
  %tmp = load i16* %arrayidx, align 2
  %tmp1 = tail call float @llvm.convert.from.fp16(i16 %tmp)
  ret float %tmp1
}

define double @load7(i16* nocapture readonly %a) nounwind {
; CHECK-LABEL: load7:
; CHECK-NEXT: ldr [[HREG:h[0-9]+]], [x0, #20]
; CHECK-NEXT: fcvt d0, [[HREG]]
; CHECK-NEXT: ret

  %arrayidx = getelementptr inbounds i16* %a, i64 10
  %tmp = load i16* %arrayidx, align 2
  %tmp1 = tail call float @llvm.convert.from.fp16(i16 %tmp)
  %conv = fpext float %tmp1 to double
  ret double %conv
}

define float @load8(i16* nocapture readonly %a) nounwind {
; CHECK-LABEL: load8:
; CHECK-NEXT: ldur [[HREG:h[0-9]+]], [x0, #-20]
; CHECK-NEXT: fcvt s0, [[HREG]]
; CHECK-NEXT: ret

  %arrayidx = getelementptr inbounds i16* %a, i64 -10
  %tmp = load i16* %arrayidx, align 2
  %tmp1 = tail call float @llvm.convert.from.fp16(i16 %tmp)
  ret float %tmp1
}

define double @load9(i16* nocapture readonly %a) nounwind {
; CHECK-LABEL: load9:
; CHECK-NEXT: ldur [[HREG:h[0-9]+]], [x0, #-20]
; CHECK-NEXT: fcvt d0, [[HREG]]
; CHECK-NEXT: ret

  %arrayidx = getelementptr inbounds i16* %a, i64 -10
  %tmp = load i16* %arrayidx, align 2
  %tmp1 = tail call float @llvm.convert.from.fp16(i16 %tmp)
  %conv = fpext float %tmp1 to double
  ret double %conv
}

declare float @llvm.convert.from.fp16(i16) nounwind readnone
