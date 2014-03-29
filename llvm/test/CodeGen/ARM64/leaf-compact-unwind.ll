; Use the -disable-cfi flag so that we get the compact unwind info in the
; emitted assembly. Compact unwind info is omitted when CFI directives
; are emitted.
;
; RUN: llc -march=arm64 -mtriple=arm64-apple-ios -disable-cfi < %s | FileCheck %s
;
; rdar://13070556

@bar = common global i32 0, align 4

; Leaf function with no stack allocation and no saving/restoring
; of non-volatile registers.
define i32 @foo1(i32 %a) #0 {
entry:
  %add = add nsw i32 %a, 42
  ret i32 %add
}

; Leaf function with stack allocation but no saving/restoring
; of non-volatile registers.
define i32 @foo2(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e, i32 %f, i32 %g, i32 %h) #0 {
entry:
  %stack = alloca [36 x i32], align 4
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv19 = phi i64 [ 0, %entry ], [ %indvars.iv.next20, %for.body ]
  %arrayidx = getelementptr inbounds [36 x i32]* %stack, i64 0, i64 %indvars.iv19
  %0 = trunc i64 %indvars.iv19 to i32
  store i32 %0, i32* %arrayidx, align 4, !tbaa !0
  %indvars.iv.next20 = add i64 %indvars.iv19, 1
  %lftr.wideiv21 = trunc i64 %indvars.iv.next20 to i32
  %exitcond22 = icmp eq i32 %lftr.wideiv21, 36
  br i1 %exitcond22, label %for.body4, label %for.body

for.body4:                                        ; preds = %for.body, %for.body4
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body4 ], [ 0, %for.body ]
  %z1.016 = phi i32 [ %add, %for.body4 ], [ 0, %for.body ]
  %arrayidx6 = getelementptr inbounds [36 x i32]* %stack, i64 0, i64 %indvars.iv
  %1 = load i32* %arrayidx6, align 4, !tbaa !0
  %add = add nsw i32 %1, %z1.016
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 36
  br i1 %exitcond, label %for.end9, label %for.body4

for.end9:                                         ; preds = %for.body4
  ret i32 %add
}

; Leaf function with no stack allocation but with saving restoring of
; non-volatile registers.
define i32 @foo3(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e, i32 %f, i32 %g, i32 %h) #1 {
entry:
  %0 = load volatile i32* @bar, align 4, !tbaa !0
  %1 = load volatile i32* @bar, align 4, !tbaa !0
  %2 = load volatile i32* @bar, align 4, !tbaa !0
  %3 = load volatile i32* @bar, align 4, !tbaa !0
  %4 = load volatile i32* @bar, align 4, !tbaa !0
  %5 = load volatile i32* @bar, align 4, !tbaa !0
  %6 = load volatile i32* @bar, align 4, !tbaa !0
  %7 = load volatile i32* @bar, align 4, !tbaa !0
  %8 = load volatile i32* @bar, align 4, !tbaa !0
  %9 = load volatile i32* @bar, align 4, !tbaa !0
  %10 = load volatile i32* @bar, align 4, !tbaa !0
  %11 = load volatile i32* @bar, align 4, !tbaa !0
  %12 = load volatile i32* @bar, align 4, !tbaa !0
  %13 = load volatile i32* @bar, align 4, !tbaa !0
  %14 = load volatile i32* @bar, align 4, !tbaa !0
  %15 = load volatile i32* @bar, align 4, !tbaa !0
  %16 = load volatile i32* @bar, align 4, !tbaa !0
  %17 = load volatile i32* @bar, align 4, !tbaa !0
  %factor = mul i32 %h, -2
  %factor56 = mul i32 %g, -2
  %factor57 = mul i32 %f, -2
  %factor58 = mul i32 %e, -2
  %factor59 = mul i32 %d, -2
  %factor60 = mul i32 %c, -2
  %factor61 = mul i32 %b, -2
  %sum = add i32 %1, %0
  %sum62 = add i32 %sum, %2
  %sum63 = add i32 %sum62, %3
  %sum64 = add i32 %sum63, %4
  %sum65 = add i32 %sum64, %5
  %sum66 = add i32 %sum65, %6
  %sum67 = add i32 %sum66, %7
  %sum68 = add i32 %sum67, %8
  %sum69 = add i32 %sum68, %9
  %sum70 = add i32 %sum69, %10
  %sum71 = add i32 %sum70, %11
  %sum72 = add i32 %sum71, %12
  %sum73 = add i32 %sum72, %13
  %sum74 = add i32 %sum73, %14
  %sum75 = add i32 %sum74, %15
  %sum76 = add i32 %sum75, %16
  %sub10 = sub i32 %17, %sum76
  %sub11 = add i32 %sub10, %factor
  %sub12 = add i32 %sub11, %factor56
  %sub13 = add i32 %sub12, %factor57
  %sub14 = add i32 %sub13, %factor58
  %sub15 = add i32 %sub14, %factor59
  %sub16 = add i32 %sub15, %factor60
  %add = add i32 %sub16, %factor61
  ret i32 %add
}

; Leaf function with stack allocation and saving/restoring of non-volatile
; registers.
define i32 @foo4(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e, i32 %f, i32 %g, i32 %h) #0 {
entry:
  %stack = alloca [128 x i32], align 4
  %0 = zext i32 %a to i64
  br label %for.body

for.cond2.preheader:                              ; preds = %for.body
  %1 = sext i32 %f to i64
  br label %for.body4

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv22 = phi i64 [ 0, %entry ], [ %indvars.iv.next23, %for.body ]
  %2 = add nsw i64 %indvars.iv22, %0
  %arrayidx = getelementptr inbounds [128 x i32]* %stack, i64 0, i64 %indvars.iv22
  %3 = trunc i64 %2 to i32
  store i32 %3, i32* %arrayidx, align 4, !tbaa !0
  %indvars.iv.next23 = add i64 %indvars.iv22, 1
  %lftr.wideiv25 = trunc i64 %indvars.iv.next23 to i32
  %exitcond26 = icmp eq i32 %lftr.wideiv25, 128
  br i1 %exitcond26, label %for.cond2.preheader, label %for.body

for.body4:                                        ; preds = %for.body4, %for.cond2.preheader
  %indvars.iv = phi i64 [ 0, %for.cond2.preheader ], [ %indvars.iv.next, %for.body4 ]
  %z1.018 = phi i32 [ 0, %for.cond2.preheader ], [ %add8, %for.body4 ]
  %4 = add nsw i64 %indvars.iv, %1
  %arrayidx7 = getelementptr inbounds [128 x i32]* %stack, i64 0, i64 %4
  %5 = load i32* %arrayidx7, align 4, !tbaa !0
  %add8 = add nsw i32 %5, %z1.018
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 128
  br i1 %exitcond, label %for.end11, label %for.body4

for.end11:                                        ; preds = %for.body4
  ret i32 %add8
}

attributes #0 = { readnone "target-cpu"="cyclone" }
attributes #1 = { "target-cpu"="cyclone" }

!0 = metadata !{metadata !"int", metadata !1}
!1 = metadata !{metadata !"omnipotent char", metadata !2}
!2 = metadata !{metadata !"Simple C/C++ TBAA"}

; CHECK:        .section        __LD,__compact_unwind,regular,debug
; CHECK:        .quad   _foo1                   ; Range Start
; CHECK:        .long   33554432                ; Compact Unwind Encoding: 0x2000000
; CHECK:        .quad   _foo2                   ; Range Start
; CHECK:        .long   33591296                ; Compact Unwind Encoding: 0x2009000
; CHECK:        .quad   _foo3                   ; Range Start
; CHECK:        .long   33570831                ; Compact Unwind Encoding: 0x200400f
; CHECK:        .quad   _foo4                   ; Range Start
; CHECK:        .long   33689616                ; Compact Unwind Encoding: 0x2021010
