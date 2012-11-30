; RUN: opt < %s -msan -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; Check the presence of __msan_init
; CHECK: @llvm.global_ctors {{.*}} @__msan_init

; load followed by cmp: check that we load the shadow and call __msan_warning.
define void @LoadAndCmp(i32* nocapture %a) nounwind uwtable {
entry:
  %0 = load i32* %a, align 4
  %tobool = icmp eq i32 %0, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  tail call void (...)* @foo() nounwind
  br label %if.end

if.end:                                           ; preds = %entry, %if.then
  ret void
}

declare void @foo(...)

; CHECK: define void @LoadAndCmp
; CHECK: = load
; CHECK: = load
; CHECK: call void @__msan_warning_noreturn()
; CHECK-NEXT: call void asm sideeffect
; CHECK-NEXT: unreachable
; CHECK: }

; Check that we store the shadow for the retval.
define i32 @ReturnInt() nounwind uwtable readnone {
entry:
  ret i32 123
}

; CHECK: define i32 @ReturnInt()
; CHECK: store i32 0,{{.*}}__msan_retval_tls
; CHECK: }

; Check that we get the shadow for the retval.
define void @CopyRetVal(i32* nocapture %a) nounwind uwtable {
entry:
  %call = tail call i32 @ReturnInt() nounwind
  store i32 %call, i32* %a, align 4
  ret void
}

; CHECK: define void @CopyRetVal
; CHECK: load{{.*}}__msan_retval_tls
; CHECK: store
; CHECK: store
; CHECK: }


; Check that we generate PHIs for shadow.
define void @FuncWithPhi(i32* nocapture %a, i32* %b, i32* nocapture %c) nounwind uwtable {
entry:
  %tobool = icmp eq i32* %b, null
  br i1 %tobool, label %if.else, label %if.then

  if.then:                                          ; preds = %entry
  %0 = load i32* %b, align 4
  br label %if.end

  if.else:                                          ; preds = %entry
  %1 = load i32* %c, align 4
  br label %if.end

  if.end:                                           ; preds = %if.else, %if.then
  %t.0 = phi i32 [ %0, %if.then ], [ %1, %if.else ]
  store i32 %t.0, i32* %a, align 4
  ret void
}

; CHECK: define void @FuncWithPhi
; CHECK: = phi
; CHECK-NEXT: = phi
; CHECK: store
; CHECK: store
; CHECK: }

; Compute shadow for "x << 10"
define void @ShlConst(i32* nocapture %x) nounwind uwtable {
entry:
  %0 = load i32* %x, align 4
  %1 = shl i32 %0, 10
  store i32 %1, i32* %x, align 4
  ret void
}

; CHECK: define void @ShlConst
; CHECK: = load
; CHECK: = load
; CHECK: shl
; CHECK: shl
; CHECK: store
; CHECK: store
; CHECK: }

; Compute shadow for "10 << x": it should have 'sext i1'.
define void @ShlNonConst(i32* nocapture %x) nounwind uwtable {
entry:
  %0 = load i32* %x, align 4
  %1 = shl i32 10, %0
  store i32 %1, i32* %x, align 4
  ret void
}

; CHECK: define void @ShlNonConst
; CHECK: = load
; CHECK: = load
; CHECK: = sext i1
; CHECK: store
; CHECK: store
; CHECK: }

; SExt
define void @SExt(i32* nocapture %a, i16* nocapture %b) nounwind uwtable {
entry:
  %0 = load i16* %b, align 2
  %1 = sext i16 %0 to i32
  store i32 %1, i32* %a, align 4
  ret void
}

; CHECK: define void @SExt
; CHECK: = load
; CHECK: = load
; CHECK: = sext
; CHECK: = sext
; CHECK: store
; CHECK: store
; CHECK: }


; memset
define void @MemSet(i8* nocapture %x) nounwind uwtable {
entry:
  call void @llvm.memset.p0i8.i64(i8* %x, i8 42, i64 10, i32 1, i1 false)
  ret void
}

declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i32, i1) nounwind

; CHECK: define void @MemSet
; CHECK: call i8* @__msan_memset
; CHECK: }


; memcpy
define void @MemCpy(i8* nocapture %x, i8* nocapture %y) nounwind uwtable {
entry:
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %x, i8* %y, i64 10, i32 1, i1 false)
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i32, i1) nounwind

; CHECK: define void @MemCpy
; CHECK: call i8* @__msan_memcpy
; CHECK: }


; memmove is lowered to a call
define void @MemMove(i8* nocapture %x, i8* nocapture %y) nounwind uwtable {
entry:
  call void @llvm.memmove.p0i8.p0i8.i64(i8* %x, i8* %y, i64 10, i32 1, i1 false)
  ret void
}

declare void @llvm.memmove.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i32, i1) nounwind

; CHECK: define void @MemMove
; CHECK: call i8* @__msan_memmove
; CHECK: }


; Check that we propagate shadow for "select"

define i32 @Select(i32 %a, i32 %b, i32 %c) nounwind uwtable readnone {
entry:
  %tobool = icmp ne i32 %c, 0
  %cond = select i1 %tobool, i32 %a, i32 %b
  ret i32 %cond
}

; CHECK: define i32 @Select
; CHECK: select
; CHECK-NEXT: select
; CHECK: }


define i8* @IntToPtr(i64 %x) nounwind uwtable readnone {
entry:
  %0 = inttoptr i64 %x to i8*
  ret i8* %0
}

; CHECK: define i8* @IntToPtr
; CHECK: load i64*{{.*}}__msan_param_tls
; CHECK-NEXT: inttoptr
; CHECK-NEXT: store i64{{.*}}__msan_retval_tls
; CHECK: }


define i8* @IntToPtr_ZExt(i16 %x) nounwind uwtable readnone {
entry:
  %0 = inttoptr i16 %x to i8*
  ret i8* %0
}

; CHECK: define i8* @IntToPtr_ZExt
; CHECK: zext
; CHECK-NEXT: inttoptr
; CHECK: }


; Check that we insert exactly one check on udiv
; (2nd arg shadow is checked, 1st arg shadow is propagated)

define i32 @Div(i32 %a, i32 %b) nounwind uwtable readnone {
entry:
  %div = udiv i32 %a, %b
  ret i32 %div
}

; CHECK: define i32 @Div
; CHECK: icmp
; CHECK: br
; CHECK-NOT: icmp
; CHECK: udiv
; CHECK-NOT: icmp
; CHECK: }


; Check that we propagate shadow for x<0, x>=0, etc (i.e. sign bit tests)

define zeroext i1 @ICmpSLT(i32 %x) nounwind uwtable readnone {
  %1 = icmp slt i32 %x, 0
  ret i1 %1
}

; CHECK: define zeroext i1 @ICmpSLT
; CHECK: icmp slt
; CHECK: icmp slt
; CHECK-NOT: br
; CHECK: }

define zeroext i1 @ICmpSGE(i32 %x) nounwind uwtable readnone {
  %1 = icmp sge i32 %x, 0
  ret i1 %1
}

; CHECK: define zeroext i1 @ICmpSGE
; CHECK: icmp slt
; CHECK: icmp sge
; CHECK-NOT: br
; CHECK: }

define zeroext i1 @ICmpSGT(i32 %x) nounwind uwtable readnone {
  %1 = icmp sgt i32 0, %x
  ret i1 %1
}

; CHECK: define zeroext i1 @ICmpSGT
; CHECK: icmp slt
; CHECK: icmp sgt
; CHECK-NOT: br
; CHECK: }

define zeroext i1 @ICmpSLE(i32 %x) nounwind uwtable readnone {
  %1 = icmp sle i32 0, %x
  ret i1 %1
}

; CHECK: define zeroext i1 @ICmpSLE
; CHECK: icmp slt
; CHECK: icmp sle
; CHECK-NOT: br
; CHECK: }


; Check that loads from shadow have the same aligment as the original loads.

define i32 @ShadowLoadAlignmentLarge() nounwind uwtable {
  %y = alloca i32, align 64
  %1 = load volatile i32* %y, align 64
  ret i32 %1
}

; CHECK: define i32 @ShadowLoadAlignmentLarge
; CHECK: load i32* {{.*}} align 64
; CHECK: load volatile i32* {{.*}} align 64
; CHECK: }

define i32 @ShadowLoadAlignmentSmall() nounwind uwtable {
  %y = alloca i32, align 2
  %1 = load volatile i32* %y, align 2
  ret i32 %1
}

; CHECK: define i32 @ShadowLoadAlignmentSmall
; CHECK: load i32* {{.*}} align 2
; CHECK: load volatile i32* {{.*}} align 2
; CHECK: }


; Test vector manipulation instructions.

define i32 @ExtractElement(<4 x i32> %vec, i32 %idx) {
  %x = extractelement <4 x i32> %vec, i32 %idx
  ret i32 %x
}

; CHECK: define i32 @ExtractElement
; CHECK: extractelement
; CHECK: br
; CHECK: extractelement
; CHECK: }

define <4 x i32> @InsertElement(<4 x i32> %vec, i32 %idx, i32 %x) {
  %vec1 = insertelement <4 x i32> %vec, i32 %x, i32 %idx
  ret <4 x i32> %vec1
}

; CHECK: define <4 x i32> @InsertElement
; CHECK: insertelement
; CHECK: br
; CHECK: insertelement
; CHECK: }

define <4 x i32> @ShuffleVector(<4 x i32> %vec, <4 x i32> %vec1) {
  %vec2 = shufflevector <4 x i32> %vec, <4 x i32> %vec1,
                        <4 x i32> <i32 0, i32 4, i32 1, i32 5>
  ret <4 x i32> %vec2
}

; CHECK: define <4 x i32> @ShuffleVector
; CHECK: shufflevector
; CHECK-NOT: br
; CHECK: shufflevector
; CHECK: }
