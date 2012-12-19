; RUN: opt < %s -msan -msan-check-access-address=0 -S | FileCheck %s
; RUN: opt < %s -msan -msan-check-access-address=0 -msan-track-origins=1 -S | FileCheck -check-prefix=CHECK-ORIGINS %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; Check the presence of __msan_init
; CHECK: @llvm.global_ctors {{.*}} @__msan_init

; Check the presence and the linkage type of __msan_track_origins
; CHECK: @__msan_track_origins = weak_odr constant i32 0


; Check instrumentation of stores

define void @Store(i32* nocapture %p, i32 %x) nounwind uwtable {
entry:
  store i32 %x, i32* %p, align 4
  ret void
}

; CHECK: @Store
; CHECK: load {{.*}} @__msan_param_tls
; CHECK: store
; CHECK: store
; CHECK: ret void
; CHECK-ORIGINS: @Store
; CHECK-ORIGINS: load {{.*}} @__msan_param_tls
; CHECK-ORIGINS: store
; CHECK-ORIGINS: icmp
; CHECK-ORIGINS: br i1
; CHECK-ORIGINS: <label>
; CHECK-ORIGINS: store
; CHECK-ORIGINS: br label
; CHECK-ORIGINS: <label>
; CHECK-ORIGINS: store
; CHECK-ORIGINS: ret void


; Check instrumentation of aligned stores
; Shadow store has the same alignment as the original store; origin store
; does not specify explicit alignment.

define void @AlignedStore(i32* nocapture %p, i32 %x) nounwind uwtable {
entry:
  store i32 %x, i32* %p, align 32
  ret void
}

; CHECK: @AlignedStore
; CHECK: load {{.*}} @__msan_param_tls
; CHECK: store {{.*}} align 32
; CHECK: store {{.*}} align 32
; CHECK: ret void
; CHECK-ORIGINS: @AlignedStore
; CHECK-ORIGINS: load {{.*}} @__msan_param_tls
; CHECK-ORIGINS: store {{.*}} align 32
; CHECK-ORIGINS: icmp
; CHECK-ORIGINS: br i1
; CHECK-ORIGINS: <label>
; CHECK-ORIGINS-NOT: store {{.*}} align
; CHECK-ORIGINS: br label
; CHECK-ORIGINS: <label>
; CHECK-ORIGINS: store {{.*}} align 32
; CHECK-ORIGINS: ret void


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

; CHECK: @LoadAndCmp
; CHECK: = load
; CHECK: = load
; CHECK: call void @__msan_warning_noreturn()
; CHECK-NEXT: call void asm sideeffect
; CHECK-NEXT: unreachable
; CHECK: ret void

; Check that we store the shadow for the retval.
define i32 @ReturnInt() nounwind uwtable readnone {
entry:
  ret i32 123
}

; CHECK: @ReturnInt
; CHECK: store i32 0,{{.*}}__msan_retval_tls
; CHECK: ret i32

; Check that we get the shadow for the retval.
define void @CopyRetVal(i32* nocapture %a) nounwind uwtable {
entry:
  %call = tail call i32 @ReturnInt() nounwind
  store i32 %call, i32* %a, align 4
  ret void
}

; CHECK: @CopyRetVal
; CHECK: load{{.*}}__msan_retval_tls
; CHECK: store
; CHECK: store
; CHECK: ret void


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

; CHECK: @FuncWithPhi
; CHECK: = phi
; CHECK-NEXT: = phi
; CHECK: store
; CHECK: store
; CHECK: ret void

; Compute shadow for "x << 10"
define void @ShlConst(i32* nocapture %x) nounwind uwtable {
entry:
  %0 = load i32* %x, align 4
  %1 = shl i32 %0, 10
  store i32 %1, i32* %x, align 4
  ret void
}

; CHECK: @ShlConst
; CHECK: = load
; CHECK: = load
; CHECK: shl
; CHECK: shl
; CHECK: store
; CHECK: store
; CHECK: ret void

; Compute shadow for "10 << x": it should have 'sext i1'.
define void @ShlNonConst(i32* nocapture %x) nounwind uwtable {
entry:
  %0 = load i32* %x, align 4
  %1 = shl i32 10, %0
  store i32 %1, i32* %x, align 4
  ret void
}

; CHECK: @ShlNonConst
; CHECK: = load
; CHECK: = load
; CHECK: = sext i1
; CHECK: store
; CHECK: store
; CHECK: ret void

; SExt
define void @SExt(i32* nocapture %a, i16* nocapture %b) nounwind uwtable {
entry:
  %0 = load i16* %b, align 2
  %1 = sext i16 %0 to i32
  store i32 %1, i32* %a, align 4
  ret void
}

; CHECK: @SExt
; CHECK: = load
; CHECK: = load
; CHECK: = sext
; CHECK: = sext
; CHECK: store
; CHECK: store
; CHECK: ret void


; memset
define void @MemSet(i8* nocapture %x) nounwind uwtable {
entry:
  call void @llvm.memset.p0i8.i64(i8* %x, i8 42, i64 10, i32 1, i1 false)
  ret void
}

declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i32, i1) nounwind

; CHECK: @MemSet
; CHECK: call i8* @__msan_memset
; CHECK: ret void


; memcpy
define void @MemCpy(i8* nocapture %x, i8* nocapture %y) nounwind uwtable {
entry:
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %x, i8* %y, i64 10, i32 1, i1 false)
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i32, i1) nounwind

; CHECK: @MemCpy
; CHECK: call i8* @__msan_memcpy
; CHECK: ret void


; memmove is lowered to a call
define void @MemMove(i8* nocapture %x, i8* nocapture %y) nounwind uwtable {
entry:
  call void @llvm.memmove.p0i8.p0i8.i64(i8* %x, i8* %y, i64 10, i32 1, i1 false)
  ret void
}

declare void @llvm.memmove.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i32, i1) nounwind

; CHECK: @MemMove
; CHECK: call i8* @__msan_memmove
; CHECK: ret void


; Check that we propagate shadow for "select"

define i32 @Select(i32 %a, i32 %b, i32 %c) nounwind uwtable readnone {
entry:
  %tobool = icmp ne i32 %c, 0
  %cond = select i1 %tobool, i32 %a, i32 %b
  ret i32 %cond
}

; CHECK: @Select
; CHECK: select
; CHECK-NEXT: select
; CHECK: ret i32


define i8* @IntToPtr(i64 %x) nounwind uwtable readnone {
entry:
  %0 = inttoptr i64 %x to i8*
  ret i8* %0
}

; CHECK: @IntToPtr
; CHECK: load i64*{{.*}}__msan_param_tls
; CHECK-NEXT: inttoptr
; CHECK-NEXT: store i64{{.*}}__msan_retval_tls
; CHECK: ret i8


define i8* @IntToPtr_ZExt(i16 %x) nounwind uwtable readnone {
entry:
  %0 = inttoptr i16 %x to i8*
  ret i8* %0
}

; CHECK: @IntToPtr_ZExt
; CHECK: zext
; CHECK-NEXT: inttoptr
; CHECK: ret i8


; Check that we insert exactly one check on udiv
; (2nd arg shadow is checked, 1st arg shadow is propagated)

define i32 @Div(i32 %a, i32 %b) nounwind uwtable readnone {
entry:
  %div = udiv i32 %a, %b
  ret i32 %div
}

; CHECK: @Div
; CHECK: icmp
; CHECK: call void @__msan_warning
; CHECK-NOT: icmp
; CHECK: udiv
; CHECK-NOT: icmp
; CHECK: ret i32


; Check that we propagate shadow for x<0, x>=0, etc (i.e. sign bit tests)

define zeroext i1 @ICmpSLT(i32 %x) nounwind uwtable readnone {
  %1 = icmp slt i32 %x, 0
  ret i1 %1
}

; CHECK: @ICmpSLT
; CHECK: icmp slt
; CHECK-NOT: call void @__msan_warning
; CHECK: icmp slt
; CHECK-NOT: call void @__msan_warning
; CHECK: ret i1

define zeroext i1 @ICmpSGE(i32 %x) nounwind uwtable readnone {
  %1 = icmp sge i32 %x, 0
  ret i1 %1
}

; CHECK: @ICmpSGE
; CHECK: icmp slt
; CHECK-NOT: call void @__msan_warning
; CHECK: icmp sge
; CHECK-NOT: call void @__msan_warning
; CHECK: ret i1

define zeroext i1 @ICmpSGT(i32 %x) nounwind uwtable readnone {
  %1 = icmp sgt i32 0, %x
  ret i1 %1
}

; CHECK: @ICmpSGT
; CHECK: icmp slt
; CHECK-NOT: call void @__msan_warning
; CHECK: icmp sgt
; CHECK-NOT: call void @__msan_warning
; CHECK: ret i1

define zeroext i1 @ICmpSLE(i32 %x) nounwind uwtable readnone {
  %1 = icmp sle i32 0, %x
  ret i1 %1
}

; CHECK: @ICmpSLE
; CHECK: icmp slt
; CHECK-NOT: call void @__msan_warning
; CHECK: icmp sle
; CHECK-NOT: call void @__msan_warning
; CHECK: ret i1


; Check that loads from shadow have the same aligment as the original loads.

define i32 @ShadowLoadAlignmentLarge() nounwind uwtable {
  %y = alloca i32, align 64
  %1 = load volatile i32* %y, align 64
  ret i32 %1
}

; CHECK: @ShadowLoadAlignmentLarge
; CHECK: load i32* {{.*}} align 64
; CHECK: load volatile i32* {{.*}} align 64
; CHECK: ret i32

define i32 @ShadowLoadAlignmentSmall() nounwind uwtable {
  %y = alloca i32, align 2
  %1 = load volatile i32* %y, align 2
  ret i32 %1
}

; CHECK: @ShadowLoadAlignmentSmall
; CHECK: load i32* {{.*}} align 2
; CHECK: load volatile i32* {{.*}} align 2
; CHECK: ret i32


; Test vector manipulation instructions.
; Check that the same bit manipulation is applied to the shadow values.
; Check that there is a zero test of the shadow of %idx argument, where present.

define i32 @ExtractElement(<4 x i32> %vec, i32 %idx) {
  %x = extractelement <4 x i32> %vec, i32 %idx
  ret i32 %x
}

; CHECK: @ExtractElement
; CHECK: extractelement
; CHECK: call void @__msan_warning
; CHECK: extractelement
; CHECK: ret i32

define <4 x i32> @InsertElement(<4 x i32> %vec, i32 %idx, i32 %x) {
  %vec1 = insertelement <4 x i32> %vec, i32 %x, i32 %idx
  ret <4 x i32> %vec1
}

; CHECK: @InsertElement
; CHECK: insertelement
; CHECK: call void @__msan_warning
; CHECK: insertelement
; CHECK: ret <4 x i32>

define <4 x i32> @ShuffleVector(<4 x i32> %vec, <4 x i32> %vec1) {
  %vec2 = shufflevector <4 x i32> %vec, <4 x i32> %vec1,
                        <4 x i32> <i32 0, i32 4, i32 1, i32 5>
  ret <4 x i32> %vec2
}

; CHECK: @ShuffleVector
; CHECK: shufflevector
; CHECK-NOT: call void @__msan_warning
; CHECK: shufflevector
; CHECK: ret <4 x i32>


; Test bswap intrinsic instrumentation
define i32 @BSwap(i32 %x) nounwind uwtable readnone {
  %y = tail call i32 @llvm.bswap.i32(i32 %x)
  ret i32 %y
}

declare i32 @llvm.bswap.i32(i32) nounwind readnone

; CHECK: @BSwap
; CHECK-NOT: call void @__msan_warning
; CHECK: @llvm.bswap.i32
; CHECK-NOT: call void @__msan_warning
; CHECK: @llvm.bswap.i32
; CHECK-NOT: call void @__msan_warning
; CHECK: ret i32


; Store intrinsic.

define void @StoreIntrinsic(i8* %p, <4 x float> %x) nounwind uwtable {
  call void @llvm.x86.sse.storeu.ps(i8* %p, <4 x float> %x)
  ret void
}

declare void @llvm.x86.sse.storeu.ps(i8*, <4 x float>) nounwind

; CHECK: @StoreIntrinsic
; CHECK-NOT: br
; CHECK-NOT: = or
; CHECK: store <4 x i32> {{.*}} align 1
; CHECK: call void @llvm.x86.sse.storeu.ps
; CHECK: ret void


; Load intrinsic.

define <16 x i8> @LoadIntrinsic(i8* %p) nounwind uwtable {
  %call = call <16 x i8> @llvm.x86.sse3.ldu.dq(i8* %p)
  ret <16 x i8> %call
}

declare <16 x i8> @llvm.x86.sse3.ldu.dq(i8* %p) nounwind

; CHECK: @LoadIntrinsic
; CHECK: load <16 x i8>* {{.*}} align 1
; CHECK-NOT: br
; CHECK-NOT: = or
; CHECK: call <16 x i8> @llvm.x86.sse3.ldu.dq
; CHECK: store <16 x i8> {{.*}} @__msan_retval_tls
; CHECK: ret <16 x i8>

; CHECK-ORIGINS: @LoadIntrinsic
; CHECK-ORIGINS: [[ORIGIN:%[01-9a-z]+]] = load i32* {{.*}}
; CHECK-ORIGINS: call <16 x i8> @llvm.x86.sse3.ldu.dq
; CHECK-ORIGINS: store i32 {{.*}}[[ORIGIN]], i32* @__msan_retval_origin_tls
; CHECK-ORIGINS: ret <16 x i8>


; Simple NoMem intrinsic
; Check that shadow is OR'ed, and origin is Select'ed
; And no shadow checks!

define <8 x i16> @Paddsw128(<8 x i16> %a, <8 x i16> %b) nounwind uwtable {
  %call = call <8 x i16> @llvm.x86.sse2.padds.w(<8 x i16> %a, <8 x i16> %b)
  ret <8 x i16> %call
}

declare <8 x i16> @llvm.x86.sse2.padds.w(<8 x i16> %a, <8 x i16> %b) nounwind

; CHECK: @Paddsw128
; CHECK-NEXT: load <8 x i16>* {{.*}} @__msan_param_tls
; CHECK-NEXT: load <8 x i16>* {{.*}} @__msan_param_tls
; CHECK-NEXT: = or <8 x i16>
; CHECK-NEXT: call <8 x i16> @llvm.x86.sse2.padds.w
; CHECK-NEXT: store <8 x i16> {{.*}} @__msan_retval_tls
; CHECK-NEXT: ret <8 x i16>

; CHECK-ORIGINS: @Paddsw128
; CHECK-ORIGINS: load i32* {{.*}} @__msan_param_origin_tls
; CHECK-ORIGINS: load i32* {{.*}} @__msan_param_origin_tls
; CHECK-ORIGINS: = bitcast <8 x i16> {{.*}} to i128
; CHECK-ORIGINS-NEXT: = icmp ne i128 {{.*}}, 0
; CHECK-ORIGINS-NEXT: = select i1 {{.*}}, i32 {{.*}}, i32
; CHECK-ORIGINS: call <8 x i16> @llvm.x86.sse2.padds.w
; CHECK-ORIGINS: store i32 {{.*}} @__msan_retval_origin_tls
; CHECK-ORIGINS: ret <8 x i16>
