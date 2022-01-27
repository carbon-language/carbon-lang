; RUN: opt -S -passes=globalopt < %s | FileCheck %s

; CHECK: @llvm.global_ctors = appending global [1 x {{.*}}@_GLOBAL__I_c
@llvm.global_ctors = appending global [3 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__I_a, i8* null }, { i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__I_b, i8* null }, { i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__I_c, i8* null }]

; CHECK: @tmp = local_unnamed_addr global i32 42
; CHECK: @tmp2 = local_unnamed_addr global i32 42
; CHECK: @tmp3 = global i32 42
@tmp = global i32 0
@tmp2 = global i32 0
@tmp3 = global i32 0
@ptrToTmp3 = global i32* null

define i32 @TheAnswerToLifeTheUniverseAndEverything() {
  ret i32 42
}

define void @_GLOBAL__I_a() {
enter:
  call void @_optimizable()
  call void @_not_optimizable()
  ret void
}

define void @_optimizable() {
enter:
  %valptr = alloca i32

  %val = call i32 @TheAnswerToLifeTheUniverseAndEverything()
  store i32 %val, i32* @tmp
  store i32 %val, i32* %valptr

  %0 = bitcast i32* %valptr to i8*
  %barr = call i8* @llvm.launder.invariant.group(i8* %0)
  %1 = getelementptr i8, i8* %barr, i32 0
  %2 = bitcast i8* %1 to i32*

  %val2 = load i32, i32* %2
  store i32 %val2, i32* @tmp2
  ret void
}

; We can't step through launder.invariant.group here, because that would change
; this load in @usage_of_globals()
; %val = load i32, i32* %ptrVal, !invariant.group !0
; into
; %val = load i32, i32* @tmp3, !invariant.group !0
; and then we could assume that %val and %val2 to be the same, which coud be
; false, because @changeTmp3ValAndCallBarrierInside() may change the value
; of @tmp3.
define void @_not_optimizable() {
enter:
  store i32 13, i32* @tmp3, !invariant.group !0

  %0 = bitcast i32* @tmp3 to i8*
  %barr = call i8* @llvm.launder.invariant.group(i8* %0)
  %1 = bitcast i8* %barr to i32*

  store i32* %1, i32** @ptrToTmp3
  store i32 42, i32* %1, !invariant.group !0

  ret void
}

define void @usage_of_globals() {
entry:
  %ptrVal = load i32*, i32** @ptrToTmp3
  %val = load i32, i32* %ptrVal, !invariant.group !0

  call void @changeTmp3ValAndCallBarrierInside()
  %val2 = load i32, i32* @tmp3, !invariant.group !0
  ret void;
}

@tmp4 = global i32 0

define void @_GLOBAL__I_b() {
enter:
  %val = call i32 @TheAnswerToLifeTheUniverseAndEverything()
  %p1 = bitcast i32* @tmp4 to i8*
  %p2 = call i8* @llvm.strip.invariant.group.p0i8(i8* %p1)
  %p3 = bitcast i8* %p2 to i32*
  store i32 %val, i32* %p3
  ret void
}

@tmp5 = global i32 0
@tmp6 = global i32* null
; CHECK: @tmp6 = local_unnamed_addr global i32* null

define i32* @_dont_return_param(i32* %p) {
  %p1 = bitcast i32* %p to i8*
  %p2 = call i8* @llvm.launder.invariant.group(i8* %p1)
  %p3 = bitcast i8* %p2 to i32*
  ret i32* %p3
}

; We should bail out if we return any pointers derived via invariant.group intrinsics at any point.
define void @_GLOBAL__I_c() {
enter:
  %tmp5 = call i32* @_dont_return_param(i32* @tmp5)
  store i32* %tmp5, i32** @tmp6
  ret void
}


declare void @changeTmp3ValAndCallBarrierInside()

declare i8* @llvm.launder.invariant.group(i8*)
declare i8* @llvm.strip.invariant.group.p0i8(i8*)

!0 = !{}
