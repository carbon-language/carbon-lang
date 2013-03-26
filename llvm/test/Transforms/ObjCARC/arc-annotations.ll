; This file consists of various tests which ensure that the objc-arc-annotations
; are working correctly. In the future, I will use this in other lit tests to
; check the data flow analysis of ARC.

; RUN: opt -S -objc-arc -enable-objc-arc-annotations < %s | FileCheck %s
; XFAIL: *

declare i8* @objc_retain(i8*)
declare i8* @objc_retainAutoreleasedReturnValue(i8*)
declare void @objc_release(i8*)
declare i8* @objc_autorelease(i8*)
declare i8* @objc_autoreleaseReturnValue(i8*)
declare void @objc_autoreleasePoolPop(i8*)
declare i8* @objc_autoreleasePoolPush()
declare i8* @objc_retainBlock(i8*)

declare i8* @objc_retainedObject(i8*)
declare i8* @objc_unretainedObject(i8*)
declare i8* @objc_unretainedPointer(i8*)

declare void @use_pointer(i8*)
declare void @callee()
declare void @callee_fnptr(void ()*)
declare void @invokee()
declare i8* @returner()

; Simple retain+release pair deletion, with some intervening control
; flow and harmless instructions.

; CHECK: define void @test0(
; CHECK: entry:
; CHECK:   call void @llvm.arc.annotation.bottomup.bbstart(i8** @x, i8** @S_None)
; CHECK:   %0 = tail call i8* @objc_retain(i8* %a) #0, !llvm.arc.annotation.bottomup !0, !llvm.arc.annotation.topdown !1
; CHECK:   call void @llvm.arc.annotation.bottomup.bbend(i8** @x, i8** @S_Use)
; CHECK:   call void @llvm.arc.annotation.topdown.bbend(i8** @x, i8** @S_Retain)
; CHECK: t:
; CHECK:   call void @llvm.arc.annotation.topdown.bbstart(i8** @x, i8** @S_Retain)
; CHECK:   call void @llvm.arc.annotation.bottomup.bbstart(i8** @x, i8** @S_Use)
; CHECK:   store float 2.000000e+00, float* %b, !llvm.arc.annotation.bottomup !2
; CHECK:   call void @llvm.arc.annotation.bottomup.bbend(i8** @x, i8** @S_Release)
; CHECK:   call void @llvm.arc.annotation.topdown.bbend(i8** @x, i8** @S_Retain)
; CHECK: f:
; CHECK:   call void @llvm.arc.annotation.topdown.bbstart(i8** @x, i8** @S_Retain)
; CHECK:   call void @llvm.arc.annotation.bottomup.bbstart(i8** @x, i8** @S_Use)
; CHECK:   store i32 7, i32* %x, !llvm.arc.annotation.bottomup !2
; CHECK:   call void @llvm.arc.annotation.bottomup.bbend(i8** @x, i8** @S_Release)
; CHECK:   call void @llvm.arc.annotation.topdown.bbend(i8** @x, i8** @S_Retain)
; CHECK: return:
; CHECK:   call void @llvm.arc.annotation.topdown.bbstart(i8** @x, i8** @S_Retain)
; CHECK:   call void @llvm.arc.annotation.bottomup.bbstart(i8** @x, i8** @S_Release)
; CHECK:   call void @objc_release(i8* %c) #0, !llvm.arc.annotation.bottomup !3, !llvm.arc.annotation.topdown !4
; CHECK:   call void @llvm.arc.annotation.topdown.bbend(i8** @x, i8** @S_None)
; CHECK: }
define void @test0(i32* %x, i1 %p) nounwind {
entry:
  %a = bitcast i32* %x to i8*
  %0 = call i8* @objc_retain(i8* %a) nounwind
  br i1 %p, label %t, label %f

t:
  store i8 3, i8* %a
  %b = bitcast i32* %x to float*
  store float 2.0, float* %b
  br label %return

f:
  store i32 7, i32* %x
  br label %return

return:
  %c = bitcast i32* %x to i8*
  call void @objc_release(i8* %c) nounwind
  ret void
}

; Like test0 but the release isn't always executed when the retain is,
; so the optimization is not safe.

; TODO: Make the objc_release's argument be %0.

; CHECK: define void @test1(
; CHECK: entry:
; CHECK:   call void @llvm.arc.annotation.bottomup.bbstart(i8** @x, i8** @S_None)
; CHECK:   %0 = tail call i8* @objc_retain(i8* %a) #0, !llvm.arc.annotation.bottomup !5, !llvm.arc.annotation.topdown !6
; CHECK:   call void @llvm.arc.annotation.bottomup.bbend(i8** @x, i8** @S_None)
; CHECK:   call void @llvm.arc.annotation.topdown.bbend(i8** @x, i8** @S_Retain)
; CHECK: t:
; CHECK:   call void @llvm.arc.annotation.topdown.bbstart(i8** @x, i8** @S_Retain)
; CHECK:   call void @llvm.arc.annotation.bottomup.bbstart(i8** @x, i8** @S_Use)
; CHECK:   store float 2.000000e+00, float* %b, !llvm.arc.annotation.bottomup !7
; CHECK:   call void @llvm.arc.annotation.bottomup.bbend(i8** @x, i8** @S_Release)
; CHECK:   call void @llvm.arc.annotation.topdown.bbend(i8** @x, i8** @S_Retain)
; CHECK: f:
; CHECK:   call void @llvm.arc.annotation.topdown.bbstart(i8** @x, i8** @S_Retain)
; CHECK:   call void @llvm.arc.annotation.bottomup.bbstart(i8** @x, i8** @S_None)
; CHECK:   call void @callee(), !llvm.arc.annotation.topdown !8
; CHECK:   call void @llvm.arc.annotation.bottomup.bbend(i8** @x, i8** @S_None)
; CHECK:   call void @llvm.arc.annotation.topdown.bbend(i8** @x, i8** @S_CanRelease)
; CHECK: return:
; CHECK:   call void @llvm.arc.annotation.topdown.bbstart(i8** @x, i8** @S_None)
; CHECK:   call void @llvm.arc.annotation.bottomup.bbstart(i8** @x, i8** @S_Release)
; CHECK:   call void @objc_release(i8* %c) #0, !llvm.arc.annotation.bottomup !9
; CHECK:   call void @llvm.arc.annotation.topdown.bbend(i8** @x, i8** @S_None)
; CHECK: alt_return:
; CHECK:   call void @llvm.arc.annotation.topdown.bbstart(i8** @x, i8** @S_None)
; CHECK:   call void @llvm.arc.annotation.topdown.bbend(i8** @x, i8** @S_None)
; CHECK: }
define void @test1(i32* %x, i1 %p, i1 %q) nounwind {
entry:
  %a = bitcast i32* %x to i8*
  %0 = call i8* @objc_retain(i8* %a) nounwind
  br i1 %p, label %t, label %f

t:
  store i8 3, i8* %a
  %b = bitcast i32* %x to float*
  store float 2.0, float* %b
  br label %return

f:
  store i32 7, i32* %x
  call void @callee()
  br i1 %q, label %return, label %alt_return

return:
  %c = bitcast i32* %x to i8*
  call void @objc_release(i8* %c) nounwind
  ret void

alt_return:
  ret void
}

; Don't do partial elimination into two different CFG diamonds.

; CHECK: define void @test1b(
; CHECK: entry:
; CHECK:   call void @llvm.arc.annotation.bottomup.bbstart(i8** @x, i8** @S_None)
; CHECK:   %0 = tail call i8* @objc_retain(i8* %x) #0, !llvm.arc.annotation.bottomup !10, !llvm.arc.annotation.topdown !11
; CHECK:   call void @llvm.arc.annotation.bottomup.bbend(i8** @x, i8** @S_None)
; CHECK:   call void @llvm.arc.annotation.topdown.bbend(i8** @x, i8** @S_Retain)
; CHECK: if.then:
; CHECK:   call void @llvm.arc.annotation.topdown.bbstart(i8** @x, i8** @S_Retain)
; CHECK:   call void @llvm.arc.annotation.bottomup.bbstart(i8** @x, i8** @S_CanRelease)
; CHECK:   tail call void @callee(), !llvm.arc.annotation.bottomup !12, !llvm.arc.annotation.topdown !13
; CHECK:   call void @llvm.arc.annotation.bottomup.bbend(i8** @x, i8** @S_Use)
; CHECK:   call void @llvm.arc.annotation.topdown.bbend(i8** @x, i8** @S_CanRelease)
; CHECK: if.end:
; CHECK:   call void @llvm.arc.annotation.topdown.bbstart(i8** @x, i8** @S_CanRelease)
; CHECK:   call void @llvm.arc.annotation.bottomup.bbstart(i8** @x, i8** @S_Use)
; CHECK:   call void @llvm.arc.annotation.bottomup.bbend(i8** @x, i8** @S_Use)
; CHECK:   call void @llvm.arc.annotation.topdown.bbend(i8** @x, i8** @S_CanRelease)
; CHECK: if.then3:
; CHECK:   call void @llvm.arc.annotation.topdown.bbstart(i8** @x, i8** @S_CanRelease)
; CHECK:   call void @llvm.arc.annotation.bottomup.bbstart(i8** @x, i8** @S_Use)
; CHECK:   tail call void @use_pointer(i8* %x), !llvm.arc.annotation.bottomup !14, !llvm.arc.annotation.topdown !15
; CHECK:   call void @llvm.arc.annotation.bottomup.bbend(i8** @x, i8** @S_MovableRelease)
; CHECK:   call void @llvm.arc.annotation.topdown.bbend(i8** @x, i8** @S_Use)
; CHECK: if.end5:
; CHECK:   call void @llvm.arc.annotation.topdown.bbstart(i8** @x, i8** @S_None)
; CHECK:   call void @llvm.arc.annotation.bottomup.bbstart(i8** @x, i8** @S_MovableRelease)
; CHECK:   tail call void @objc_release(i8* %x) #0, !clang.imprecise_release !16, !llvm.arc.annotation.bottomup !17
; CHECK:   call void @llvm.arc.annotation.topdown.bbend(i8** @x, i8** @S_None)
; CHECK: }
define void @test1b(i8* %x, i1 %p, i1 %q) {
entry:
  tail call i8* @objc_retain(i8* %x) nounwind
  br i1 %p, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void @callee()
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  br i1 %q, label %if.then3, label %if.end5

if.then3:                                         ; preds = %if.end
  tail call void @use_pointer(i8* %x)
  br label %if.end5

if.end5:                                          ; preds = %if.then3, %if.end
  tail call void @objc_release(i8* %x) nounwind, !clang.imprecise_release !0
  ret void
}

; Like test0 but the pointer is passed to an intervening call,
; so the optimization is not safe.

; CHECK: define void @test2(
; CHECK: entry:
; CHECK:   call void @llvm.arc.annotation.bottomup.bbstart(i8** @x, i8** @S_None)
; CHECK:   %e = tail call i8* @objc_retain(i8* %a) #0, !llvm.arc.annotation.bottomup !18, !llvm.arc.annotation.topdown !19
; CHECK:   call void @llvm.arc.annotation.bottomup.bbend(i8** @x, i8** @S_CanRelease)
; CHECK:   call void @llvm.arc.annotation.topdown.bbend(i8** @x, i8** @S_Retain)
; CHECK: t:
; CHECK:   call void @llvm.arc.annotation.topdown.bbstart(i8** @x, i8** @S_Retain)
; CHECK:   call void @llvm.arc.annotation.bottomup.bbstart(i8** @x, i8** @S_Use)
; CHECK:   store float 2.000000e+00, float* %b, !llvm.arc.annotation.bottomup !20
; CHECK:   call void @llvm.arc.annotation.bottomup.bbend(i8** @x, i8** @S_Release)
; CHECK:   call void @llvm.arc.annotation.topdown.bbend(i8** @x, i8** @S_Retain)
; CHECK: f:
; CHECK:   call void @llvm.arc.annotation.topdown.bbstart(i8** @x, i8** @S_Retain)
; CHECK:   call void @llvm.arc.annotation.bottomup.bbstart(i8** @x, i8** @S_CanRelease)
; CHECK:   call void @use_pointer(i8* %e), !llvm.arc.annotation.bottomup !21, !llvm.arc.annotation.topdown !22
; CHECK:   store float 3.000000e+00, float* %d, !llvm.arc.annotation.bottomup !20, !llvm.arc.annotation.topdown !23
; CHECK:   call void @llvm.arc.annotation.bottomup.bbend(i8** @x, i8** @S_Release)
; CHECK:   call void @llvm.arc.annotation.topdown.bbend(i8** @x, i8** @S_Use)
; CHECK: return:
; CHECK:   call void @llvm.arc.annotation.topdown.bbstart(i8** @x, i8** @S_Use)
; CHECK:   call void @llvm.arc.annotation.bottomup.bbstart(i8** @x, i8** @S_Release)
; CHECK:   call void @objc_release(i8* %c) #0, !llvm.arc.annotation.bottomup !24, !llvm.arc.annotation.topdown !25
; CHECK:   call void @llvm.arc.annotation.topdown.bbend(i8** @x, i8** @S_None)
; CHECK: }
define void @test2(i32* %x, i1 %p) nounwind {
entry:
  %a = bitcast i32* %x to i8*
  %e = call i8* @objc_retain(i8* %a) nounwind
  br i1 %p, label %t, label %f

t:
  store i8 3, i8* %a
  %b = bitcast i32* %x to float*
  store float 2.0, float* %b
  br label %return

f:
  store i32 7, i32* %x
  call void @use_pointer(i8* %e)
  %d = bitcast i32* %x to float*
  store float 3.0, float* %d
  br label %return

return:
  %c = bitcast i32* %x to i8*
  call void @objc_release(i8* %c) nounwind
  ret void
}

; Like test0 but the release is in a loop,
; so the optimization is not safe.

; TODO: For now, assume this can't happen.

; CHECK: define void @test3(
; CHECK: entry:
; CHECK:   call void @llvm.arc.annotation.bottomup.bbstart(i8** @x, i8** @S_None)
; CHECK:   tail call i8* @objc_retain(i8* %a) #0, !llvm.arc.annotation.bottomup !26, !llvm.arc.annotation.topdown !27
; CHECK:   call void @llvm.arc.annotation.bottomup.bbend(i8** @x, i8** @S_Release)
; CHECK:   call void @llvm.arc.annotation.topdown.bbend(i8** @x, i8** @S_Retain)
; CHECK: loop:
; CHECK:   call void @llvm.arc.annotation.topdown.bbstart(i8** @x, i8** @S_Retain)
; CHECK:   call void @llvm.arc.annotation.bottomup.bbstart(i8** @x, i8** @S_Release)
; CHECK:   call void @objc_release(i8* %c) #0, !llvm.arc.annotation.bottomup !28, !llvm.arc.annotation.topdown !29
; CHECK:   call void @llvm.arc.annotation.topdown.bbend(i8** @x, i8** @S_None)
; CHECK: return:
; CHECK:   call void @llvm.arc.annotation.topdown.bbstart(i8** @x, i8** @S_None)
; CHECK:   call void @llvm.arc.annotation.topdown.bbend(i8** @x, i8** @S_None)
; CHECK: }
define void @test3(i32* %x, i1* %q) nounwind {
entry:
  %a = bitcast i32* %x to i8*
  %0 = call i8* @objc_retain(i8* %a) nounwind
  br label %loop

loop:
  %c = bitcast i32* %x to i8*
  call void @objc_release(i8* %c) nounwind
  %j = load volatile i1* %q
  br i1 %j, label %loop, label %return

return:
  ret void
}

!0 = metadata !{}

; CHECK: !0 = metadata !{metadata !"(test0,%x)", metadata !"S_Use", metadata !"S_None"}
; CHECK: !1 = metadata !{metadata !"(test0,%x)", metadata !"S_None", metadata !"S_Retain"}
; CHECK: !2 = metadata !{metadata !"(test0,%x)", metadata !"S_Release", metadata !"S_Use"}
; CHECK: !3 = metadata !{metadata !"(test0,%x)", metadata !"S_None", metadata !"S_Release"}
; CHECK: !4 = metadata !{metadata !"(test0,%x)", metadata !"S_Retain", metadata !"S_None"}
; CHECK: !5 = metadata !{metadata !"(test1,%x)", metadata !"S_None", metadata !"S_None"}
; CHECK: !6 = metadata !{metadata !"(test1,%x)", metadata !"S_None", metadata !"S_Retain"}
; CHECK: !7 = metadata !{metadata !"(test1,%x)", metadata !"S_Release", metadata !"S_Use"}
; CHECK: !8 = metadata !{metadata !"(test1,%x)", metadata !"S_Retain", metadata !"S_CanRelease"}
; CHECK: !9 = metadata !{metadata !"(test1,%x)", metadata !"S_None", metadata !"S_Release"}
; CHECK: !10 = metadata !{metadata !"(test1b,%x)", metadata !"S_None", metadata !"S_None"}
; CHECK: !11 = metadata !{metadata !"(test1b,%x)", metadata !"S_None", metadata !"S_Retain"}
; CHECK: !12 = metadata !{metadata !"(test1b,%x)", metadata !"S_Use", metadata !"S_CanRelease"}
; CHECK: !13 = metadata !{metadata !"(test1b,%x)", metadata !"S_Retain", metadata !"S_CanRelease"}
; CHECK: !14 = metadata !{metadata !"(test1b,%x)", metadata !"S_MovableRelease", metadata !"S_Use"}
; CHECK: !15 = metadata !{metadata !"(test1b,%x)", metadata !"S_CanRelease", metadata !"S_Use"}
; CHECK: !16 = metadata !{}
; CHECK: !17 = metadata !{metadata !"(test1b,%x)", metadata !"S_None", metadata !"S_MovableRelease"}
; CHECK: !18 = metadata !{metadata !"(test2,%x)", metadata !"S_CanRelease", metadata !"S_None"}
; CHECK: !19 = metadata !{metadata !"(test2,%x)", metadata !"S_None", metadata !"S_Retain"}
; CHECK: !20 = metadata !{metadata !"(test2,%x)", metadata !"S_Release", metadata !"S_Use"}
; CHECK: !21 = metadata !{metadata !"(test2,%x)", metadata !"S_Use", metadata !"S_CanRelease"}
; CHECK: !22 = metadata !{metadata !"(test2,%x)", metadata !"S_Retain", metadata !"S_CanRelease"}
; CHECK: !23 = metadata !{metadata !"(test2,%x)", metadata !"S_CanRelease", metadata !"S_Use"}
; CHECK: !24 = metadata !{metadata !"(test2,%x)", metadata !"S_None", metadata !"S_Release"}
; CHECK: !25 = metadata !{metadata !"(test2,%x)", metadata !"S_Use", metadata !"S_None"}
; CHECK: !26 = metadata !{metadata !"(test3,%x)", metadata !"S_Release", metadata !"S_None"}
; CHECK: !27 = metadata !{metadata !"(test3,%x)", metadata !"S_None", metadata !"S_Retain"}
; CHECK: !28 = metadata !{metadata !"(test3,%x)", metadata !"S_None", metadata !"S_Release"}
; CHECK: !29 = metadata !{metadata !"(test3,%x)", metadata !"S_Retain", metadata !"S_None"}

