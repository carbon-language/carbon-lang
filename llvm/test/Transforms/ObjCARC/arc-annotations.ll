; This file consists of various tests which ensure that the objc-arc-annotations
; are working correctly. In the future, I will use this in other lit tests to
; check the data flow analysis of ARC.

; REQUIRES: asserts
; RUN: opt -S -objc-arc -enable-objc-arc-annotations < %s | FileCheck %s

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
; CHECK:   %0 = tail call i8* @objc_retain(i8* %a) #0, !llvm.arc.annotation.bottomup ![[ANN0:[0-9]+]], !llvm.arc.annotation.topdown ![[ANN1:[0-9]+]]
; CHECK:   call void @llvm.arc.annotation.bottomup.bbend(i8** @x, i8** @S_Use)
; CHECK:   call void @llvm.arc.annotation.topdown.bbend(i8** @x, i8** @S_Retain)
; CHECK: t:
; CHECK:   call void @llvm.arc.annotation.topdown.bbstart(i8** @x, i8** @S_Retain)
; CHECK:   call void @llvm.arc.annotation.bottomup.bbstart(i8** @x, i8** @S_Use)
; CHECK:   store float 2.000000e+00, float* %b, !llvm.arc.annotation.bottomup ![[ANN2:[0-9]+]]
; CHECK:   call void @llvm.arc.annotation.bottomup.bbend(i8** @x, i8** @S_Release)
; CHECK:   call void @llvm.arc.annotation.topdown.bbend(i8** @x, i8** @S_Retain)
; CHECK: f:
; CHECK:   call void @llvm.arc.annotation.topdown.bbstart(i8** @x, i8** @S_Retain)
; CHECK:   call void @llvm.arc.annotation.bottomup.bbstart(i8** @x, i8** @S_Use)
; CHECK:   store i32 7, i32* %x, !llvm.arc.annotation.bottomup ![[ANN2]]
; CHECK:   call void @llvm.arc.annotation.bottomup.bbend(i8** @x, i8** @S_Release)
; CHECK:   call void @llvm.arc.annotation.topdown.bbend(i8** @x, i8** @S_Retain)
; CHECK: return:
; CHECK:   call void @llvm.arc.annotation.topdown.bbstart(i8** @x, i8** @S_Retain)
; CHECK:   call void @llvm.arc.annotation.bottomup.bbstart(i8** @x, i8** @S_Release)
; CHECK:   call void @objc_release(i8* %c) #0, !llvm.arc.annotation.bottomup ![[ANN3:[0-9]+]], !llvm.arc.annotation.topdown ![[ANN4:[0-9]+]]
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

!0 = metadata !{}

; CHECK: ![[ANN0]] = metadata !{metadata !"(test0,%x)", metadata !"S_Use", metadata !"S_None"}
; CHECK: ![[ANN1]] = metadata !{metadata !"(test0,%x)", metadata !"S_None", metadata !"S_Retain"}
; CHECK: ![[ANN2]] = metadata !{metadata !"(test0,%x)", metadata !"S_Release", metadata !"S_Use"}
; CHECK: ![[ANN3]] = metadata !{metadata !"(test0,%x)", metadata !"S_None", metadata !"S_Release"}
; CHECK: ![[ANN4]] = metadata !{metadata !"(test0,%x)", metadata !"S_Retain", metadata !"S_None"}

