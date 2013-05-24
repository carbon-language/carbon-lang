; RUN: opt -objc-arc -S < %s | FileCheck %s

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
declare void @bar(i32 ()*)
declare void @use_alloca(i8**)

declare void @llvm.dbg.value(metadata, i64, metadata)

declare i8* @objc_msgSend(i8*, i8*, ...)


; In the presense of allocas, unconditionally remove retain/release pairs only
; if they are known safe in both directions. This prevents matching up an inner
; retain with the boundary guarding release in the following situation:
; 
; %A = alloca
; retain(%x)
; retain(%x) <--- Inner Retain
; store %x, %A
; %y = load %A
; ... DO STUFF ...
; release(%y)
; release(%x) <--- Guarding Release
;
; rdar://13750319

; CHECK: define void @test1a(i8* %x)
; CHECK: @objc_retain(i8* %x)
; CHECK: @objc_retain(i8* %x)
; CHECK: @objc_release(i8* %y)
; CHECK: @objc_release(i8* %x)
; CHECK: ret void
; CHECK: }
define void @test1a(i8* %x) {
entry:
  %A = alloca i8*
  tail call i8* @objc_retain(i8* %x)
  tail call i8* @objc_retain(i8* %x)
  store i8* %x, i8** %A, align 8
  %y = load i8** %A
  call void @use_alloca(i8** %A)
  call void @objc_release(i8* %y), !clang.imprecise_release !0
  call void @use_pointer(i8* %x)
  call void @objc_release(i8* %x), !clang.imprecise_release !0
  ret void
}

; CHECK: define void @test1b(i8* %x)
; CHECK: @objc_retain(i8* %x)
; CHECK: @objc_retain(i8* %x)
; CHECK: @objc_release(i8* %y)
; CHECK: @objc_release(i8* %x)
; CHECK: ret void
; CHECK: }
define void @test1b(i8* %x) {
entry:
  %A = alloca i8*
  %gep = getelementptr i8** %A, i32 0
  tail call i8* @objc_retain(i8* %x)
  tail call i8* @objc_retain(i8* %x)
  store i8* %x, i8** %gep, align 8
  %y = load i8** %A
  call void @use_alloca(i8** %A)
  call void @objc_release(i8* %y), !clang.imprecise_release !0
  call void @use_pointer(i8* %x)
  call void @objc_release(i8* %x), !clang.imprecise_release !0
  ret void
}


; CHECK: define void @test1c(i8* %x)
; CHECK: @objc_retain(i8* %x)
; CHECK: @objc_retain(i8* %x)
; CHECK: @objc_release(i8* %y)
; CHECK: @objc_release(i8* %x)
; CHECK: ret void
; CHECK: }
define void @test1c(i8* %x) {
entry:
  %A = alloca i8*, i32 3
  %gep = getelementptr i8** %A, i32 2
  tail call i8* @objc_retain(i8* %x)
  tail call i8* @objc_retain(i8* %x)
  store i8* %x, i8** %gep, align 8
  %y = load i8** %gep
  call void @use_alloca(i8** %A)
  call void @objc_release(i8* %y), !clang.imprecise_release !0
  call void @use_pointer(i8* %x)
  call void @objc_release(i8* %x), !clang.imprecise_release !0
  ret void
}


; CHECK: define void @test1d(i8* %x)
; CHECK: @objc_retain(i8* %x)
; CHECK: @objc_retain(i8* %x)
; CHECK: @objc_release(i8* %y)
; CHECK: @objc_release(i8* %x)
; CHECK: ret void
; CHECK: }
define void @test1d(i8* %x) {
entry:
  br i1 undef, label %use_allocaA, label %use_allocaB

use_allocaA:
  %allocaA = alloca i8*
  br label %exit

use_allocaB:
  %allocaB = alloca i8*
  br label %exit

exit:
  %A = phi i8** [ %allocaA, %use_allocaA ], [ %allocaB, %use_allocaB ]
  %gep = getelementptr i8** %A, i32 0
  tail call i8* @objc_retain(i8* %x)
  tail call i8* @objc_retain(i8* %x)
  store i8* %x, i8** %gep, align 8
  %y = load i8** %gep
  call void @use_alloca(i8** %A)
  call void @objc_release(i8* %y), !clang.imprecise_release !0
  call void @use_pointer(i8* %x)
  call void @objc_release(i8* %x), !clang.imprecise_release !0
  ret void
}

; CHECK: define void @test1e(i8* %x)
; CHECK: @objc_retain(i8* %x)
; CHECK: @objc_retain(i8* %x)
; CHECK: @objc_release(i8* %y)
; CHECK: @objc_release(i8* %x)
; CHECK: ret void
; CHECK: }
define void @test1e(i8* %x) {
entry:
  br i1 undef, label %use_allocaA, label %use_allocaB

use_allocaA:
  %allocaA = alloca i8*, i32 4
  br label %exit

use_allocaB:
  %allocaB = alloca i8*, i32 4
  br label %exit

exit:
  %A = phi i8** [ %allocaA, %use_allocaA ], [ %allocaB, %use_allocaB ]
  %gep = getelementptr i8** %A, i32 2
  tail call i8* @objc_retain(i8* %x)
  tail call i8* @objc_retain(i8* %x)
  store i8* %x, i8** %gep, align 8
  %y = load i8** %gep
  call void @use_alloca(i8** %A)
  call void @objc_release(i8* %y), !clang.imprecise_release !0
  call void @use_pointer(i8* %x)
  call void @objc_release(i8* %x), !clang.imprecise_release !0
  ret void
}

; CHECK: define void @test1f(i8* %x)
; CHECK: @objc_retain(i8* %x)
; CHECK: @objc_retain(i8* %x)
; CHECK: @objc_release(i8* %y)
; CHECK: @objc_release(i8* %x)
; CHECK: ret void
; CHECK: }
define void @test1f(i8* %x) {
entry:
  %allocaOne = alloca i8*
  %allocaTwo = alloca i8*
  %A = select i1 undef, i8** %allocaOne, i8** %allocaTwo
  tail call i8* @objc_retain(i8* %x)
  tail call i8* @objc_retain(i8* %x)
  store i8* %x, i8** %A, align 8
  %y = load i8** %A
  call void @use_alloca(i8** %A)
  call void @objc_release(i8* %y), !clang.imprecise_release !0
  call void @use_pointer(i8* %x)
  call void @objc_release(i8* %x), !clang.imprecise_release !0
  ret void
}

; Make sure that if a store is in a different basic block we handle known safe
; conservatively.


; CHECK: define void @test2a(i8* %x)
; CHECK: @objc_retain(i8* %x)
; CHECK: @objc_retain(i8* %x)
; CHECK: @objc_release(i8* %y)
; CHECK: @objc_release(i8* %x)
; CHECK: ret void
; CHECK: }
define void @test2a(i8* %x) {
entry:
  %A = alloca i8*
  store i8* %x, i8** %A, align 8
  %y = load i8** %A
  br label %bb1

bb1:
  br label %bb2

bb2:
  br label %bb3

bb3:
  tail call i8* @objc_retain(i8* %x)
  tail call i8* @objc_retain(i8* %x)
  call void @use_alloca(i8** %A)
  call void @objc_release(i8* %y), !clang.imprecise_release !0
  call void @use_pointer(i8* %x)
  call void @objc_release(i8* %x), !clang.imprecise_release !0
  ret void
}

; CHECK: define void @test2b(i8* %x)
; CHECK: @objc_retain(i8* %x)
; CHECK: @objc_retain(i8* %x)
; CHECK: @objc_release(i8* %y)
; CHECK: @objc_release(i8* %x)
; CHECK: ret void
; CHECK: }
define void @test2b(i8* %x) {
entry:
  %A = alloca i8*
  %gep1 = getelementptr i8** %A, i32 0
  store i8* %x, i8** %gep1, align 8
  %gep2 = getelementptr i8** %A, i32 0
  %y = load i8** %gep2
  br label %bb1

bb1:
  br label %bb2

bb2:
  br label %bb3

bb3:
  tail call i8* @objc_retain(i8* %x)
  tail call i8* @objc_retain(i8* %x)
  call void @use_alloca(i8** %A)
  call void @objc_release(i8* %y), !clang.imprecise_release !0
  call void @use_pointer(i8* %x)
  call void @objc_release(i8* %x), !clang.imprecise_release !0
  ret void
}

; CHECK: define void @test2c(i8* %x)
; CHECK: @objc_retain(i8* %x)
; CHECK: @objc_retain(i8* %x)
; CHECK: @objc_release(i8* %y)
; CHECK: @objc_release(i8* %x)
; CHECK: ret void
; CHECK: }
define void @test2c(i8* %x) {
entry:
  %A = alloca i8*, i32 3
  %gep1 = getelementptr i8** %A, i32 2
  store i8* %x, i8** %gep1, align 8
  %gep2 = getelementptr i8** %A, i32 2
  %y = load i8** %gep2
  tail call i8* @objc_retain(i8* %x)
  br label %bb1

bb1:
  br label %bb2

bb2:
  br label %bb3

bb3:
  tail call i8* @objc_retain(i8* %x)
  call void @use_alloca(i8** %A)
  call void @objc_release(i8* %y), !clang.imprecise_release !0
  call void @use_pointer(i8* %x)
  call void @objc_release(i8* %x), !clang.imprecise_release !0
  ret void
}

!0 = metadata !{}

declare i32 @__gxx_personality_v0(...)
