; RUN: llc < %s -march=x86-64 | FileCheck %s

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i32, i1) nounwind


; Variable memcpy's should lower to calls.
define i8* @test1(i8* %a, i8* %b, i64 %n) nounwind {
entry:
	tail call void @llvm.memcpy.p0i8.p0i8.i64( i8* %a, i8* %b, i64 %n, i32 1, i1 0 )
	ret i8* %a
        
; CHECK: test1:
; CHECK: memcpy
}

; Variable memcpy's should lower to calls.
define i8* @test2(i64* %a, i64* %b, i64 %n) nounwind {
entry:
	%tmp14 = bitcast i64* %a to i8*
	%tmp25 = bitcast i64* %b to i8*
	tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %tmp14, i8* %tmp25, i64 %n, i32 8, i1 0 )
	ret i8* %tmp14
        
; CHECK: test2:
; CHECK: memcpy
}

; Large constant memcpy's should lower to a call when optimizing for size.
; PR6623
define void @test3(i8* nocapture %A, i8* nocapture %B) nounwind optsize noredzone {
entry:
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %A, i8* %B, i64 64, i32 1, i1 false)
  ret void
; CHECK: test3:
; CHECK: memcpy
}

; Large constant memcpy's should be inlined when not optimizing for size.
define void @test4(i8* nocapture %A, i8* nocapture %B) nounwind noredzone {
entry:
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %A, i8* %B, i64 64, i32 1, i1 false)
  ret void
; CHECK: test4:
; CHECK: movq
; CHECK: movq
; CHECK: movq
; CHECK: movq
; CHECK: movq
; CHECK: movq
; CHECK: movq
; CHECK: movq
; CHECK: movq
; CHECK: movq
; CHECK: movq
; CHECK: movq
}

