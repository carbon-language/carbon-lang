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

