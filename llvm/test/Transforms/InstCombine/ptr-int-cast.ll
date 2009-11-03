; RUN: opt < %s -instcombine -S > %t
target datalayout = "E-p:64:64:64-a0:0:8-f32:32:32-f64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-v64:64:64-v128:128:128"

define i1 @test1(i32 *%x) nounwind {
entry:
; RUN: grep {ptrtoint i32\\* %x to i64} %t
	%tmp = ptrtoint i32* %x to i1
	ret i1 %tmp
}

define i32* @test2(i128 %x) nounwind {
entry:
; RUN: grep {inttoptr i64 %.mp1 to i32\\*} %t
	%tmp = inttoptr i128 %x to i32*
	ret i32* %tmp
}

