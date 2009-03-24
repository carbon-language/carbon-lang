; RUN: llvm-as < %s | opt -instcombine | llvm-dis > %t

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

