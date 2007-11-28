; RUN: not llvm-as < %s |& not grep Asserti
; RUN: not llvm-as < %s |& grep Redefinition

define void @test() {
	%tmp.1 = add i32 0, 1
	br label %return
return:
	%tmp.1 = add i32 0, 1
	ret void
}

