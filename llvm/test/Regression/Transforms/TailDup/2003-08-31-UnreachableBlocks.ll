; RUN: llvm-as < %s | opt -tailduplicate -disable-output

int %foo() {
entry:
	br label %return.i

after_ret.i:
	br label %return.i

return.i:
	%tmp.3 = cast int* null to int
	br label %return.i1

after_ret.i1:
	br label %return.i1

return.i1:
	%tmp.8 = sub int %tmp.3, 0
	ret int 0
}
