; This testcase shows a bug where an common subexpression exists, but there
; is no shared dominator block that the expression can be hoisted out to.
;
; RUN: llvm-as < %s | opt -gcse | llvm-dis

int "test"(int %X, int %Y) {
	%Z = add int %X, %Y
	ret int %Z

Unreachable:
	%Q = add int %X, %Y
	ret int %Q
}
