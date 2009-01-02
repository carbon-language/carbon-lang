; RUN: llvm-as < %s | opt -functionattrs | llvm-dis | not grep {nocapture *%%q}
; RUN: llvm-as < %s | opt -functionattrs | llvm-dis | grep {nocapture *%%p}

@g = external global i32**

define i32 @f(i32* %p, i32* %q) {
	%a1 = alloca i32*
	%a2 = alloca i32**
	store i32* %p, i32** %a1
	store i32** %a1, i32*** %a2
	%reload1 = load i32*** %a2
	%reload2 = load i32** %reload1
	%load_p = load i32* %reload2
	store i32 0, i32* %reload2

	%b1 = alloca i32*
	%b2 = alloca i32**
	store i32* %q, i32** %b1
	store i32** %b1, i32*** %b2
	%reload3 = load i32*** %b2
	store i32** %reload3, i32*** @g
	ret i32 %load_p
}
