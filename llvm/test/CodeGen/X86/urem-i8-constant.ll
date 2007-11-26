; RUN: llvm-as < %s | llc -march=x86 | not grep mul

define i8 @foo(i8 %tmp325) {
	%t546 = urem i8 %tmp325, 37
	ret i8 %t546
}
