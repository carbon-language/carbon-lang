; RUN: llvm-as < %s | llc -fast-isel -mtriple=i386-apple-darwin -mattr=sse2 | \
; RUN:   grep mov | grep lazy_ptr | count 1

@src = external global i32

define i32 @loadgv() nounwind {
entry:
	%0 = load i32* @src, align 4
	%1 = load i32* @src, align 4
        %2 = add i32 %0, %1
  store i32 %2, i32* @src
	ret i32 %2
}
