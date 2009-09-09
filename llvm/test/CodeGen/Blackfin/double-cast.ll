; RUN: llc < %s -march=bfin

declare i32 @printf(i8*, ...)

define i32 @main() {
	%1 = call i32 (i8*, ...)* @printf(i8* undef, double undef)
	ret i32 0
}
