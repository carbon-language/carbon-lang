; RUN: llvm-as %s -o %t.bc
; RUN: lli %t.bc > /dev/null

define i32 @main() {
	ret i32 0
}

