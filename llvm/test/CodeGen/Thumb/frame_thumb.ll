; RUN: llvm-as < %s | llc -mtriple=thumb-apple-darwin \
; RUN:     -disable-fp-elim | not grep {r11}
; RUN: llvm-as < %s | llc -mtriple=thumb-linux-gnueabi \
; RUN:     -disable-fp-elim | not grep {r11}

define i32 @f() {
entry:
	ret i32 10
}
