;RUN: llvm-as < %s | opt -analyze -print-cfg-only -disable-output
;PR 1497

define void @foo() {
entry:
	br label %return

return:		; preds = %entry
	ret void
}

