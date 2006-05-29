; RUN: llvm-as < %s | opt -lcssa | llvm-dis | grep "%lcssa = phi int" &&
; RUN: llvm-as < %s | opt -lcssa | llvm-dis | grep "%X4 = add int 3, %lcssa"

void %lcssa(bool %S2) {
entry:
	br label %loop.interior

loop.interior:		; preds = %entry
	br bool %S2, label %if.true, label %if.false
	
if.true:
	%X1 = add int 0, 0
	br label %post.if

if.false:
	%X2 = add int 0, 1
	br label %post.if

post.if:
	%X3 = phi int [%X1, %if.true], [%X2, %if.false]
	br bool %S2, label %loop.exit, label %loop.interior

loop.exit:
	%X4 = add int 3, %X3
	ret void
}