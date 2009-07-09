; RUN: llvm-as < %s | llc -mtriple=i686-pc-linux-gnu -relocation-model=pic | FileCheck %s

; CHECK: bar:
; CHECK: call	.Lllvm$1.$piclabel
; CHECK: popl	%ebx
; CHECK: addl	$_GLOBAL_OFFSET_TABLE_ + [.-.Lllvm$1.$piclabel], %ebx
; CHECK: call	foo@PLT


define void @bar() nounwind {
entry:
    call void(...)* @foo()
    br label %return
return:
    ret void
}

declare void @foo(...)
