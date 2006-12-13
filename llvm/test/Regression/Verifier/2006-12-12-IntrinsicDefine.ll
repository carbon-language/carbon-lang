; RUN: not llvm-as < %s
; PR1047

void %llvm.memcpy.i32(sbyte*, sbyte*, uint, uint) {
entry:
	ret void
}
