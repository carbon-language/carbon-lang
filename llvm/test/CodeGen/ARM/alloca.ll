; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm -mtriple=arm-linux-gnu | \
; RUN:   grep {mov r11, sp}
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm -mtriple=arm-linux-gnu | \
; RUN:   grep {mov sp, r11}

void %f(uint %a) {
entry:
	%tmp = alloca sbyte, uint %a
	call void %g( sbyte* %tmp, uint %a, uint 1, uint 2, uint 3 )
	ret void
}

declare void %g(sbyte*, uint, uint, uint, uint)
