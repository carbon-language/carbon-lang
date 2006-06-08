; RUN: llvm-as < %s | llc -march=x86

int %test1() {
	; Dest is AX, dest type = i32.
        %tmp4 = call int asm sideeffect "FROB $0", "={ax}"()
        ret int %tmp4
}

void %test2(int %V) {
	; input is AX, in type = i32.
        call void asm sideeffect "FROB $0", "{ax}"(int %V)
        ret void
}

