; RUN: llvm-as < %s | llc -march=x86

define i32 @test1() {
	; Dest is AX, dest type = i32.
        %tmp4 = call i32 asm sideeffect "FROB $0", "={ax}"()
        ret i32 %tmp4
}

define void @test2(i32 %V) {
	; input is AX, in type = i32.
        call void asm sideeffect "FROB $0", "{ax}"(i32 %V)
        ret void
}

define void @test3() {
        ; FP constant as a memory operand.
        tail call void asm sideeffect "frob $0", "m"( float 0x41E0000000000000)
        ret void
}


