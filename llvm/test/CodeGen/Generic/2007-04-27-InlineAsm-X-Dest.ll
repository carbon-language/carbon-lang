; RUN: llvm-as < %s | llc

; Test that we can have an "X" output constraint.

define void @test(i16 * %t) {
        call void asm sideeffect "fwait", "=*X,~{dirflag},~{fpsr},~{flags},~{memory}"( i16* %t )
        ret void
}
