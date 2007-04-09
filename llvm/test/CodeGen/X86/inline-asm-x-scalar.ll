; RUN: llvm-as < %s | llc -march=x86 -mcpu=yonah

define void @test1() {
        tail call void asm sideeffect "ucomiss $0", "x"( float 0x41E0000000000000)
        ret void
}

define void @test2() {
        %tmp53 = tail call i32 asm "ucomiss $1, $3\0Acmovae  $2, $0 ", "=r,mx,mr,x,0,~{dirflag},~{fpsr},~{flags},~{cc}"( float 0x41E0000000000000, i32 2147483647, float 0.000000e+00, i32 0 )         ; <i32> [#uses
        unreachable
}

define void @test3() {
        tail call void asm sideeffect "ucomiss $0, $1", "mx,x,~{dirflag},~{fpsr},~{flags},~{cc}"( float 0x41E0000000000000, i32 65536 )
        ret void
}

