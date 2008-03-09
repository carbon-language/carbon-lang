; RUN: llvm-as < %s | llc -march=x86

define x86_fp80 @test1() {
        %tmp85 = call x86_fp80 asm sideeffect "fld0", "={st(0)}"()
        ret x86_fp80 %tmp85
}

define double @test2() {
        %tmp85 = call double asm sideeffect "fld0", "={st(0)}"()
        ret double %tmp85
}

define void @test3(x86_fp80 %X) {
        call void asm sideeffect "frob ", "{st(0)},~{dirflag},~{fpsr},~{flags}"( x86_fp80 %X)
        ret void
}

define void @test4(double %X) {
        call void asm sideeffect "frob ", "{st(0)},~{dirflag},~{fpsr},~{flags}"( double %X)
        ret void
}

define void @test5(double %X) {
        %Y = add double %X, 123.0
        call void asm sideeffect "frob ", "{st(0)},~{dirflag},~{fpsr},~{flags}"( double %Y)
        ret void
}


