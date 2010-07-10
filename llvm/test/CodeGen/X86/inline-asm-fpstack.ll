; RUN: llc < %s -mtriple=i386-apple-darwin | FileCheck %s

; There should be no stack manipulations between the inline asm and ret.
; CHECK: test1
; CHECK: InlineAsm End
; CHECK-NEXT: ret
define x86_fp80 @test1() {
        %tmp85 = call x86_fp80 asm sideeffect "fld0", "={st(0)}"()
        ret x86_fp80 %tmp85
}

; CHECK: test2
; CHECK: InlineAsm End
; CHECK-NEXT: ret
define double @test2() {
        %tmp85 = call double asm sideeffect "fld0", "={st(0)}"()
        ret double %tmp85
}

; Setting up argument in st(0) should be a single fld.
; CHECK: test3
; CHECK: fld
; CHECK-NEXT: InlineAsm Start
; Asm consumes stack, nothing should be popped.
; CHECK: InlineAsm End
; CHECK-NOT: fstp
; CHECK: ret
define void @test3(x86_fp80 %X) {
        call void asm sideeffect "frob ", "{st(0)},~{dirflag},~{fpsr},~{flags}"( x86_fp80 %X)
        ret void
}

; CHECK: test4
; CHECK: fld
; CHECK-NEXT: InlineAsm Start
; CHECK: InlineAsm End
; CHECK-NOT: fstp
; CHECK: ret
define void @test4(double %X) {
        call void asm sideeffect "frob ", "{st(0)},~{dirflag},~{fpsr},~{flags}"( double %X)
        ret void
}

; Same as test3/4, but using value from fadd.
; The fadd can be done in xmm or x87 regs - we don't test that.
; CHECK: test5
; CHECK: InlineAsm End
; CHECK-NOT: fstp
; CHECK: ret
define void @test5(double %X) {
        %Y = fadd double %X, 123.0
        call void asm sideeffect "frob ", "{st(0)},~{dirflag},~{fpsr},~{flags}"( double %Y)
        ret void
}

; CHECK: test6
define void @test6(double %A, double %B, double %C, 
                   double %D, double %E) nounwind  {
entry:
; Uses the same value twice, should have one fstp after the asm.
; CHECK: foo
; CHECK: InlineAsm End
; CHECK-NEXT: fstp
; CHECK-NOT: fstp
	tail call void asm sideeffect "foo $0 $1", "f,f,~{dirflag},~{fpsr},~{flags}"( double %A, double %A ) nounwind 
; Uses two different values, should be in st(0)/st(1) and both be popped.
; CHECK: bar
; CHECK: InlineAsm End
; CHECK-NEXT: fstp
; CHECK-NEXT: fstp
	tail call void asm sideeffect "bar $0 $1", "f,f,~{dirflag},~{fpsr},~{flags}"( double %B, double %C ) nounwind 
; Uses two different values, one of which isn't killed in this asm, it
; should not be popped after the asm.
; CHECK: baz
; CHECK: InlineAsm End
; CHECK-NEXT: fstp
; CHECK-NOT: fstp
	tail call void asm sideeffect "baz $0 $1", "f,f,~{dirflag},~{fpsr},~{flags}"( double %D, double %E ) nounwind 
; This is the last use of %D, so it should be popped after.
; CHECK: baz
; CHECK: InlineAsm End
; CHECK-NEXT: fstp
; CHECK-NOT: fstp
; CHECK: ret
	tail call void asm sideeffect "baz $0", "f,~{dirflag},~{fpsr},~{flags}"( double %D ) nounwind 
	ret void
}

