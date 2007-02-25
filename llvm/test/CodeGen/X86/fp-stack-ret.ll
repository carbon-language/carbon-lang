; RUN: llvm-as < %s | llc -mtriple=i686-apple-darwin8 -march=x86 | grep fldl &&
; RUN: llvm-as < %s | llc -mtriple=i686-apple-darwin8 -march=x86 | not grep xmm &&
; RUN: llvm-as < %s | llc -mtriple=i686-apple-darwin8 -march=x86 | not grep 'sub.*esp'

; These testcases shouldn't require loading into an XMM register then storing 
; to memory, then reloading into an FPStack reg.

define double @test1(double *%P) {
        %A = load double* %P
        ret double %A
}

