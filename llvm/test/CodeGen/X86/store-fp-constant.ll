; RUN: llvm-upgrade < %s | llvm-as | llc -march=x86 | not grep rodata
; RUN: llvm-upgrade < %s | llvm-as | llc -march=x86 | not grep literal
;
; Check that no FP constants in this testcase ends up in the 
; constant pool.
%G = external global float 


declare void %extfloat(float %F)
declare void %extdouble(double)

implementation

void %testfloatstore() {
        call void %extfloat(float 0x40934999A0000000)
        call void %extdouble(double 0x409349A631F8A090)
	store float 0x402A064C20000000, float* %G
        ret void
}

