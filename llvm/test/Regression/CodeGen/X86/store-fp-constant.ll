; RUN: llvm-as < %s | llc -march=x86 | not grep rodata
;
; Check that no FP constants in this testcase ends up in the 
; constant pool.
%G = external global float 


declare void %extfloat(float %F)
declare void %extdouble(double)

implementation

void %testfloatstore() {
        call void %extfloat(float 1234.4)
        call void %extdouble(double 1234.4123)
	store float 13.0123, float* %G
        ret void
}

