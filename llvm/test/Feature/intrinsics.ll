; RUN: llvm-as %s -o - | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll


declare bool %llvm.isunordered(float, float)
declare bool %llvm.isunordered(double, double)

implementation

; Test llvm intrinsics
;
void %libm() {
        call bool %llvm.isunordered(float 0.0, float 1.0)
        call bool %llvm.isunordered(double 0.0, double 0x7FF8000000000000)
	ret void
}
