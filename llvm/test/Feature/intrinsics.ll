
declare bool %llvm.isunordered(float, float)
declare bool %llvm.isunordered(double, double)

implementation

; Test llvm intrinsics
;
void %libm() {
        call bool %llvm.isunordered(float 0.0, float 1.0)
        call bool %llvm.isunordered(double 0.0, double double 0x7FF8000000000000)
	ret void
}
