
declare bool %llvm.isnan(float)
declare bool %llvm.isnan(double)

declare bool %llvm.isunordered(float, float)
declare bool %llvm.isunordered(double, double)

implementation

; Test llvm intrinsics
;
void %libm() {
	call bool %llvm.isnan(float 0.0)
	call bool %llvm.isnan(double 10.0)
        call bool %llvm.isunordered(float 0.0, float 1.0)
        call bool %llvm.isunordered(double 0.0, double 1.0)
	ret void
}
