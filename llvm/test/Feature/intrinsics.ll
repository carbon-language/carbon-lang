
declare bool %llvm.isnan(float)
declare bool %llvm.isnan(double)

implementation

; Test llvm intrinsics
;
void %libm() {
	call bool %llvm.isnan(float 0.0)
	call bool %llvm.isnan(double 10.0)
	ret void
}
