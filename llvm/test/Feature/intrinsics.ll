declare bool "llvm.isunordered"(float, float)

implementation

; Test llvm intrinsics
;
void "void"(int, int)
begin
	%c = call bool %llvm.isunordered(float 0.0, float 1.0)
	ret void
end

