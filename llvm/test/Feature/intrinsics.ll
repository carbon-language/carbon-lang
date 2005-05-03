; RUN: llvm-as %s -o - | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll


declare bool %llvm.isunordered(float, float)
declare bool %llvm.isunordered(double, double)

declare void %llvm.prefetch(sbyte*, uint, uint)

declare uint %llvm.ctpop.32(uint)
declare ushort %llvm.cttz.16(ushort)
declare ulong %llvm.ctlz.64(ulong)

implementation

; Test llvm intrinsics
;
void %libm() {
        call bool %llvm.isunordered(float 0.0, float 1.0)
        call bool %llvm.isunordered(double 0.0, double 0x7FF8000000000000)
	call void %llvm.prefetch(sbyte* null, uint 1, uint 3)
        call uint %llvm.ctpop(uint 3)
        call ushort %llvm.cttz(ushort 7)
        call ulong %llvm.ctlz(ulong 65000)
	ret void
}
