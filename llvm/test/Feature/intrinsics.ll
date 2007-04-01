; RUN: llvm-upgrade < %s | llvm-as | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll


declare bool %llvm.isunordered.f32(float,float)
declare bool %llvm.isunordered.f64(double,double)

declare void %llvm.prefetch(sbyte*, uint, uint)

declare uint %llvm.ctpop.i8(ubyte)
declare uint %llvm.ctpop.i16(ushort)
declare uint %llvm.ctpop.i32(uint)
declare uint %llvm.ctpop.i64(ulong)

declare uint %llvm.cttz.i8(ubyte)
declare uint %llvm.cttz.i16(ushort)
declare uint %llvm.cttz.i32(uint)
declare uint %llvm.cttz.i64(ulong)

declare uint %llvm.ctlz.i8(ubyte)
declare uint %llvm.ctlz.i16(ushort)
declare uint %llvm.ctlz.i32(uint)
declare uint %llvm.ctlz.i64(ulong)

declare float %llvm.sqrt.f32(float)
declare double %llvm.sqrt.f64(double)

implementation

; Test llvm intrinsics
;
void %libm() {
        call bool %llvm.isunordered.f32(float 1.0, float 2.0)
        call bool %llvm.isunordered.f64(double 3.0, double 4.0)

	call void %llvm.prefetch(sbyte* null, uint 1, uint 3)

        call float %llvm.sqrt.f32(float 5.0)
        call double %llvm.sqrt.f64(double 6.0)

        call uint %llvm.ctpop.i8(ubyte 10)
        call uint %llvm.ctpop.i16(ushort 11)
        call uint %llvm.ctpop.i32(uint 12)
        call uint %llvm.ctpop.i64(ulong 13)

        call uint %llvm.ctlz.i8(ubyte 14)
        call uint %llvm.ctlz.i16(ushort 15)
        call uint %llvm.ctlz.i32(uint 16)
        call uint %llvm.ctlz.i64(ulong 17)

        call uint %llvm.cttz.i8(ubyte 18)
        call uint %llvm.cttz.i16(ushort 19)
        call uint %llvm.cttz.i32(uint 20)
        call uint %llvm.cttz.i64(ulong 21)
	ret void
}

; FIXME: test ALL the intrinsics in this file.
