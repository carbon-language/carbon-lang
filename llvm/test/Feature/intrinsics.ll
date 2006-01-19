; RUN: llvm-as %s -o - | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll


declare bool %llvm.isunordered(float, float)
declare bool %llvm.isunordered(double, double)
declare bool %llvm.isunordered.f32(float,float)
declare bool %llvm.isunordered.f64(double,double)

declare void %llvm.prefetch(sbyte*, uint, uint)

declare uint %llvm.ctpop(uint)
declare ubyte %llvm.ctpop.i8(ubyte)
declare ushort %llvm.ctpop.i16(ushort)
declare uint %llvm.ctpop.i32(uint)
declare ulong %llvm.ctpop.i64(ulong)

declare ushort %llvm.cttz(ushort)
declare ubyte %llvm.cttz.i8(ubyte)
declare ushort %llvm.cttz.i16(ushort)
declare uint %llvm.cttz.i32(uint)
declare ulong %llvm.cttz.i64(ulong)

declare ulong %llvm.ctlz(ulong)
declare ubyte %llvm.ctlz.i8(ubyte)
declare ushort %llvm.ctlz.i16(ushort)
declare uint %llvm.ctlz.i32(uint)
declare ulong %llvm.ctlz.i64(ulong)

declare float %llvm.sqrt(float)
declare double %llvm.sqrt(double)
declare float %llvm.sqrt.f32(float)
declare double %llvm.sqrt.f64(double)

implementation

; Test llvm intrinsics
;
void %libm() {
        call bool %llvm.isunordered(float 0.0, float 1.0)
        call bool %llvm.isunordered(double 0.0, double 0x7FF8000000000000)
        call bool %llvm.isunordered.f32(float 1.0, float 2.0)
        call bool %llvm.isunordered.f64(double 3.0, double 4.0)

	call void %llvm.prefetch(sbyte* null, uint 1, uint 3)

        call float %llvm.sqrt(float 4.0)
        call double %llvm.sqrt(double 4.0)
        call float %llvm.sqrt.f32(float 5.0)
        call double %llvm.sqrt.f64(double 6.0)

        call uint %llvm.ctpop(uint 3)
        call ubyte %llvm.ctpop.i8(ubyte 10)
        call ushort %llvm.ctpop.i16(ushort 11)
        call uint %llvm.ctpop.i32(uint 12)
        call ulong %llvm.ctpop.i64(ulong 13)

        call ulong %llvm.ctlz(ulong 65000)
        call ubyte %llvm.ctlz.i8(ubyte 14)
        call ushort %llvm.ctlz.i16(ushort 15)
        call uint %llvm.ctlz.i32(uint 16)
        call ulong %llvm.ctlz.i64(ulong 17)

        call ushort %llvm.cttz(ushort 7)
        call ubyte %llvm.cttz.i8(ubyte 18)
        call ushort %llvm.cttz.i16(ushort 19)
        call uint %llvm.cttz.i32(uint 20)
        call ulong %llvm.cttz.i64(ulong 21)
	ret void
}

; FIXME: test ALL the intrinsics in this file.
