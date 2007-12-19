; RUN: llvm-as -o - %s | llc -march=cellspu > %t1.s
; RUN: grep cntb    %t1.s | count 3 &&
; RUN: grep andi    %t1.s | count 3 &&
; RUN: grep rotmi   %t1.s | count 2 &&
; RUN: grep rothmi  %t1.s | count 1

declare i32 @llvm.ctpop.i8(i8)
declare i32 @llvm.ctpop.i16(i16)
declare i32 @llvm.ctpop.i32(i32)

define i32 @test_i8(i8 %X) {
	call i32 @llvm.ctpop.i8(i8 %X)
	%Y = bitcast i32 %1 to i32
	ret i32 %Y
}

define i32 @test_i16(i16 %X) {
        call i32 @llvm.ctpop.i16(i16 %X)
	%Y = bitcast i32 %1 to i32
        ret i32 %Y
}

define i32 @test_i32(i32 %X) {
        call i32 @llvm.ctpop.i32(i32 %X)
	%Y = bitcast i32 %1 to i32
        ret i32 %Y
}

