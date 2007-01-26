; RUN: llvm-as < %s | llc -march=arm &&
; RUN: llvm-as < %s | llc -march=arm -enable-thumb

define void @test1() {
    %tmp = alloca [ 64 x i32 ] , align 4
    ret void
}

define void @test2() {
    %tmp = alloca [ 4168 x i8 ] , align 4
    ret void
}
