; RUN: llvm-as < %s | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

%ZFunTy = type i32(i8 zext)
%SFunTy = type i32(i8 sext)

declare i16 @"test"(i16 sext %arg) sext 
declare i8 @"test2" (i16 zext %a2) zext 

declare void @exit(i32) noreturn nounwind

define i32 @main(i32 %argc, i8 **%argv) nounwind inreg {
    %val = trunc i32 %argc to i16
    %res1 = call i16 (i16 sext) sext *@test(i16 %val)
    %two = add i16 %res1, %res1
    %res2 = call i8 @test2(i16 %two zext) zext 
    %retVal = sext i16 %two to i32
    ret i32 %retVal
}
