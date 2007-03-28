; RUN: llvm-as < %s | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

%ZFunTy = type i33(i8 zext)
%SFunTy = type i33(i8 sext)

declare i16 @"test"(i16 sext %arg) sext 
declare i8  @"test2" (i16 zext %a2) zext 


define i33 @main(i33 %argc, i8 **%argv) {
    %val = trunc i33 %argc to i16
    %res = call i16 (i16 sext) sext *@test(i16 %val)
    %two = add i16 %res, %res
    %res2 = call i8 @test2(i16 %two zext) zext 
    %retVal = sext i16 %two to i33
    ret i33 %retVal
}
