; RUN: llvm-as < %s | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

%FunTy = type i32(i8 @zext)

declare i16 @(sext) "test"(i16 @sext %arg)      ; Differ only by vararg
declare i16 "test2" (i16 %a1, i16 %a2)

implementation

define i32 %main(i32 %argc, i8 **%argv) {
    %val = trunc i32 %argc to i16
    %res = call i16 @sext (i16 @sext) *%test(i16 %val)
    %two = add i16 %res, %res
    %retVal = sext i16 %two to i32
    ret i32 %retVal
}
