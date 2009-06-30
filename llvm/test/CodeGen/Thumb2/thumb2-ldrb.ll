; RUN: llvm-as < %s | llc -march=thumb -mattr=+thumb2 | grep {ldrb r0} | count 7
; RUN: llvm-as < %s | llc -march=thumb -mattr=+thumb2 | grep mov | grep 1
; RUN: llvm-as < %s | llc -march=thumb -mattr=+thumb2 | not grep mvn
; RUN: llvm-as < %s | llc -march=thumb -mattr=+thumb2 | grep ldrb | grep lsl
; RUN: llvm-as < %s | llc -march=thumb -mattr=+thumb2 | grep lsr | not grep ldrb

define i8 @f1(i8* %v) {
entry:
        %tmp = load i8* %v
        ret i8 %tmp
}

define i8 @f2(i8* %v) {
entry:
        %tmp2 = getelementptr i8* %v, i8 1023
        %tmp = load i8* %tmp2
        ret i8 %tmp
}

define i8 @f3(i32 %base) {
entry:
        %tmp1 = add i32 %base, 4096
        %tmp2 = inttoptr i32 %tmp1 to i8*
        %tmp3 = load i8* %tmp2
        ret i8 %tmp3
}

define i8 @f4(i32 %base) {
entry:
        %tmp1 = sub i32 %base, 128
        %tmp2 = inttoptr i32 %tmp1 to i8*
        %tmp3 = load i8* %tmp2
        ret i8 %tmp3
}

define i8 @f5(i32 %base, i32 %offset) {
entry:
        %tmp1 = add i32 %base, %offset
        %tmp2 = inttoptr i32 %tmp1 to i8*
        %tmp3 = load i8* %tmp2
        ret i8 %tmp3
}

define i8 @f6(i32 %base, i32 %offset) {
entry:
        %tmp1 = shl i32 %offset, 2
        %tmp2 = add i32 %base, %tmp1
        %tmp3 = inttoptr i32 %tmp2 to i8*
        %tmp4 = load i8* %tmp3
        ret i8 %tmp4
}

define i8 @f7(i32 %base, i32 %offset) {
entry:
        %tmp1 = lshr i32 %offset, 2
        %tmp2 = add i32 %base, %tmp1
        %tmp3 = inttoptr i32 %tmp2 to i8*
        %tmp4 = load i8* %tmp3
        ret i8 %tmp4
}
