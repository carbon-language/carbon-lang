; RUN: llvm-as < %s | llc -march=thumb -mattr=+thumb2 | grep {ldr\\.w r0} | count 6
; RUN: llvm-as < %s | llc -march=thumb -mattr=+thumb2 | grep {ldr r0} | count 1
; RUN: llvm-as < %s | llc -march=thumb -mattr=+thumb2 | grep mov\\.w | grep 1
; RUN: llvm-as < %s | llc -march=thumb -mattr=+thumb2 | not grep mvn\\.w
; RUN: llvm-as < %s | llc -march=thumb -mattr=+thumb2 | grep ldr\\.w | grep lsl
; RUN: llvm-as < %s | llc -march=thumb -mattr=+thumb2 | grep lsr\\.w | not grep ldr

define i32 @f1(i32* %v) {
entry:
        %tmp = load i32* %v
        ret i32 %tmp
}

define i32 @f2(i32* %v) {
entry:
        %tmp2 = getelementptr i32* %v, i32 1023
        %tmp = load i32* %tmp2
        ret i32 %tmp
}

define i32 @f3(i32* %v) {
entry:
        %tmp2 = getelementptr i32* %v, i32 1024
        %tmp = load i32* %tmp2
        ret i32 %tmp
}

define i32 @f4(i32 %base) {
entry:
        %tmp1 = sub i32 %base, 128
        %tmp2 = inttoptr i32 %tmp1 to i32*
        %tmp3 = load i32* %tmp2
        ret i32 %tmp3
}

define i32 @f5(i32 %base, i32 %offset) {
entry:
        %tmp1 = add i32 %base, %offset
        %tmp2 = inttoptr i32 %tmp1 to i32*
        %tmp3 = load i32* %tmp2
        ret i32 %tmp3
}

define i32 @f6(i32 %base, i32 %offset) {
entry:
        %tmp1 = shl i32 %offset, 2
        %tmp2 = add i32 %base, %tmp1
        %tmp3 = inttoptr i32 %tmp2 to i32*
        %tmp4 = load i32* %tmp3
        ret i32 %tmp4
}

define i32 @f7(i32 %base, i32 %offset) {
entry:
        %tmp1 = lshr i32 %offset, 2
        %tmp2 = add i32 %base, %tmp1
        %tmp3 = inttoptr i32 %tmp2 to i32*
        %tmp4 = load i32* %tmp3
        ret i32 %tmp4
}
