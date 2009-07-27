; RUN: llvm-as < %s | llc -march=thumb -mattr=+thumb2 | grep {ldrh\\.w r0} | count 6
; RUN: llvm-as < %s | llc -march=thumb -mattr=+thumb2 | grep {ldrh r0} | count 1
; RUN: llvm-as < %s | llc -march=thumb -mattr=+thumb2 | grep mov\\.w | grep 1
; RUN: llvm-as < %s | llc -march=thumb -mattr=+thumb2 | not grep mvn\\.w
; RUN: llvm-as < %s | llc -march=thumb -mattr=+thumb2 | grep ldrh\\.w | grep lsl
; RUN: llvm-as < %s | llc -march=thumb -mattr=+thumb2 | grep lsr\\.w | not grep ldrh

define i16 @f1(i16* %v) {
entry:
        %tmp = load i16* %v
        ret i16 %tmp
}

define i16 @f2(i16* %v) {
entry:
        %tmp2 = getelementptr i16* %v, i16 1023
        %tmp = load i16* %tmp2
        ret i16 %tmp
}

define i16 @f3(i16* %v) {
entry:
        %tmp2 = getelementptr i16* %v, i16 2048
        %tmp = load i16* %tmp2
        ret i16 %tmp
}

define i16 @f4(i32 %base) {
entry:
        %tmp1 = sub i32 %base, 128
        %tmp2 = inttoptr i32 %tmp1 to i16*
        %tmp3 = load i16* %tmp2
        ret i16 %tmp3
}

define i16 @f5(i32 %base, i32 %offset) {
entry:
        %tmp1 = add i32 %base, %offset
        %tmp2 = inttoptr i32 %tmp1 to i16*
        %tmp3 = load i16* %tmp2
        ret i16 %tmp3
}

define i16 @f6(i32 %base, i32 %offset) {
entry:
        %tmp1 = shl i32 %offset, 2
        %tmp2 = add i32 %base, %tmp1
        %tmp3 = inttoptr i32 %tmp2 to i16*
        %tmp4 = load i16* %tmp3
        ret i16 %tmp4
}

define i16 @f7(i32 %base, i32 %offset) {
entry:
        %tmp1 = lshr i32 %offset, 2
        %tmp2 = add i32 %base, %tmp1
        %tmp3 = inttoptr i32 %tmp2 to i16*
        %tmp4 = load i16* %tmp3
        ret i16 %tmp4
}
