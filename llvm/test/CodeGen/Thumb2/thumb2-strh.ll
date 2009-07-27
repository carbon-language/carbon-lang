; RUN: llvm-as < %s | llc -march=thumb -mattr=+thumb2 | grep {strh\\.w\\W*r\[0-9\],\\W*\\\[r\[0-9\]*\\\]$} | count 1
; RUN: llvm-as < %s | llc -march=thumb -mattr=+thumb2 | grep {strh\\.w\\W*r\[0-9\],\\W*\\\[r\[0-9\]*,\\W*#+4092\\\]$} | count 1
; RUN: llvm-as < %s | llc -march=thumb -mattr=+thumb2 | grep {strh\\W*r\[0-9\],\\W*\\\[r\[0-9\]*,\\W*#-128\\\]$} | count 2
; RUN: llvm-as < %s | llc -march=thumb -mattr=+thumb2 | not grep {strh\\.w\\W*r\[0-9\],\\W*\\\[r\[0-9\]*,\\W*#+4096\\\]$}
; RUN: llvm-as < %s | llc -march=thumb -mattr=+thumb2 | grep {strh\\.w\\W*r\[0-9\],\\W*\\\[r\[0-9\]*,\\W*+r\[0-9\]*\\\]$} | count 3
; RUN: llvm-as < %s | llc -march=thumb -mattr=+thumb2 | grep {strh\\.w\\W*r\[0-9\],\\W*\\\[r\[0-9\]*,\\W*+r\[0-9\]*,\\Wlsl #2\\\]$} | count 1

define i16 @f1(i16 %a, i16* %v) {
        store i16 %a, i16* %v
        ret i16 %a
}

define i16 @f2(i16 %a, i16* %v) {
        %tmp2 = getelementptr i16* %v, i32 2046
        store i16 %a, i16* %tmp2
        ret i16 %a
}

define i16 @f2a(i16 %a, i16* %v) {
        %tmp2 = getelementptr i16* %v, i32 -64
        store i16 %a, i16* %tmp2
        ret i16 %a
}

define i16 @f3(i16 %a, i16* %v) {
        %tmp2 = getelementptr i16* %v, i32 2048
        store i16 %a, i16* %tmp2
        ret i16 %a
}

define i16 @f4(i16 %a, i32 %base) {
entry:
        %tmp1 = sub i32 %base, 128
        %tmp2 = inttoptr i32 %tmp1 to i16*
        store i16 %a, i16* %tmp2
        ret i16 %a
}

define i16 @f5(i16 %a, i32 %base, i32 %offset) {
entry:
        %tmp1 = add i32 %base, %offset
        %tmp2 = inttoptr i32 %tmp1 to i16*
        store i16 %a, i16* %tmp2
        ret i16 %a
}

define i16 @f6(i16 %a, i32 %base, i32 %offset) {
entry:
        %tmp1 = shl i32 %offset, 2
        %tmp2 = add i32 %base, %tmp1
        %tmp3 = inttoptr i32 %tmp2 to i16*
        store i16 %a, i16* %tmp3
        ret i16 %a
}

define i16 @f7(i16 %a, i32 %base, i32 %offset) {
entry:
        %tmp1 = lshr i32 %offset, 2
        %tmp2 = add i32 %base, %tmp1
        %tmp3 = inttoptr i32 %tmp2 to i16*
        store i16 %a, i16* %tmp3
        ret i16 %a
}
