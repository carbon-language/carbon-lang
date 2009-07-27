; RUN: llvm-as < %s | llc -march=thumb -mattr=+thumb2 | grep {str\\.w\\W*r\[0-9\],\\W*\\\[r\[0-9\]*\\\]$} | count 1
; RUN: llvm-as < %s | llc -march=thumb -mattr=+thumb2 | grep {str\\.w\\W*r\[0-9\],\\W*\\\[r\[0-9\]*,\\W*#+4092\\\]$} | count 1
; RUN: llvm-as < %s | llc -march=thumb -mattr=+thumb2 | grep {str\\W*r\[0-9\],\\W*\\\[r\[0-9\]*,\\W*#-128\\\]$} | count 2
; RUN: llvm-as < %s | llc -march=thumb -mattr=+thumb2 | not grep {str\\.w\\W*r\[0-9\],\\W*\\\[r\[0-9\]*,\\W*#+4096\\\]$}
; RUN: llvm-as < %s | llc -march=thumb -mattr=+thumb2 | grep {str\\.w\\W*r\[0-9\],\\W*\\\[r\[0-9\]*,\\W*+r\[0-9\]*\\\]$} | count 3
; RUN: llvm-as < %s | llc -march=thumb -mattr=+thumb2 | grep {str\\.w\\W*r\[0-9\],\\W*\\\[r\[0-9\]*,\\W*+r\[0-9\]*,\\Wlsl #2\\\]$} | count 1

define i32 @f1(i32 %a, i32* %v) {
        store i32 %a, i32* %v
        ret i32 %a
}

define i32 @f2(i32 %a, i32* %v) {
        %tmp2 = getelementptr i32* %v, i32 1023
        store i32 %a, i32* %tmp2
        ret i32 %a
}

define i32 @f2a(i32 %a, i32* %v) {
        %tmp2 = getelementptr i32* %v, i32 -32
        store i32 %a, i32* %tmp2
        ret i32 %a
}

define i32 @f3(i32 %a, i32* %v) {
        %tmp2 = getelementptr i32* %v, i32 1024
        store i32 %a, i32* %tmp2
        ret i32 %a
}

define i32 @f4(i32 %a, i32 %base) {
entry:
        %tmp1 = sub i32 %base, 128
        %tmp2 = inttoptr i32 %tmp1 to i32*
        store i32 %a, i32* %tmp2
        ret i32 %a
}

define i32 @f5(i32 %a, i32 %base, i32 %offset) {
entry:
        %tmp1 = add i32 %base, %offset
        %tmp2 = inttoptr i32 %tmp1 to i32*
        store i32 %a, i32* %tmp2
        ret i32 %a
}

define i32 @f6(i32 %a, i32 %base, i32 %offset) {
entry:
        %tmp1 = shl i32 %offset, 2
        %tmp2 = add i32 %base, %tmp1
        %tmp3 = inttoptr i32 %tmp2 to i32*
        store i32 %a, i32* %tmp3
        ret i32 %a
}

define i32 @f7(i32 %a, i32 %base, i32 %offset) {
entry:
        %tmp1 = lshr i32 %offset, 2
        %tmp2 = add i32 %base, %tmp1
        %tmp3 = inttoptr i32 %tmp2 to i32*
        store i32 %a, i32* %tmp3
        ret i32 %a
}
