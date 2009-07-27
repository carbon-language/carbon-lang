; RUN: llvm-as < %s | llc -march=thumb -mattr=+thumb2 | grep {strb\\.w\\W*r\[0-9\],\\W*\\\[r\[0-9\]*\\\]$} | count 1
; RUN: llvm-as < %s | llc -march=thumb -mattr=+thumb2 | grep {strb\\.w\\W*r\[0-9\],\\W*\\\[r\[0-9\]*,\\W*#+4092\\\]$} | count 1
; RUN: llvm-as < %s | llc -march=thumb -mattr=+thumb2 | grep {strb\\W*r\[0-9\],\\W*\\\[r\[0-9\]*,\\W*#-128\\\]$} | count 2
; RUN: llvm-as < %s | llc -march=thumb -mattr=+thumb2 | not grep {strb\\.w\\W*r\[0-9\],\\W*\\\[r\[0-9\]*,\\W*#+4096\\\]$}
; RUN: llvm-as < %s | llc -march=thumb -mattr=+thumb2 | grep {strb\\.w\\W*r\[0-9\],\\W*\\\[r\[0-9\]*,\\W*+r\[0-9\]*\\\]$} | count 3
; RUN: llvm-as < %s | llc -march=thumb -mattr=+thumb2 | grep {strb\\.w\\W*r\[0-9\],\\W*\\\[r\[0-9\]*,\\W*+r\[0-9\]*,\\Wlsl #2\\\]$} | count 1

define i8 @f1(i8 %a, i8* %v) {
        store i8 %a, i8* %v
        ret i8 %a
}

define i8 @f2(i8 %a, i8* %v) {
        %tmp2 = getelementptr i8* %v, i32 4092
        store i8 %a, i8* %tmp2
        ret i8 %a
}

define i8 @f2a(i8 %a, i8* %v) {
        %tmp2 = getelementptr i8* %v, i32 -128
        store i8 %a, i8* %tmp2
        ret i8 %a
}

define i8 @f3(i8 %a, i8* %v) {
        %tmp2 = getelementptr i8* %v, i32 4096
        store i8 %a, i8* %tmp2
        ret i8 %a
}

define i8 @f4(i8 %a, i32 %base) {
entry:
        %tmp1 = sub i32 %base, 128
        %tmp2 = inttoptr i32 %tmp1 to i8*
        store i8 %a, i8* %tmp2
        ret i8 %a
}

define i8 @f5(i8 %a, i32 %base, i32 %offset) {
entry:
        %tmp1 = add i32 %base, %offset
        %tmp2 = inttoptr i32 %tmp1 to i8*
        store i8 %a, i8* %tmp2
        ret i8 %a
}

define i8 @f6(i8 %a, i32 %base, i32 %offset) {
entry:
        %tmp1 = shl i32 %offset, 2
        %tmp2 = add i32 %base, %tmp1
        %tmp3 = inttoptr i32 %tmp2 to i8*
        store i8 %a, i8* %tmp3
        ret i8 %a
}

define i8 @f7(i8 %a, i32 %base, i32 %offset) {
entry:
        %tmp1 = lshr i32 %offset, 2
        %tmp2 = add i32 %base, %tmp1
        %tmp3 = inttoptr i32 %tmp2 to i8*
        store i8 %a, i8* %tmp3
        ret i8 %a
}
