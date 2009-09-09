; RUN: llc < %s -march=thumb -mattr=+thumb2 | grep "bfc " | count 3

; 4278190095 = 0xff00000f
define i32 @f1(i32 %a) {
    %tmp = and i32 %a, 4278190095
    ret i32 %tmp
}

; 4286578688 = 0xff800000
define i32 @f2(i32 %a) {
    %tmp = and i32 %a, 4286578688
    ret i32 %tmp
}

; 4095 = 0x00000fff
define i32 @f3(i32 %a) {
    %tmp = and i32 %a, 4095
    ret i32 %tmp
}

; 2147483646 = 0x7ffffffe   not implementable w/ BFC
define i32 @f4(i32 %a) {
    %tmp = and i32 %a, 2147483646
    ret i32 %tmp
}
