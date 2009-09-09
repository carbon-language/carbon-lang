; RUN: llc < %s -march=thumb -mattr=+thumb2 | grep {and\\W*r\[0-9\],\\W*r\[0-9\],\\W*#\[0-9\]*} | grep {#171\\|#1179666\\|#872428544\\|#1448498774\\|#66846720} | count 5

; 171 = 0x000000ab
define i32 @f1(i32 %a) {
    %tmp = and i32 %a, 171
    ret i32 %tmp
}

; 1179666 = 0x00120012
define i32 @f2(i32 %a) {
    %tmp = and i32 %a, 1179666
    ret i32 %tmp
}

; 872428544 = 0x34003400
define i32 @f3(i32 %a) {
    %tmp = and i32 %a, 872428544
    ret i32 %tmp
}

; 1448498774 = 0x56565656
define i32 @f4(i32 %a) {
    %tmp = and i32 %a, 1448498774
    ret i32 %tmp
}

; 66846720 = 0x03fc0000
define i32 @f5(i32 %a) {
    %tmp = and i32 %a, 66846720
    ret i32 %tmp
}
