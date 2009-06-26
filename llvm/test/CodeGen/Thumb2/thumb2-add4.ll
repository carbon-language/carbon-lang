; RUN: llvm-as < %s | llc | grep {adds\\W*r\[0-9\],\\W*r\[0-9\],\\W*#\[0-9\]*} | grep {#171\\|#1179666\\|#872428544\\|#1448498774\\|#66846720} | Count 5

target triple = "thumbv7-apple-darwin"

; 171 = 0x000000ab
define i64 @f1(i64 %a) {
    %tmp = add i64 %a, 171
    ret i64 %tmp
}

; 1179666 = 0x00120012
define i64 @f2(i64 %a) {
    %tmp = add i64 %a, 1179666
    ret i64 %tmp
}

; 872428544 = 0x34003400
define i64 @f3(i64 %a) {
    %tmp = add i64 %a, 872428544
    ret i64 %tmp
}

; 1448498774 = 0x56565656
define i64 @f4(i64 %a) {
    %tmp = add i64 %a, 1448498774
    ret i64 %tmp
}

; 66846720 = 0x03fc0000
define i64 @f5(i64 %a) {
    %tmp = add i64 %a, 66846720
    ret i64 %tmp
}
