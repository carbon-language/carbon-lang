; RUN: llc < %s -march=thumb -mattr=+thumb2 | grep {orr\\W*r\[0-9\]*,\\W*r\[0-9\]*,\\W*#\[0-9\]*} | grep {#187\\|#11141290\\|#-872363008\\|#1145324612\\|#1114112} | count 5

; 0x000000bb = 187
define i32 @f1(i32 %a) {
    %tmp2 = or i32 %a, 187
    ret i32 %tmp2
}

; 0x00aa00aa = 11141290
define i32 @f2(i32 %a) {
    %tmp2 = or i32 %a, 11141290 
    ret i32 %tmp2
}

; 0xcc00cc00 = 3422604288
define i32 @f3(i32 %a) {
    %tmp2 = or i32 %a, 3422604288
    ret i32 %tmp2
}

; 0x44444444 = 1145324612
define i32 @f4(i32 %a) {
    %tmp2 = or i32 %a, 1145324612
    ret i32 %tmp2
}

; 0x00110000 = 1114112
define i32 @f5(i32 %a) {
    %tmp2 = or i32 %a, 1114112
    ret i32 %tmp2
}
