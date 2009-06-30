; RUN: llvm-as < %s | llc -march=thumb -mattr=+thumb2 | grep {cmp\\W*r\[0-9\],\\W*#\[0-9\]*$} | grep {#187\\|#11141290\\|#3422604288\\|#1114112\\|#3722304989} | count 5

; 0x000000bb = 187
define i1 @f1(i32 %a) {
    %tmp = icmp ne i32 %a, 187
    ret i1 %tmp
}

; 0x00aa00aa = 11141290
define i1 @f2(i32 %a) {
    %tmp = icmp eq i32 %a, 11141290 
    ret i1 %tmp
}

; 0xcc00cc00 = 3422604288
define i1 @f3(i32 %a) {
    %tmp = icmp ne i32 %a, 3422604288
    ret i1 %tmp
}

; 0xdddddddd = 3722304989
define i1 @f4(i32 %a) {
    %tmp = icmp ne i32 %a, 3722304989
    ret i1 %tmp
}

; 0x00110000 = 1114112
define i1 @f5(i32 %a) {
    %tmp = icmp eq i32 %a, 1114112
    ret i1 %tmp
}
