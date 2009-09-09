; RUN: llc < %s -march=thumb -mattr=+thumb2 | grep "cmn\\.w "  | grep {#187\\|#11141290\\|#-872363008\\|#1114112} | count 4

; -0x000000bb = 4294967109
define i1 @f1(i32 %a) {
    %tmp = icmp ne i32 %a, 4294967109
    ret i1 %tmp
}

; -0x00aa00aa = 4283826006
define i1 @f2(i32 %a) {
    %tmp = icmp eq i32 %a, 4283826006
    ret i1 %tmp
}

; -0xcc00cc00 = 872363008
define i1 @f3(i32 %a) {
    %tmp = icmp ne i32 %a, 872363008
    ret i1 %tmp
}

; -0x00110000 = 4293853184
define i1 @f4(i32 %a) {
    %tmp = icmp eq i32 %a, 4293853184
    ret i1 %tmp
}
