; RUN: llvm-as < %s | llc -march=thumb -mattr=+thumb2 | grep "bic "  | grep {#187\\|#11141290\\|#3422604288\\|#1114112} | count 4

; ~0x000000bb = 4294967108
define i32 @f1(i32 %a) {
    %tmp = and i32 %a, 4294967108
    ret i32 %tmp
}

; ~0x00aa00aa = 4283826005
define i32 @f2(i32 %a) {
    %tmp = and i32 %a, 4283826005
    ret i32 %tmp
}

; ~0xcc00cc00 = 872363007
define i32 @f3(i32 %a) {
    %tmp = and i32 %a, 872363007
    ret i32 %tmp
}

; ~0x00110000 = 4293853183
define i32 @f4(i32 %a) {
    %tmp = and i32 %a, 4293853183
    ret i32 %tmp
}
