; RUN: llvm-as < %s | llc | grep {mvn\\W*r\[0-9\],\\W*#\[0-9\]*} | grep {#187\\|#11141290\\|#3422604288\\|#1114112} | count 4

target triple = "thumbv7-apple-darwin"

; 0x000000bb = 187
define i32 @f1(i32 %a) {
    %tmp = xor i32 4294967295, 187
    ret i32 %tmp
}

; 0x00aa00aa = 11141290
define i32 @f2(i32 %a) {
    %tmp = xor i32 4294967295, 11141290 
    ret i32 %tmp
}

; 0xcc00cc00 = 3422604288
define i32 @f3(i32 %a) {
    %tmp = xor i32 4294967295, 3422604288
    ret i32 %tmp
}

; 0x00110000 = 1114112
define i32 @f5(i32 %a) {
    %tmp = xor i32 4294967295, 1114112
    ret i32 %tmp
}
