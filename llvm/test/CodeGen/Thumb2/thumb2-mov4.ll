; RUN: llvm-as < %s | llc | grep {movw\\W*r\[0-9\],\\W*#\[0-9\]*} | grep {#65535} | Count 1

target triple = "thumbv7-apple-darwin"

define i32 @f6(i32 %a) {
    %tmp = add i32 0, 65535
    ret i32 %tmp
}
