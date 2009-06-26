; RUN: llvm-as < %s | llc | grep {addw\\W*r\[0-9\],\\W*r\[0-9\],\\W*#\[0-9\]*} | grep {#4095} | Count 1

target triple = "thumbv7-apple-darwin"

define i32 @f1(i32 %a) {
    %tmp = add i32 %a, 4095
    ret i32 %tmp
}
