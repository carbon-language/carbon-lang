; RUN: llvm-as < %s | llc -march=thumb -mattr=+thumb2 | grep {adcs\\W*r\[0-9\],\\W*r\[0-9\],\\W*r\[0-9\]*} | Count 1

define i64 @f1(i64 %a, i64 %b) {
    %tmp = add i64 %a, %b
    ret i64 %tmp
}
