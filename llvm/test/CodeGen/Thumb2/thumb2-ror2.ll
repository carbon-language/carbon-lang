; RUN: llvm-as < %s | llc -march=thumb -mattr=+thumb2 | grep {ror\\W*r\[0-9\]*,\\W*r\[0-9\]*,\\W*r\[0-9\]*} | count 1

define i32 @f1(i32 %a, i32 %b) {
    %db = sub i32 32, %b
    %l8 = shl i32 %a, %b
    %r8 = lshr i32 %a, %db
    %tmp = or i32 %l8, %r8
    ret i32 %tmp
}
