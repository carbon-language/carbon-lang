; RUN: llc < %s -march=systemz | grep lcr | count 1

define i32 @foo(i32 %a) {
entry:
    %c = sub i32 0, %a
    ret i32 %c
}

