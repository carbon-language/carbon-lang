; RUN: llc < %s -march=systemz | grep lcgr | count 1

define i64 @foo(i64 %a) {
entry:
    %c = sub i64 0, %a
    ret i64 %c
}