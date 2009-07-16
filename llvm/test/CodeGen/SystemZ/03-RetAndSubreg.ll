; RUN: llvm-as < %s | llc -march=systemz | grep ngr | count 4

define i32 @foo(i32 %a, i32 %b) {
entry:
    %c = and i32 %a, %b
    ret i32 %c
}

define i32 @foo1(i32 %a, i32 %b) zeroext {
entry:
    %c = and i32 %a, %b
    ret i32 %c
}

define i32 @foo2(i32 %a, i32 %b) signext {
entry:
    %c = and i32 %a, %b
    ret i32 %c
}

