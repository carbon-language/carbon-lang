; RUN: llvm-as < %s | llc -march=systemz | grep ogr   | count 3
; RUN: llvm-as < %s | llc -march=systemz | grep llilf | count 1
; RUN: llvm-as < %s | llc -march=systemz | grep lgfr  | count 1


define i32 @foo(i32 %a, i32 %b) {
entry:
    %c = or i32 %a, %b
    ret i32 %c
}

define i32 @foo1(i32 %a, i32 %b) zeroext {
entry:
    %c = or i32 %a, %b
    ret i32 %c
}

define i32 @foo2(i32 %a, i32 %b) signext {
entry:
    %c = or i32 %a, %b
    ret i32 %c
}

