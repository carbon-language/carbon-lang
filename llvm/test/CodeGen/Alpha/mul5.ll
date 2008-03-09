; Make sure this testcase does not use mulq
; RUN: llvm-as < %s | llc -march=alpha | \
; RUN:   not grep -i mul
; XFAIL: *

define i64 @foo1(i64 %x) {
entry:
        %tmp.1 = mul i64 %x, 9          ; <i64> [#uses=1]
        ret i64 %tmp.1
}

define i64 @foo3(i64 %x) {
entry:
        %tmp.1 = mul i64 %x, 259                ; <i64> [#uses=1]
        ret i64 %tmp.1
}

define i64 @foo4l(i64 %x) {
entry:
        %tmp.1 = mul i64 %x, 260                ; <i64> [#uses=1]
        ret i64 %tmp.1
}

define i64 @foo4ln(i64 %x) {
entry:
        %tmp.1 = mul i64 %x, 508                ; <i64> [#uses=1]
        ret i64 %tmp.1
}

define i64 @foo4ln_more(i64 %x) {
entry:
        %tmp.1 = mul i64 %x, 252                ; <i64> [#uses=1]
        ret i64 %tmp.1
}

define i64 @foo1n(i64 %x) {
entry:
        %tmp.1 = mul i64 %x, 511                ; <i64> [#uses=1]
        ret i64 %tmp.1
}

define i64 @foo8l(i64 %x) {
entry:
        %tmp.1 = mul i64 %x, 768                ; <i64> [#uses=1]
        ret i64 %tmp.1
}

define i64 @bar(i64 %x) {
entry:
        %tmp.1 = mul i64 %x, 5          ; <i64> [#uses=1]
        ret i64 %tmp.1
}

