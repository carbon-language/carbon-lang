; This testcase causes an infinite loop in the instruction combiner,
; because it things that the constant value is a not expression... and 
; constantly inverts the branch back and forth.
;
; RUN: llvm-as < %s | opt -instcombine -disable-output

ubyte %test19(bool %c) {
        br bool true, label %True, label %False
True:
        ret ubyte 1
False:
        ret ubyte 3
}

