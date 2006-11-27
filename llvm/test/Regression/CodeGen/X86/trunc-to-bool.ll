; An integer truncation to bool should be done with an and instruction to make
; sure only the LSBit survives. Test that this is the case both for a returned
; value and as the operand of a branch.
; RUN: llvm-as < %s | llc -march=x86 &&
; RUN: llvm-as < %s | llc -march=x86 | grep '\(and\)\|\(test.*\$1\)' | wc -l | grep 3
bool %test1(int %X) {
    %Y = trunc int %X to bool
    ret bool %Y
}

bool %test2(int %val, int %mask) {
entry:
    %mask     = trunc int %mask to ubyte
    %shifted  = ashr int %val, ubyte %mask
    %anded    = and int %shifted, 1
    %trunced  = trunc int %anded to bool
    br bool %trunced, label %ret_true, label %ret_false
ret_true:
    ret bool true
ret_false:
    ret bool false
}

int %test3(sbyte* %ptr) {
    %val = load sbyte* %ptr
    %tmp = trunc sbyte %val to bool             ; %<bool> [#uses=1]
    br bool %tmp, label %cond_true, label %cond_false
cond_true:
    ret int 21
cond_false:
    ret int 42
}
