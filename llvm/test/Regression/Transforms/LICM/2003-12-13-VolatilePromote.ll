; RUN: llvm-as < %s | opt -licm | llvm-dis | grep -C1 volatile | grep Loop

%X = global int 7

void %testfunc(int %i) {
        br label %Loop

Loop:
        %x = volatile load int* %X  ; Should not promote this to a register
        %x2 = add int %x, 1
        store int %x2, int* %X
        br bool true, label %Out, label %Loop

Out:
        ret void
}

