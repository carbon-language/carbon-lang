; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine

%bob = type { int }

int %alias() {
    %pbob1 = alloca %bob
    %pbob2 = getelementptr %bob* %pbob1
    %pbobel = getelementptr %bob* %pbob2, long 0, uint 0
    %rval = load int* %pbobel
    ret int %rval
}

