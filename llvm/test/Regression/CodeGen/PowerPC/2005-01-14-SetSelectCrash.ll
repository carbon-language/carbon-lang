; RUN: llvm-as < %s | llc -march=ppc32 

int %main() {
        %setle = setle long 1, 0
        %select = select bool true, bool %setle, bool true
        ret int 0
}

