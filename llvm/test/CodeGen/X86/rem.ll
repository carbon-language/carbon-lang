; RUN: llvm-upgrade < %s | llvm-as | llc -march=x86 | not grep div

int %test1(int %X) {
        %tmp1 = rem int %X, 255
        ret int %tmp1
}

int %test2(int %X) {
        %tmp1 = rem int %X, 256 
        ret int %tmp1
}

uint %test3(uint %X) {
        %tmp1 = rem uint %X, 255
        ret uint %tmp1
}

uint %test4(uint %X) {
        %tmp1 = rem uint %X, 256  ; just an and
        ret uint %tmp1
}

