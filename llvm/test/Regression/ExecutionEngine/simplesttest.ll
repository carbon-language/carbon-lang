; RUN: llvm-as -f %s -o %t.bc
; RUN: lli %t.bc > /dev/null


implementation

int %main() {
        ret int 0
}

