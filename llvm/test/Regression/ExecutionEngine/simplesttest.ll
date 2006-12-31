; RUN: llvm-upgrade < %s | llvm-as -f -o %t.bc
; RUN: lli %t.bc > /dev/null


implementation

int %main() {
        ret int 0
}

