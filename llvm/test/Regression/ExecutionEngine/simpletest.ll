; RUN: llvm-as -f %s -o %t.bc
; RUN: lli %t.bc > /dev/null

implementation

int %bar() { ret int 0 }

int %main() {
        %r = call int %bar()
        ret int %r
}

