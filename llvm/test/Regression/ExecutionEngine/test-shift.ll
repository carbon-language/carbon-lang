; RUN: llvm-upgrade %s | llvm-as -f -o %t.bc
; RUN: lli %t.bc > /dev/null

; test shifts
int %main() {
    %shamt = add ubyte 0, 1

    ; Left shifts...
    %t1.s = shl int 1, ubyte %shamt
    %t2.s = shl int 1, ubyte 4

    %t1 = shl uint 1, ubyte %shamt
    %t2 = shl uint 1, ubyte 5

    ;%t1 = shl long 1, ubyte %shamt
    %t2.s = shl long 1, ubyte 4

    ;%t1 = shl ulong 1, ubyte %shamt
    %t2 = shl ulong 1, ubyte 5

    ; Right shifts...
    %tr1.s = shr int 1, ubyte %shamt
    %tr2.s = shr int 1, ubyte 4

    %tr1 = shr uint 1, ubyte %shamt
    %tr2 = shr uint 1, ubyte 5

    ;%tr1 = shr long 1, ubyte %shamt
    %tr1.l = shr long 1, ubyte 4
    %tr2.l = shr long 1, ubyte %shamt
    %tr3.l = shl long 1, ubyte 4
    %tr4.l = shl long 1, ubyte %shamt

    ;%t1 = shr ulong 1, ubyte %shamt
    %tr1.u = shr ulong 1, ubyte 5
    %tr2.u = shr ulong 1, ubyte %shamt
    %tr3.u = shl ulong 1, ubyte 5
    %tr4.u = shl ulong 1, ubyte %shamt
    ret int 0
}
