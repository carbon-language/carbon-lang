; test shifts
int %main() {
    %shamt = add ubyte 0, 1

    ; Left shifts...
    %t1 = shl int 1, ubyte %shamt
    %t2 = shl int 1, ubyte 4

    %t1 = shl uint 1, ubyte %shamt
    %t2 = shl uint 1, ubyte 5

    ;%t1 = shl long 1, ubyte %shamt
    %t2 = shl long 1, ubyte 4

    ;%t1 = shl ulong 1, ubyte %shamt
    %t2 = shl ulong 1, ubyte 5

    ; Right shifts...
    %tr1 = shr int 1, ubyte %shamt
    %tr2 = shr int 1, ubyte 4

    %tr1 = shr uint 1, ubyte %shamt
    %tr2 = shr uint 1, ubyte 5

    ;%tr1 = shr long 1, ubyte %shamt
    %tr1 = shr long 1, ubyte 4
    %tr2 = shr long 1, ubyte %shamt
    %tr3 = shl long 1, ubyte 4
    %tr4 = shl long 1, ubyte %shamt

    ;%t1 = shr ulong 1, ubyte %shamt
    %tr1 = shr ulong 1, ubyte 5
    %tr2 = shr ulong 1, ubyte %shamt
    %tr3 = shl ulong 1, ubyte 5
    %tr4 = shl ulong 1, ubyte %shamt
    ret int 0
}
