target endian = big
target pointersize = 64

%.str_1 = internal constant [42 x sbyte] c"   ui = %u (0x%x)\09\09UL-ui = %lld (0x%llx)\0A\00"

implementation   ; Functions:

declare int %printf(sbyte*, ...)

internal ulong %getL() {
entry:          ; No predecessors!
        ret ulong 12659530247033960611
}

int %main(int %argc.1, sbyte** %argv.1) {
entry:          ; No predecessors!
        %tmp.11 = call ulong %getL( )
        %tmp.5 = cast ulong %tmp.11 to uint
        %tmp.23 = and ulong %tmp.11, 18446744069414584320
        %tmp.16 = call int (sbyte*, ...)* %printf( sbyte* getelementptr ([42 x sbyte]* %.str_1, long 0, long 0), uint %tmp.5, uint %tmp.5, ulong %tmp.23, ulong %tmp.23 )
        ret int 0
}
