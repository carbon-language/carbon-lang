; test shifts
int %main() {
    %i = add int 10, 0
    %u = add uint 20, 0
    %shamt = add ubyte 0, 0
    %shamt2 = add ubyte 1, 0
    %shamt3 = add ubyte 2, 0
    %shamt4 = add ubyte 3, 0
    ; constantShiftAmount  isRightShift   isOperandUnsigned
    ;  0                     0               0
    %temp01 = shl int %i, ubyte %shamt
    ;  0                     0               1
    %temp02 = shl uint %u, ubyte %shamt2
    ;  0                     1               0
    %temp03 = shr int %i, ubyte %shamt3
    ;  0                     1               1
    %temp04 = shr uint %u, ubyte %shamt4
    ;  1                     0               0
    %temp05 = shl int %i, ubyte 4
    ;  1                     0               1
    %temp06 = shl uint %u, ubyte 5
    ;  1                     1               0
    %temp07 = shr int %i, ubyte 6
    ;  1                     1               1
    %temp08 = shr uint %u, ubyte 7
    ret int 0
}
