; RUN: llvm-as < %s | llc -march=ppc32 -mcpu=g5 &&
; RUN: llvm-as < %s | llc -march=ppc32 -mcpu=g5 | not grep CPI


; Tests spltw(0x80000000) and spltw(0x7FFFFFFF).
void %test1(<4 x int>* %P1, <4 x int>* %P2, <4 x float>* %P3) {
        %tmp = load <4 x int>* %P1              
        %tmp4 = and <4 x int> %tmp, < int -2147483648, int -2147483648, int -2147483648, int -2147483648 >                
        store <4 x int> %tmp4, <4 x int>* %P1
        %tmp7 = load <4 x int>* %P2             
        %tmp9 = and <4 x int> %tmp7, < int 2147483647, int 2147483647, int 2147483647, int 2147483647 >           
        store <4 x int> %tmp9, <4 x int>* %P2
        %tmp = load <4 x float>* %P3            
        %tmp11 = cast <4 x float> %tmp to <4 x int>             
        %tmp12 = and <4 x int> %tmp11, < int 2147483647, int 2147483647, int 2147483647, int 2147483647 >
        %tmp13 = cast <4 x int> %tmp12 to <4 x float>
        store <4 x float> %tmp13, <4 x float>* %P3
        ret void
}

<4 x int> %test2() {
        ret <4 x int> <int 30, int 30, int 30, int 30>
}
