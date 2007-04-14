; This is the sequence of stuff that the Java front-end expands for a single 
; <= comparison.  Check to make sure we turn it into a <= (only)

; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine | llvm-dis | \
; RUN:    grep -v {icmp sle} | not grep #uses

bool %le(int %A, int %B) {
        %c1 = setgt int %A, %B;
        %tmp = select bool %c1, int 1, int 0;
        %c2 = setlt int %A, %B;
        %result = select bool %c2, int -1, int %tmp;
        %c3 = setle int %result, 0;
        ret bool %c3;
}

