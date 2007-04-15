; RUN: llvm-upgrade < %s | llvm-as | llc -march=x86 -x86-asm-syntax=intel | \
; RUN:   grep {lea EAX, DWORD PTR \\\[... + 4\\*... - 5\\\]}
; RUN: llvm-upgrade < %s | llvm-as | llc -march=x86 -x86-asm-syntax=intel | \
; RUN:   not grep add

int %test1(int %A, int %B) {
        %tmp1 = shl int %A, ubyte 2             ; <int> [#uses=1]
        %tmp3 = add int %B, -5          ; <int> [#uses=1]
        %tmp4 = add int %tmp3, %tmp1            ; <int> [#uses=1]
        ret int %tmp4
}

