; This testcase caused the combiner to go into an infinite loop, moving the 
; cast back and forth, changing the seteq to operate on int vs uint and back.

; RUN: llvm-as < %s | opt -instcombine -disable-output

bool %test(uint %A, int %B) {
        %C = sub uint 0, %A
        %Cc = cast uint %C to int
        %D = sub int 0, %B
        %E = seteq int %Cc, %D
        ret bool %E
}

