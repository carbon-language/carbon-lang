; The program should not just cast 2143289344 to float and store it!
;
; RUN: llvm-as < %s | opt -raise | llvm-dis | not grep 41DFF

void %test() {
       %mem_tmp = alloca float
       %tmp.0 = cast float* %mem_tmp to uint*
       store uint 2143289344, uint* %tmp.0
       ret void
}
