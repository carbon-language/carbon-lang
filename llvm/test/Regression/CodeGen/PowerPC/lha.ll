; RUN: llvm-as < %s | llc -march=ppc32 | grep lha
; XFAIL: *

uint %test(short* %a) {
entry:
    %tmp.1 = load short* %a
    %tmp.2 = cast short %tmp.1 to uint
    ret uint %tmp.2
}
