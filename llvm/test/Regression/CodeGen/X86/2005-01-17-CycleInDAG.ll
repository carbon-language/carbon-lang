; This testcase was distilled from 132.ijpeg.  Bsaically we cannot fold the
; load into the sub instruction here as it induces a cycle in the dag, which
; is invalid code (there is no correct way to order the instruction).  Check
; that we do not fold the load into the sub.

; RUN: llvm-as < %s | llc -march=x86 -disable-pattern-isel=0 | not grep 'sub.*GLOBAL'

%GLOBAL = external global int

int %test(int* %P1, int* %P2, int* %P3) {
   %L = load int* %GLOBAL
   store int 12, int* %P2
   %Y = load int* %P3
   %Z = sub int %Y, %L
   ret int %Z
}
