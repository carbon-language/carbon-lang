; RUN: llvm-upgrade < %s | llvm-as | llc -march=x86 -relocation-model=static | \
; RUN:   grep {test1 \$_GV}
; RUN: llvm-upgrade < %s | llvm-as | llc -march=x86 -relocation-model=static | \
; RUN:   grep {test2 _GV}
; PR882

target datalayout = "e-p:32:32"
target endian = little
target pointersize = 32
target triple = "i686-apple-darwin9.0.0d2"
%GV = weak global int 0         ; <int*> [#uses=2]
%str = external global [12 x sbyte]             ; <[12 x sbyte]*> [#uses=1]

implementation   ; Functions:

void %foo() {
entry:
        tail call void asm sideeffect "test1 $0", "i,~{dirflag},~{fpsr},~{flags}"( int* %GV )
        tail call void asm sideeffect "test2 ${0:c}", "i,~{dirflag},~{fpsr},~{flags}"( int* %GV )
        ret void
}


void %unknown_bootoption() {
entry:
        call void asm sideeffect "ud2\0A\09.word ${0:c}\0A\09.long ${1:c}\0A",
"i,i,~{dirflag},~{fpsr},~{flags}"( int 235, sbyte* getelementptr ([12 x sbyte]*
%str, int 0, uint 0) )
        ret void
}

