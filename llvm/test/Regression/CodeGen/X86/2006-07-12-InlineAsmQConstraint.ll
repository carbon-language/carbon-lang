; RUN: llvm-as < %s | llc -march=x86
; PR828

target endian = little
target pointersize = 32
target triple = "i686-pc-linux-gnu"

implementation   ; Functions:

void %_ZN5() {

cond_true9:             ; preds = %entry
        %tmp3.i.i = call int asm sideeffect "lock; cmpxchg $1,$2",
"={ax},q,m,0,~{dirflag},~{fpsr},~{flags},~{memory}"( int 0, int* null, int 0 ) 
              ; <int> [#uses=0]
        ret void
}

