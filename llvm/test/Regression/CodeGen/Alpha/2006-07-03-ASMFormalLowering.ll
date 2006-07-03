;RUN: llvm-as < %s | llc -march=alpha

; ModuleID = 'atomicity.cc'
target endian = little
target pointersize = 64
target triple = "alphaev67-unknown-linux-gnu"

implementation   ; Functions:

int %_ZN9__gnu_cxx18__exchange_and_addEPVii(int* %__mem, int %__val) {
entry:
        %__tmp = alloca int, align 4            ; <int*> [#uses=1]
        %tmp3 = call int asm sideeffect "\0A$$Lxadd_0:\0A\09ldl_l  $0,$3\0A\09addl   $0,$4,$1\0A\09stl_c  $1,$2\0A\09beq    $1,$$Lxadd_0\0A\09mb", "=&r,==&r,==m,m,r"( int* %__tmp, int* %__mem, int* %__mem, int %__val )          ; <int> [#uses=1]
        ret int %tmp3
}

void %_ZN9__gnu_cxx12__atomic_addEPVii(int* %__mem, int %__val) {
entry:
        %tmp2 = call int asm sideeffect "\0A$$Ladd_1:\0A\09ldl_l  $0,$2\0A\09addl   $0,$3,$0\0A\09stl_c  $0,$1\0A\09beq    $0,$$Ladd_1\0A\09mb", "=&r,==m,m,r"( int* %__mem, int* %__mem, int %__val )              ; <int> [#uses=0]
        ret void
}
