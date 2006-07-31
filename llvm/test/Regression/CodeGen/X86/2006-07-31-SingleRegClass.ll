; RUN: llvm-as < %s | llc -march=x86 -x86-asm-syntax=att | grep 'movl 4(%eax),%ebp' &&
; RUN: llvm-as < %s | llc -march=x86 -x86-asm-syntax=att | grep 'movl 0(%eax), %ebx'

; PR850

int %foo(int %__s.i.i, int %tmp5.i.i, int %tmp6.i.i, int %tmp7.i.i, int %tmp8.i.i ) {

%tmp9.i.i = call int asm sideeffect "push %ebp\0Apush %ebx\0Amovl 4($2),%ebp\0Amovl 0($2), %ebx\0Amovl $1,%eax\0Aint  $$0x80\0Apop  %ebx\0Apop %ebp", "={ax},i,0,{cx},{dx},{si},{di}"(int 192, int %__s.i.i, int %tmp5.i.i, int %tmp6.i.i, int %tmp7.i.i, int %tmp8.i.i )
  ret int %tmp9.i.i
}
