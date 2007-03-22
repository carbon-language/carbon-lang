//===---------------------------------------------------------------------===//
// Random ideas for the X86 backend: MMX-specific stuff.
//===---------------------------------------------------------------------===//

//===---------------------------------------------------------------------===//

We should compile 

#include <mmintrin.h>

extern __m64 C;

void baz(__v2si *A, __v2si *B)
{
  *A = __builtin_ia32_psllq(*B, C);
  _mm_empty();
}

to:

.globl _baz
_baz:
	call	L3
"L00000000001$pb":
L3:
	popl	%ecx
	subl	$12, %esp
	movl	20(%esp), %eax
	movq	(%eax), %mm0
	movl	L_C$non_lazy_ptr-"L00000000001$pb"(%ecx), %eax
	movq	(%eax), %mm1
	movl	16(%esp), %eax
	psllq	%mm1, %mm0
	movq	%mm0, (%eax)
	emms
	addl	$12, %esp
	ret

not:

_baz:
	subl $12, %esp
	call "L1$pb"
"L1$pb":
	popl %eax
	movl L_C$non_lazy_ptr-"L1$pb"(%eax), %eax
	movl (%eax), %ecx
	movl %ecx, (%esp)
	movl 4(%eax), %eax
	movl %eax, 4(%esp)
	movl 20(%esp), %eax
	movq (%eax), %mm0
	movq (%esp), %mm1
	psllq %mm1, %mm0
	movl 16(%esp), %eax
	movq %mm0, (%eax)
	emms
	addl $12, %esp
	ret
