//===---------------------------------------------------------------------===//
// Random ideas for the X86 backend: MMX-specific stuff.
//===---------------------------------------------------------------------===//

//===---------------------------------------------------------------------===//

This:

#include <mmintrin.h>

__v2si qux(int A) {
  return (__v2si){ 0, A };
}

is compiled into:

_qux:
        subl $28, %esp
        movl 32(%esp), %eax
        movd %eax, %mm0
        movq %mm0, (%esp)
        movl (%esp), %eax
        movl %eax, 20(%esp)
        movq %mm0, 8(%esp)
        movl 12(%esp), %eax
        movl %eax, 16(%esp)
        movq 16(%esp), %mm0
        addl $28, %esp
        ret

Yuck!

GCC gives us:

_qux:
        subl    $12, %esp
        movl    16(%esp), %eax
        movl    20(%esp), %edx
        movl    $0, (%eax)
        movl    %edx, 4(%eax)
        addl    $12, %esp
        ret     $4

//===---------------------------------------------------------------------===//

int main() {
  __m64 A[1] = { _mm_cvtsi32_si64(1)  };
  __m64 B[1] = { _mm_cvtsi32_si64(10) };
  __m64 sum = _mm_cvtsi32_si64(0);

  sum = __builtin_ia32_paddq(__builtin_ia32_paddq(A[0], B[0]), sum);

  printf("Sum = %d\n", _mm_cvtsi64_si32(sum));
  return 0;
}

Generates:

        movl $11, %eax
###     movd %eax, %mm0
###     movq %mm0, 8(%esp)
###     movl 8(%esp), %eax
        movl %eax, 4(%esp)
        movl $_str, (%esp)
        call L_printf$stub
        xorl %eax, %eax
        addl $28, %esp

These instructions are unnecessary.
