//===---------------------------------------------------------------------===//
// Random ideas for the X86 backend: SSE-specific stuff.
//===---------------------------------------------------------------------===//

//===---------------------------------------------------------------------===//

Expand libm rounding functions inline:  Significant speedups possible.
http://gcc.gnu.org/ml/gcc-patches/2006-10/msg00909.html

//===---------------------------------------------------------------------===//

When compiled with unsafemath enabled, "main" should enable SSE DAZ mode and
other fast SSE modes.

//===---------------------------------------------------------------------===//

Think about doing i64 math in SSE regs.

//===---------------------------------------------------------------------===//

This testcase should have no SSE instructions in it, and only one load from
a constant pool:

double %test3(bool %B) {
        %C = select bool %B, double 123.412, double 523.01123123
        ret double %C
}

Currently, the select is being lowered, which prevents the dag combiner from
turning 'select (load CPI1), (load CPI2)' -> 'load (select CPI1, CPI2)'

The pattern isel got this one right.

//===---------------------------------------------------------------------===//

SSE doesn't have [mem] op= reg instructions.  If we have an SSE instruction
like this:

  X += y

and the register allocator decides to spill X, it is cheaper to emit this as:

Y += [xslot]
store Y -> [xslot]

than as:

tmp = [xslot]
tmp += y
store tmp -> [xslot]

..and this uses one fewer register (so this should be done at load folding
time, not at spiller time).  *Note* however that this can only be done
if Y is dead.  Here's a testcase:

%.str_3 = external global [15 x sbyte]          ; <[15 x sbyte]*> [#uses=0]
implementation   ; Functions:
declare void %printf(int, ...)
void %main() {
build_tree.exit:
        br label %no_exit.i7
no_exit.i7:             ; preds = %no_exit.i7, %build_tree.exit
        %tmp.0.1.0.i9 = phi double [ 0.000000e+00, %build_tree.exit ], [ %tmp.34.i18, %no_exit.i7 ]      ; <double> [#uses=1]
        %tmp.0.0.0.i10 = phi double [ 0.000000e+00, %build_tree.exit ], [ %tmp.28.i16, %no_exit.i7 ]     ; <double> [#uses=1]
        %tmp.28.i16 = add double %tmp.0.0.0.i10, 0.000000e+00
        %tmp.34.i18 = add double %tmp.0.1.0.i9, 0.000000e+00
        br bool false, label %Compute_Tree.exit23, label %no_exit.i7
Compute_Tree.exit23:            ; preds = %no_exit.i7
        tail call void (int, ...)* %printf( int 0 )
        store double %tmp.34.i18, double* null
        ret void
}

We currently emit:

.BBmain_1:
        xorpd %XMM1, %XMM1
        addsd %XMM0, %XMM1
***     movsd %XMM2, QWORD PTR [%ESP + 8]
***     addsd %XMM2, %XMM1
***     movsd QWORD PTR [%ESP + 8], %XMM2
        jmp .BBmain_1   # no_exit.i7

This is a bugpoint reduced testcase, which is why the testcase doesn't make
much sense (e.g. its an infinite loop). :)

//===---------------------------------------------------------------------===//

SSE should implement 'select_cc' using 'emulated conditional moves' that use
pcmp/pand/pandn/por to do a selection instead of a conditional branch:

double %X(double %Y, double %Z, double %A, double %B) {
        %C = setlt double %A, %B
        %z = add double %Z, 0.0    ;; select operand is not a load
        %D = select bool %C, double %Y, double %z
        ret double %D
}

We currently emit:

_X:
        subl $12, %esp
        xorpd %xmm0, %xmm0
        addsd 24(%esp), %xmm0
        movsd 32(%esp), %xmm1
        movsd 16(%esp), %xmm2
        ucomisd 40(%esp), %xmm1
        jb LBB_X_2
LBB_X_1:
        movsd %xmm0, %xmm2
LBB_X_2:
        movsd %xmm2, (%esp)
        fldl (%esp)
        addl $12, %esp
        ret

//===---------------------------------------------------------------------===//

It's not clear whether we should use pxor or xorps / xorpd to clear XMM
registers. The choice may depend on subtarget information. We should do some
more experiments on different x86 machines.

//===---------------------------------------------------------------------===//

Currently the x86 codegen isn't very good at mixing SSE and FPStack
code:

unsigned int foo(double x) { return x; }

foo:
	subl $20, %esp
	movsd 24(%esp), %xmm0
	movsd %xmm0, 8(%esp)
	fldl 8(%esp)
	fisttpll (%esp)
	movl (%esp), %eax
	addl $20, %esp
	ret

This will be solved when we go to a dynamic programming based isel.

//===---------------------------------------------------------------------===//

Lower memcpy / memset to a series of SSE 128 bit move instructions when it's
feasible.

//===---------------------------------------------------------------------===//

Teach the coalescer to commute 2-addr instructions, allowing us to eliminate
the reg-reg copy in this example:

float foo(int *x, float *y, unsigned c) {
  float res = 0.0;
  unsigned i;
  for (i = 0; i < c; i++) {
    float xx = (float)x[i];
    xx = xx * y[i];
    xx += res;
    res = xx;
  }
  return res;
}

LBB_foo_3:      # no_exit
        cvtsi2ss %XMM0, DWORD PTR [%EDX + 4*%ESI]
        mulss %XMM0, DWORD PTR [%EAX + 4*%ESI]
        addss %XMM0, %XMM1
        inc %ESI
        cmp %ESI, %ECX
****    movaps %XMM1, %XMM0
        jb LBB_foo_3    # no_exit

//===---------------------------------------------------------------------===//

Codegen:
  if (copysign(1.0, x) == copysign(1.0, y))
into:
  if (x^y & mask)
when using SSE.

//===---------------------------------------------------------------------===//

Use movhps to update upper 64-bits of a v4sf value. Also movlps on lower half
of a v4sf value.

//===---------------------------------------------------------------------===//

Better codegen for vector_shuffles like this { x, 0, 0, 0 } or { x, 0, x, 0}.
Perhaps use pxor / xorp* to clear a XMM register first?

//===---------------------------------------------------------------------===//

How to decide when to use the "floating point version" of logical ops? Here are
some code fragments:

	movaps LCPI5_5, %xmm2
	divps %xmm1, %xmm2
	mulps %xmm2, %xmm3
	mulps 8656(%ecx), %xmm3
	addps 8672(%ecx), %xmm3
	andps LCPI5_6, %xmm2
	andps LCPI5_1, %xmm3
	por %xmm2, %xmm3
	movdqa %xmm3, (%edi)

	movaps LCPI5_5, %xmm1
	divps %xmm0, %xmm1
	mulps %xmm1, %xmm3
	mulps 8656(%ecx), %xmm3
	addps 8672(%ecx), %xmm3
	andps LCPI5_6, %xmm1
	andps LCPI5_1, %xmm3
	orps %xmm1, %xmm3
	movaps %xmm3, 112(%esp)
	movaps %xmm3, (%ebx)

Due to some minor source change, the later case ended up using orps and movaps
instead of por and movdqa. Does it matter?

//===---------------------------------------------------------------------===//

X86RegisterInfo::copyRegToReg() returns X86::MOVAPSrr for VR128. Is it possible
to choose between movaps, movapd, and movdqa based on types of source and
destination?

How about andps, andpd, and pand? Do we really care about the type of the packed
elements? If not, why not always use the "ps" variants which are likely to be
shorter.

//===---------------------------------------------------------------------===//

External test Nurbs exposed some problems. Look for
__ZN15Nurbs_SSE_Cubic17TessellateSurfaceE, bb cond_next140. This is what icc
emits:

        movaps    (%edx), %xmm2                                 #59.21
        movaps    (%edx), %xmm5                                 #60.21
        movaps    (%edx), %xmm4                                 #61.21
        movaps    (%edx), %xmm3                                 #62.21
        movl      40(%ecx), %ebp                                #69.49
        shufps    $0, %xmm2, %xmm5                              #60.21
        movl      100(%esp), %ebx                               #69.20
        movl      (%ebx), %edi                                  #69.20
        imull     %ebp, %edi                                    #69.49
        addl      (%eax), %edi                                  #70.33
        shufps    $85, %xmm2, %xmm4                             #61.21
        shufps    $170, %xmm2, %xmm3                            #62.21
        shufps    $255, %xmm2, %xmm2                            #63.21
        lea       (%ebp,%ebp,2), %ebx                           #69.49
        negl      %ebx                                          #69.49
        lea       -3(%edi,%ebx), %ebx                           #70.33
        shll      $4, %ebx                                      #68.37
        addl      32(%ecx), %ebx                                #68.37
        testb     $15, %bl                                      #91.13
        jne       L_B1.24       # Prob 5%                       #91.13

This is the llvm code after instruction scheduling:

cond_next140 (0xa910740, LLVM BB @0xa90beb0):
	%reg1078 = MOV32ri -3
	%reg1079 = ADD32rm %reg1078, %reg1068, 1, %NOREG, 0
	%reg1037 = MOV32rm %reg1024, 1, %NOREG, 40
	%reg1080 = IMUL32rr %reg1079, %reg1037
	%reg1081 = MOV32rm %reg1058, 1, %NOREG, 0
	%reg1038 = LEA32r %reg1081, 1, %reg1080, -3
	%reg1036 = MOV32rm %reg1024, 1, %NOREG, 32
	%reg1082 = SHL32ri %reg1038, 4
	%reg1039 = ADD32rr %reg1036, %reg1082
	%reg1083 = MOVAPSrm %reg1059, 1, %NOREG, 0
	%reg1034 = SHUFPSrr %reg1083, %reg1083, 170
	%reg1032 = SHUFPSrr %reg1083, %reg1083, 0
	%reg1035 = SHUFPSrr %reg1083, %reg1083, 255
	%reg1033 = SHUFPSrr %reg1083, %reg1083, 85
	%reg1040 = MOV32rr %reg1039
	%reg1084 = AND32ri8 %reg1039, 15
	CMP32ri8 %reg1084, 0
	JE mbb<cond_next204,0xa914d30>

Still ok. After register allocation:

cond_next140 (0xa910740, LLVM BB @0xa90beb0):
	%EAX = MOV32ri -3
	%EDX = MOV32rm <fi#3>, 1, %NOREG, 0
	ADD32rm %EAX<def&use>, %EDX, 1, %NOREG, 0
	%EDX = MOV32rm <fi#7>, 1, %NOREG, 0
	%EDX = MOV32rm %EDX, 1, %NOREG, 40
	IMUL32rr %EAX<def&use>, %EDX
	%ESI = MOV32rm <fi#5>, 1, %NOREG, 0
	%ESI = MOV32rm %ESI, 1, %NOREG, 0
	MOV32mr <fi#4>, 1, %NOREG, 0, %ESI
	%EAX = LEA32r %ESI, 1, %EAX, -3
	%ESI = MOV32rm <fi#7>, 1, %NOREG, 0
	%ESI = MOV32rm %ESI, 1, %NOREG, 32
	%EDI = MOV32rr %EAX
	SHL32ri %EDI<def&use>, 4
	ADD32rr %EDI<def&use>, %ESI
	%XMM0 = MOVAPSrm %ECX, 1, %NOREG, 0
	%XMM1 = MOVAPSrr %XMM0
	SHUFPSrr %XMM1<def&use>, %XMM1, 170
	%XMM2 = MOVAPSrr %XMM0
	SHUFPSrr %XMM2<def&use>, %XMM2, 0
	%XMM3 = MOVAPSrr %XMM0
	SHUFPSrr %XMM3<def&use>, %XMM3, 255
	SHUFPSrr %XMM0<def&use>, %XMM0, 85
	%EBX = MOV32rr %EDI
	AND32ri8 %EBX<def&use>, 15
	CMP32ri8 %EBX, 0
	JE mbb<cond_next204,0xa914d30>

This looks really bad. The problem is shufps is a destructive opcode. Since it
appears as operand two in more than one shufps ops. It resulted in a number of
copies. Note icc also suffers from the same problem. Either the instruction
selector should select pshufd or The register allocator can made the two-address
to three-address transformation.

It also exposes some other problems. See MOV32ri -3 and the spills.

//===---------------------------------------------------------------------===//

http://gcc.gnu.org/bugzilla/show_bug.cgi?id=25500

LLVM is producing bad code.

LBB_main_4:	# cond_true44
	addps %xmm1, %xmm2
	subps %xmm3, %xmm2
	movaps (%ecx), %xmm4
	movaps %xmm2, %xmm1
	addps %xmm4, %xmm1
	addl $16, %ecx
	incl %edx
	cmpl $262144, %edx
	movaps %xmm3, %xmm2
	movaps %xmm4, %xmm3
	jne LBB_main_4	# cond_true44

There are two problems. 1) No need to two loop induction variables. We can
compare against 262144 * 16. 2) Known register coalescer issue. We should
be able eliminate one of the movaps:

	addps %xmm2, %xmm1    <=== Commute!
	subps %xmm3, %xmm1
	movaps (%ecx), %xmm4
	movaps %xmm1, %xmm1   <=== Eliminate!
	addps %xmm4, %xmm1
	addl $16, %ecx
	incl %edx
	cmpl $262144, %edx
	movaps %xmm3, %xmm2
	movaps %xmm4, %xmm3
	jne LBB_main_4	# cond_true44

//===---------------------------------------------------------------------===//

Consider:

__m128 test(float a) {
  return _mm_set_ps(0.0, 0.0, 0.0, a*a);
}

This compiles into:

movss 4(%esp), %xmm1
mulss %xmm1, %xmm1
xorps %xmm0, %xmm0
movss %xmm1, %xmm0
ret

Because mulss doesn't modify the top 3 elements, the top elements of 
xmm1 are already zero'd.  We could compile this to:

movss 4(%esp), %xmm0
mulss %xmm0, %xmm0
ret

//===---------------------------------------------------------------------===//

Here's a sick and twisted idea.  Consider code like this:

__m128 test(__m128 a) {
  float b = *(float*)&A;
  ...
  return _mm_set_ps(0.0, 0.0, 0.0, b);
}

This might compile to this code:

movaps c(%esp), %xmm1
xorps %xmm0, %xmm0
movss %xmm1, %xmm0
ret

Now consider if the ... code caused xmm1 to get spilled.  This might produce
this code:

movaps c(%esp), %xmm1
movaps %xmm1, c2(%esp)
...

xorps %xmm0, %xmm0
movaps c2(%esp), %xmm1
movss %xmm1, %xmm0
ret

However, since the reload is only used by these instructions, we could 
"fold" it into the uses, producing something like this:

movaps c(%esp), %xmm1
movaps %xmm1, c2(%esp)
...

movss c2(%esp), %xmm0
ret

... saving two instructions.

The basic idea is that a reload from a spill slot, can, if only one 4-byte 
chunk is used, bring in 3 zeros the the one element instead of 4 elements.
This can be used to simplify a variety of shuffle operations, where the
elements are fixed zeros.

//===---------------------------------------------------------------------===//

For this:

#include <emmintrin.h>
void test(__m128d *r, __m128d *A, double B) {
  *r = _mm_loadl_pd(*A, &B);
}

We generates:

	subl $12, %esp
	movsd 24(%esp), %xmm0
	movsd %xmm0, (%esp)
	movl 20(%esp), %eax
	movapd (%eax), %xmm0
	movlpd (%esp), %xmm0
	movl 16(%esp), %eax
	movapd %xmm0, (%eax)
	addl $12, %esp
	ret

icc generates:

        movl      4(%esp), %edx                                 #3.6
        movl      8(%esp), %eax                                 #3.6
        movapd    (%eax), %xmm0                                 #4.22
        movlpd    12(%esp), %xmm0                               #4.8
        movapd    %xmm0, (%edx)                                 #4.3
        ret                                                     #5.1

So icc is smart enough to know that B is in memory so it doesn't load it and
store it back to stack.

//===---------------------------------------------------------------------===//

__m128d test1( __m128d A, __m128d B) {
  return _mm_shuffle_pd(A, B, 0x3);
}

compiles to

shufpd $3, %xmm1, %xmm0

Perhaps it's better to use unpckhpd instead?

unpckhpd %xmm1, %xmm0

Don't know if unpckhpd is faster. But it is shorter.

//===---------------------------------------------------------------------===//

This code generates ugly code, probably due to costs being off or something:

void %test(float* %P, <4 x float>* %P2 ) {
        %xFloat0.688 = load float* %P
        %loadVector37.712 = load <4 x float>* %P2
        %inFloat3.713 = insertelement <4 x float> %loadVector37.712, float 0.000000e+00, uint 3
        store <4 x float> %inFloat3.713, <4 x float>* %P2
        ret void
}

Generates:

_test:
        pxor %xmm0, %xmm0
        movd %xmm0, %eax        ;; EAX = 0!
        movl 8(%esp), %ecx
        movaps (%ecx), %xmm0
        pinsrw $6, %eax, %xmm0
        shrl $16, %eax          ;; EAX = 0 again!
        pinsrw $7, %eax, %xmm0
        movaps %xmm0, (%ecx)
        ret

It would be better to generate:

_test:
        movl 8(%esp), %ecx
        movaps (%ecx), %xmm0
	xor %eax, %eax
        pinsrw $6, %eax, %xmm0
        pinsrw $7, %eax, %xmm0
        movaps %xmm0, (%ecx)
        ret

or use pxor (to make a zero vector) and shuffle (to insert it).

//===---------------------------------------------------------------------===//

Some useful information in the Apple Altivec / SSE Migration Guide:

http://developer.apple.com/documentation/Performance/Conceptual/
Accelerate_sse_migration/index.html

e.g. SSE select using and, andnot, or. Various SSE compare translations.

//===---------------------------------------------------------------------===//

Add hooks to commute some CMPP operations.

//===---------------------------------------------------------------------===//

Apply the same transformation that merged four float into a single 128-bit load
to loads from constant pool.

//===---------------------------------------------------------------------===//

Floating point max / min are commutable when -enable-unsafe-fp-path is
specified. We should turn int_x86_sse_max_ss and X86ISD::FMIN etc. into other
nodes which are selected to max / min instructions that are marked commutable.

//===---------------------------------------------------------------------===//

We should compile this:
#include <xmmintrin.h>
typedef union {
  int i[4];
  float f[4];
  __m128 v;
} vector4_t;
void swizzle (const void *a, vector4_t * b, vector4_t * c) {
  b->v = _mm_loadl_pi (b->v, (__m64 *) a);
  c->v = _mm_loadl_pi (c->v, ((__m64 *) a) + 1);
}

to:

_swizzle:
        movl    4(%esp), %eax
        movl    8(%esp), %edx
        movl    12(%esp), %ecx
        movlps  (%eax), %xmm0
        movlps  %xmm0, (%edx)
        movlps  8(%eax), %xmm0
        movlps  %xmm0, (%ecx)
        ret

not:

swizzle:
        movl 8(%esp), %eax
        movaps (%eax), %xmm0
        movl 4(%esp), %ecx
        movlps (%ecx), %xmm0
        movaps %xmm0, (%eax)
        movl 12(%esp), %eax
        movaps (%eax), %xmm0
        movlps 8(%ecx), %xmm0
        movaps %xmm0, (%eax)
        ret

//===---------------------------------------------------------------------===//
