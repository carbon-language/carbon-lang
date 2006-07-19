//===---------------------------------------------------------------------===//
// Random ideas for the X86 backend.
//===---------------------------------------------------------------------===//

Add a MUL2U and MUL2S nodes to represent a multiply that returns both the
Hi and Lo parts (combination of MUL and MULH[SU] into one node).  Add this to
X86, & make the dag combiner produce it when needed.  This will eliminate one
imul from the code generated for:

long long test(long long X, long long Y) { return X*Y; }

by using the EAX result from the mul.  We should add a similar node for
DIVREM.

another case is:

long long test(int X, int Y) { return (long long)X*Y; }

... which should only be one imul instruction.

//===---------------------------------------------------------------------===//

This should be one DIV/IDIV instruction, not a libcall:

unsigned test(unsigned long long X, unsigned Y) {
        return X/Y;
}

This can be done trivially with a custom legalizer.  What about overflow 
though?  http://gcc.gnu.org/bugzilla/show_bug.cgi?id=14224

//===---------------------------------------------------------------------===//

Improvements to the multiply -> shift/add algorithm:
http://gcc.gnu.org/ml/gcc-patches/2004-08/msg01590.html

//===---------------------------------------------------------------------===//

Improve code like this (occurs fairly frequently, e.g. in LLVM):
long long foo(int x) { return 1LL << x; }

http://gcc.gnu.org/ml/gcc-patches/2004-09/msg01109.html
http://gcc.gnu.org/ml/gcc-patches/2004-09/msg01128.html
http://gcc.gnu.org/ml/gcc-patches/2004-09/msg01136.html

Another useful one would be  ~0ULL >> X and ~0ULL << X.

//===---------------------------------------------------------------------===//

Compile this:
_Bool f(_Bool a) { return a!=1; }

into:
        movzbl  %dil, %eax
        xorl    $1, %eax
        ret

//===---------------------------------------------------------------------===//

Some isel ideas:

1. Dynamic programming based approach when compile time if not an
   issue.
2. Code duplication (addressing mode) during isel.
3. Other ideas from "Register-Sensitive Selection, Duplication, and
   Sequencing of Instructions".
4. Scheduling for reduced register pressure.  E.g. "Minimum Register 
   Instruction Sequence Problem: Revisiting Optimal Code Generation for DAGs" 
   and other related papers.
   http://citeseer.ist.psu.edu/govindarajan01minimum.html

//===---------------------------------------------------------------------===//

Should we promote i16 to i32 to avoid partial register update stalls?

//===---------------------------------------------------------------------===//

Leave any_extend as pseudo instruction and hint to register
allocator. Delay codegen until post register allocation.

//===---------------------------------------------------------------------===//

Model X86 EFLAGS as a real register to avoid redudant cmp / test. e.g.

	cmpl $1, %eax
	setg %al
	testb %al, %al  # unnecessary
	jne .BB7

//===---------------------------------------------------------------------===//

Count leading zeros and count trailing zeros:

int clz(int X) { return __builtin_clz(X); }
int ctz(int X) { return __builtin_ctz(X); }

$ gcc t.c -S -o - -O3  -fomit-frame-pointer -masm=intel
clz:
        bsr     %eax, DWORD PTR [%esp+4]
        xor     %eax, 31
        ret
ctz:
        bsf     %eax, DWORD PTR [%esp+4]
        ret

however, check that these are defined for 0 and 32.  Our intrinsics are, GCC's
aren't.

//===---------------------------------------------------------------------===//

Use push/pop instructions in prolog/epilog sequences instead of stores off 
ESP (certain code size win, perf win on some [which?] processors).
Also, it appears icc use push for parameter passing. Need to investigate.

//===---------------------------------------------------------------------===//

Only use inc/neg/not instructions on processors where they are faster than
add/sub/xor.  They are slower on the P4 due to only updating some processor
flags.

//===---------------------------------------------------------------------===//

The instruction selector sometimes misses folding a load into a compare.  The
pattern is written as (cmp reg, (load p)).  Because the compare isn't 
commutative, it is not matched with the load on both sides.  The dag combiner
should be made smart enough to cannonicalize the load into the RHS of a compare
when it can invert the result of the compare for free.

How about intrinsics? An example is:
  *res = _mm_mulhi_epu16(*A, _mm_mul_epu32(*B, *C));

compiles to
	pmuludq (%eax), %xmm0
	movl 8(%esp), %eax
	movdqa (%eax), %xmm1
	pmulhuw %xmm0, %xmm1

The transformation probably requires a X86 specific pass or a DAG combiner
target specific hook.

//===---------------------------------------------------------------------===//

The DAG Isel doesn't fold the loads into the adds in this testcase.  The
pattern selector does.  This is because the chain value of the load gets 
selected first, and the loads aren't checking to see if they are only used by
and add.

.ll:

int %test(int* %x, int* %y, int* %z) {
        %X = load int* %x
        %Y = load int* %y
        %Z = load int* %z
        %a = add int %X, %Y
        %b = add int %a, %Z
        ret int %b
}

dag isel:

_test:
        movl 4(%esp), %eax
        movl (%eax), %eax
        movl 8(%esp), %ecx
        movl (%ecx), %ecx
        addl %ecx, %eax
        movl 12(%esp), %ecx
        movl (%ecx), %ecx
        addl %ecx, %eax
        ret

pattern isel:

_test:
        movl 12(%esp), %ecx
        movl 4(%esp), %edx
        movl 8(%esp), %eax
        movl (%eax), %eax
        addl (%edx), %eax
        addl (%ecx), %eax
        ret

This is bad for register pressure, though the dag isel is producing a 
better schedule. :)

//===---------------------------------------------------------------------===//

In many cases, LLVM generates code like this:

_test:
        movl 8(%esp), %eax
        cmpl %eax, 4(%esp)
        setl %al
        movzbl %al, %eax
        ret

on some processors (which ones?), it is more efficient to do this:

_test:
        movl 8(%esp), %ebx
	xor %eax, %eax
        cmpl %ebx, 4(%esp)
        setl %al
        ret

Doing this correctly is tricky though, as the xor clobbers the flags.

//===---------------------------------------------------------------------===//

We should generate 'test' instead of 'cmp' in various cases, e.g.:

bool %test(int %X) {
        %Y = shl int %X, ubyte 1
        %C = seteq int %Y, 0
        ret bool %C
}
bool %test(int %X) {
        %Y = and int %X, 8
        %C = seteq int %Y, 0
        ret bool %C
}

This may just be a matter of using 'test' to write bigger patterns for X86cmp.

An important case is comparison against zero:

if (X == 0) ...

instead of:

	cmpl $0, %eax
	je LBB4_2	#cond_next

use:
	test %eax, %eax
	jz LBB4_2

which is smaller.

//===---------------------------------------------------------------------===//

We should generate bts/btr/etc instructions on targets where they are cheap or
when codesize is important.  e.g., for:

void setbit(int *target, int bit) {
    *target |= (1 << bit);
}
void clearbit(int *target, int bit) {
    *target &= ~(1 << bit);
}

//===---------------------------------------------------------------------===//

Instead of the following for memset char*, 1, 10:

	movl $16843009, 4(%edx)
	movl $16843009, (%edx)
	movw $257, 8(%edx)

It might be better to generate

	movl $16843009, %eax
	movl %eax, 4(%edx)
	movl %eax, (%edx)
	movw al, 8(%edx)
	
when we can spare a register. It reduces code size.

//===---------------------------------------------------------------------===//

Evaluate what the best way to codegen sdiv X, (2^C) is.  For X/8, we currently
get this:

int %test1(int %X) {
        %Y = div int %X, 8
        ret int %Y
}

_test1:
        movl 4(%esp), %eax
        movl %eax, %ecx
        sarl $31, %ecx
        shrl $29, %ecx
        addl %ecx, %eax
        sarl $3, %eax
        ret

GCC knows several different ways to codegen it, one of which is this:

_test1:
        movl    4(%esp), %eax
        cmpl    $-1, %eax
        leal    7(%eax), %ecx
        cmovle  %ecx, %eax
        sarl    $3, %eax
        ret

which is probably slower, but it's interesting at least :)

//===---------------------------------------------------------------------===//

Should generate min/max for stuff like:

void minf(float a, float b, float *X) {
  *X = a <= b ? a : b;
}

Make use of floating point min / max instructions. Perhaps introduce ISD::FMIN
and ISD::FMAX node types?

//===---------------------------------------------------------------------===//

The first BB of this code:

declare bool %foo()
int %bar() {
        %V = call bool %foo()
        br bool %V, label %T, label %F
T:
        ret int 1
F:
        call bool %foo()
        ret int 12
}

compiles to:

_bar:
        subl $12, %esp
        call L_foo$stub
        xorb $1, %al
        testb %al, %al
        jne LBB_bar_2   # F

It would be better to emit "cmp %al, 1" than a xor and test.

//===---------------------------------------------------------------------===//

Enable X86InstrInfo::convertToThreeAddress().

//===---------------------------------------------------------------------===//

Investigate whether it is better to codegen the following

        %tmp.1 = mul int %x, 9
to

	movl	4(%esp), %eax
	leal	(%eax,%eax,8), %eax

as opposed to what llc is currently generating:

	imull $9, 4(%esp), %eax

Currently the load folding imull has a higher complexity than the LEA32 pattern.

//===---------------------------------------------------------------------===//

We are currently lowering large (1MB+) memmove/memcpy to rep/stosl and rep/movsl
We should leave these as libcalls for everything over a much lower threshold,
since libc is hand tuned for medium and large mem ops (avoiding RFO for large
stores, TLB preheating, etc)

//===---------------------------------------------------------------------===//

Optimize this into something reasonable:
 x * copysign(1.0, y) * copysign(1.0, z)

//===---------------------------------------------------------------------===//

Optimize copysign(x, *y) to use an integer load from y.

//===---------------------------------------------------------------------===//

%X = weak global int 0

void %foo(int %N) {
	%N = cast int %N to uint
	%tmp.24 = setgt int %N, 0
	br bool %tmp.24, label %no_exit, label %return

no_exit:
	%indvar = phi uint [ 0, %entry ], [ %indvar.next, %no_exit ]
	%i.0.0 = cast uint %indvar to int
	volatile store int %i.0.0, int* %X
	%indvar.next = add uint %indvar, 1
	%exitcond = seteq uint %indvar.next, %N
	br bool %exitcond, label %return, label %no_exit

return:
	ret void
}

compiles into:

	.text
	.align	4
	.globl	_foo
_foo:
	movl 4(%esp), %eax
	cmpl $1, %eax
	jl LBB_foo_4	# return
LBB_foo_1:	# no_exit.preheader
	xorl %ecx, %ecx
LBB_foo_2:	# no_exit
	movl L_X$non_lazy_ptr, %edx
	movl %ecx, (%edx)
	incl %ecx
	cmpl %eax, %ecx
	jne LBB_foo_2	# no_exit
LBB_foo_3:	# return.loopexit
LBB_foo_4:	# return
	ret

We should hoist "movl L_X$non_lazy_ptr, %edx" out of the loop after
remateralization is implemented. This can be accomplished with 1) a target
dependent LICM pass or 2) makeing SelectDAG represent the whole function. 

//===---------------------------------------------------------------------===//

The following tests perform worse with LSR:

lambda, siod, optimizer-eval, ackermann, hash2, nestedloop, strcat, and Treesor.

//===---------------------------------------------------------------------===//

Teach the coalescer to coalesce vregs of different register classes. e.g. FR32 /
FR64 to VR128.

//===---------------------------------------------------------------------===//

mov $reg, 48(%esp)
...
leal 48(%esp), %eax
mov %eax, (%esp)
call _foo

Obviously it would have been better for the first mov (or any op) to store
directly %esp[0] if there are no other uses.

//===---------------------------------------------------------------------===//

Adding to the list of cmp / test poor codegen issues:

int test(__m128 *A, __m128 *B) {
  if (_mm_comige_ss(*A, *B))
    return 3;
  else
    return 4;
}

_test:
	movl 8(%esp), %eax
	movaps (%eax), %xmm0
	movl 4(%esp), %eax
	movaps (%eax), %xmm1
	comiss %xmm0, %xmm1
	setae %al
	movzbl %al, %ecx
	movl $3, %eax
	movl $4, %edx
	cmpl $0, %ecx
	cmove %edx, %eax
	ret

Note the setae, movzbl, cmpl, cmove can be replaced with a single cmovae. There
are a number of issues. 1) We are introducing a setcc between the result of the
intrisic call and select. 2) The intrinsic is expected to produce a i32 value
so a any extend (which becomes a zero extend) is added.

We probably need some kind of target DAG combine hook to fix this.

//===---------------------------------------------------------------------===//

We generate significantly worse code for this than GCC:
http://gcc.gnu.org/bugzilla/show_bug.cgi?id=21150
http://gcc.gnu.org/bugzilla/attachment.cgi?id=8701

There is also one case we do worse on PPC.

//===---------------------------------------------------------------------===//

If shorter, we should use things like:
movzwl %ax, %eax
instead of:
andl $65535, %EAX

The former can also be used when the two-addressy nature of the 'and' would
require a copy to be inserted (in X86InstrInfo::convertToThreeAddress).

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

Bad codegen:

char foo(int x) { return x; }

_foo:
	movl 4(%esp), %eax
	shll $24, %eax
	sarl $24, %eax
	ret

SIGN_EXTEND_INREG can be implemented as (sext (trunc)) to take advantage of 
sub-registers.

//===---------------------------------------------------------------------===//

Consider this:

typedef struct pair { float A, B; } pair;
void pairtest(pair P, float *FP) {
        *FP = P.A+P.B;
}

We currently generate this code with llvmgcc4:

_pairtest:
        subl $12, %esp
        movl 20(%esp), %eax
        movl %eax, 4(%esp)
        movl 16(%esp), %eax
        movl %eax, (%esp)
        movss (%esp), %xmm0
        addss 4(%esp), %xmm0
        movl 24(%esp), %eax
        movss %xmm0, (%eax)
        addl $12, %esp
        ret

we should be able to generate:
_pairtest:
        movss 4(%esp), %xmm0
        movl 12(%esp), %eax
        addss 8(%esp), %xmm0
        movss %xmm0, (%eax)
        ret

The issue is that llvmgcc4 is forcing the struct to memory, then passing it as
integer chunks.  It does this so that structs like {short,short} are passed in
a single 32-bit integer stack slot.  We should handle the safe cases above much
nicer, while still handling the hard cases.

//===---------------------------------------------------------------------===//

Some ideas for instruction selection code simplification: 1. A pre-pass to
determine which chain producing node can or cannot be folded. The generated
isel code would then use the information. 2. The same pre-pass can force
ordering of TokenFactor operands to allow load / store folding. 3. During isel,
instead of recursively going up the chain operand chain, mark the chain operand
as available and put it in some work list. Select other nodes in the normal
manner. The chain operands are selected after all other nodes are selected. Uses
of chain nodes are modified after instruction selection is completed.

//===---------------------------------------------------------------------===//

Another instruction selector deficiency:

void %bar() {
	%tmp = load int (int)** %foo
	%tmp = tail call int %tmp( int 3 )
	ret void
}

_bar:
	subl $12, %esp
	movl L_foo$non_lazy_ptr, %eax
	movl (%eax), %eax
	call *%eax
	addl $12, %esp
	ret

The current isel scheme will not allow the load to be folded in the call since
the load's chain result is read by the callseq_start.

//===---------------------------------------------------------------------===//

Don't forget to find a way to squash noop truncates in the JIT environment.

//===---------------------------------------------------------------------===//

Implement anyext in the same manner as truncate that would allow them to be
eliminated.

//===---------------------------------------------------------------------===//

How about implementing truncate / anyext as a property of machine instruction
operand? i.e. Print as 32-bit super-class register / 16-bit sub-class register.
Do this for the cases where a truncate / anyext is guaranteed to be eliminated.
For IA32 that is truncate from 32 to 16 and anyext from 16 to 32.

//===---------------------------------------------------------------------===//

For this:

int test(int a)
{
  return a * 3;
}

We currently emits
	imull $3, 4(%esp), %eax

Perhaps this is what we really should generate is? Is imull three or four
cycles? Note: ICC generates this:
	movl	4(%esp), %eax
	leal	(%eax,%eax,2), %eax

The current instruction priority is based on pattern complexity. The former is
more "complex" because it folds a load so the latter will not be emitted.

Perhaps we should use AddedComplexity to give LEA32r a higher priority? We
should always try to match LEA first since the LEA matching code does some
estimate to determine whether the match is profitable.

However, if we care more about code size, then imull is better. It's two bytes
shorter than movl + leal.

//===---------------------------------------------------------------------===//

Implement CTTZ, CTLZ with bsf and bsr.

//===---------------------------------------------------------------------===//

It appears gcc place string data with linkonce linkage in
.section __TEXT,__const_coal,coalesced instead of
.section __DATA,__const_coal,coalesced.
Take a look at darwin.h, there are other Darwin assembler directives that we
do not make use of.

//===---------------------------------------------------------------------===//

We should handle __attribute__ ((__visibility__ ("hidden"))).

//===---------------------------------------------------------------------===//

Consider:
int foo(int *a, int t) {
int x;
for (x=0; x<40; ++x)
   t = t + a[x] + x;
return t;
}

We generate:
LBB1_1: #cond_true
        movl %ecx, %esi
        movl (%edx,%eax,4), %edi
        movl %esi, %ecx
        addl %edi, %ecx
        addl %eax, %ecx
        incl %eax
        cmpl $40, %eax
        jne LBB1_1      #cond_true

GCC generates:

L2:
        addl    (%ecx,%edx,4), %eax
        addl    %edx, %eax
        addl    $1, %edx
        cmpl    $40, %edx
        jne     L2

Smells like a register coallescing/reassociation issue.

//===---------------------------------------------------------------------===//

Use cpuid to auto-detect CPU features such as SSE, SSE2, and SSE3.

//===---------------------------------------------------------------------===//

JIT should resolve __cxa_atexit on Mac OS X. In a non-jit environment, the
symbol is a dynamically resolved by the linker.
