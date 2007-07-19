//===---------------------------------------------------------------------===//
// Random ideas for the X86 backend.
//===---------------------------------------------------------------------===//

Missing features:
  - Support for SSE4: http://www.intel.com/software/penryn
http://softwarecommunity.intel.com/isn/Downloads/Intel%20SSE4%20Programming%20Reference.pdf
  - support for 3DNow!
  - weird abis?

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

or:

unsigned long long int t2(unsigned int a, unsigned int b) {
       return (unsigned long long)a * b;
}

... which should be one mul instruction.


This can be done with a custom expander, but it would be nice to move this to
generic code.

//===---------------------------------------------------------------------===//

CodeGen/X86/lea-3.ll:test3 should be a single LEA, not a shift/move.  The X86
backend knows how to three-addressify this shift, but it appears the register
allocator isn't even asking it to do so in this case.  We should investigate
why this isn't happening, it could have significant impact on other important
cases for X86 as well.

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

One better solution for 1LL << x is:
        xorl    %eax, %eax
        xorl    %edx, %edx
        testb   $32, %cl
        sete    %al
        setne   %dl
        sall    %cl, %eax
        sall    %cl, %edx

But that requires good 8-bit subreg support.

64-bit shifts (in general) expand to really bad code.  Instead of using
cmovs, we should expand to a conditional branch like GCC produces.

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

Another example (use predsimplify to eliminate a select):

int foo (unsigned long j) {
  if (j)
    return __builtin_ffs (j) - 1;
  else
    return 0;
}

//===---------------------------------------------------------------------===//

It appears icc use push for parameter passing. Need to investigate.

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

//===---------------------------------------------------------------------===//

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
        xor  %eax, %eax
        cmpl %ebx, 4(%esp)
        setl %al
        ret

Doing this correctly is tricky though, as the xor clobbers the flags.

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

We are generating far worse code than gcc:

volatile short X, Y;

void foo(int N) {
  int i;
  for (i = 0; i < N; i++) { X = i; Y = i*4; }
}

LBB1_1:	#bb.preheader
	xorl %ecx, %ecx
	xorw %dx, %dx
LBB1_2:	#bb
	movl L_X$non_lazy_ptr, %esi
	movw %dx, (%esi)
	movw %dx, %si
	shlw $2, %si
	movl L_Y$non_lazy_ptr, %edi
	movw %si, (%edi)
	incl %ecx
	incw %dx
	cmpl %eax, %ecx
	jne LBB1_2	#bb

vs.

	xorl	%edx, %edx
	movl	L_X$non_lazy_ptr-"L00000000001$pb"(%ebx), %esi
	movl	L_Y$non_lazy_ptr-"L00000000001$pb"(%ebx), %ecx
L4:
	movw	%dx, (%esi)
	leal	0(,%edx,4), %eax
	movw	%ax, (%ecx)
	addl	$1, %edx
	cmpl	%edx, %edi
	jne	L4

There are 3 issues:

1. Lack of post regalloc LICM.
2. Poor sub-regclass support. That leads to inability to promote the 16-bit
   arithmetic op to 32-bit and making use of leal.
3. LSR unable to reused IV for a different type (i16 vs. i32) even though
   the cast would be free.

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
        movl 8(%esp), %eax
        movl 4(%esp), %ecx
        movd %eax, %xmm0
        movd %ecx, %xmm1
        addss %xmm0, %xmm1
        movl 12(%esp), %eax
        movss %xmm1, (%eax)
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

While true in general, in this specific case we could do better by promoting
load int + bitcast to float -> load fload.  This basically needs alignment info,
the code is already implemented (but disabled) in dag combine).

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

int %foo(int* %a, int %t) {
entry:
        br label %cond_true

cond_true:              ; preds = %cond_true, %entry
        %x.0.0 = phi int [ 0, %entry ], [ %tmp9, %cond_true ]  
        %t_addr.0.0 = phi int [ %t, %entry ], [ %tmp7, %cond_true ]
        %tmp2 = getelementptr int* %a, int %x.0.0              
        %tmp3 = load int* %tmp2         ; <int> [#uses=1]
        %tmp5 = add int %t_addr.0.0, %x.0.0             ; <int> [#uses=1]
        %tmp7 = add int %tmp5, %tmp3            ; <int> [#uses=2]
        %tmp9 = add int %x.0.0, 1               ; <int> [#uses=2]
        %tmp = setgt int %tmp9, 39              ; <bool> [#uses=1]
        br bool %tmp, label %bb12, label %cond_true

bb12:           ; preds = %cond_true
        ret int %tmp7
}

is pessimized by -loop-reduce and -indvars

//===---------------------------------------------------------------------===//

u32 to float conversion improvement:

float uint32_2_float( unsigned u ) {
  float fl = (int) (u & 0xffff);
  float fh = (int) (u >> 16);
  fh *= 0x1.0p16f;
  return fh + fl;
}

00000000        subl    $0x04,%esp
00000003        movl    0x08(%esp,1),%eax
00000007        movl    %eax,%ecx
00000009        shrl    $0x10,%ecx
0000000c        cvtsi2ss        %ecx,%xmm0
00000010        andl    $0x0000ffff,%eax
00000015        cvtsi2ss        %eax,%xmm1
00000019        mulss   0x00000078,%xmm0
00000021        addss   %xmm1,%xmm0
00000025        movss   %xmm0,(%esp,1)
0000002a        flds    (%esp,1)
0000002d        addl    $0x04,%esp
00000030        ret

//===---------------------------------------------------------------------===//

When using fastcc abi, align stack slot of argument of type double on 8 byte
boundary to improve performance.

//===---------------------------------------------------------------------===//

Codegen:

int f(int a, int b) {
  if (a == 4 || a == 6)
    b++;
  return b;
}


as:

or eax, 2
cmp eax, 6
jz label

//===---------------------------------------------------------------------===//

GCC's ix86_expand_int_movcc function (in i386.c) has a ton of interesting
simplifications for integer "x cmp y ? a : b".  For example, instead of:

int G;
void f(int X, int Y) {
  G = X < 0 ? 14 : 13;
}

compiling to:

_f:
        movl $14, %eax
        movl $13, %ecx
        movl 4(%esp), %edx
        testl %edx, %edx
        cmovl %eax, %ecx
        movl %ecx, _G
        ret

it could be:
_f:
        movl    4(%esp), %eax
        sarl    $31, %eax
        notl    %eax
        addl    $14, %eax
        movl    %eax, _G
        ret

etc.

//===---------------------------------------------------------------------===//

Currently we don't have elimination of redundant stack manipulations. Consider
the code:

int %main() {
entry:
	call fastcc void %test1( )
	call fastcc void %test2( sbyte* cast (void ()* %test1 to sbyte*) )
	ret int 0
}

declare fastcc void %test1()

declare fastcc void %test2(sbyte*)


This currently compiles to:

	subl $16, %esp
	call _test5
	addl $12, %esp
	subl $16, %esp
	movl $_test5, (%esp)
	call _test6
	addl $12, %esp

The add\sub pair is really unneeded here.

//===---------------------------------------------------------------------===//

We currently compile sign_extend_inreg into two shifts:

long foo(long X) {
  return (long)(signed char)X;
}

becomes:

_foo:
        movl 4(%esp), %eax
        shll $24, %eax
        sarl $24, %eax
        ret

This could be:

_foo:
        movsbl  4(%esp),%eax
        ret

//===---------------------------------------------------------------------===//

Consider the expansion of:

uint %test3(uint %X) {
        %tmp1 = rem uint %X, 255
        ret uint %tmp1
}

Currently it compiles to:

...
        movl $2155905153, %ecx
        movl 8(%esp), %esi
        movl %esi, %eax
        mull %ecx
...

This could be "reassociated" into:

        movl $2155905153, %eax
        movl 8(%esp), %ecx
        mull %ecx

to avoid the copy.  In fact, the existing two-address stuff would do this
except that mul isn't a commutative 2-addr instruction.  I guess this has
to be done at isel time based on the #uses to mul?

//===---------------------------------------------------------------------===//

Make sure the instruction which starts a loop does not cross a cacheline
boundary. This requires knowning the exact length of each machine instruction.
That is somewhat complicated, but doable. Example 256.bzip2:

In the new trace, the hot loop has an instruction which crosses a cacheline
boundary.  In addition to potential cache misses, this can't help decoding as I
imagine there has to be some kind of complicated decoder reset and realignment
to grab the bytes from the next cacheline.

532  532 0x3cfc movb     (1809(%esp, %esi), %bl   <<<--- spans 2 64 byte lines
942  942 0x3d03 movl     %dh, (1809(%esp, %esi)                                                                          
937  937 0x3d0a incl     %esi                           
3    3   0x3d0b cmpb     %bl, %dl                                               
27   27  0x3d0d jnz      0x000062db <main+11707>

//===---------------------------------------------------------------------===//

In c99 mode, the preprocessor doesn't like assembly comments like #TRUNCATE.

//===---------------------------------------------------------------------===//

This could be a single 16-bit load.

int f(char *p) {
    if ((p[0] == 1) & (p[1] == 2)) return 1;
    return 0;
}

//===---------------------------------------------------------------------===//

We should inline lrintf and probably other libc functions.

//===---------------------------------------------------------------------===//

Start using the flags more.  For example, compile:

int add_zf(int *x, int y, int a, int b) {
     if ((*x += y) == 0)
          return a;
     else
          return b;
}

to:
       addl    %esi, (%rdi)
       movl    %edx, %eax
       cmovne  %ecx, %eax
       ret
instead of:

_add_zf:
        addl (%rdi), %esi
        movl %esi, (%rdi)
        testl %esi, %esi
        cmove %edx, %ecx
        movl %ecx, %eax
        ret

and:

int add_zf(int *x, int y, int a, int b) {
     if ((*x + y) < 0)
          return a;
     else
          return b;
}

to:

add_zf:
        addl    (%rdi), %esi
        movl    %edx, %eax
        cmovns  %ecx, %eax
        ret

instead of:

_add_zf:
        addl (%rdi), %esi
        testl %esi, %esi
        cmovs %edx, %ecx
        movl %ecx, %eax
        ret

//===---------------------------------------------------------------------===//

This:
#include <math.h>
int foo(double X) { return isnan(X); }

compiles to (-m64):

_foo:
        pxor %xmm1, %xmm1
        ucomisd %xmm1, %xmm0
        setp %al
        movzbl %al, %eax
        ret

the pxor is not needed, we could compare the value against itself.

//===---------------------------------------------------------------------===//

These two functions have identical effects:

unsigned int f(unsigned int i, unsigned int n) {++i; if (i == n) ++i; return i;}
unsigned int f2(unsigned int i, unsigned int n) {++i; i += i == n; return i;}

We currently compile them to:

_f:
        movl 4(%esp), %eax
        movl %eax, %ecx
        incl %ecx
        movl 8(%esp), %edx
        cmpl %edx, %ecx
        jne LBB1_2      #UnifiedReturnBlock
LBB1_1: #cond_true
        addl $2, %eax
        ret
LBB1_2: #UnifiedReturnBlock
        movl %ecx, %eax
        ret
_f2:
        movl 4(%esp), %eax
        movl %eax, %ecx
        incl %ecx
        cmpl 8(%esp), %ecx
        sete %cl
        movzbl %cl, %ecx
        leal 1(%ecx,%eax), %eax
        ret

both of which are inferior to GCC's:

_f:
        movl    4(%esp), %edx
        leal    1(%edx), %eax
        addl    $2, %edx
        cmpl    8(%esp), %eax
        cmove   %edx, %eax
        ret
_f2:
        movl    4(%esp), %eax
        addl    $1, %eax
        xorl    %edx, %edx
        cmpl    8(%esp), %eax
        sete    %dl
        addl    %edx, %eax
        ret

//===---------------------------------------------------------------------===//

This code:

void test(int X) {
  if (X) abort();
}

is currently compiled to:

_test:
        subl $12, %esp
        cmpl $0, 16(%esp)
        jne LBB1_1
        addl $12, %esp
        ret
LBB1_1:
        call L_abort$stub

It would be better to produce:

_test:
        subl $12, %esp
        cmpl $0, 16(%esp)
        jne L_abort$stub
        addl $12, %esp
        ret

This can be applied to any no-return function call that takes no arguments etc.
Alternatively, the stack save/restore logic could be shrink-wrapped, producing
something like this:

_test:
        cmpl $0, 4(%esp)
        jne LBB1_1
        ret
LBB1_1:
        subl $12, %esp
        call L_abort$stub

Both are useful in different situations.  Finally, it could be shrink-wrapped
and tail called, like this:

_test:
        cmpl $0, 4(%esp)
        jne LBB1_1
        ret
LBB1_1:
        pop %eax   # realign stack.
        call L_abort$stub

Though this probably isn't worth it.

//===---------------------------------------------------------------------===//

We need to teach the codegen to convert two-address INC instructions to LEA
when the flags are dead.  For example, on X86-64, compile:

int foo(int A, int B) {
  return A+1;
}

to:

_foo:
        leal    1(%edi), %eax
        ret

instead of:

_foo:
        incl %edi
        movl %edi, %eax
        ret

Another example is:

;; X's live range extends beyond the shift, so the register allocator
;; cannot coalesce it with Y.  Because of this, a copy needs to be
;; emitted before the shift to save the register value before it is
;; clobbered.  However, this copy is not needed if the register
;; allocator turns the shift into an LEA.  This also occurs for ADD.

; Check that the shift gets turned into an LEA.
; RUN: llvm-upgrade < %s | llvm-as | llc -march=x86 -x86-asm-syntax=intel | \
; RUN:   not grep {mov E.X, E.X}

%G = external global int

int %test1(int %X, int %Y) {
        %Z = add int %X, %Y
        volatile store int %Y, int* %G
        volatile store int %Z, int* %G
        ret int %X
}

int %test2(int %X) {
        %Z = add int %X, 1  ;; inc
        volatile store int %Z, int* %G
        ret int %X
}

//===---------------------------------------------------------------------===//

This:
#include <xmmintrin.h>
unsigned test(float f) {
 return _mm_cvtsi128_si32( (__m128i) _mm_set_ss( f ));
}

Compiles to:
_test:
        movss 4(%esp), %xmm0
        movd %xmm0, %eax
        ret

it should compile to a move from the stack slot directly into eax.  DAGCombine
has this xform, but it is currently disabled until the alignment fields of 
the load/store nodes are trustworthy.

//===---------------------------------------------------------------------===//

Sometimes it is better to codegen subtractions from a constant (e.g. 7-x) with
a neg instead of a sub instruction.  Consider:

int test(char X) { return 7-X; }

we currently produce:
_test:
        movl $7, %eax
        movsbl 4(%esp), %ecx
        subl %ecx, %eax
        ret

We would use one fewer register if codegen'd as:

        movsbl 4(%esp), %eax
	neg %eax
        add $7, %eax
        ret

Note that this isn't beneficial if the load can be folded into the sub.  In
this case, we want a sub:

int test(int X) { return 7-X; }
_test:
        movl $7, %eax
        subl 4(%esp), %eax
        ret

//===---------------------------------------------------------------------===//

For code like:
phi (undef, x)

We get an implicit def on the undef side. If the phi is spilled, we then get:
implicitdef xmm1
store xmm1 -> stack

It should be possible to teach the x86 backend to "fold" the store into the
implicitdef, which just deletes the implicit def.

These instructions should go away:
#IMPLICIT_DEF %xmm1 
movaps %xmm1, 192(%esp) 
movaps %xmm1, 224(%esp) 
movaps %xmm1, 176(%esp)
