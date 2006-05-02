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

Some targets (e.g. athlons) prefer freep to fstp ST(0):
http://gcc.gnu.org/ml/gcc-patches/2004-04/msg00659.html

//===---------------------------------------------------------------------===//

This should use fiadd on chips where it is profitable:
double foo(double P, int *I) { return P+*I; }

We have fiadd patterns now but the followings have the same cost and
complexity. We need a way to specify the later is more profitable.

def FpADD32m  : FpI<(ops RFP:$dst, RFP:$src1, f32mem:$src2), OneArgFPRW,
                    [(set RFP:$dst, (fadd RFP:$src1,
                                     (extloadf64f32 addr:$src2)))]>;
                // ST(0) = ST(0) + [mem32]

def FpIADD32m : FpI<(ops RFP:$dst, RFP:$src1, i32mem:$src2), OneArgFPRW,
                    [(set RFP:$dst, (fadd RFP:$src1,
                                     (X86fild addr:$src2, i32)))]>;
                // ST(0) = ST(0) + [mem32int]

//===---------------------------------------------------------------------===//

The FP stackifier needs to be global.  Also, it should handle simple permutates
to reduce number of shuffle instructions, e.g. turning:

fld P	->		fld Q
fld Q			fld P
fxch

or:

fxch	->		fucomi
fucomi			jl X
jg X

Ideas:
http://gcc.gnu.org/ml/gcc-patches/2004-11/msg02410.html


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

Add a target specific hook to DAG combiner to handle SINT_TO_FP and
FP_TO_SINT when the source operand is already in memory.

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

Open code rint,floor,ceil,trunc:
http://gcc.gnu.org/ml/gcc-patches/2004-08/msg02006.html
http://gcc.gnu.org/ml/gcc-patches/2004-08/msg02011.html

//===---------------------------------------------------------------------===//

Combine: a = sin(x), b = cos(x) into a,b = sincos(x).

Expand these to calls of sin/cos and stores:
      double sincos(double x, double *sin, double *cos);
      float sincosf(float x, float *sin, float *cos);
      long double sincosl(long double x, long double *sin, long double *cos);

Doing so could allow SROA of the destination pointers.  See also:
http://gcc.gnu.org/bugzilla/show_bug.cgi?id=17687

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

LSR should be turned on for the X86 backend and tuned to take advantage of its
addressing modes.

//===---------------------------------------------------------------------===//

When compiled with unsafemath enabled, "main" should enable SSE DAZ mode and
other fast SSE modes.

//===---------------------------------------------------------------------===//

Think about doing i64 math in SSE regs.

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

We need to lower switch statements to tablejumps when appropriate instead of
always into binary branch trees.

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

None of the FPStack instructions are handled in
X86RegisterInfo::foldMemoryOperand, which prevents the spiller from
folding spill code into the instructions.

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

It's not clear whether we should use pxor or xorps / xorpd to clear XMM
registers. The choice may depend on subtarget information. We should do some
more experiments on different x86 machines.

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

Use movhps to update upper 64-bits of a v4sf value. Also movlps on lower half
of a v4sf value.

//===---------------------------------------------------------------------===//

Better codegen for vector_shuffles like this { x, 0, 0, 0 } or { x, 0, x, 0}.
Perhaps use pxor / xorp* to clear a XMM register first?

//===---------------------------------------------------------------------===//

Better codegen for:

void f(float a, float b, vector float * out) { *out = (vector float){ a, 0.0, 0.0, b}; }
void f(float a, float b, vector float * out) { *out = (vector float){ a, b, 0.0, 0}; }

For the later we generate:

_f:
        pxor %xmm0, %xmm0
        movss 8(%esp), %xmm1
        movaps %xmm0, %xmm2
        unpcklps %xmm1, %xmm2
        movss 4(%esp), %xmm1
        unpcklps %xmm0, %xmm1
        unpcklps %xmm2, %xmm1
        movl 12(%esp), %eax
        movaps %xmm1, (%eax)
        ret

This seems like it should use shufps, one for each of a & b.

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

Use movddup to splat a v2f64 directly from a memory source. e.g.

#include <emmintrin.h>

void test(__m128d *r, double A) {
  *r = _mm_set1_pd(A);
}

llc:

_test:
	movsd 8(%esp), %xmm0
	unpcklpd %xmm0, %xmm0
	movl 4(%esp), %eax
	movapd %xmm0, (%eax)
	ret

icc:

_test:
	movl 4(%esp), %eax
	movddup 8(%esp), %xmm0
	movapd %xmm0, (%eax)
	ret

//===---------------------------------------------------------------------===//

A Mac OS X IA-32 specific ABI bug wrt returning value > 8 bytes:
http://llvm.org/bugs/show_bug.cgi?id=729

//===---------------------------------------------------------------------===//

X86RegisterInfo::copyRegToReg() returns X86::MOVAPSrr for VR128. Is it possible
to choose between movaps, movapd, and movdqa based on types of source and
destination?

How about andps, andpd, and pand? Do we really care about the type of the packed
elements? If not, why not always use the "ps" variants which are likely to be
shorter.

//===---------------------------------------------------------------------===//

We are emitting bad code for this:

float %test(float* %V, int %I, int %D, float %V) {
entry:
	%tmp = seteq int %D, 0
	br bool %tmp, label %cond_true, label %cond_false23

cond_true:
	%tmp3 = getelementptr float* %V, int %I
	%tmp = load float* %tmp3
	%tmp5 = setgt float %tmp, %V
	%tmp6 = tail call bool %llvm.isunordered.f32( float %tmp, float %V )
	%tmp7 = or bool %tmp5, %tmp6
	br bool %tmp7, label %UnifiedReturnBlock, label %cond_next

cond_next:
	%tmp10 = add int %I, 1
	%tmp12 = getelementptr float* %V, int %tmp10
	%tmp13 = load float* %tmp12
	%tmp15 = setle float %tmp13, %V
	%tmp16 = tail call bool %llvm.isunordered.f32( float %tmp13, float %V )
	%tmp17 = or bool %tmp15, %tmp16
	%retval = select bool %tmp17, float 0.000000e+00, float 1.000000e+00
	ret float %retval

cond_false23:
	%tmp28 = tail call float %foo( float* %V, int %I, int %D, float %V )
	ret float %tmp28

UnifiedReturnBlock:		; preds = %cond_true
	ret float 0.000000e+00
}

declare bool %llvm.isunordered.f32(float, float)

declare float %foo(float*, int, int, float)


It exposes a known load folding problem:

	movss (%edx,%ecx,4), %xmm1
	ucomiss %xmm1, %xmm0

As well as this:

LBB_test_2:	# cond_next
	movss LCPI1_0, %xmm2
	pxor %xmm3, %xmm3
	ucomiss %xmm0, %xmm1
	jbe LBB_test_6	# cond_next
LBB_test_5:	# cond_next
	movaps %xmm2, %xmm3
LBB_test_6:	# cond_next
	movss %xmm3, 40(%esp)
	flds 40(%esp)
	addl $44, %esp
	ret

Clearly it's unnecessary to clear %xmm3. It's also not clear why we are emitting
three moves (movss, movaps, movss).

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

We generate significantly worse code for this than GCC:
http://gcc.gnu.org/bugzilla/show_bug.cgi?id=21150
http://gcc.gnu.org/bugzilla/attachment.cgi?id=8701

There is also one case we do worse on PPC.

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

This testcase:

%G1 = weak global <4 x float> zeroinitializer           ; <<4 x float>*> [#uses=1]
%G2 = weak global <4 x float> zeroinitializer           ; <<4 x float>*> [#uses=1]
%G3 = weak global <4 x float> zeroinitializer           ; <<4 x float>*> [#uses=1]
%G4 = weak global <4 x float> zeroinitializer           ; <<4 x float>*> [#uses=1]

implementation   ; Functions:

void %test() {
        %tmp = load <4 x float>* %G1            ; <<4 x float>> [#uses=2]
        %tmp2 = load <4 x float>* %G2           ; <<4 x float>> [#uses=2]
        %tmp135 = shufflevector <4 x float> %tmp, <4 x float> %tmp2, <4 x uint> < uint 0, uint 4, uint 1, uint 5 >            ; <<4 x float>> [#uses=1]
        store <4 x float> %tmp135, <4 x float>* %G3
        %tmp293 = shufflevector <4 x float> %tmp, <4 x float> %tmp2, <4 x uint> < uint 1, uint undef, uint 3, uint 4 >        ; <<4 x float>> [#uses=1]
        store <4 x float> %tmp293, <4 x float>* %G4
        ret void
}

Compiles (llc -march=x86 -mcpu=yonah -relocation-model=static) to:

_test:
        movaps _G2, %xmm0
        movaps _G1, %xmm1
        movaps %xmm1, %xmm2
2)      shufps $3, %xmm0, %xmm2
        movaps %xmm1, %xmm3
2)      shufps $1, %xmm0, %xmm3
1)      unpcklps %xmm0, %xmm1
2)      shufps $128, %xmm2, %xmm3
1)      movaps %xmm1, _G3
        movaps %xmm3, _G4
        ret

The 1) marked instructions could be scheduled better for reduced register 
pressure.  The scheduling issue is more pronounced without -static.

The 2) marked instructions are the lowered form of the 1,undef,3,4 
shufflevector.  It seems that there should be a better way to do it :)


