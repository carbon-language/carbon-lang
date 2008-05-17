Target Independent Opportunities:

//===---------------------------------------------------------------------===//

We should make the various target's "IMPLICIT_DEF" instructions be a single
target-independent opcode like TargetInstrInfo::INLINEASM.  This would allow
us to eliminate the TargetInstrDesc::isImplicitDef() method, and would allow
us to avoid having to define this for every target for every register class.

//===---------------------------------------------------------------------===//

With the recent changes to make the implicit def/use set explicit in
machineinstrs, we should change the target descriptions for 'call' instructions
so that the .td files don't list all the call-clobbered registers as implicit
defs.  Instead, these should be added by the code generator (e.g. on the dag).

This has a number of uses:

1. PPC32/64 and X86 32/64 can avoid having multiple copies of call instructions
   for their different impdef sets.
2. Targets with multiple calling convs (e.g. x86) which have different clobber
   sets don't need copies of call instructions.
3. 'Interprocedural register allocation' can be done to reduce the clobber sets
   of calls.

//===---------------------------------------------------------------------===//

Make the PPC branch selector target independant

//===---------------------------------------------------------------------===//

Get the C front-end to expand hypot(x,y) -> llvm.sqrt(x*x+y*y) when errno and
precision don't matter (ffastmath).  Misc/mandel will like this. :)

//===---------------------------------------------------------------------===//

Solve this DAG isel folding deficiency:

int X, Y;

void fn1(void)
{
  X = X | (Y << 3);
}

compiles to

fn1:
	movl Y, %eax
	shll $3, %eax
	orl X, %eax
	movl %eax, X
	ret

The problem is the store's chain operand is not the load X but rather
a TokenFactor of the load X and load Y, which prevents the folding.

There are two ways to fix this:

1. The dag combiner can start using alias analysis to realize that y/x
   don't alias, making the store to X not dependent on the load from Y.
2. The generated isel could be made smarter in the case it can't
   disambiguate the pointers.

Number 1 is the preferred solution.

This has been "fixed" by a TableGen hack. But that is a short term workaround
which will be removed once the proper fix is made.

//===---------------------------------------------------------------------===//

On targets with expensive 64-bit multiply, we could LSR this:

for (i = ...; ++i) {
   x = 1ULL << i;

into:
 long long tmp = 1;
 for (i = ...; ++i, tmp+=tmp)
   x = tmp;

This would be a win on ppc32, but not x86 or ppc64.

//===---------------------------------------------------------------------===//

Shrink: (setlt (loadi32 P), 0) -> (setlt (loadi8 Phi), 0)

//===---------------------------------------------------------------------===//

Reassociate should turn: X*X*X*X -> t=(X*X) (t*t) to eliminate a multiply.

//===---------------------------------------------------------------------===//

Interesting? testcase for add/shift/mul reassoc:

int bar(int x, int y) {
  return x*x*x+y+x*x*x*x*x*y*y*y*y;
}
int foo(int z, int n) {
  return bar(z, n) + bar(2*z, 2*n);
}

Reassociate should handle the example in GCC PR16157.

//===---------------------------------------------------------------------===//

These two functions should generate the same code on big-endian systems:

int g(int *j,int *l)  {  return memcmp(j,l,4);  }
int h(int *j, int *l) {  return *j - *l; }

this could be done in SelectionDAGISel.cpp, along with other special cases,
for 1,2,4,8 bytes.

//===---------------------------------------------------------------------===//

It would be nice to revert this patch:
http://lists.cs.uiuc.edu/pipermail/llvm-commits/Week-of-Mon-20060213/031986.html

And teach the dag combiner enough to simplify the code expanded before 
legalize.  It seems plausible that this knowledge would let it simplify other
stuff too.

//===---------------------------------------------------------------------===//

For vector types, TargetData.cpp::getTypeInfo() returns alignment that is equal
to the type size. It works but can be overly conservative as the alignment of
specific vector types are target dependent.

//===---------------------------------------------------------------------===//

We should add 'unaligned load/store' nodes, and produce them from code like
this:

v4sf example(float *P) {
  return (v4sf){P[0], P[1], P[2], P[3] };
}

//===---------------------------------------------------------------------===//

Add support for conditional increments, and other related patterns.  Instead
of:

	movl 136(%esp), %eax
	cmpl $0, %eax
	je LBB16_2	#cond_next
LBB16_1:	#cond_true
	incl _foo
LBB16_2:	#cond_next

emit:
	movl	_foo, %eax
	cmpl	$1, %edi
	sbbl	$-1, %eax
	movl	%eax, _foo

//===---------------------------------------------------------------------===//

Combine: a = sin(x), b = cos(x) into a,b = sincos(x).

Expand these to calls of sin/cos and stores:
      double sincos(double x, double *sin, double *cos);
      float sincosf(float x, float *sin, float *cos);
      long double sincosl(long double x, long double *sin, long double *cos);

Doing so could allow SROA of the destination pointers.  See also:
http://gcc.gnu.org/bugzilla/show_bug.cgi?id=17687

//===---------------------------------------------------------------------===//

Scalar Repl cannot currently promote this testcase to 'ret long cst':

        %struct.X = type { i32, i32 }
        %struct.Y = type { %struct.X }

define i64 @bar() {
        %retval = alloca %struct.Y, align 8
        %tmp12 = getelementptr %struct.Y* %retval, i32 0, i32 0, i32 0
        store i32 0, i32* %tmp12
        %tmp15 = getelementptr %struct.Y* %retval, i32 0, i32 0, i32 1
        store i32 1, i32* %tmp15
        %retval.upgrd.1 = bitcast %struct.Y* %retval to i64*
        %retval.upgrd.2 = load i64* %retval.upgrd.1
        ret i64 %retval.upgrd.2
}

it should be extended to do so.

//===---------------------------------------------------------------------===//

-scalarrepl should promote this to be a vector scalar.

        %struct..0anon = type { <4 x float> }

define void @test1(<4 x float> %V, float* %P) {
        %u = alloca %struct..0anon, align 16
        %tmp = getelementptr %struct..0anon* %u, i32 0, i32 0
        store <4 x float> %V, <4 x float>* %tmp
        %tmp1 = bitcast %struct..0anon* %u to [4 x float]*
        %tmp.upgrd.1 = getelementptr [4 x float]* %tmp1, i32 0, i32 1
        %tmp.upgrd.2 = load float* %tmp.upgrd.1
        %tmp3 = mul float %tmp.upgrd.2, 2.000000e+00
        store float %tmp3, float* %P
        ret void
}

//===---------------------------------------------------------------------===//

Turn this into a single byte store with no load (the other 3 bytes are
unmodified):

void %test(uint* %P) {
	%tmp = load uint* %P
        %tmp14 = or uint %tmp, 3305111552
        %tmp15 = and uint %tmp14, 3321888767
        store uint %tmp15, uint* %P
        ret void
}

//===---------------------------------------------------------------------===//

dag/inst combine "clz(x)>>5 -> x==0" for 32-bit x.

Compile:

int bar(int x)
{
  int t = __builtin_clz(x);
  return -(t>>5);
}

to:

_bar:   addic r3,r3,-1
        subfe r3,r3,r3
        blr

//===---------------------------------------------------------------------===//

Legalize should lower ctlz like this:
  ctlz(x) = popcnt((x-1) & ~x)

on targets that have popcnt but not ctlz.  itanium, what else?

//===---------------------------------------------------------------------===//

quantum_sigma_x in 462.libquantum contains the following loop:

      for(i=0; i<reg->size; i++)
	{
	  /* Flip the target bit of each basis state */
	  reg->node[i].state ^= ((MAX_UNSIGNED) 1 << target);
	} 

Where MAX_UNSIGNED/state is a 64-bit int.  On a 32-bit platform it would be just
so cool to turn it into something like:

   long long Res = ((MAX_UNSIGNED) 1 << target);
   if (target < 32) {
     for(i=0; i<reg->size; i++)
       reg->node[i].state ^= Res & 0xFFFFFFFFULL;
   } else {
     for(i=0; i<reg->size; i++)
       reg->node[i].state ^= Res & 0xFFFFFFFF00000000ULL
   }
   
... which would only do one 32-bit XOR per loop iteration instead of two.

It would also be nice to recognize the reg->size doesn't alias reg->node[i], but
alas...

//===---------------------------------------------------------------------===//

This isn't recognized as bswap by instcombine:

unsigned int swap_32(unsigned int v) {
  v = ((v & 0x00ff00ffU) << 8)  | ((v & 0xff00ff00U) >> 8);
  v = ((v & 0x0000ffffU) << 16) | ((v & 0xffff0000U) >> 16);
  return v;
}

Nor is this (yes, it really is bswap):

unsigned long reverse(unsigned v) {
    unsigned t;
    t = v ^ ((v << 16) | (v >> 16));
    t &= ~0xff0000;
    v = (v << 24) | (v >> 8);
    return v ^ (t >> 8);
}

//===---------------------------------------------------------------------===//

These should turn into single 16-bit (unaligned?) loads on little/big endian
processors.

unsigned short read_16_le(const unsigned char *adr) {
  return adr[0] | (adr[1] << 8);
}
unsigned short read_16_be(const unsigned char *adr) {
  return (adr[0] << 8) | adr[1];
}

//===---------------------------------------------------------------------===//

-instcombine should handle this transform:
   icmp pred (sdiv X / C1 ), C2
when X, C1, and C2 are unsigned.  Similarly for udiv and signed operands. 

Currently InstCombine avoids this transform but will do it when the signs of
the operands and the sign of the divide match. See the FIXME in 
InstructionCombining.cpp in the visitSetCondInst method after the switch case 
for Instruction::UDiv (around line 4447) for more details.

The SingleSource/Benchmarks/Shootout-C++/hash and hash2 tests have examples of
this construct. 

//===---------------------------------------------------------------------===//

Instcombine misses several of these cases (see the testcase in the patch):
http://gcc.gnu.org/ml/gcc-patches/2006-10/msg01519.html

//===---------------------------------------------------------------------===//

viterbi speeds up *significantly* if the various "history" related copy loops
are turned into memcpy calls at the source level.  We need a "loops to memcpy"
pass.

//===---------------------------------------------------------------------===//

Consider:

typedef unsigned U32;
typedef unsigned long long U64;
int test (U32 *inst, U64 *regs) {
    U64 effective_addr2;
    U32 temp = *inst;
    int r1 = (temp >> 20) & 0xf;
    int b2 = (temp >> 16) & 0xf;
    effective_addr2 = temp & 0xfff;
    if (b2) effective_addr2 += regs[b2];
    b2 = (temp >> 12) & 0xf;
    if (b2) effective_addr2 += regs[b2];
    effective_addr2 &= regs[4];
     if ((effective_addr2 & 3) == 0)
        return 1;
    return 0;
}

Note that only the low 2 bits of effective_addr2 are used.  On 32-bit systems,
we don't eliminate the computation of the top half of effective_addr2 because
we don't have whole-function selection dags.  On x86, this means we use one
extra register for the function when effective_addr2 is declared as U64 than
when it is declared U32.

//===---------------------------------------------------------------------===//

Promote for i32 bswap can use i64 bswap + shr.  Useful on targets with 64-bit
regs and bswap, like itanium.

//===---------------------------------------------------------------------===//

LSR should know what GPR types a target has.  This code:

volatile short X, Y; // globals

void foo(int N) {
  int i;
  for (i = 0; i < N; i++) { X = i; Y = i*4; }
}

produces two identical IV's (after promotion) on PPC/ARM:

LBB1_1: @bb.preheader
        mov r3, #0
        mov r2, r3
        mov r1, r3
LBB1_2: @bb
        ldr r12, LCPI1_0
        ldr r12, [r12]
        strh r2, [r12]
        ldr r12, LCPI1_1
        ldr r12, [r12]
        strh r3, [r12]
        add r1, r1, #1    <- [0,+,1]
        add r3, r3, #4
        add r2, r2, #1    <- [0,+,1]
        cmp r1, r0
        bne LBB1_2      @bb


//===---------------------------------------------------------------------===//

Tail call elim should be more aggressive, checking to see if the call is
followed by an uncond branch to an exit block.

; This testcase is due to tail-duplication not wanting to copy the return
; instruction into the terminating blocks because there was other code
; optimized out of the function after the taildup happened.
; RUN: llvm-as < %s | opt -tailcallelim | llvm-dis | not grep call

define i32 @t4(i32 %a) {
entry:
	%tmp.1 = and i32 %a, 1		; <i32> [#uses=1]
	%tmp.2 = icmp ne i32 %tmp.1, 0		; <i1> [#uses=1]
	br i1 %tmp.2, label %then.0, label %else.0

then.0:		; preds = %entry
	%tmp.5 = add i32 %a, -1		; <i32> [#uses=1]
	%tmp.3 = call i32 @t4( i32 %tmp.5 )		; <i32> [#uses=1]
	br label %return

else.0:		; preds = %entry
	%tmp.7 = icmp ne i32 %a, 0		; <i1> [#uses=1]
	br i1 %tmp.7, label %then.1, label %return

then.1:		; preds = %else.0
	%tmp.11 = add i32 %a, -2		; <i32> [#uses=1]
	%tmp.9 = call i32 @t4( i32 %tmp.11 )		; <i32> [#uses=1]
	br label %return

return:		; preds = %then.1, %else.0, %then.0
	%result.0 = phi i32 [ 0, %else.0 ], [ %tmp.3, %then.0 ],
                            [ %tmp.9, %then.1 ]
	ret i32 %result.0
}

//===---------------------------------------------------------------------===//

Tail recursion elimination is not transforming this function, because it is
returning n, which fails the isDynamicConstant check in the accumulator 
recursion checks.

long long fib(const long long n) {
  switch(n) {
    case 0:
    case 1:
      return n;
    default:
      return fib(n-1) + fib(n-2);
  }
}

//===---------------------------------------------------------------------===//

Argument promotion should promote arguments for recursive functions, like 
this:

; RUN: llvm-as < %s | opt -argpromotion | llvm-dis | grep x.val

define internal i32 @foo(i32* %x) {
entry:
	%tmp = load i32* %x		; <i32> [#uses=0]
	%tmp.foo = call i32 @foo( i32* %x )		; <i32> [#uses=1]
	ret i32 %tmp.foo
}

define i32 @bar(i32* %x) {
entry:
	%tmp3 = call i32 @foo( i32* %x )		; <i32> [#uses=1]
	ret i32 %tmp3
}

//===---------------------------------------------------------------------===//

"basicaa" should know how to look through "or" instructions that act like add
instructions.  For example in this code, the x*4+1 is turned into x*4 | 1, and
basicaa can't analyze the array subscript, leading to duplicated loads in the
generated code:

void test(int X, int Y, int a[]) {
int i;
  for (i=2; i<1000; i+=4) {
  a[i+0] = a[i-1+0]*a[i-2+0];
  a[i+1] = a[i-1+1]*a[i-2+1];
  a[i+2] = a[i-1+2]*a[i-2+2];
  a[i+3] = a[i-1+3]*a[i-2+3];
  }
}

//===---------------------------------------------------------------------===//

We should investigate an instruction sinking pass.  Consider this silly
example in pic mode:

#include <assert.h>
void foo(int x) {
  assert(x);
  //...
}

we compile this to:
_foo:
	subl	$28, %esp
	call	"L1$pb"
"L1$pb":
	popl	%eax
	cmpl	$0, 32(%esp)
	je	LBB1_2	# cond_true
LBB1_1:	# return
	# ...
	addl	$28, %esp
	ret
LBB1_2:	# cond_true
...

The PIC base computation (call+popl) is only used on one path through the 
code, but is currently always computed in the entry block.  It would be 
better to sink the picbase computation down into the block for the 
assertion, as it is the only one that uses it.  This happens for a lot of 
code with early outs.

Another example is loads of arguments, which are usually emitted into the 
entry block on targets like x86.  If not used in all paths through a 
function, they should be sunk into the ones that do.

In this case, whole-function-isel would also handle this.

//===---------------------------------------------------------------------===//

Investigate lowering of sparse switch statements into perfect hash tables:
http://burtleburtle.net/bob/hash/perfect.html

//===---------------------------------------------------------------------===//

We should turn things like "load+fabs+store" and "load+fneg+store" into the
corresponding integer operations.  On a yonah, this loop:

double a[256];
void foo() {
  int i, b;
  for (b = 0; b < 10000000; b++)
  for (i = 0; i < 256; i++)
    a[i] = -a[i];
}

is twice as slow as this loop:

long long a[256];
void foo() {
  int i, b;
  for (b = 0; b < 10000000; b++)
  for (i = 0; i < 256; i++)
    a[i] ^= (1ULL << 63);
}

and I suspect other processors are similar.  On X86 in particular this is a
big win because doing this with integers allows the use of read/modify/write
instructions.

//===---------------------------------------------------------------------===//

DAG Combiner should try to combine small loads into larger loads when 
profitable.  For example, we compile this C++ example:

struct THotKey { short Key; bool Control; bool Shift; bool Alt; };
extern THotKey m_HotKey;
THotKey GetHotKey () { return m_HotKey; }

into (-O3 -fno-exceptions -static -fomit-frame-pointer):

__Z9GetHotKeyv:
	pushl	%esi
	movl	8(%esp), %eax
	movb	_m_HotKey+3, %cl
	movb	_m_HotKey+4, %dl
	movb	_m_HotKey+2, %ch
	movw	_m_HotKey, %si
	movw	%si, (%eax)
	movb	%ch, 2(%eax)
	movb	%cl, 3(%eax)
	movb	%dl, 4(%eax)
	popl	%esi
	ret	$4

GCC produces:

__Z9GetHotKeyv:
	movl	_m_HotKey, %edx
	movl	4(%esp), %eax
	movl	%edx, (%eax)
	movzwl	_m_HotKey+4, %edx
	movw	%dx, 4(%eax)
	ret	$4

The LLVM IR contains the needed alignment info, so we should be able to 
merge the loads and stores into 4-byte loads:

	%struct.THotKey = type { i16, i8, i8, i8 }
define void @_Z9GetHotKeyv(%struct.THotKey* sret  %agg.result) nounwind  {
...
	%tmp2 = load i16* getelementptr (@m_HotKey, i32 0, i32 0), align 8
	%tmp5 = load i8* getelementptr (@m_HotKey, i32 0, i32 1), align 2
	%tmp8 = load i8* getelementptr (@m_HotKey, i32 0, i32 2), align 1
	%tmp11 = load i8* getelementptr (@m_HotKey, i32 0, i32 3), align 2

Alternatively, we should use a small amount of base-offset alias analysis
to make it so the scheduler doesn't need to hold all the loads in regs at
once.

//===---------------------------------------------------------------------===//

We should extend parameter attributes to capture more information about
pointer parameters for alias analysis.  Some ideas:

1. Add a "nocapture" attribute, which indicates that the callee does not store
   the address of the parameter into a global or any other memory location
   visible to the callee.  This can be used to make basicaa and other analyses
   more powerful.  It is true for things like memcpy, strcat, and many other
   things, including structs passed by value, most C++ references, etc.
2. Generalize readonly to be set on parameters.  This is important mod/ref 
   info for the function, which is important for basicaa and others.  It can
   also be used by the inliner to avoid inserting a memcpy for byval 
   arguments when the function is inlined.

These functions can be inferred by various analysis passes such as the 
globalsmodrefaa pass.  Note that getting #2 right is actually really tricky.
Consider this code:

struct S;  S G;
void caller(S byvalarg) { G.field = 1; ... }
void callee() { caller(G); }

The fact that the caller does not modify byval arg is not enough, we need
to know that it doesn't modify G either.  This is very tricky.

//===---------------------------------------------------------------------===//

We should add an FRINT node to the DAG to model targets that have legal
implementations of ceil/floor/rint.

//===---------------------------------------------------------------------===//

This GCC bug: http://gcc.gnu.org/bugzilla/show_bug.cgi?id=34043
contains a testcase that compiles down to:

	%struct.XMM128 = type { <4 x float> }
..
	%src = alloca %struct.XMM128
..
	%tmp6263 = bitcast %struct.XMM128* %src to <2 x i64>*
	%tmp65 = getelementptr %struct.XMM128* %src, i32 0, i32 0
	store <2 x i64> %tmp5899, <2 x i64>* %tmp6263, align 16
	%tmp66 = load <4 x float>* %tmp65, align 16		
	%tmp71 = add <4 x float> %tmp66, %tmp66		

If the mid-level optimizer turned the bitcast of pointer + store of tmp5899
into a bitcast of the vector value and a store to the pointer, then the 
store->load could be easily removed.

//===---------------------------------------------------------------------===//

Consider:

int test() {
  long long input[8] = {1,1,1,1,1,1,1,1};
  foo(input);
}

We currently compile this into a memcpy from a global array since the 
initializer is fairly large and not memset'able.  This is good, but the memcpy
gets lowered to load/stores in the code generator.  This is also ok, except
that the codegen lowering for memcpy doesn't handle the case when the source
is a constant global.  This gives us atrocious code like this:

	call	"L1$pb"
"L1$pb":
	popl	%eax
	movl	_C.0.1444-"L1$pb"+32(%eax), %ecx
	movl	%ecx, 40(%esp)
	movl	_C.0.1444-"L1$pb"+20(%eax), %ecx
	movl	%ecx, 28(%esp)
	movl	_C.0.1444-"L1$pb"+36(%eax), %ecx
	movl	%ecx, 44(%esp)
	movl	_C.0.1444-"L1$pb"+44(%eax), %ecx
	movl	%ecx, 52(%esp)
	movl	_C.0.1444-"L1$pb"+40(%eax), %ecx
	movl	%ecx, 48(%esp)
	movl	_C.0.1444-"L1$pb"+12(%eax), %ecx
	movl	%ecx, 20(%esp)
	movl	_C.0.1444-"L1$pb"+4(%eax), %ecx
...

instead of:
	movl	$1, 16(%esp)
	movl	$0, 20(%esp)
	movl	$1, 24(%esp)
	movl	$0, 28(%esp)
	movl	$1, 32(%esp)
	movl	$0, 36(%esp)
	...

//===---------------------------------------------------------------------===//

http://llvm.org/PR717:

The following code should compile into "ret int undef". Instead, LLVM
produces "ret int 0":

int f() {
  int x = 4;
  int y;
  if (x == 3) y = 0;
  return y;
}

//===---------------------------------------------------------------------===//

The loop unroller should partially unroll loops (instead of peeling them)
when code growth isn't too bad and when an unroll count allows simplification
of some code within the loop.  One trivial example is:

#include <stdio.h>
int main() {
    int nRet = 17;
    int nLoop;
    for ( nLoop = 0; nLoop < 1000; nLoop++ ) {
        if ( nLoop & 1 )
            nRet += 2;
        else
            nRet -= 1;
    }
    return nRet;
}

Unrolling by 2 would eliminate the '&1' in both copies, leading to a net
reduction in code size.  The resultant code would then also be suitable for
exit value computation.

//===---------------------------------------------------------------------===//

We miss a bunch of rotate opportunities on various targets, including ppc, x86,
etc.  On X86, we miss a bunch of 'rotate by variable' cases because the rotate
matching code in dag combine doesn't look through truncates aggressively 
enough.  Here are some testcases reduces from GCC PR17886:

unsigned long long f(unsigned long long x, int y) {
  return (x << y) | (x >> 64-y); 
} 
unsigned f2(unsigned x, int y){
  return (x << y) | (x >> 32-y); 
} 
unsigned long long f3(unsigned long long x){
  int y = 9;
  return (x << y) | (x >> 64-y); 
} 
unsigned f4(unsigned x){
  int y = 10;
  return (x << y) | (x >> 32-y); 
}
unsigned long long f5(unsigned long long x, unsigned long long y) {
  return (x << 8) | ((y >> 48) & 0xffull);
}
unsigned long long f6(unsigned long long x, unsigned long long y, int z) {
  switch(z) {
  case 1:
    return (x << 8) | ((y >> 48) & 0xffull);
  case 2:
    return (x << 16) | ((y >> 40) & 0xffffull);
  case 3:
    return (x << 24) | ((y >> 32) & 0xffffffull);
  case 4:
    return (x << 32) | ((y >> 24) & 0xffffffffull);
  default:
    return (x << 40) | ((y >> 16) & 0xffffffffffull);
  }
}

On X86-64, we only handle f3/f4 right.  On x86-32, several of these 
generate truly horrible code, instead of using shld and friends.  On
ARM, we end up with calls to L___lshrdi3/L___ashldi3 in f, which is
badness.  PPC64 misses f, f5 and f6.  CellSPU aborts in isel.

//===---------------------------------------------------------------------===//

We do a number of simplifications in simplify libcalls to strength reduce
standard library functions, but we don't currently merge them together.  For
example, it is useful to merge memcpy(a,b,strlen(b)) -> strcpy.  This can only
be done safely if "b" isn't modified between the strlen and memcpy of course.

//===---------------------------------------------------------------------===//

define i32 @test2(float %X, float %Y) {
entry:
        %tmp3 = fcmp uno float %X, %Y           ; <i1> [#uses=1]
        %tmp34 = zext i1 %tmp3 to i8            ; <i8> [#uses=1]
        %tmp = xor i8 %tmp34, 1         ; <i8> [#uses=1]
        %toBoolnot5 = zext i8 %tmp to i32               ; <i32> [#uses=1]
        ret i32 %toBoolnot5
}

could be optimized further. Instcombine should use its bitwise analysis to
collapse the zext/xor/zext structure to an xor/zext and then remove the 
xor by reversing the fcmp.

Desired output:

define i32 @test2(float %X, float %Y) {
entry:
        %tmp3 = fcmp ord float %X, %Y           ; <i1> [#uses=1]
        %tmp34 = zext i1 %tmp3 to i32           ; <i32> [#uses=1]
        ret i32 %tmp34
}

To fix this, we need to make CanEvaluateInDifferentType smarter.

//===---------------------------------------------------------------------===//

We should be able to evaluate this loop:

int test(int x_offs) {
  while (x_offs > 4)
     x_offs -= 4;
  return x_offs;
}

//===---------------------------------------------------------------------===//
