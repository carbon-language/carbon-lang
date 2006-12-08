Target Independent Opportunities:

//===---------------------------------------------------------------------===//

We should make the following changes to clean up MachineInstr:

1. Add an Opcode field to TargetInstrDescriptor, so you can tell the opcode of
   an instruction with just a TargetInstrDescriptor*.
2. Remove the Opcode field from MachineInstr, replacing it with a
   TargetInstrDescriptor*.
3. Getting information about a machine instr then becomes:
     MI->getInfo()->isTwoAddress()
   instead of:
     const TargetInstrInfo &TII = ...
     TII.isTwoAddrInstr(MI->getOpcode())

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

FreeBench/mason contains code like this:

static p_type m0u(p_type p) {
  int m[]={0, 8, 1, 2, 16, 5, 13, 7, 14, 9, 3, 4, 11, 12, 15, 10, 17, 6};
  p_type pu;
  pu.a = m[p.a];
  pu.b = m[p.b];
  pu.c = m[p.c];
  return pu;
}

We currently compile this into a memcpy from a static array into 'm', then
a bunch of loads from m.  It would be better to avoid the memcpy and just do
loads from the static array.

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

//===---------------------------------------------------------------------===//

These two functions should generate the same code on big-endian systems:

int g(int *j,int *l)  {  return memcmp(j,l,4);  }
int h(int *j, int *l) {  return *j - *l; }

this could be done in SelectionDAGISel.cpp, along with other special cases,
for 1,2,4,8 bytes.

//===---------------------------------------------------------------------===//

This code:
int rot(unsigned char b) { int a = ((b>>1) ^ (b<<7)) & 0xff; return a; }

Can be improved in two ways:

1. The instcombiner should eliminate the type conversions.
2. The X86 backend should turn this into a rotate by one bit.

//===---------------------------------------------------------------------===//

Add LSR exit value substitution. It'll probably be a win for Ackermann, etc.

//===---------------------------------------------------------------------===//

It would be nice to revert this patch:
http://lists.cs.uiuc.edu/pipermail/llvm-commits/Week-of-Mon-20060213/031986.html

And teach the dag combiner enough to simplify the code expanded before 
legalize.  It seems plausible that this knowledge would let it simplify other
stuff too.

//===---------------------------------------------------------------------===//

For packed types, TargetData.cpp::getTypeInfo() returns alignment that is equal
to the type size. It works but can be overly conservative as the alignment of
specific packed types are target dependent.

//===---------------------------------------------------------------------===//

We should add 'unaligned load/store' nodes, and produce them from code like
this:

v4sf example(float *P) {
  return (v4sf){P[0], P[1], P[2], P[3] };
}

//===---------------------------------------------------------------------===//

We should constant fold packed type casts at the LLVM level, regardless of the
cast.  Currently we cannot fold some casts because we don't have TargetData
information in the constant folder, so we don't know the endianness of the 
target!

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

        %struct.X = type { int, int }
        %struct.Y = type { %struct.X }
ulong %bar() {
        %retval = alloca %struct.Y, align 8             ; <%struct.Y*> [#uses=3]
        %tmp12 = getelementptr %struct.Y* %retval, int 0, uint 0, uint 0
        store int 0, int* %tmp12
        %tmp15 = getelementptr %struct.Y* %retval, int 0, uint 0, uint 1
        store int 1, int* %tmp15
        %retval = cast %struct.Y* %retval to ulong*
        %retval = load ulong* %retval           ; <ulong> [#uses=1]
        ret ulong %retval
}

it should be extended to do so.

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

-scalarrepl should promote this to be a vector scalar.

        %struct..0anon = type { <4 x float> }
implementation   ; Functions:
void %test1(<4 x float> %V, float* %P) {
entry:
        %u = alloca %struct..0anon, align 16            ; <%struct..0anon*> [#uses=2]
        %tmp = getelementptr %struct..0anon* %u, int 0, uint 0          ; <<4 x float>*> [#uses=1]
        store <4 x float> %V, <4 x float>* %tmp
        %tmp1 = cast %struct..0anon* %u to [4 x float]*         ; <[4 x float]*> [#uses=1]
        %tmp = getelementptr [4 x float]* %tmp1, int 0, int 1           ; <float*> [#uses=1]
        %tmp = load float* %tmp         ; <float> [#uses=1]
        %tmp3 = mul float %tmp, 2.000000e+00            ; <float> [#uses=1]
        store float %tmp3, float* %P
        ret void
}

//===---------------------------------------------------------------------===//

-instcombine should handle this transform:
   setcc (sdiv X / C1 ), C2
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

-predsimplify should transform this:

void bad(unsigned x)
{
  if (x > 4)
    bar(12);
  else if (x > 3)
    bar(523);
  else if (x > 2)
    bar(36);
  else if (x > 1)
    bar(65);
  else if (x > 0)
    bar(45);
  else
    bar(367);
}

into:

void good(unsigned x)
{
  if (x == 4)
    bar(523);
  else if (x == 3)
    bar(36);
  else if (x == 2)
    bar(65);
  else if (x == 1)
    bar(45);
  else if (x == 0)
    bar(367);
  else
    bar(12);
}

to enable further optimizations.

//===---------------------------------------------------------------------===//
