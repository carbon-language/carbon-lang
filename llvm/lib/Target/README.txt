Target Independent Opportunities:

===-------------------------------------------------------------------------===

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

Turn this into a signed shift right in instcombine:

int f(unsigned x) {
  return x >> 31 ? -1 : 0;
}

http://gcc.gnu.org/bugzilla/show_bug.cgi?id=25600
http://gcc.gnu.org/ml/gcc-patches/2006-02/msg01492.html

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
        %tmp12 = getelementptr %struct.Y* %retval, int 0, uint 0, uint 0                ; <int*> [#uses=1]
        store int 0, int* %tmp12
        %tmp15 = getelementptr %struct.Y* %retval, int 0, uint 0, uint 1                ; <int*> [#uses=1]
        store int 1, int* %tmp15
        %retval = cast %struct.Y* %retval to ulong*             ; <ulong*> [#uses=1]
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


