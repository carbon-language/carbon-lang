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

===-------------------------------------------------------------------------===

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

//===---------------------------------------------------------------------===//

DAG combine this into mul A, 8:

int %test(int %A) {
  %B = mul int %A, 8  ;; shift
  %C = add int %B, 7  ;; dead, no demanded bits.
  %D = and int %C, -8 ;; dead once add is gone.
  ret int %D
}

This sort of thing occurs in the alloca lowering code and other places that
are generating alignment of an already aligned value.

//===---------------------------------------------------------------------===//

Turn this into a signed shift right in instcombine:

int f(unsigned x) {
  return x >> 31 ? -1 : 0;
}

http://gcc.gnu.org/bugzilla/show_bug.cgi?id=25600
http://gcc.gnu.org/ml/gcc-patches/2006-02/msg01492.html

//===---------------------------------------------------------------------===//

We should reassociate:
int f(int a, int b){ return a * a + 2 * a * b + b * b; }
into:
int f(int a, int b) { return a * (a + 2 * b) + b * b; }
to eliminate a multiply.

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

Reassociate is missing this:

int test(int X, int Y) {
 return (X+X+Y+Y);  // (X+Y) << 1;
}

it needs to turn the shifts into multiplies to get it.

//===---------------------------------------------------------------------===//

These two functions should generate the same code on big-endian systems:

int g(int *j,int *l)  {  return memcmp(j,l,4);  }
int h(int *j, int *l) {  return *j - *l; }

this could be done in SelectionDAGISel.cpp, along with other special cases,
for 1,2,4,8 bytes.

//===---------------------------------------------------------------------===//

