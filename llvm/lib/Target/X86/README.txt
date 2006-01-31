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

Should support emission of the bswap instruction, probably by adding a new
DAG node for byte swapping.  Also useful on PPC which has byte-swapping loads.

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

//===---------------------------------------------------------------------===//

Should we promote i16 to i32 to avoid partial register update stalls?

//===---------------------------------------------------------------------===//

Leave any_extend as pseudo instruction and hint to register
allocator. Delay codegen until post register allocation.

//===---------------------------------------------------------------------===//

Add a target specific hook to DAG combiner to handle SINT_TO_FP and
FP_TO_SINT when the source operand is already in memory.

//===---------------------------------------------------------------------===//

Check if load folding would add a cycle in the dag.

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

//===---------------------------------------------------------------------===//

For all targets, not just X86:
When llvm.memcpy, llvm.memset, or llvm.memmove are lowered, they should be 
optimized to a few store instructions if the source is constant and the length
is smallish (< 8). This will greatly help some tests like Shootout/strcat.c

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

The instruction selector sometimes misses folding a load into a compare.  The
pattern is written as (cmp reg, (load p)).  Because the compare isn't 
commutative, it is not matched with the load on both sides.  The dag combiner
should be made smart enough to cannonicalize the load into the RHS of a compare
when it can invert the result of the compare for free.

//===---------------------------------------------------------------------===//

The code generated for 'abs' is truly aweful:

float %foo(float %tmp.38) {
       %tmp.39 = setgt float %tmp.38, 0.000000e+00
        %tmp.45 = sub float -0.000000e+00, %tmp.38
        %mem_tmp.0.0 = select bool %tmp.39, float %tmp.38, float %tmp.45
        ret float %mem_tmp.0.0
}

_foo:
        subl $4, %esp
        movss LCPI1_0, %xmm0
        movss 8(%esp), %xmm1
        subss %xmm1, %xmm0
        xorps %xmm2, %xmm2
        ucomiss %xmm2, %xmm1
        setp %al
        seta %cl
        orb %cl, %al
        testb %al, %al
        jne LBB_foo_2   # 
LBB_foo_1:      # 
        movss %xmm0, %xmm1
LBB_foo_2:      # 
        movss %xmm1, (%esp)
        flds (%esp)
        addl $4, %esp
        ret

This should be a high-priority to fix.  With the fp-stack, this is a single
instruction.  With SSE it could be far better than this.  Why is the sequence
above using 'setp'?  It shouldn't care about nan's.

//===---------------------------------------------------------------------===//

Is there a better way to implement Y = -X (fneg) than the literal code:

float %test(float %X) {
        %Y = sub float -0.0, %X
        ret float %Y
}

        movss LCPI1_0, %xmm0   ;; load -0.0
        subss 8(%esp), %xmm0   ;; subtract

//===---------------------------------------------------------------------===//

None of the SSE instructions are handled in X86RegisterInfo::foldMemoryOperand,
which prevents the spiller from folding spill code into the instructions.

This leads to code like this:

mov %eax, 8(%esp)
cvtsi2sd %eax, %xmm0
instead of:
cvtsi2sd 8(%esp), %xmm0

//===---------------------------------------------------------------------===//

This instruction selector selects 'int X = 0' as 'mov Reg, 0' not 'xor Reg,Reg'
This is bigger and slower.

//===---------------------------------------------------------------------===//

LSR should be turned on for the X86 backend and tuned to take advantage of its
addressing modes.

//===---------------------------------------------------------------------===//

When compiled with unsafemath enabled, "main" should enable SSE DAZ mode and
other fast SSE modes.

//===---------------------------------------------------------------------===//

cd Regression/CodeGen/X86
llvm-as < setuge.ll | llc -march=x86 -mcpu=yonah -enable-x86-sse

_cmp:
        subl $4, %esp
1)      leal 20(%esp), %eax
        movss 12(%esp), %xmm0
1)      leal 16(%esp), %ecx
        ucomiss 8(%esp), %xmm0
        cmovb %ecx, %eax
2)      movss (%eax), %xmm0
2)      movss %xmm0, (%esp)
        flds (%esp)
        addl $4, %esp
        ret


1) These LEA's should be adds.  This is tricky because they are FrameIndex's
   before prolog-epilog rewriting.
2) We shouldn't load into XMM regs only to store it back.

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


