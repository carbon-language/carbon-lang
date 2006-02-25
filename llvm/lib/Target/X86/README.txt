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
