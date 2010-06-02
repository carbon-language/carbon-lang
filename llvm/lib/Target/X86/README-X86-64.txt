//===- README_X86_64.txt - Notes for X86-64 code gen ----------------------===//

AMD64 Optimization Manual 8.2 has some nice information about optimizing integer
multiplication by a constant. How much of it applies to Intel's X86-64
implementation? There are definite trade-offs to consider: latency vs. register
pressure vs. code size.

//===---------------------------------------------------------------------===//

Are we better off using branches instead of cmove to implement FP to
unsigned i64?

_conv:
	ucomiss	LC0(%rip), %xmm0
	cvttss2siq	%xmm0, %rdx
	jb	L3
	subss	LC0(%rip), %xmm0
	movabsq	$-9223372036854775808, %rax
	cvttss2siq	%xmm0, %rdx
	xorq	%rax, %rdx
L3:
	movq	%rdx, %rax
	ret

instead of

_conv:
	movss LCPI1_0(%rip), %xmm1
	cvttss2siq %xmm0, %rcx
	movaps %xmm0, %xmm2
	subss %xmm1, %xmm2
	cvttss2siq %xmm2, %rax
	movabsq $-9223372036854775808, %rdx
	xorq %rdx, %rax
	ucomiss %xmm1, %xmm0
	cmovb %rcx, %rax
	ret

Seems like the jb branch has high likelyhood of being taken. It would have
saved a few instructions.

//===---------------------------------------------------------------------===//

Poor codegen:

int X[2];
int b;
void test(void) {
  memset(X, b, 2*sizeof(X[0]));
}

llc:
	movq _b@GOTPCREL(%rip), %rax
	movzbq (%rax), %rax
	movq %rax, %rcx
	shlq $8, %rcx
	orq %rax, %rcx
	movq %rcx, %rax
	shlq $16, %rax
	orq %rcx, %rax
	movq %rax, %rcx
	shlq $32, %rcx
	movq _X@GOTPCREL(%rip), %rdx
	orq %rax, %rcx
	movq %rcx, (%rdx)
	ret

gcc:
	movq	_b@GOTPCREL(%rip), %rax
	movabsq	$72340172838076673, %rdx
	movzbq	(%rax), %rax
	imulq	%rdx, %rax
	movq	_X@GOTPCREL(%rip), %rdx
	movq	%rax, (%rdx)
	ret

//===---------------------------------------------------------------------===//

It's not possible to reference AH, BH, CH, and DH registers in an instruction
requiring REX prefix. However, divb and mulb both produce results in AH. If isel
emits a CopyFromReg which gets turned into a movb and that can be allocated a
r8b - r15b.

To get around this, isel emits a CopyFromReg from AX and then right shift it
down by 8 and truncate it. It's not pretty but it works. We need some register
allocation magic to make the hack go away (e.g. putting additional constraints
on the result of the movb).

//===---------------------------------------------------------------------===//

The x86-64 ABI for hidden-argument struct returns requires that the
incoming value of %rdi be copied into %rax by the callee upon return.

The idea is that it saves callers from having to remember this value,
which would often require a callee-saved register. Callees usually
need to keep this value live for most of their body anyway, so it
doesn't add a significant burden on them.

We currently implement this in codegen, however this is suboptimal
because it means that it would be quite awkward to implement the
optimization for callers.

A better implementation would be to relax the LLVM IR rules for sret
arguments to allow a function with an sret argument to have a non-void
return type, and to have the front-end to set up the sret argument value
as the return value of the function. The front-end could more easily
emit uses of the returned struct value to be in terms of the function's
lowered return value, and it would free non-C frontends from a
complication only required by a C-based ABI.

//===---------------------------------------------------------------------===//

We get a redundant zero extension for code like this:

int mask[1000];
int foo(unsigned x) {
 if (x < 10)
   x = x * 45;
 else
   x = x * 78;
 return mask[x];
}

_foo:
LBB1_0:	## entry
	cmpl	$9, %edi
	jbe	LBB1_3	## bb
LBB1_1:	## bb1
	imull	$78, %edi, %eax
LBB1_2:	## bb2
	movl	%eax, %eax                    <----
	movq	_mask@GOTPCREL(%rip), %rcx
	movl	(%rcx,%rax,4), %eax
	ret
LBB1_3:	## bb
	imull	$45, %edi, %eax
	jmp	LBB1_2	## bb2
  
Before regalloc, we have:

        %reg1025<def> = IMUL32rri8 %reg1024, 45, %EFLAGS<imp-def>
        JMP mbb<bb2,0x203afb0>
    Successors according to CFG: 0x203afb0 (#3)

bb1: 0x203af60, LLVM BB @0x1e02310, ID#2:
    Predecessors according to CFG: 0x203aec0 (#0)
        %reg1026<def> = IMUL32rri8 %reg1024, 78, %EFLAGS<imp-def>
    Successors according to CFG: 0x203afb0 (#3)

bb2: 0x203afb0, LLVM BB @0x1e02340, ID#3:
    Predecessors according to CFG: 0x203af10 (#1) 0x203af60 (#2)
        %reg1027<def> = PHI %reg1025, mbb<bb,0x203af10>,
                            %reg1026, mbb<bb1,0x203af60>
        %reg1029<def> = MOVZX64rr32 %reg1027

so we'd have to know that IMUL32rri8 leaves the high word zero extended and to
be able to recognize the zero extend.  This could also presumably be implemented
if we have whole-function selectiondags.

//===---------------------------------------------------------------------===//
