TODO:
* gpr0 allocation
* implement do-loop -> bdnz transform
* implement powerpc-64 for darwin
* use stfiwx in float->int

* Fold add and sub with constant into non-extern, non-weak addresses so this:
	lis r2, ha16(l2__ZTV4Cell)
	la r2, lo16(l2__ZTV4Cell)(r2)
	addi r2, r2, 8
becomes:
	lis r2, ha16(l2__ZTV4Cell+8)
	la r2, lo16(l2__ZTV4Cell+8)(r2)


* Teach LLVM how to codegen this:
unsigned short foo(float a) { return a; }
as:
_foo:
        fctiwz f0,f1
        stfd f0,-8(r1)
        lhz r3,-2(r1)
        blr
not:
_foo:
        fctiwz f0, f1
        stfd f0, -8(r1)
        lwz r2, -4(r1)
        rlwinm r3, r2, 0, 16, 31
        blr

* Support 'update' load/store instructions.  These are cracked on the G5, but
  are still a codesize win.

* should hint to the branch select pass that it doesn't need to print the
  second unconditional branch, so we don't end up with things like:
	b .LBBl42__2E_expand_function_8_674	; loopentry.24
	b .LBBl42__2E_expand_function_8_42	; NewDefault
	b .LBBl42__2E_expand_function_8_42	; NewDefault

===-------------------------------------------------------------------------===

* Codegen this:

   void test2(int X) {
     if (X == 0x12345678) bar();
   }

    as:

       xoris r0,r3,0x1234
       cmpwi cr0,r0,0x5678
       beq cr0,L6

    not:

        lis r2, 4660
        ori r2, r2, 22136 
        cmpw cr0, r3, r2  
        bne .LBB_test2_2

===-------------------------------------------------------------------------===

Lump the constant pool for each function into ONE pic object, and reference
pieces of it as offsets from the start.  For functions like this (contrived
to have lots of constants obviously):

double X(double Y) { return (Y*1.23 + 4.512)*2.34 + 14.38; }

We generate:

_X:
        lis r2, ha16(.CPI_X_0)
        lfd f0, lo16(.CPI_X_0)(r2)
        lis r2, ha16(.CPI_X_1)
        lfd f2, lo16(.CPI_X_1)(r2)
        fmadd f0, f1, f0, f2
        lis r2, ha16(.CPI_X_2)
        lfd f1, lo16(.CPI_X_2)(r2)
        lis r2, ha16(.CPI_X_3)
        lfd f2, lo16(.CPI_X_3)(r2)
        fmadd f1, f0, f1, f2
        blr

It would be better to materialize .CPI_X into a register, then use immediates
off of the register to avoid the lis's.  This is even more important in PIC 
mode.

Note that this (and the static variable version) is discussed here for GCC:
http://gcc.gnu.org/ml/gcc-patches/2006-02/msg00133.html

===-------------------------------------------------------------------------===

Implement Newton-Rhapson method for improving estimate instructions to the
correct accuracy, and implementing divide as multiply by reciprocal when it has
more than one use.  Itanium will want this too.

===-------------------------------------------------------------------------===

#define  ARRAY_LENGTH  16

union bitfield {
	struct {
#ifndef	__ppc__
		unsigned int                       field0 : 6;
		unsigned int                       field1 : 6;
		unsigned int                       field2 : 6;
		unsigned int                       field3 : 6;
		unsigned int                       field4 : 3;
		unsigned int                       field5 : 4;
		unsigned int                       field6 : 1;
#else
		unsigned int                       field6 : 1;
		unsigned int                       field5 : 4;
		unsigned int                       field4 : 3;
		unsigned int                       field3 : 6;
		unsigned int                       field2 : 6;
		unsigned int                       field1 : 6;
		unsigned int                       field0 : 6;
#endif
	} bitfields, bits;
	unsigned int	u32All;
	signed int	i32All;
	float	f32All;
};


typedef struct program_t {
	union bitfield    array[ARRAY_LENGTH];
    int               size;
    int               loaded;
} program;


void AdjustBitfields(program* prog, unsigned int fmt1)
{
	unsigned int shift = 0;
	unsigned int texCount = 0;
	unsigned int i;
	
	for (i = 0; i < 8; i++)
	{
		prog->array[i].bitfields.field0 = texCount;
		prog->array[i].bitfields.field1 = texCount + 1;
		prog->array[i].bitfields.field2 = texCount + 2;
		prog->array[i].bitfields.field3 = texCount + 3;

		texCount += (fmt1 >> shift) & 0x7;
		shift    += 3;
	}
}

In the loop above, the bitfield adds get generated as 
(add (shl bitfield, C1), (shl C2, C1)) where C2 is 1, 2 or 3.

Since the input to the (or and, and) is an (add) rather than a (shl), the shift
doesn't get folded into the rlwimi instruction.  We should ideally see through
things like this, rather than forcing llvm to generate the equivalent

(shl (add bitfield, C2), C1) with some kind of mask.

===-------------------------------------------------------------------------===

Compile this:

int %f1(int %a, int %b) {
        %tmp.1 = and int %a, 15         ; <int> [#uses=1]
        %tmp.3 = and int %b, 240                ; <int> [#uses=1]
        %tmp.4 = or int %tmp.3, %tmp.1          ; <int> [#uses=1]
        ret int %tmp.4
}

without a copy.  We make this currently:

_f1:
        rlwinm r2, r4, 0, 24, 27
        rlwimi r2, r3, 0, 28, 31
        or r3, r2, r2
        blr

The two-addr pass or RA needs to learn when it is profitable to commute an
instruction to avoid a copy AFTER the 2-addr instruction.  The 2-addr pass
currently only commutes to avoid inserting a copy BEFORE the two addr instr.

===-------------------------------------------------------------------------===

176.gcc contains a bunch of code like this (this occurs dozens of times):

int %test(uint %mode.0.i.0) {
        %tmp.79 = cast uint %mode.0.i.0 to sbyte        ; <sbyte> [#uses=1]
        %tmp.80 = cast sbyte %tmp.79 to int             ; <int> [#uses=1]
        %tmp.81 = shl int %tmp.80, ubyte 16             ; <int> [#uses=1]
        %tmp.82 = and int %tmp.81, 16711680
        ret int %tmp.82
}

which we compile to:

_test:
        extsb r2, r3
        rlwinm r3, r2, 16, 8, 15
        blr

The extsb is obviously dead.  This can be handled by a future thing like 
MaskedValueIsZero that checks to see if bits are ever demanded (in this case, 
the sign bits are never used, so we can fold the sext_inreg to nothing).

I'm seeing code like this:

        srwi r3, r3, 16
        extsb r3, r3
        rlwimi r4, r3, 16, 8, 15

in which the extsb is preventing the srwi from being nuked.

===-------------------------------------------------------------------------===

Another example that occurs is:

uint %test(int %specbits.6.1) {
        %tmp.2540 = shr int %specbits.6.1, ubyte 11     ; <int> [#uses=1]
        %tmp.2541 = cast int %tmp.2540 to uint          ; <uint> [#uses=1]
        %tmp.2542 = shl uint %tmp.2541, ubyte 13        ; <uint> [#uses=1]
        %tmp.2543 = and uint %tmp.2542, 8192            ; <uint> [#uses=1]
        ret uint %tmp.2543
}

which we codegen as:

l1_test:
        srawi r2, r3, 11
        rlwinm r3, r2, 13, 18, 18
        blr

the srawi can be nuked by turning the SAR into a logical SHR (the sext bits are 
dead), which I think can then be folded into the rlwinm.

===-------------------------------------------------------------------------===

Compile offsets from allocas:

int *%test() {
        %X = alloca { int, int }
        %Y = getelementptr {int,int}* %X, int 0, uint 1
        ret int* %Y
}

into a single add, not two:

_test:
        addi r2, r1, -8
        addi r3, r2, 4
        blr

--> important for C++.

===-------------------------------------------------------------------------===

int test3(int a, int b) { return (a < 0) ? a : 0; }

should be branch free code.  LLVM is turning it into < 1 because of the RHS.

===-------------------------------------------------------------------------===

No loads or stores of the constants should be needed:

struct foo { double X, Y; };
void xxx(struct foo F);
void bar() { struct foo R = { 1.0, 2.0 }; xxx(R); }

===-------------------------------------------------------------------------===

Darwin Stub LICM optimization:

Loops like this:
  
  for (...)  bar();

Have to go through an indirect stub if bar is external or linkonce.  It would 
be better to compile it as:

     fp = &bar;
     for (...)  fp();

which only computes the address of bar once (instead of each time through the 
stub).  This is Darwin specific and would have to be done in the code generator.
Probably not a win on x86.

===-------------------------------------------------------------------------===

PowerPC i1/setcc stuff (depends on subreg stuff):

Check out the PPC code we get for 'compare' in this testcase:
http://gcc.gnu.org/bugzilla/show_bug.cgi?id=19672

oof.  on top of not doing the logical crnand instead of (mfcr, mfcr,
invert, invert, or), we then have to compare it against zero instead of
using the value already in a CR!

that should be something like
        cmpw cr7, r8, r5
        cmpw cr0, r7, r3
        crnand cr0, cr0, cr7
        bne cr0, LBB_compare_4

instead of
        cmpw cr7, r8, r5
        cmpw cr0, r7, r3
        mfcr r7, 1
        mcrf cr7, cr0
        mfcr r8, 1
        rlwinm r7, r7, 30, 31, 31
        rlwinm r8, r8, 30, 31, 31
        xori r7, r7, 1
        xori r8, r8, 1
        addi r2, r2, 1
        or r7, r8, r7
        cmpwi cr0, r7, 0
        bne cr0, LBB_compare_4  ; loopexit

===-------------------------------------------------------------------------===

Simple IPO for argument passing, change:
  void foo(int X, double Y, int Z) -> void foo(int X, int Z, double Y)

the Darwin ABI specifies that any integer arguments in the first 32 bytes worth
of arguments get assigned to r3 through r10. That is, if you have a function
foo(int, double, int) you get r3, f1, r6, since the 64 bit double ate up the
argument bytes for r4 and r5. The trick then would be to shuffle the argument
order for functions we can internalize so that the maximum number of 
integers/pointers get passed in regs before you see any of the fp arguments.

Instead of implementing this, it would actually probably be easier to just 
implement a PPC fastcc, where we could do whatever we wanted to the CC, 
including having this work sanely.

===-------------------------------------------------------------------------===

Fix Darwin FP-In-Integer Registers ABI

Darwin passes doubles in structures in integer registers, which is very very 
bad.  Add something like a BIT_CONVERT to LLVM, then do an i-p transformation 
that percolates these things out of functions.

Check out how horrible this is:
http://gcc.gnu.org/ml/gcc/2005-10/msg01036.html

This is an extension of "interprocedural CC unmunging" that can't be done with
just fastcc.

===-------------------------------------------------------------------------===

Code Gen IPO optimization:

Squish small scalar globals together into a single global struct, allowing the 
address of the struct to be CSE'd, avoiding PIC accesses (also reduces the size
of the GOT on targets with one).

===-------------------------------------------------------------------------===

Generate lwbrx and other byteswapping load/store instructions when reasonable.

===-------------------------------------------------------------------------===

Implement TargetConstantVec, and set up PPC to custom lower ConstantVec into
TargetConstantVec's if it's one of the many forms that are algorithmically
computable using the spiffy altivec instructions.

===-------------------------------------------------------------------------===

Compile this:

double %test(double %X) {
        %Y = cast double %X to long
        %Z = cast long %Y to double
        ret double %Z
}

to this:

_test:
        fctidz f0, f1
        stfd f0, -8(r1)
        lwz r2, -4(r1)
        lwz r3, -8(r1)
        stw r2, -12(r1)
        stw r3, -16(r1)
        lfd f0, -16(r1)
        fcfid f1, f0
        blr

without the lwz/stw's.

===-------------------------------------------------------------------------===

Compile this:

int foo(int a) {
  int b = (a < 8);
  if (b) {
    return b * 3;     // ignore the fact that this is always 3.
  } else {
    return 2;
  }
}

into something not this:

_foo:
1)      cmpwi cr7, r3, 8
        mfcr r2, 1
        rlwinm r2, r2, 29, 31, 31
1)      cmpwi cr0, r3, 7
        bgt cr0, LBB1_2 ; UnifiedReturnBlock
LBB1_1: ; then
        rlwinm r2, r2, 0, 31, 31
        mulli r3, r2, 3
        blr
LBB1_2: ; UnifiedReturnBlock
        li r3, 2
        blr

In particular, the two compares (marked 1) could be shared by reversing one.
This could be done in the dag combiner, by swapping a BR_CC when a SETCC of the
same operands (but backwards) exists.  In this case, this wouldn't save us 
anything though, because the compares still wouldn't be shared.

===-------------------------------------------------------------------------===

The legalizer should lower this:

bool %test(ulong %x) {
  %tmp = setlt ulong %x, 4294967296
  ret bool %tmp
}

into "if x.high == 0", not:

_test:
        addi r2, r3, -1
        cntlzw r2, r2
        cntlzw r3, r3
        srwi r2, r2, 5
        srwi r4, r3, 5
        li r3, 0
        cmpwi cr0, r2, 0
        bne cr0, LBB1_2 ; 
LBB1_1:
        or r3, r4, r4
LBB1_2:
        blr

noticed in 2005-05-11-Popcount-ffs-fls.c.


===-------------------------------------------------------------------------===

We should custom expand setcc instead of pretending that we have it.  That
would allow us to expose the access of the crbit after the mfcr, allowing
that access to be trivially folded into other ops.  A simple example:

int foo(int a, int b) { return (a < b) << 4; }

compiles into:

_foo:
        cmpw cr7, r3, r4
        mfcr r2, 1
        rlwinm r2, r2, 29, 31, 31
        slwi r3, r2, 4
        blr

