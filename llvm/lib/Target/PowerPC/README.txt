//===- README.txt - Notes for improving PowerPC-specific code gen ---------===//

TODO:
* gpr0 allocation
* implement do-loop -> bdnz transform

===-------------------------------------------------------------------------===

Support 'update' load/store instructions.  These are cracked on the G5, but are
still a codesize win.

===-------------------------------------------------------------------------===

Teach the .td file to pattern match PPC::BR_COND to appropriate bc variant, so
we don't have to always run the branch selector for small functions.

===-------------------------------------------------------------------------===

* Codegen this:

   void test2(int X) {
     if (X == 0x12345678) bar();
   }

    as:

       xoris r0,r3,0x1234
       cmplwi cr0,r0,0x5678
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

PIC Code Gen IPO optimization:

Squish small scalar globals together into a single global struct, allowing the 
address of the struct to be CSE'd, avoiding PIC accesses (also reduces the size
of the GOT on targets with one).

Note that this is discussed here for GCC:
http://gcc.gnu.org/ml/gcc-patches/2006-02/msg00133.html

===-------------------------------------------------------------------------===

Implement Newton-Rhapson method for improving estimate instructions to the
correct accuracy, and implementing divide as multiply by reciprocal when it has
more than one use.  Itanium will want this too.

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

FreeBench/mason has a basic block that looks like this:

         %tmp.130 = seteq int %p.0__, 5          ; <bool> [#uses=1]
         %tmp.134 = seteq int %p.1__, 6          ; <bool> [#uses=1]
         %tmp.139 = seteq int %p.2__, 12         ; <bool> [#uses=1]
         %tmp.144 = seteq int %p.3__, 13         ; <bool> [#uses=1]
         %tmp.149 = seteq int %p.4__, 14         ; <bool> [#uses=1]
         %tmp.154 = seteq int %p.5__, 15         ; <bool> [#uses=1]
         %bothcond = and bool %tmp.134, %tmp.130         ; <bool> [#uses=1]
         %bothcond123 = and bool %bothcond, %tmp.139             ; <bool>
         %bothcond124 = and bool %bothcond123, %tmp.144          ; <bool>
         %bothcond125 = and bool %bothcond124, %tmp.149          ; <bool>
         %bothcond126 = and bool %bothcond125, %tmp.154          ; <bool>
         br bool %bothcond126, label %shortcirc_next.5, label %else.0

This is a particularly important case where handling CRs better will help.

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

===-------------------------------------------------------------------------===

Fold add and sub with constant into non-extern, non-weak addresses so this:

static int a;
void bar(int b) { a = b; }
void foo(unsigned char *c) {
  *c = a;
}

So that 

_foo:
        lis r2, ha16(_a)
        la r2, lo16(_a)(r2)
        lbz r2, 3(r2)
        stb r2, 0(r3)
        blr

Becomes

_foo:
        lis r2, ha16(_a+3)
        lbz r2, lo16(_a+3)(r2)
        stb r2, 0(r3)
        blr

===-------------------------------------------------------------------------===

We generate really bad code for this:

int f(signed char *a, _Bool b, _Bool c) {
   signed char t = 0;
  if (b)  t = *a;
  if (c)  *a = t;
}

===-------------------------------------------------------------------------===

This:
int test(unsigned *P) { return *P >> 24; }

Should compile to:

_test:
        lbz r3,0(r3)
        blr

not:

_test:
        lwz r2, 0(r3)
        srwi r3, r2, 24
        blr

===-------------------------------------------------------------------------===

On the G5, logical CR operations are more expensive in their three
address form: ops that read/write the same register are half as expensive as
those that read from two registers that are different from their destination.

We should model this with two separate instructions.  The isel should generate
the "two address" form of the instructions.  When the register allocator 
detects that it needs to insert a copy due to the two-addresness of the CR
logical op, it will invoke PPCInstrInfo::convertToThreeAddress.  At this point
we can convert to the "three address" instruction, to save code space.

This only matters when we start generating cr logical ops.

===-------------------------------------------------------------------------===

We should compile these two functions to the same thing:

#include <stdlib.h>
void f(int a, int b, int *P) {
  *P = (a-b)>=0?(a-b):(b-a);
}
void g(int a, int b, int *P) {
  *P = abs(a-b);
}

Further, they should compile to something better than:

_g:
        subf r2, r4, r3
        subfic r3, r2, 0
        cmpwi cr0, r2, -1
        bgt cr0, LBB2_2 ; entry
LBB2_1: ; entry
        mr r2, r3
LBB2_2: ; entry
        stw r2, 0(r5)
        blr

GCC produces:

_g:
        subf r4,r4,r3
        srawi r2,r4,31
        xor r0,r2,r4
        subf r0,r2,r0
        stw r0,0(r5)
        blr

... which is much nicer.

This theoretically may help improve twolf slightly (used in dimbox.c:142?).

===-------------------------------------------------------------------------===

int foo(int N, int ***W, int **TK, int X) {
  int t, i;
  
  for (t = 0; t < N; ++t)
    for (i = 0; i < 4; ++i)
      W[t / X][i][t % X] = TK[i][t];
      
  return 5;
}

We generate relatively atrocious code for this loop compared to gcc.

We could also strength reduce the rem and the div:
http://www.lcs.mit.edu/pubs/pdf/MIT-LCS-TM-600.pdf

===-------------------------------------------------------------------------===

float foo(float X) { return (int)(X); }

Currently produces:

_foo:
        fctiwz f0, f1
        stfd f0, -8(r1)
        lwz r2, -4(r1)
        extsw r2, r2
        std r2, -16(r1)
        lfd f0, -16(r1)
        fcfid f0, f0
        frsp f1, f0
        blr

We could use a target dag combine to turn the lwz/extsw into an lwa when the 
lwz has a single use.  Since LWA is cracked anyway, this would be a codesize
win only.

===-------------------------------------------------------------------------===

We generate ugly code for this:

void func(unsigned int *ret, float dx, float dy, float dz, float dw) {
  unsigned code = 0;
  if(dx < -dw) code |= 1;
  if(dx > dw)  code |= 2;
  if(dy < -dw) code |= 4;
  if(dy > dw)  code |= 8;
  if(dz < -dw) code |= 16;
  if(dz > dw)  code |= 32;
  *ret = code;
}

===-------------------------------------------------------------------------===

Complete the signed i32 to FP conversion code using 64-bit registers
transformation, good for PI.  See PPCISelLowering.cpp, this comment:

     // FIXME: disable this lowered code.  This generates 64-bit register values,
     // and we don't model the fact that the top part is clobbered by calls.  We
     // need to flag these together so that the value isn't live across a call.
     //setOperationAction(ISD::SINT_TO_FP, MVT::i32, Custom);

Also, if the registers are spilled to the stack, we have to ensure that all
64-bits of them are save/restored, otherwise we will miscompile the code.  It
sounds like we need to get the 64-bit register classes going.

===-------------------------------------------------------------------------===

%struct.B = type { ubyte, [3 x ubyte] }

void %foo(%struct.B* %b) {
entry:
        %tmp = cast %struct.B* %b to uint*              ; <uint*> [#uses=1]
        %tmp = load uint* %tmp          ; <uint> [#uses=1]
        %tmp3 = cast %struct.B* %b to uint*             ; <uint*> [#uses=1]
        %tmp4 = load uint* %tmp3                ; <uint> [#uses=1]
        %tmp8 = cast %struct.B* %b to uint*             ; <uint*> [#uses=2]
        %tmp9 = load uint* %tmp8                ; <uint> [#uses=1]
        %tmp4.mask17 = shl uint %tmp4, ubyte 1          ; <uint> [#uses=1]
        %tmp1415 = and uint %tmp4.mask17, 2147483648            ; <uint> [#uses=1]
        %tmp.masked = and uint %tmp, 2147483648         ; <uint> [#uses=1]
        %tmp11 = or uint %tmp1415, %tmp.masked          ; <uint> [#uses=1]
        %tmp12 = and uint %tmp9, 2147483647             ; <uint> [#uses=1]
        %tmp13 = or uint %tmp12, %tmp11         ; <uint> [#uses=1]
        store uint %tmp13, uint* %tmp8
        ret void
}

We emit:

_foo:
        lwz r2, 0(r3)
        slwi r4, r2, 1
        or r4, r4, r2
        rlwimi r2, r4, 0, 0, 0
        stw r2, 0(r3)
        blr

We could collapse a bunch of those ORs and ANDs and generate the following
equivalent code:

_foo:
        lwz r2, 0(r3)
        rlwinm r4, r2, 1, 0, 0
        or r2, r2, r4
        stw r2, 0(r3)
        blr
