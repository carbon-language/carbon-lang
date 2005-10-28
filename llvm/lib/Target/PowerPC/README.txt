TODO:
* gpr0 allocation
* implement do-loop -> bdnz transform
* implement powerpc-64 for darwin
* use stfiwx in float->int
* be able to combine sequences like the following into 2 instructions:
	lis r2, ha16(l2__ZTV4Cell)
	la r2, lo16(l2__ZTV4Cell)(r2)
	addi r2, r2, 8

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

and:
  extern int X, Y; int* test(int C) { return C? &X : &Y; }
as one load when using --enable-pic.

* Support 'update' load/store instructions.  These are cracked on the G5, but
  are still a codesize win.

* Add a custom legalizer for the GlobalAddress node, to move the funky darwin
  stub stuff from the instruction selector to the legalizer (exposing low-level
  operations to the dag for optzn.  For example, we want to codegen this:

        int A = 0;
        void B() { A++; }
  as:
        lis r9,ha16(_A)
        lwz r2,lo16(_A)(r9)
        addi r2,r2,1
        stw r2,lo16(_A)(r9)
  not:
        lis r2, ha16(_A)
        lwz r2, lo16(_A)(r2)
        addi r2, r2, 1
        lis r3, ha16(_A)
        stw r2, lo16(_A)(r3)

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

===-------------------------------------------------------------------------===

Implement Newton-Rhapson method for improving estimate instructions to the
correct accuracy, and implementing divide as multiply by reciprocal when it has
more than one use.  Itanium will want this too.

===-------------------------------------------------------------------------===

int foo(int a, int b) { return a == b ? 16 : 0; }
_foo:
        cmpw cr7, r3, r4
        mfcr r2
        rlwinm r2, r2, 31, 31, 31
        slwi r3, r2, 4
        blr

If we exposed the srl & mask ops after the MFCR that we are doing to select
the correct CR bit, then we could fold the slwi into the rlwinm before it.

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

Compile this (standard bitfield insert of a constant):
void %test(uint* %tmp1) {
        %tmp2 = load uint* %tmp1                ; <uint> [#uses=1]
        %tmp5 = or uint %tmp2, 257949696                ; <uint> [#uses=1]
        %tmp6 = and uint %tmp5, 4018143231              ; <uint> [#uses=1]
        store uint %tmp6, uint* %tmp1
        ret void
}

to:

_test:
        lwz r0,0(r3)
        li r2,123
        rlwimi r0,r2,21,3,10
        stw r0,0(r3)
        blr

instead of:

_test:
        lis r2, -4225
        lwz r4, 0(r3)
        ori r2, r2, 65535
        oris r4, r4, 3936
        and r2, r4, r2
        stw r2, 0(r3)
        blr



