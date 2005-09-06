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
