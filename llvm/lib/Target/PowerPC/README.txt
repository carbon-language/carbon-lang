TODO:
* gpr0 allocation
* implement do-loop -> bdnz transform
* implement powerpc-64 for darwin
* use stfiwx in float->int
* be able to combine sequences like the following into 2 instructions:
	lis r2, ha16(l2__ZTV4Cell)
	la r2, lo16(l2__ZTV4Cell)(r2)
	addi r2, r2, 8

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
