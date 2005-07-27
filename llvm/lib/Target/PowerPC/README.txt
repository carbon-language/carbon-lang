TODO:
* gpr0 allocation
* implement do-loop -> bdnz transform
* implement powerpc-64 for darwin
* use stfiwx in float->int
* be able to combine sequences like the following into 2 instructions:
	lis r2, ha16(l2__ZTV4Cell)
	la r2, lo16(l2__ZTV4Cell)(r2)
	addi r2, r2, 8

* should hint to the branch select pass that it doesn't need to print the
  second unconditional branch, so we don't end up with things like:
	b .LBBl42__2E_expand_function_8_674	; loopentry.24
	b .LBBl42__2E_expand_function_8_42	; NewDefault
	b .LBBl42__2E_expand_function_8_42	; NewDefault
