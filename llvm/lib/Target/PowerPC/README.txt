TODO:
* implement not-R0 register GPR class
* implement scheduling info
* implement do-loop pass
* implement do-loop -> bdnz transform
* implement powerpc-64 for darwin
* implement powerpc-64 for aix
* use stfiwx in float->int
* should hint to the branch select pass that it doesn't need to print the
  second unconditional branch, so we don't end up with things like:
	b .LBBl42__2E_expand_function_8_674	; loopentry.24
	b .LBBl42__2E_expand_function_8_42	; NewDefault
	b .LBBl42__2E_expand_function_8_42	; NewDefault

Currently failing tests that should pass:
* MultiSource
  |- Applications
  |  `- hbd: miscompilation
