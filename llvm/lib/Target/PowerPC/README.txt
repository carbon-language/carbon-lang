TODO:
* use stfiwx in float->int
* implement cast fp to bool
* implement algebraic shift right long by reg
* implement scheduling info
* implement powerpc-64 for darwin
* implement powerpc-64 for aix
* fix rlwimi generation to be use-and-def
* fix ulong to double:
  floatdidf assumes signed longs.  so if the high but of a ulong
  just happens to be set, you get the wrong sign.  The fix for this
  is to call cmpdi2 to compare against zero, if so shift right by one,
  convert to fp, and multiply by (add to itself).  the sequence would
  look like:
  {r3:r4} holds ulong a;
  li r5, 0
  li r6, 0 (set r5:r6 to ulong 0)
  call cmpdi2 ==> sets r3 <, =, > 0
  if r3 > 0
  call floatdidf as usual
  else
  shift right ulong a, 1 (we could use emitShift)
  call floatdidf
  fadd f1, f1, f1 (fp left shift by 1)
* cast elimination pass (uint -> sbyte -> short, kill the byte -> short)
* should hint to the branch select pass that it doesn't need to print the
  second unconditional branch, so we don't end up with things like:
	b .LBBl42__2E_expand_function_8_674	; loopentry.24
	b .LBBl42__2E_expand_function_8_42	; NewDefault
	b .LBBl42__2E_expand_function_8_42	; NewDefault

Currently failing tests that should pass:
* SingleSource
  `- Regression
  |  `- casts (ulong to fp failure)
* MultiSource
  |- Applications
  |  `- hbd: miscompilation
