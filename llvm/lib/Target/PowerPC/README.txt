Currently unimplemented:
* cast fp to bool
* signed right shift of long by reg

Current bugs:
* large fixed-size allocas not correct, although should
  be closer to working.  Added code in PPCRegisterInfo.cpp
  to do >16bit subtractions to the stack pointer.

Codegen improvements needed:
* no alias analysis causes us to generate slow code for Shootout/matrix
* setCondInst needs to know branchless versions of seteq/setne/etc
* cast elimination pass (uint -> sbyte -> short, kill the byte -> short)
* should hint to the branch select pass that it doesn't need to print the
  second unconditional branch, so we don't end up with things like:
.LBBl42__2E_expand_function_8_21:	; LeafBlock37
	cmplwi cr0, r29, 11
	bne cr0, $+8
	b .LBBl42__2E_expand_function_8_674	; loopentry.24
	b .LBBl42__2E_expand_function_8_42	; NewDefault
	b .LBBl42__2E_expand_function_8_42	; NewDefault

Current hacks:
* lazy insert of GlobalBaseReg definition at front of first MBB
  A prime candidate for sabre's "slightly above ISel" passes.
* cast code is huge, unwieldy.  Should probably be broken up into
  smaller pieces.
* visitLoadInst is getting awfully cluttered as well.

Currently failing tests:
* Regression
* SingleSource
  `- Benchmarks
  |  `- Shootout-C++ : most programs fail, miscompilations
  `- UnitTests
  |  `- 2003-05-26-Shorts
  |  `- 2003-07-09-LoadShorts
  |  `- 2004-06-20-StaticBitfieldInt
  `- C++Catch
  `- SimpleC++Test
  `- ConditionalExpr
  `- casts
  `- sumarray2d: large alloca miscompiled
  `- test_indvars
* MultiSource
  |- Applications
  |  `- burg: miscompilation
  |  `- siod: llc bus error
  |  `- hbd: miscompilation
  |  `- d (make_dparser): miscompilation
  `- Benchmarks
     `- MallocBench/make: branch target too far
