Currently unimplemented:
* cast fp to bool
* signed right shift of long by reg

Current bugs:
* large fixed-size allocas not correct, although should
  be closer to working.  Added code in PPCRegisterInfo.cpp
  to do >16bit subtractions to the stack pointer.
* ulong to double.  ahhh, here's the problem:
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
* linking llvmg++ .s files with gcc instead of g++

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
* conditional restore of link register (tricky, temporarily backed out
  part of first attempt)

Current hacks:
* lazy insert of GlobalBaseReg definition at front of first MBB
  A prime candidate for sabre's "slightly above ISel" passes.
* cast code is huge, unwieldy.  Should probably be broken up into
  smaller pieces.
* visitLoadInst is getting awfully cluttered as well.

Currently failing tests:
* SingleSource
  `- Regression
  |  `- 2003-05-22-VarSizeArray
  |  `- casts (ulong to fp failure)
  `- Benchmarks
  |  `- Shootout-C++ : most programs fail, miscompilations
  `- UnitTests
  |  `- C++Catch
  |  `- SimpleC++Test
  |  `- ConditionalExpr (also C++)
* MultiSource
  |- Applications
  |  `- burg: miscompilation
  |  `- siod: llc bus error
  |  `- hbd: miscompilation
  |  `- d (make_dparser): miscompilation
  `- Benchmarks
     `- MallocBench/make: miscompilation
