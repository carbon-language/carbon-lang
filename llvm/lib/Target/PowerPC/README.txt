Currently unimplemented:
* cast fp to bool
* signed right shift of long by reg

Current bugs:
* conditional branches assume target is within 32k bytes
* large fixed-size allocas not correct, although should
  be closer to working.  Added code in PPCRegisterInfo.cpp
  to do >16bit subtractions to the stack pointer.

Codegen improvements needed:
* we unconditionally emit save/restore of LR even if we don't use it
* no alias analysis causes us to generate slow code for Shootout/matrix
* setCondInst needs to know branchless versions of seteq/setne/etc
* cast elimination pass (uint -> sbyte -> short, kill the byte -> short)

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
