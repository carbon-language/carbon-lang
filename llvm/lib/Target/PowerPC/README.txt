Currently unimplemented:
* cast fp to bool
* signed right shift

Current bugs:
* use of a cByte/cShort by setCC not first truncated or sign extended
  (uByte r3 = 250, r3 + 100; setlt r3, 200 will get wrong result).
* conditional branches assume target is within 32k bytes
* large fixed-size allocas not correct

Currently failing tests:
* Regression
* SingleSource
  `- Benchmarks
  |  `- Shootout-C++ : most programs fail, miscompilations
  `- UnitTests
  |  `- 2002-05-02-CastTest
  |  `- 2003-05-07-VarArgs
  |  `- 2003-05-26-Shorts
  |  `- 2003-07-09-LoadShorts
  |  `- 2003-07-09-SignedArgs
  |  `- 2003-08-11-VaListArg
  |  `- 2003-05-22-VarSizeArray
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
     `- McCat/12-IOtest: miscompilation
     `- Ptrdist/bc: branch target too far
     `- FreeBench/pifft
     `- MallocBench/espresso: same as bc
     `- MallocBench/make: same as bc
