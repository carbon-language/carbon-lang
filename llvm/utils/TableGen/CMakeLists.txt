set(LLVM_REQUIRES_EH 1)
set(LLVM_REQUIRES_RTTI 1)

add_tablegen(llvm-tblgen LLVM
  ARMDecoderEmitter.cpp
  AsmMatcherEmitter.cpp
  AsmWriterEmitter.cpp
  AsmWriterInst.cpp
  CallingConvEmitter.cpp
  CodeEmitterGen.cpp
  CodeGenDAGPatterns.cpp
  CodeGenInstruction.cpp
  CodeGenRegisters.cpp
  CodeGenTarget.cpp
  DAGISelEmitter.cpp
  DAGISelMatcherEmitter.cpp
  DAGISelMatcherGen.cpp
  DAGISelMatcherOpt.cpp
  DAGISelMatcher.cpp
  DisassemblerEmitter.cpp
  EDEmitter.cpp
  FastISelEmitter.cpp
  FixedLenDecoderEmitter.cpp
  InstrEnumEmitter.cpp
  InstrInfoEmitter.cpp
  IntrinsicEmitter.cpp
  PseudoLoweringEmitter.cpp
  RegisterInfoEmitter.cpp
  SetTheory.cpp
  StringMatcher.cpp
  SubtargetEmitter.cpp
  TGValueTypes.cpp
  TableGen.cpp
  X86DisassemblerTables.cpp
  X86RecognizableInstr.cpp
  )
