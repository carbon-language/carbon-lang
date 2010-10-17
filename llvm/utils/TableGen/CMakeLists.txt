set(LLVM_REQUIRES_EH 1)
set(LLVM_REQUIRES_RTTI 1)

add_executable(tblgen
  ARMDecoderEmitter.cpp
  AsmMatcherEmitter.cpp
  AsmWriterEmitter.cpp
  AsmWriterInst.cpp
  CallingConvEmitter.cpp
  ClangASTNodesEmitter.cpp
  ClangAttrEmitter.cpp
  ClangDiagnosticsEmitter.cpp
  CodeEmitterGen.cpp
  CodeGenDAGPatterns.cpp
  CodeGenInstruction.cpp
  CodeGenTarget.cpp
  DAGISelEmitter.cpp
  DAGISelMatcherEmitter.cpp
  DAGISelMatcherGen.cpp
  DAGISelMatcherOpt.cpp
  DAGISelMatcher.cpp
  DisassemblerEmitter.cpp
  EDEmitter.cpp
  FastISelEmitter.cpp
  InstrEnumEmitter.cpp
  InstrInfoEmitter.cpp
  IntrinsicEmitter.cpp
  LLVMCConfigurationEmitter.cpp
  NeonEmitter.cpp
  OptParserEmitter.cpp
  Record.cpp
  RegisterInfoEmitter.cpp
  StringMatcher.cpp
  SubtargetEmitter.cpp
  TGLexer.cpp
  TGParser.cpp
  TGValueTypes.cpp
  TableGen.cpp
  TableGenBackend.cpp
  X86DisassemblerTables.cpp
  X86RecognizableInstr.cpp
  )

target_link_libraries(tblgen LLVMSupport LLVMSystem)
if( MINGW )
  target_link_libraries(tblgen imagehlp psapi)
endif( MINGW )
if( LLVM_ENABLE_THREADS AND HAVE_LIBPTHREAD AND NOT BEOS )
  target_link_libraries(tblgen pthread)
endif()
