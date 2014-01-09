add_llvm_library(LLVMCore
  AsmWriter.cpp
  Attributes.cpp
  AutoUpgrade.cpp
  BasicBlock.cpp
  ConstantFold.cpp
  Constants.cpp
  Core.cpp
  DiagnosticInfo.cpp
  DiagnosticPrinter.cpp
  DIBuilder.cpp
  DataLayout.cpp
  DebugInfo.cpp
  DebugLoc.cpp
  Dominators.cpp
  Function.cpp
  GCOV.cpp
  GVMaterializer.cpp
  Globals.cpp
  IRBuilder.cpp
  IRPrintingPasses.cpp
  InlineAsm.cpp
  Instruction.cpp
  Instructions.cpp
  IntrinsicInst.cpp
  LLVMContext.cpp
  LLVMContextImpl.cpp
  LeakDetector.cpp
  LegacyPassManager.cpp
  Mangler.cpp
  Metadata.cpp
  Module.cpp
  Pass.cpp
  PassManager.cpp
  PassRegistry.cpp
  Type.cpp
  TypeFinder.cpp
  Use.cpp
  User.cpp
  Value.cpp
  ValueSymbolTable.cpp
  ValueTypes.cpp
  Verifier.cpp
  )

# Workaround: It takes over 20 minutes to compile with msvc10.
# FIXME: Suppressing optimizations to core libraries would not be good thing.
if( MSVC_VERSION LESS 1700 )
set_property(
  SOURCE Function.cpp
  PROPERTY COMPILE_FLAGS "/Og-"
  )
endif()

add_dependencies(LLVMCore intrinsics_gen)
