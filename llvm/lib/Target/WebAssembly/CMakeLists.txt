set(LLVM_TARGET_DEFINITIONS WebAssembly.td)

tablegen(LLVM WebAssemblyGenMCCodeEmitter.inc -gen-emitter)
tablegen(LLVM WebAssemblyGenSubtargetInfo.inc -gen-subtarget)
add_public_tablegen_target(WebAssemblyCommonTableGen)

add_llvm_target(WebAssemblyCodeGen
  WebAssemblyFrameLowering.cpp
  WebAssemblyInstrInfo.cpp
  WebAssemblyISelDAGToDAG.cpp
  WebAssemblyISelLowering.cpp
  WebAssemblyMachineFunctionInfo.cpp
  WebAssemblyRegisterInfo.cpp
  WebAssemblySelectionDAGInfo.cpp
  WebAssemblySubtarget.cpp
  WebAssemblyTargetMachine.cpp
  WebAssemblyTargetTransformInfo.cpp
)

add_dependencies(LLVMWebAssemblyCodeGen intrinsics_gen)

add_subdirectory(InstPrinter)
add_subdirectory(TargetInfo)
add_subdirectory(MCTargetDesc)
