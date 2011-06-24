set(LLVM_TARGET_DEFINITIONS X86.td)

tablegen(X86GenRegisterNames.inc -gen-register-enums)
tablegen(X86GenRegisterDesc.inc -gen-register-desc)
tablegen(X86GenRegisterInfo.h.inc -gen-register-info-header)
tablegen(X86GenRegisterInfo.inc -gen-register-info)
tablegen(X86GenDisassemblerTables.inc -gen-disassembler)
tablegen(X86GenInstrNames.inc -gen-instr-enums)
tablegen(X86GenInstrInfo.inc -gen-instr-desc)
tablegen(X86GenAsmWriter.inc -gen-asm-writer)
tablegen(X86GenAsmWriter1.inc -gen-asm-writer -asmwriternum=1)
tablegen(X86GenAsmMatcher.inc -gen-asm-matcher)
tablegen(X86GenDAGISel.inc -gen-dag-isel)
tablegen(X86GenFastISel.inc -gen-fast-isel)
tablegen(X86GenCallingConv.inc -gen-callingconv)
tablegen(X86GenSubtarget.inc -gen-subtarget)
tablegen(X86GenEDInfo.inc -gen-enhanced-disassembly-info)

set(sources
  SSEDomainFix.cpp
  X86AsmBackend.cpp
  X86AsmPrinter.cpp
  X86COFFMachineModuleInfo.cpp
  X86CodeEmitter.cpp
  X86ELFWriterInfo.cpp
  X86FastISel.cpp
  X86FloatingPoint.cpp
  X86FrameLowering.cpp
  X86ISelDAGToDAG.cpp
  X86ISelLowering.cpp
  X86InstrInfo.cpp
  X86JITInfo.cpp
  X86MachObjectWriter.cpp
  X86MCAsmInfo.cpp
  X86MCCodeEmitter.cpp 
  X86MCInstLower.cpp
  X86RegisterInfo.cpp
  X86SelectionDAGInfo.cpp
  X86Subtarget.cpp
  X86TargetMachine.cpp
  X86TargetObjectFile.cpp
  )

if( CMAKE_CL_64 )
  # A workaround for a bug in cmake 2.8.3. See PR 8885.
  if( CMAKE_VERSION STREQUAL "2.8.3" )
    include(CMakeDetermineCompilerId)
  endif()
  # end of workaround.
  enable_language(ASM_MASM)
  ADD_CUSTOM_COMMAND(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/X86CompilationCallback_Win64.obj
    MAIN_DEPENDENCY X86CompilationCallback_Win64.asm
    COMMAND ${CMAKE_ASM_MASM_COMPILER} /Fo ${CMAKE_CURRENT_BINARY_DIR}/X86CompilationCallback_Win64.obj /c ${CMAKE_CURRENT_SOURCE_DIR}/X86CompilationCallback_Win64.asm
   )
   set(sources ${sources} ${CMAKE_CURRENT_BINARY_DIR}/X86CompilationCallback_Win64.obj)
endif()

add_llvm_target(X86CodeGen ${sources})

add_subdirectory(AsmParser)
add_subdirectory(Disassembler)
add_subdirectory(InstPrinter)
add_subdirectory(TargetInfo)
add_subdirectory(Utils)
