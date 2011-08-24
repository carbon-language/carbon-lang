add_llvm_library(LLVMTarget
  Mangler.cpp
  Target.cpp
  TargetData.cpp
  TargetELFWriterInfo.cpp
  TargetFrameLowering.cpp
  TargetInstrInfo.cpp
  TargetIntrinsicInfo.cpp
  TargetLibraryInfo.cpp
  TargetLoweringObjectFile.cpp
  TargetMachine.cpp
  TargetRegisterInfo.cpp
  TargetSubtargetInfo.cpp
  )

add_llvm_library_dependencies(LLVMTarget
  LLVMCore
  LLVMMC
  LLVMSupport
  )

set(LLVM_ENUM_ASM_PRINTERS "")
set(LLVM_ENUM_ASM_PARSERS "")
set(LLVM_ENUM_DISASSEMBLERS "")
foreach(t ${LLVM_TARGETS_TO_BUILD})
  message(STATUS "Targeting ${t}")
  add_subdirectory(${t})
  set( td ${LLVM_MAIN_SRC_DIR}/lib/Target/${t} )
  file(GLOB asmp_file "${td}/*AsmPrinter.cpp")
  if( asmp_file )
    set(LLVM_ENUM_ASM_PRINTERS
      "${LLVM_ENUM_ASM_PRINTERS}LLVM_ASM_PRINTER(${t})\n")
  endif()
  if( EXISTS ${td}/AsmParser/CMakeLists.txt )
    set(LLVM_ENUM_ASM_PARSERS
      "${LLVM_ENUM_ASM_PARSERS}LLVM_ASM_PARSER(${t})\n")
  endif()
  if( EXISTS ${td}/Disassembler/CMakeLists.txt )
    set(LLVM_ENUM_DISASSEMBLERS
      "${LLVM_ENUM_DISASSEMBLERS}LLVM_DISASSEMBLER(${t})\n")
  endif()
endforeach(t)

# Produce llvm/Config/AsmPrinters.def
configure_file(
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Config/AsmPrinters.def.in
  ${LLVM_BINARY_DIR}/include/llvm/Config/AsmPrinters.def
  )

# Produce llvm/Config/AsmParsers.def
configure_file(
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Config/AsmParsers.def.in
  ${LLVM_BINARY_DIR}/include/llvm/Config/AsmParsers.def
  )

# Produce llvm/Config/Disassemblers.def
configure_file(
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Config/Disassemblers.def.in
  ${LLVM_BINARY_DIR}/include/llvm/Config/Disassemblers.def
  )
