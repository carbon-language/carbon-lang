add_llvm_library(LLVMMCDisassembler
  Disassembler.cpp
  EDDisassembler.cpp
  EDInst.cpp
  EDOperand.cpp
  EDToken.cpp
  )

add_llvm_library_dependencies(LLVMMCDisassembler
  LLVMMC
  LLVMMCParser
  LLVMSupport
  LLVMTarget
  )

foreach(t ${LLVM_TARGETS_TO_BUILD})
  set(td ${LLVM_MAIN_SRC_DIR}/lib/Target/${t})
  if(EXISTS ${td}/Disassembler/CMakeLists.txt)
    add_llvm_library_dependencies(LLVMMCDisassembler "LLVM${t}Disassembler")
  endif()
endforeach(t)
