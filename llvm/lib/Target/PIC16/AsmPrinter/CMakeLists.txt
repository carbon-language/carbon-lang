include_directories(
  ${CMAKE_CURRENT_BINARY_DIR}/..
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  )

add_llvm_library(LLVMPIC16AsmPrinter
  PIC16AsmPrinter.cpp
  )

target_link_libraries(LLVMPIC16AsmPrinter
  LLVMPIC16CodeGen
  )

add_dependencies(LLVMPIC16AsmPrinter PIC16CodeGenTable_gen)
