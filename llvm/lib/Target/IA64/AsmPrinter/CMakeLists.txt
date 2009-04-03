include_directories(
  ${CMAKE_CURRENT_BINARY_DIR}/..
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  )

add_partially_linked_object(LLVMIA64AsmPrinter
  IA64AsmPrinter.cpp
  )

target_name_of_partially_linked_object(LLVMIA64CodeGen n)

add_dependencies(LLVMIA64AsmPrinter ${n})
