include_directories(
  ${CMAKE_CURRENT_BINARY_DIR}/..
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  )

add_partially_linked_object(LLVMMipsAsmPrinter
  MipsAsmPrinter.cpp
  )

target_name_of_partially_linked_object(LLVMMipsCodeGen n)

add_dependencies(LLVMMipsAsmPrinter ${n})
