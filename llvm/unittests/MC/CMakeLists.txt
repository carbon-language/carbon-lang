set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  MC
  MCDisassembler
  Support
  )

add_llvm_unittest(MCTests
  Disassembler.cpp
  StringTableBuilderTest.cpp
  YAMLTest.cpp
  )

foreach(t ${LLVM_TARGETS_TO_BUILD})
  if (IS_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/${t}")
    add_subdirectory(${t})
  endif (IS_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/${t}")
endforeach()
