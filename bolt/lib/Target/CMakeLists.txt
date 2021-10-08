foreach(t ${LLVM_TARGETS_TO_BUILD})
  message(STATUS "Targeting llvm-bolt ${t}")
  add_subdirectory(${t})
endforeach()
