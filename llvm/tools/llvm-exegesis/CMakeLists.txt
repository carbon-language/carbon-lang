set(LLVM_LINK_COMPONENTS
  AllTargetsAsmPrinters
  AllTargetsDescs
  AllTargetsInfos
  X86
  )

add_llvm_tool(llvm-exegesis
  llvm-exegesis.cpp
  )

add_subdirectory(lib)
target_link_libraries(llvm-exegesis PRIVATE LLVMExegesis)

if(HAVE_LIBPFM)
  target_link_libraries(llvm-exegesis PRIVATE pfm)
endif()

