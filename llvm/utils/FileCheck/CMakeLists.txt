add_executable(FileCheck
  FileCheck.cpp
  )

target_link_libraries(FileCheck LLVMSupport LLVMSystem)
if( MINGW )
  target_link_libraries(FileCheck imagehlp psapi)
endif( MINGW )
if( LLVM_ENABLE_THREADS AND HAVE_LIBPTHREAD )
  target_link_libraries(FileCheck pthread)
endif()
