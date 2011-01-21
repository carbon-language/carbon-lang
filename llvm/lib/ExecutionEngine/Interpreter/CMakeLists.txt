# If the user required a custom path for ffi headers, use it on its
# abolute form (see config-ix.cmake):
if( FFI_INCLUDE_DIR )
  include_directories( ${FFI_INCLUDE_PATH} )
endif()

# If the user required a custom path for libffi, use it on its abolute
# form (see config-ix.cmake):
if( FFI_LIBRARY_DIR )
  link_directories( ${FFI_LIBRARY_PATH} )
endif()

add_llvm_library(LLVMInterpreter
  Execution.cpp
  ExternalFunctions.cpp
  Interpreter.cpp
  )

if( LLVM_ENABLE_FFI )
  target_link_libraries( LLVMInterpreter ffi )
endif()
