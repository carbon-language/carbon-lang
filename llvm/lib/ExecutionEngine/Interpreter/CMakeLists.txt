# Make sure that the path to libffi headers is on the command
# line. That path can be a compiler's non-default path even when
# FFI_INCLUDE_DIR was not used, because cmake has its own paths for
# searching for headers (CMAKE_SYSTEM_INCLUDE_PATH, for instance):
if( FFI_INCLUDE_PATH )
  include_directories( ${FFI_INCLUDE_PATH} )
endif()

# Make sure that the path to libffi archive is on the command
# line. That path can be a linker's non-default path even when
# FFI_LIBRARY_DIR was not used, because cmake has its own paths for
# searching for libraries (CMAKE_SYSTEM_LIBRARY_PATH, for instance):
if( FFI_LIBRARY_PATH )
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
