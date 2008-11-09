# TODO: Build a native tblgen when cross-compiling, if the user
# didn't set LLVM_TABLEGEN. Then, remove this check:
if( CMAKE_CROSSCOMPILING AND ${LLVM_TABLEGEN} STREQUAL "tblgen" )
  message(FATAL_ERROR
    "Set LLVM_TABLEGEN to the full route to a native tblgen executable")
endif()
