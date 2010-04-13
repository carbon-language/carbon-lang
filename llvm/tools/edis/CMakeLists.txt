set(SHARED_LIBRARY TRUE)
set(LLVM_NO_RTTI 1)

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/EDInfo.inc
  COMMAND ${LLVM_TABLEGEN_EXE} -o ${CMAKE_CURRENT_BINARY_DIR}/EDInfo.inc
          -gen-enhanced-disassembly-header ${CMAKE_CURRENT_SOURCE_DIR}/EDInfo.td
  DEPENDS tblgen
  COMMENT "Building enhanced disassembly semantic information header (EDInfo.inc)")
set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/EDInfo.inc PROPERTIES GENERATED 1)

include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_llvm_library(EnhancedDisassembly
  EDDisassembler.cpp
  EDInst.cpp
  EDMain.cpp
  EDOperand.cpp
  EDToken.cpp
  ../../include/llvm-c/EnhancedDisassembly.h
  ${CMAKE_CURRENT_BINARY_DIR}/EDInfo.inc
)

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  # FIXME: Deal with EDIS version?
  set_target_properties(EnhancedDisassembly
    PROPERTIES
    LINK_FLAGS "-avoid-version -Wl,-exported_symbols_list -Wl,${CMAKE_CURRENT_SOURCE_DIR}/EnhancedDisassembly.exports -Wl,-dead_strip -Wl"
    )
endif()

if(MSVC)
  # windows.h doesn't compile with /Za
  get_target_property(NON_ANSI_COMPILE_FLAGS CIndex COMPILE_FLAGS)
  string(REPLACE /Za "" NON_ANSI_COMPILE_FLAGS ${NON_ANSI_COMPILE_FLAGS})
  set_target_properties(EnhancedDisassembly PROPERTIES COMPILE_FLAGS ${NON_ANSI_COMPILE_FLAGS})
endif(MSVC)

set_target_properties(EnhancedDisassembly
  PROPERTIES
  LINKER_LANGUAGE CXX)
  