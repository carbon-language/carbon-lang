# LLVM_TARGET_DEFINITIONS must contain the name of the .td file to process.
# Extra parameters for `tblgen' may come after `ofn' parameter.
# Adds the name of the generated file to TABLEGEN_OUTPUT.

macro(tablegen ofn)
  file(GLOB local_tds "*.td")
  file(GLOB_RECURSE global_tds "${LLVM_MAIN_SRC_DIR}/include/llvm/*.td")

  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${ofn}
    COMMAND ${LLVM_TABLEGEN_EXE} ${ARGN} -I ${CMAKE_CURRENT_SOURCE_DIR}
    -I ${LLVM_MAIN_SRC_DIR}/lib/Target -I ${LLVM_MAIN_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/${LLVM_TARGET_DEFINITIONS} 
    -o ${CMAKE_CURRENT_BINARY_DIR}/${ofn}
    DEPENDS tblgen ${local_tds} ${global_tds}
    COMMENT "Building ${ofn}..."
    )
  set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT} ${CMAKE_CURRENT_BINARY_DIR}/${ofn})
  set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/${ofn} 
    PROPERTIES GENERATED 1)
endmacro(tablegen)
