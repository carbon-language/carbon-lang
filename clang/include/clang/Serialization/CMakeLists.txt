set(LLVM_TARGET_DEFINITIONS ../Basic/Attr.td)
tablegen(AttrPCHRead.inc
         -gen-clang-attr-pch-read
         -I ${CMAKE_CURRENT_SOURCE_DIR}/../../)
add_custom_target(ClangAttrPCHRead
  DEPENDS AttrPCHRead.inc)

tablegen(AttrPCHWrite.inc
         -gen-clang-attr-pch-write
         -I ${CMAKE_CURRENT_SOURCE_DIR}/../../)
add_custom_target(ClangAttrPCHWrite
  DEPENDS AttrPCHWrite.inc)
