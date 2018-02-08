set(LLVM_LINK_COMPONENTS
  Support
  )

if (MSVC)
  set_source_files_properties(SemaExpr.cpp PROPERTIES COMPILE_FLAGS /bigobj)
endif()

add_clang_library(clangSema
  Scope.cpp
  Sema.cpp
  SemaExpr.cpp

  LINK_LIBS
  clangBasic
  )
