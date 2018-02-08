set(LLVM_LINK_COMPONENTS
  Support
  )

if (MSVC)
  set_source_files_properties(SemaExpr.cpp PROPERTIES COMPILE_FLAGS /bigobj)
endif()

add_flang_library(flangSema
  Scope.cpp
  Sema.cpp
  SemaExpr.cpp

  LINK_LIBS
  flangBasic
  )

