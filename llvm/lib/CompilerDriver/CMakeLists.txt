set(LLVM_LINK_COMPONENTS support)

# We don't want this library to appear in `llvm-config --libs` output,
# so its name doesn't start with "LLVM".

add_llvm_library(CompilerDriver
  Action.cpp
  BuiltinOptions.cpp
  CompilationGraph.cpp
  Main.cpp
  Tool.cpp
  )
