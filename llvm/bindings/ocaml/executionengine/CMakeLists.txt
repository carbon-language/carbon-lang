add_ocaml_library(llvm_executionengine
  OCAML    llvm_executionengine
  OCAMLDEP llvm llvm_target
  C        executionengine_ocaml
  LLVM     executionengine mcjit native
  PKG      ctypes)
