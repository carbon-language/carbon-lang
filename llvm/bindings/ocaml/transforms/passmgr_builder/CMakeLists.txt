add_ocaml_library(llvm_passmgr_builder
  OCAML    llvm_passmgr_builder
  OCAMLDEP llvm
  C        passmgr_builder_ocaml
  LLVM     ipo)
