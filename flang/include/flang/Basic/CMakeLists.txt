# README: The content of the various .td tiles was copied verbatim  from Clang. 
# The purpose is to show how TableGen can be used to create include files.
# If we decide to reuse the same diagnostic infrastrure than Clang then the 
# .td files will eventually be adapted for Flang. 
# The directory .../clang/include/clang/Basic  contains more examples.
# The generated file can be found in .../build/include/flang/Basic/
# 

macro(flang_diag_gen component)
  flang_tablegen(Diagnostic${component}Kinds.inc
    -gen-flang-diags-defs -flang-component=${component}
    SOURCE Diagnostic.td
    TARGET FlangDiagnostic${component})
endmacro(flang_diag_gen)

flang_diag_gen(Common)


