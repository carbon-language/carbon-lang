set(LLVM_LINK_COMPONENTS Support)

add_tablegen(flang-tblgen FLANG
  FlangDiagnosticsEmitter.cpp
  FlangOptionDocEmitter.cpp
  TableGen.cpp
  )
