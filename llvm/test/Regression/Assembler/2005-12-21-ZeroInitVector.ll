; RUN: llvm-upgrade < %s | llvm-as > /dev/null

< 4 x int> %foo() {
  ret <4 x int> zeroinitializer
}
