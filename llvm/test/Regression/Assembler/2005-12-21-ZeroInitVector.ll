; RUN: llvm-as < %s > /dev/null

< 4 x int> %foo() {
  ret <4 x int> zeroinitializer
}
