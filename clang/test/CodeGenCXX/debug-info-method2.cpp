// RUN: %clang_cc1 -flimit-debug-info -x c++ -g -S -emit-llvm < %s | FileCheck %s
// rdar://10336845
// Preserve type qualifiers in -flimit-debug-info mode.

// 720934 = DW_TAG_const_type | LLVMDebugVersion
// CHECK:  metadata !{i32 720934
class A {
public:
  int bar(int arg) const;
};

int A::bar(int arg) const{
  return arg+2;
}

int main() {
  A a;
  int i = a.bar(2);
  return i;
}
