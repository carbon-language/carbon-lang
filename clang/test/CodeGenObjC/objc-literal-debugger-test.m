// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fdebugger-objc-literal -emit-llvm -o - %s | FileCheck %s

int main() {
  id l = @'a';
  l = @'a';
  l = @42;
  l = @-42;
  l = @42u;
  l = @3.141592654f;
  l = @__objc_yes;
  l = @__objc_no;
  l = @{ @"name":@666 };
  l = @[ @"foo", @"bar" ];
}

// CHECK: declare i8* @objc_msgSend(i8*, i8*, ...) nonlazybind
