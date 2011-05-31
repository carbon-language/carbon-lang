// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo '[{"directory":".","command":"clang++ '$(llvm-config --cppflags all)' -c %s","file":"%s"}]' > %t/compile_commands.json
// RUN: remove-cstr-calls %t %s | FileCheck %s

#include <string>

namespace llvm { struct StringRef { StringRef(const char *p); }; }

void f1(const std::string &s) {
  f1(s.c_str());  // CHECK:remove-cstr-calls.cpp:11:6:11:14:s
}
void f2(const llvm::StringRef r) {
  std::string s;
  f2(s.c_str());  // CHECK:remove-cstr-calls.cpp:15:6:15:14:s
}
void f3(const llvm::StringRef &r) {
  std::string s;
  f3(s.c_str());  // CHECK:remove-cstr-calls.cpp:19:6:19:14:s
}
