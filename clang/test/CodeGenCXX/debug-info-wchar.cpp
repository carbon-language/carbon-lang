// RUN: %clang_cc1 -emit-llvm -g %s -o -| FileCheck %s
void foo() {
// CHECK:  !"0x24\00wchar_t\00{{.*}}", null, null} ; [ DW_TAG_base_type ] [wchar_t]
  const wchar_t w = L'x';
}
