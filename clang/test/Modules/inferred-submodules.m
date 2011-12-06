// RUN: rm -rf %t
// RUN: %clang_cc1 -x objective-c -Wauto-import -fmodule-cache-path %t -fauto-module-import -F %S/Inputs %s -verify

__import_module__ Module.Sub;

void test_Module_Sub() {
  int *ip = Module_Sub;
}

__import_module__ Module.Buried.Treasure;

void dig() {
  unsigned *up = Buried_Treasure;
}

