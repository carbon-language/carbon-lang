// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodule-cache-path %t -F %S/Inputs -DFOO -verify %s
// RUN: %clang_cc1 -x objective-c++ -fmodule-cache-path %t -F %S/Inputs -DFOO -verify %s
// RUN: %clang_cc1 -fmodule-cache-path %t -F %S/Inputs -DFOO -verify %s

__import_module__ Module;
void test_getModuleVersion() {
  const char *version = getModuleVersion();
  const char *version2 = [Module version];
}


