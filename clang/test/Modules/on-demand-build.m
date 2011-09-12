// RUN: mkdir -p %t
// RUN: rm -f %t/Module.pcm
// RUN: %clang_cc1 -fmodule-cache-path %t -F %S/Inputs -verify %s

__import_module__ Module;
void test_getModuleVersion() {
  int version = getModuleVersion(); // expected-warning{{incompatible pointer to integer conversion initializing 'int' with an expression of type 'const char *'}}
  int version2 = [Module version]; // expected-warning{{incompatible pointer to integer conversion initializing 'int' with an expression of type 'const char *'}}
}


