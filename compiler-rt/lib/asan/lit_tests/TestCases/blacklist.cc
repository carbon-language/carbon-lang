// Test the blacklist functionality of ASan

// RUN: echo "fun:*brokenFunction*" > %tmp
// RUN: echo "global:*badGlobal*" >> %tmp
// RUN: echo "src:*blacklist-extra.cc" >> %tmp
// RUN: %clangxx_asan -fsanitize-blacklist=%tmp -m64 -O0 %s -o %t \
// RUN: %p/Helpers/blacklist-extra.cc && %t 2>&1
// RUN: %clangxx_asan -fsanitize-blacklist=%tmp -m64 -O1 %s -o %t \
// RUN: %p/Helpers/blacklist-extra.cc && %t 2>&1
// RUN: %clangxx_asan -fsanitize-blacklist=%tmp -m64 -O2 %s -o %t \
// RUN: %p/Helpers/blacklist-extra.cc && %t 2>&1
// RUN: %clangxx_asan -fsanitize-blacklist=%tmp -m64 -O3 %s -o %t \
// RUN: %p/Helpers/blacklist-extra.cc && %t 2>&1
// RUN: %clangxx_asan -fsanitize-blacklist=%tmp -m32 -O0 %s -o %t \
// RUN: %p/Helpers/blacklist-extra.cc && %t 2>&1
// RUN: %clangxx_asan -fsanitize-blacklist=%tmp -m32 -O1 %s -o %t \
// RUN: %p/Helpers/blacklist-extra.cc && %t 2>&1
// RUN: %clangxx_asan -fsanitize-blacklist=%tmp -m32 -O2 %s -o %t \
// RUN: %p/Helpers/blacklist-extra.cc && %t 2>&1
// RUN: %clangxx_asan -fsanitize-blacklist=%tmp -m32 -O3 %s -o %t \
// RUN: %p/Helpers/blacklist-extra.cc && %t 2>&1

// badGlobal is accessed improperly, but we blacklisted it.
int badGlobal;
int readBadGlobal() {
  return (&badGlobal)[1];
}

// A function which is broken, but excluded in the blacklist.
int brokenFunction(int argc) {
  char x[10] = {0};
  return x[argc * 10];  // BOOM
}

// This function is defined in Helpers/blacklist-extra.cc, a source file which
// is blacklisted by name
int externalBrokenFunction(int x);

int main(int argc, char **argv) {
  brokenFunction(argc);
  int x = readBadGlobal();
  externalBrokenFunction(argc);
  return 0;
}
