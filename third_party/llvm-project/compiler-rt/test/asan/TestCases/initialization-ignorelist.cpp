// Test for ignorelist functionality of initialization-order checker.

// RUN: %clangxx_asan -O0 %s %p/Helpers/initialization-ignorelist-extra.cpp\
// RUN:   %p/Helpers/initialization-ignorelist-extra2.cpp \
// RUN:   -fsanitize-ignorelist=%p/Helpers/initialization-ignorelist.txt -o %t
// RUN: %env_asan_opts=check_initialization_order=true %run %t 2>&1
// RUN: %clangxx_asan -O1 %s %p/Helpers/initialization-ignorelist-extra.cpp\
// RUN:   %p/Helpers/initialization-ignorelist-extra2.cpp \
// RUN:   -fsanitize-ignorelist=%p/Helpers/initialization-ignorelist.txt -o %t
// RUN: %env_asan_opts=check_initialization_order=true %run %t 2>&1
// RUN: %clangxx_asan -O2 %s %p/Helpers/initialization-ignorelist-extra.cpp\
// RUN:   %p/Helpers/initialization-ignorelist-extra2.cpp \
// RUN:   -fsanitize-ignorelist=%p/Helpers/initialization-ignorelist.txt -o %t
// RUN: %env_asan_opts=check_initialization_order=true %run %t 2>&1

// Function is defined in another TU.
int readBadGlobal();
int x = readBadGlobal();  // init-order bug.

// Function is defined in another TU.
int accessBadObject();
int y = accessBadObject();  // init-order bug.

int readBadSrcGlobal();
int z = readBadSrcGlobal();  // init-order bug.

int main(int argc, char **argv) {
  return argc + x + y + z - 1;
}
