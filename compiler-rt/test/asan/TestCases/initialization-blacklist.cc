// Test for blacklist functionality of initialization-order checker.

// RUN: %clangxx_asan -O0 %s %p/Helpers/initialization-blacklist-extra.cc\
// RUN:   %p/Helpers/initialization-blacklist-extra2.cc \
// RUN:   -fsanitize-blacklist=%p/Helpers/initialization-blacklist.txt \
// RUN:   -fsanitize=init-order -o %t
// RUN: ASAN_OPTIONS=check_initialization_order=true %t 2>&1
// RUN: %clangxx_asan -O1 %s %p/Helpers/initialization-blacklist-extra.cc\
// RUN:   %p/Helpers/initialization-blacklist-extra2.cc \
// RUN:   -fsanitize-blacklist=%p/Helpers/initialization-blacklist.txt \
// RUN:   -fsanitize=init-order -o %t
// RUN: ASAN_OPTIONS=check_initialization_order=true %t 2>&1
// RUN: %clangxx_asan -O2 %s %p/Helpers/initialization-blacklist-extra.cc\
// RUN:   %p/Helpers/initialization-blacklist-extra2.cc \
// RUN:   -fsanitize-blacklist=%p/Helpers/initialization-blacklist.txt \
// RUN:   -fsanitize=init-order -o %t
// RUN: ASAN_OPTIONS=check_initialization_order=true %t 2>&1

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
