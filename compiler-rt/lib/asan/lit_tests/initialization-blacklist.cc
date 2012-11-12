// Test for blacklist functionality of initialization-order checker.

// RUN: %clangxx_asan -m64 -O0 %s %p/Helpers/initialization-blacklist-extra.cc\
// RUN:   -mllvm -asan-blacklist=%p/Helpers/initialization-blacklist.txt \
// RUN:   -mllvm -asan-initialization-order -o %t && %t 2>&1
// RUN: %clangxx_asan -m64 -O1 %s %p/Helpers/initialization-blacklist-extra.cc\
// RUN:   -mllvm -asan-blacklist=%p/Helpers/initialization-blacklist.txt \
// RUN:   -mllvm -asan-initialization-order -o %t && %t 2>&1
// RUN: %clangxx_asan -m64 -O2 %s %p/Helpers/initialization-blacklist-extra.cc\
// RUN:   -mllvm -asan-blacklist=%p/Helpers/initialization-blacklist.txt \
// RUN:   -mllvm -asan-initialization-order -o %t && %t 2>&1
// RUN: %clangxx_asan -m32 -O0 %s %p/Helpers/initialization-blacklist-extra.cc\
// RUN:   -mllvm -asan-blacklist=%p/Helpers/initialization-blacklist.txt \
// RUN:   -mllvm -asan-initialization-order -o %t && %t 2>&1
// RUN: %clangxx_asan -m32 -O1 %s %p/Helpers/initialization-blacklist-extra.cc\
// RUN:   -mllvm -asan-blacklist=%p/Helpers/initialization-blacklist.txt \
// RUN:   -mllvm -asan-initialization-order -o %t && %t 2>&1
// RUN: %clangxx_asan -m32 -O2 %s %p/Helpers/initialization-blacklist-extra.cc\
// RUN:   -mllvm -asan-blacklist=%p/Helpers/initialization-blacklist.txt \
// RUN:   -mllvm -asan-initialization-order -o %t && %t 2>&1

// Function is defined in another TU.
int readBadGlobal();
int x = readBadGlobal();  // init-order bug.

// Function is defined in another TU.
int accessBadObject();
int y = accessBadObject();  // init-order bug.

int main(int argc, char **argv) {
  return argc + x + y - 1;
}
