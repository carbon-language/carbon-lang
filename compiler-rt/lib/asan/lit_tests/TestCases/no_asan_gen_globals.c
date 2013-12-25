// Make sure __asan_gen_* strings do not end up in the symbol table.

// RUN: %clang_asan %s -o %t.exe
// RUN: nm %t.exe | grep __asan_gen_ || exit 0

int x, y, z;
int main() { return 0; }
