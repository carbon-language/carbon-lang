// RUN: %clangxx_asan -O0 %s -DTEMP_DIR='"'"%T"'"' -o %t && %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O1 %s -DTEMP_DIR='"'"%T"'"' -o %t && %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O2 %s -DTEMP_DIR='"'"%T"'"' -o %t && %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O3 %s -DTEMP_DIR='"'"%T"'"' -o %t && %t 2>&1 | FileCheck %s

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <dirent.h>


int main() {
  // Ensure the readdir_r interceptor doesn't erroneously mark the entire dirent
  // as written when the end of the directory pointer is reached.
  fputs("reading the " TEMP_DIR " directory...\n", stderr);
  DIR *d = opendir(TEMP_DIR);
  struct dirent entry, *result;
  unsigned count = 0;
  do {
    // Stamp the entry struct to try to trick the interceptor.
    entry.d_reclen = 9999;
    if (readdir_r(d, &entry, &result) != 0)
      abort();
    ++count;
  } while (result != NULL);
  fprintf(stderr, "read %d entries\n", count);
  // CHECK: reading the {{.*}} directory...
  // CHECK-NOT: stack-buffer-overflow
  // CHECK: read {{.*}} entries
}
