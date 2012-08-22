// RUN: %clangxx_asan -O2 %s -o %t && %t 2>&1 | FileCheck %s

#include <stdio.h>
#include <stdlib.h>

bool MySymbolizer(const void *pc, char *out_buffer, int out_size) {
  snprintf(out_buffer, out_size, "MySymbolizer");
  return true;
}

typedef bool (*asan_symbolize_callback)(const void*, char*, int);
extern "C"
void __asan_set_symbolize_callback(asan_symbolize_callback);

int main() {
  __asan_set_symbolize_callback(MySymbolizer);
  char *x = (char*)malloc(10 * sizeof(char));
  free(x);
  return x[5];
  // CHECK: MySymbolizer
}
