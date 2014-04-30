// RUN: %clangxx_msan -m64 -O0 -g -xc++ %s -o %t && %run %t
// RUN: %clangxx_msan -m64 -O3 -g -xc++ %s -o %t && %run %t

#include <stdio.h>
#include <stdlib.h>

int main(void) {
  char *buf;
  size_t buf_len = 42;
  FILE *fp = open_memstream(&buf, &buf_len);
  fprintf(fp, "hello");
  fflush(fp);
  printf("buf_len = %zu\n", buf_len);
  for (int j = 0; j < buf_len; j++) {
    printf("buf[%d] = %c\n", j, buf[j]);
  }
  return 0;
}
