// RUN: %clangxx -O0 -g %s -o %t && %run %t
// UNSUPPORTED: linux
// fgetln is BSD-only.
// UNSUPPORTED: solaris

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
  FILE *fp = fopen("/etc/hosts", "r");
  assert(fp);

  size_t len;
  char *s = fgetln(fp, &len);

  printf("%.*s\n", (int)len, s);

  assert(!fclose(fp));

  return 0;
}
