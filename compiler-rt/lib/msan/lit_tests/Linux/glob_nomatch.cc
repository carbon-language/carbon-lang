// RUN: %clangxx_msan -m64 -O0 %s -o %t && %t %p
// RUN: %clangxx_msan -m64 -O3 %s -o %t && %t %p

#include <assert.h>
#include <glob.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
  assert(argc == 2);
  char buf[1024];
  snprintf(buf, sizeof(buf), "%s/%s", argv[1], "glob_test_root/*c");

  glob_t globbuf;
  int res = glob(buf, 0, 0, &globbuf);
  assert(res == GLOB_NOMATCH);
  assert(globbuf.gl_pathc == 0);
  if (globbuf.gl_pathv == 0)
    exit(0);
  return 0;
}
