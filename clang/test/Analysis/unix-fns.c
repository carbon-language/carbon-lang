// RUN: %clang_cc1 -analyze -analyzer-check-objc-mem %s -analyzer-store=region
// RUN: %clang_cc1 -analyze -analyzer-check-objc-mem %s -analyzer-store=basic

#include <fcntl.h>

void test_open(const char *path) {
  int fd;
  fd = open(path, O_RDONLY); // no-warning
  if (!fd)
    close(fd);

  fd = open(path, O_CREAT); // expected-warning{{Call to 'open' requires a third argument when the 'O_CREAT' flag is set}}
  if (!fd)
    close(fd);
} 
