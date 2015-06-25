// RUN: %clangxx_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
#include "test.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

// dup2(oldfd, newfd) races with read(newfd).
// This is not reported as race because:
// 1. Some software dups a closed pipe in place of a socket before closing
//    the socket (to prevent races actually).
// 2. Some daemons dup /dev/null in place of stdin/stdout.

int fd;

void *Thread(void *x) {
  char buf;
  if (read(fd, &buf, 1) != 1)
    exit(printf("read failed\n"));
  return 0;
}

int main() {
  fd = open("/dev/random", O_RDONLY);
  int fd2 = open("/dev/random", O_RDONLY);
  if (fd == -1 || fd2 == -1)
    exit(printf("open failed\n"));
  pthread_t th;
  pthread_create(&th, 0, Thread, 0);
  if (dup2(fd2, fd) == -1)
    exit(printf("dup2 failed\n"));
  pthread_join(th, 0);
  if (close(fd) == -1)
    exit(printf("close failed\n"));
  if (close(fd2) == -1)
    exit(printf("close failed\n"));
  printf("DONE\n");
}

// CHECK-NOT: WARNING: ThreadSanitizer: data race
// CHECK: DONE
