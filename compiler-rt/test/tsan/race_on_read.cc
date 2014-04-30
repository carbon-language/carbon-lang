// RUN: %clangxx_tsan -O1 %s -o %t && not %run %t 2>&1 | FileCheck %s
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

int fd;
char buf;

void *Thread(void *x) {
  sleep(1);
  read(fd, &buf, 1);
  return NULL;
}

int main() {
  fd = open("/dev/random", O_RDONLY);
  if (fd < 0) {
    fprintf(stderr, "failed to open /dev/random (%d)\n", errno);
    return 1;
  }
  pthread_t t[2];
  pthread_create(&t[0], NULL, Thread, NULL);
  pthread_create(&t[1], NULL, Thread, NULL);
  pthread_join(t[0], NULL);
  pthread_join(t[1], NULL);
  close(fd);
  fprintf(stderr, "DONE\n");
}

// CHECK: WARNING: ThreadSanitizer: data race
// CHECK:   Write of size 1
// CHECK:     #0 read
// CHECK:   Previous write of size 1
// CHECK:     #0 read
// CHECK: DONE

