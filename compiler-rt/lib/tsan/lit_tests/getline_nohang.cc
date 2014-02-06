// RUN: %clangxx_tsan -O1 %s -o %t && TSAN_OPTIONS=report_thread_leaks=0 %t

// Make sure TSan doesn't deadlock on a file stream lock at program shutdown.
// See https://code.google.com/p/thread-sanitizer/issues/detail?id=47
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>

void *thread(void *unused) {
  char *line = NULL;
  size_t size;
  int fd[2];
  pipe(fd);
  // Forge a non-standard stream to make sure it's not closed.
  FILE *stream = fdopen(fd[0], "r");
  while (1) {
    volatile int res = getline(&line, &size, stream);
    (void)res;
  }
  return NULL;
}

int main() {
  pthread_t t;
  pthread_create(&t, NULL, thread, NULL);
  return 0;
  // ThreadSanitizer used to hang here because of a deadlock on a file stream.
}
