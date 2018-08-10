// RUN: %clang_safestack %s -pthread -o %t
// RUN: not --crash %run %t

// Test that unsafe stacks are deallocated correctly on thread exit.

#include <stdlib.h>
#include <string.h>
#include <pthread.h>

enum { kBufferSize = (1 << 15) };

void *t1_start(void *ptr)
{
  char buffer[kBufferSize];
  return buffer;
}

int main(int argc, char **argv)
{
  pthread_t t1;
  char *buffer = NULL;

  if (pthread_create(&t1, NULL, t1_start, NULL))
    abort();
  if (pthread_join(t1, &buffer))
    abort();

  // should segfault here
  memset(buffer, 0, kBufferSize);
  return 0;
}
