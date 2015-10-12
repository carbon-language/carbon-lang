// RUN: %clangxx -O0 -g %s -lutil -o %t && %run %t
// This test depends on the glibc layout of struct sem_t and checks that we
// don't leave sem_t::private uninitialized.
// UNSUPPORTED: android
#include <assert.h>
#include <semaphore.h>
#include <string.h>

void my_sem_init(bool priv, int value, unsigned *a, unsigned char *b) {
  sem_t sem;
  memset(&sem, 0xAB, sizeof(sem));
  sem_init(&sem, priv, value);

  char *p = (char *)&sem;
  memcpy(a, p, sizeof(unsigned));
  memcpy(b, p + sizeof(unsigned), sizeof(char));

  sem_destroy(&sem);
}

int main() {
  unsigned a;
  unsigned char b;

  my_sem_init(false, 42, &a, &b);
  assert(a == 42);
  assert(b != 0xAB);

  my_sem_init(true, 43, &a, &b);
  assert(a == 43);
  assert(b != 0xAB);
}
