// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -analyze -analyzer-checker=unix.API,osx.API %s -analyzer-store=region -fblocks -verify

struct _opaque_pthread_once_t {
  long __sig;
  char __opaque[8];
};
typedef struct _opaque_pthread_once_t    __darwin_pthread_once_t;
typedef __darwin_pthread_once_t pthread_once_t;
int pthread_once(pthread_once_t *, void (*)(void));
typedef long unsigned int __darwin_size_t;
typedef __darwin_size_t size_t;
void *malloc(size_t);

typedef void (^dispatch_block_t)(void);
typedef long dispatch_once_t;
void dispatch_once(dispatch_once_t *predicate, dispatch_block_t block);

#ifndef O_CREAT
#define O_CREAT 0x0200
#define O_RDONLY 0x0000
#endif
int open(const char *, int, ...);
int close(int fildes);

void test_open(const char *path) {
  int fd;
  fd = open(path, O_RDONLY); // no-warning
  if (!fd)
    close(fd);

  fd = open(path, O_CREAT); // expected-warning{{Call to 'open' requires a third argument when the 'O_CREAT' flag is set}}
  if (!fd)
    close(fd);
} 

void test_dispatch_once() {
  dispatch_once_t pred = 0;
  do { if (__builtin_expect(*(&pred), ~0l) != ~0l) dispatch_once((&pred), (^() {})); } while (0); // expected-warning{{Call to 'dispatch_once' uses the local variable 'pred' for the predicate value}}
}
void test_dispatch_once_neg() {
  static dispatch_once_t pred = 0;
  do { if (__builtin_expect(*(&pred), ~0l) != ~0l) dispatch_once((&pred), (^() {})); } while (0); // no-warning
}

void test_pthread_once_aux();

void test_pthread_once() {
  pthread_once_t pred = {0x30B1BCBA, {0}};
  pthread_once(&pred, test_pthread_once_aux); // expected-warning{{Call to 'pthread_once' uses the local variable 'pred' for the "control" value}}
}
void test_pthread_once_neg() {
  static pthread_once_t pred = {0x30B1BCBA, {0}};
  pthread_once(&pred, test_pthread_once_aux); // no-warning
}

// PR 2899 - warn of zero-sized allocations to malloc().
void pr2899() {
  char* foo = malloc(0); // expected-warning{{Call to 'malloc' has an allocation size of 0 bytes}}
  for (unsigned i = 0; i < 100; i++) {
    foo[i] = 0;
  }
}
void pr2899_nowarn(size_t size) {
  char* foo = malloc(size); // no-warning
  for (unsigned i = 0; i < 100; i++) {
    foo[i] = 0;
  }
}
