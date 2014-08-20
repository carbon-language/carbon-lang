// RUN: %clang_cc1 -analyze -analyzer-checker=core,unix.API -verify %s

#ifndef O_RDONLY
#define O_RDONLY 0
#endif

#ifndef NULL
#define NULL ((void*) 0)
#endif

int open(const char *, int, ...);
int close(int fildes);

void open_1(const char *path) {
  int fd;
  fd = open(path, O_RDONLY); // no-warning
  if (fd > -1)
    close(fd);
}

void open_2(const char *path) {
  int fd;
  int mode = 0x0;
  fd = open(path, O_RDONLY, mode, NULL); // expected-warning{{Call to 'open' with more than three arguments}}
  if (fd > -1)
    close(fd);
}
