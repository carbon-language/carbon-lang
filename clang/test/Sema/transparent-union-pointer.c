// RUN: clang-cc %s -fsyntax-only -verify

typedef union   {
  union wait *__uptr;
  int *__iptr;
} __WAIT_STATUS __attribute__ ((__transparent_union__));

extern int wait (__WAIT_STATUS __stat_loc);

void fastcgi_cleanup() {
  int status = 0;
  wait(&status);
}

