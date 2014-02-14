// Indirectly call a bunch of functions.

#include <assert.h>

extern int cnt;

typedef int (*F)(int, int);

// A function in the same object.
int f_local(int x, int y) {
  return x + y;
}

// A function in another object.
int f_other_object(int x, int y);

// A function in another DSO.
int f_dso(int x, int y);

// A function in another DSO that is replaced by the wrapper.
int f_replaced(int x, int y);

void run_test(void) {
  int x;
  int expected_cnt = 0;
  volatile F f;

  if (SLOW) ++expected_cnt;
  f = &f_local;
  x = f(1, 2);
  assert(x == 3);
  assert(cnt == expected_cnt);

  if (SLOW) ++expected_cnt;
  f = &f_other_object;
  x = f(2, 3);
  assert(x == 6);
  assert(cnt == expected_cnt);

  ++expected_cnt;
  f = &f_dso;
  x = f(2, 3);
  assert(x == 7);
  assert(cnt == expected_cnt);

  ++expected_cnt;
  f = &f_replaced;
  x = f(2, 3);
  assert(x == 11);
  assert(cnt == expected_cnt);
}
