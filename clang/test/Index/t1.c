#include "foo.h"

void foo_func(int param1) {
  int local_var = global_var;
  for (int for_var = 100; for_var < 500; ++for_var) {
    local_var = param1 + for_var;
  }
  bar_func();
}
