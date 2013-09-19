#include <stdio.h>
#include <unistd.h>

extern "C" void inc_global();

int slow_init() {
  sleep(1);
  inc_global();
  return 42;
}

int slowly_init_glob = slow_init();
