#include <time.h>

static void process_sleep(int sec) {
  clock_t beg = clock();
  while((clock() - beg) / CLOCKS_PER_SEC < sec)
    usleep(100);
}
