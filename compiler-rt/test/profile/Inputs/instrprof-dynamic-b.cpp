#include "instrprof-dynamic-header.h"
void b() {
  if (true) {
    bar<void>();
    bar<int>();
  }
}
