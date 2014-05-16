#include "instrprof-dynamic-header.h"
void a() {
  if (true) {
    bar<void>();
    bar<char>();
  }
}
