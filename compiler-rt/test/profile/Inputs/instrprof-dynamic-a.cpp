#include "instrprof-dynamic-header.h"
void a() {
  if (true) {
    bar<void>(1);
    bar<char>(1);
  }
}
