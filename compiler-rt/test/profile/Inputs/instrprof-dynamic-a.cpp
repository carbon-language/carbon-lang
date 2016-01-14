#include "instrprof-dynamic-header.h"
void a() {                             // COV: 1| [[@LINE]]|void a
  if (true) {                          // COV: 1| [[@LINE]]|  if
    bar<void>(1);                      // COV: 1| [[@LINE]]|    bar
    bar<char>(1);                      // COV: 1| [[@LINE]]|    bar
  }                                    // COV: 1| [[@LINE]]|  }
}
