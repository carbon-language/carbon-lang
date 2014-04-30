// RUN: %clangxx_tsan -O1 %s -o %t && not %run %t 2>&1 | FileCheck %s
#include <pthread.h>
#include <stdio.h>
#include <stddef.h>
#include <unistd.h>

int GlobalData[10];
int x;
namespace XXX {
  struct YYY {
    static int ZZZ[10];
  };
  int YYY::ZZZ[10];
}

void *Thread(void *a) {
  sleep(1);
  GlobalData[2] = 42;
  x = 1;
  XXX::YYY::ZZZ[0] = 1;
  return 0;
}

int main() {
  fprintf(stderr, "addr=%p\n", GlobalData);
  fprintf(stderr, "addr2=%p\n", &x);
  fprintf(stderr, "addr3=%p\n", XXX::YYY::ZZZ);
  pthread_t t;
  pthread_create(&t, 0, Thread, 0);
  GlobalData[2] = 43;
  x = 0;
  XXX::YYY::ZZZ[0] = 0;
  pthread_join(t, 0);
}

// CHECK: addr=[[ADDR:0x[0-9,a-f]+]]
// CHECK: addr2=[[ADDR2:0x[0-9,a-f]+]]
// CHECK: addr3=[[ADDR3:0x[0-9,a-f]+]]
// CHECK: WARNING: ThreadSanitizer: data race
// CHECK: Location is global 'GlobalData' of size 40 at [[ADDR]] ({{.*}}+0x{{[0-9,a-f]+}})
// CHECK: WARNING: ThreadSanitizer: data race
// CHECK: Location is global 'x' of size 4 at [[ADDR2]] ({{.*}}+0x{{[0-9,a-f]+}})
// CHECK: WARNING: ThreadSanitizer: data race
// CHECK: Location is global 'XXX::YYY::ZZZ' of size 40 at [[ADDR3]] ({{.*}}+0x{{[0-9,a-f]+}})
