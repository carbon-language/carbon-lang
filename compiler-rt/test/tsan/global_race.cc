// RUN: %clangxx_tsan -O1 %s -o %T/global_race.cc.exe && %deflake %run %T/global_race.cc.exe | FileCheck %s
#include "test.h"

int GlobalData[10];

void *Thread(void *a) {
  barrier_wait(&barrier);
  GlobalData[2] = 42;
  return 0;
}

int main() {
  barrier_init(&barrier, 2);
  print_address("addr=", 1, GlobalData);
  pthread_t t;
  pthread_create(&t, 0, Thread, 0);
  GlobalData[2] = 43;
  barrier_wait(&barrier);
  pthread_join(t, 0);
}

// CHECK: addr=[[ADDR:0x[0-9,a-f]+]]
// CHECK: WARNING: ThreadSanitizer: data race
// CHECK: Location is global 'GlobalData' {{(of size 40 )?}}at [[ADDR]] (global_race.cc.exe+0x{{[0-9,a-f]+}})

