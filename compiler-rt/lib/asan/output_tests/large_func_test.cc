#include <stdlib.h>
__attribute__((noinline))
static void LargeFunction(int *x, int zero) {
  x[0]++;
  x[1]++;
  x[2]++;
  x[3]++;
  x[4]++;
  x[5]++;
  x[6]++;
  x[7]++;
  x[8]++;
  x[9]++;

  x[zero + 111]++;  // we should report this exact line

  x[10]++;
  x[11]++;
  x[12]++;
  x[13]++;
  x[14]++;
  x[15]++;
  x[16]++;
  x[17]++;
  x[18]++;
  x[19]++;
}

int main(int argc, char **argv) {
  int *x = new int[100];
  LargeFunction(x, argc - 1);
  delete x;
}

// Check-Common: {{.*ERROR: AddressSanitizer heap-buffer-overflow on address}}
// Check-Common:   {{0x.* at pc 0x.* bp 0x.* sp 0x.*}}
// Check-Common: {{READ of size 4 at 0x.* thread T0}}
// Check-Common: {{    #0 0x.* in LargeFunction.*large_func_test.cc:15}}
// Check-Common: {{    #1 0x.* in main .*large_func_test.cc:3[012]}}
// Check-Common: {{0x.* is located 44 bytes to the right of 400-byte region}}
// Check-Common: {{allocated by thread T0 here:}}
// Check-Common: {{    #0 0x.* in operator new.*}}
// Check-Common: {{    #1 0x.* in main .*large_func_test.cc:30}}
