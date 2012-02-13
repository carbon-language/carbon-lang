#include <string.h>
int main(int argc, char **argv) {
  static char XXX[10];
  static char YYY[10];
  static char ZZZ[10];
  memset(XXX, 0, 10);
  memset(YYY, 0, 10);
  memset(ZZZ, 0, 10);
  int res = YYY[argc * 10];  // BOOOM
  // Check-Common: {{READ of size 1 at 0x.* thread T0}}
  // Check-Common: {{    #0 0x.* in main .*global-overflow.cc:9}}
  // Check-Common: {{0x.* is located 0 bytes to the right of global variable}}
  // Check-Common:   {{.*YYY.* of size 10}}
  res += XXX[argc] + ZZZ[argc];
  return res;
}
