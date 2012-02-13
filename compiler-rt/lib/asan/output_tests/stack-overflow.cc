#include <string.h>
int main(int argc, char **argv) {
  char x[10];
  memset(x, 0, 10);
  int res = x[argc * 10];  // BOOOM
  return res;
}

// Check-Common: {{READ of size 1 at 0x.* thread T0}}
// Check-Common: {{    #0 0x.* in main .*stack-overflow.cc:5}}
// Check-Common: {{Address 0x.* is .* frame <main>}}
