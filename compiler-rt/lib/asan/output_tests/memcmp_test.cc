#include <string.h>
int main(int argc, char **argv) {
  char a1[] = {argc, 2, 3, 4};
  char a2[] = {1, 2*argc, 3, 4};
// Check-Common: AddressSanitizer stack-buffer-overflow
// Check-Common: {{#0.*memcmp}}
// Check-Common: {{#1.*main}}
  int res = memcmp(a1, a2, 4 + argc);  // BOOM
  return res;
}
