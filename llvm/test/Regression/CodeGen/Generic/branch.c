#include <stdio.h>
int a = 1, b = 2;

int main() {
  int i,j;
  for (i=15; i>=0; --i) {
    if (a < i) printf("%d < %d\n", a, i);
    else printf("%d >= %d\n", a, i);
    for (j=2; j <= 25; j++) {
      printf("%d, ", j);
    }
    printf("\n");
  }
  return 0; 
}
