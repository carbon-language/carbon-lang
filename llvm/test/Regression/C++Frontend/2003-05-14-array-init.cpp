#include <stdio.h>

extern int table[];

int main() {
  printf("%d %d %d %d\n", table[0], table[1], table[2], table[3]);
  return table[1];
}

int table[] = { 1, 0, 3, 4 };
