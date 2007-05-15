// Test case for PR1420
// RUN: %llvmgxx %s -O0 -o %t.exe 
// RUN: %t.exe > %t.out
// RUN: grep {sizeof(bitFieldStruct) == 8} %t.out
// RUN: grep {Offset bitFieldStruct.i = 0} %t.out
// RUN: grep {Offset bitFieldStruct.c2 = 7} %t.out
// XFAIL: *

#include <stdio.h>

class bitFieldStruct {
  public:
    int i;
    unsigned char c:7;
    int s:17;
    char c2;
};

int main()
{
  printf("sizeof(bitFieldStruct) == %d\n", sizeof(bitFieldStruct));

  if (sizeof(bitFieldStruct) != 2 * sizeof(int))
    printf("bitFieldStruct should be %d but is %d \n", 
            2 * sizeof(int), sizeof(bitFieldStruct));

  bitFieldStruct x;
  
  char* xip = (char*) &x.i;
  char* xc2p = (char*) &x.c2;
  printf("Offset bitFieldStruct.i = %d\n", xip - xip);
  printf("Offset bitFieldStruct.c2 = %d\n", xc2p - xip);

  return 0;
}
