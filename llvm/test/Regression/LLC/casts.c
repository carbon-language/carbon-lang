#include <stdlib.h>
#include <stdio.h>

#define __STDC_LIMIT_MACROS 1
#include <inttypes.h>


int
main(int argc, char** argv)
{
  int8_t        C, c1;
  uint8_t       uc1;

  short         S, s1;
  unsigned short us1;

  int           i1;
  unsigned      ui1;

  int64_t       L, l1;
  uint64_t      ul1;

  float         F;
  double        D;
  
  /* input values */
  C = (char)  (argc >= 2)? atoi(argv[1]) : 0x64;           /* 100 = 'd' */
  S = (short) (argc >= 3)? atoi(argv[2]) : -769;           /* 0xfcff = -769 */
  L = (long)  (argc >= 4)? atoi(argv[3]) : 0xa3a3a3a3a3a3; /*179923220407203*/
  
  /* Test integer to integer conversions */
  uc1 = (uint8_t) C;                       /* 100 = 'd' */
  us1 = (unsigned short) C;                /* 100 = 'd' */
  ui1 = (unsigned int) C;                  /* 100 = 'd' */
  ul1 = (unsigned long) C;                 /* 100 = 'd' */

  s1  = (short) C;                         /* 100 = 'd' */
  i1  = (int) C;                           /* 100 = 'd' */
  l1  = (long) C;                          /* 100 = 'd' */
  
  printf("\nCHAR             C = '%c' (%d)\t\t(0x%x)\n", C, C, C);
  printf("char to short   s1 = %d\t\t(0x%x)\n", s1, s1);
  printf("char to int     i1 = %d\t\t(0x%x)\n", i1, i1);
  printf("char to long    l1 = %ld\t\t(0x%lx)\n", l1, l1);

  printf("\nchar to ubyte  uc1 = %u\t\t(0x%x)\n", uc1, uc1);
  printf("char to ushort us1 = %u\t\t(0x%x)\n", us1, us1);
  printf("char to uint   ui1 = %u\t\t(0x%x)\n", ui1, ui1);
  printf("char to ulong  ul1 = %lu\t\t(0x%lx)\n", ul1, ul1);

  uc1 = (uint8_t) S;                            /* 0xff = 255 */
  us1 = (unsigned short) S;                     /* 0xfcff = 64767 */
  ui1 = (unsigned int) S;                       /* 0xfffffcff = 4294966527 */
  ul1 = (unsigned long) S;                      /* */

  c1  = (int8_t) S;                             /* 0xff = -1 */
  i1  = (int) S;                                /* 0xfffffcff = -769 */
  l1  = (long) S;                               /* */

  printf("\n\nSHORT            S = %d\t\t(0x%x)\n", S, S);
  printf("short to byte    c1 = %d\t\t(0x%x)\n", c1, c1);
  printf("short to int     i1 = %d\t\t(0x%x)\n", i1, i1);
  printf("short to long    l1 = %ld\t\t(0x%lx)\n", l1, l1);

  printf("\nshort to ubyte  uc1 = %u\t\t(0x%x)\n", uc1, uc1);
  printf("short to ushort us1 = %u\t\t(0x%x)\n", us1, us1);
  printf("short to uint   ui1 = %u\t\t(0x%x)\n", ui1, ui1);
  printf("short to ulong  ul1 = %lu\t\t(0x%lx)\n", ul1, ul1);

  uc1 = (unsigned char) L;                      /* */
  c1  = (int8_t) L;                             /* */
  s1  = (short) L;                              /* */
  us1 = (unsigned short) L;                     /* */
  i1  = (int) L;                                /* */
  ui1 = (unsigned int) L;                       /* */
  ul1 = (unsigned long) L;                       /* */

  printf("\n\nLONG            L = %ld\t\t(0x%lx)\n", L, L);
  printf("long to byte    c1 = %d\t\t(0x%x)\n", c1, c1);
  printf("long to short   s1 = %d\t\t(0x%x)\n", s1, s1);
  printf("long to int     i1 = %d\t\t(0x%x)\n", i1, i1);

  printf("\nlong to ubyte  uc1 = %u\t\t(0x%x)\n", uc1, uc1);
  printf("long to ushort us1 = %u\t\t(0x%x)\n", us1, us1);
  printf("long to uint   ui1 = %u\t\t(0x%x)\n", ui1, ui1);
  printf("long to ulong  ul1 = %lu\t\t(0x%lx)\n", ul1, ul1);

  /* Test floating-point to integer conversions */
  F = (float)  (argc >= 4)? atof(argv[3]) : 1.0;
  D =          (argc >= 5)? atof(argv[4]) : 2.0;
  
  us1 = (unsigned short) F;
  ui1 = (unsigned int) F;
  ul1 = (unsigned long) F;

  s1  = (short) F;
  i1  = (int) F;
  l1  = (long) F;

  printf("\n\nFLOAT            F = %f\n", F);
  printf("float to short   s1 = %d\t\t(0x%x)\n", s1, s1);
  printf("float to int     i1 = %d\t\t(0x%x)\n", i1, i1);

  printf("float to ushort us1 = %u\t\t(0x%x)\n", us1, us1);
  printf("float to uint   ui1 = %u\t\t(0x%x)\n", ui1, ui1);
  printf("float to ulong  ul1 = %lu\t\t(0x%lx)\n", ul1, ul1);

  us1 = (unsigned short) D;
  ui1 = (unsigned int) D;
  ul1 = (unsigned long) D;
  
  s1  = (short) D;
  i1  = (int) D;
  l1  = (long) D;

  printf("\n\nDOUBLE            D = %f\n", D);
  printf("double to short   s1 = %d\t\t(0x%x)\n", s1, s1);
  printf("double to int     i1 = %d\t\t(0x%x)\n", i1, i1);
  printf("double to long    l1 = %ld\t\t(0x%lx)\n", l1, l1);

  printf("double to ushort us1 = %u\t\t(0x%x)\n", us1, us1);
  printf("double to uint   ui1 = %u\t\t(0x%x)\n", ui1, ui1);
  printf("double to ulong  ul1 = %lu\t\t(0x%lx)\n", ul1, ul1);

  return 0;
}
