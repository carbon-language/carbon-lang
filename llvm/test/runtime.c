#include <stdio.h>
#include <sys/types.h>

void
ll__main()
{ /* Empty version of GCC's initialization function */
}

void
printSByte(char c)
{
  putchar(c);
}

void
printUByte(unsigned char c)
{
  putchar(c);
}

void
printShort(short i)
{
  printf("%d", i);
}

void
printUShort(unsigned short i)
{
  printf("%d", i);
}

void
printInt(int i)
{
  printf("%d", i);
}

void
printUInt(unsigned int i)
{
  printf("%d", i);
}

void
printLong(int64_t l)
{
  printf("%d", l);
}

void
printULong(uint64_t l)
{
  printf("%d", l);
}

void
printString(const char* str)
{
  printf("%s", str);
}

void
printFloat(float f)
{
  printf("%g", f);
}

void
printDouble(double d)
{
  printf("%g", d);
}

void
printPointer(void* p)
{
  printf("0x%x", p);
}

#undef  TEST_RUNTIME
#ifdef  TEST_RUNTIME
int
main(int argc, char** argv)
{
  int i;
  printString("argc = ");
  printLong(argc);
  printString(" = (as float) ");
  printFloat(argc * 1.0);
  printString(" = (as double) ");
  printDouble(argc * 1.0);
  for (i=0; i < argc; i++)
    {
      printString("\nargv[");
      printLong(i);
      printString("] = ");
      printString(argv[i]);
      printString("\t@ ");
      printPointer(argv[i]);
    }
  printString("\n");
}
#endif
