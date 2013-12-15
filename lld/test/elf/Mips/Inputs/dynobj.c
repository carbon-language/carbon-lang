// clang -O0 -EL -fPIC -target mipsel-linux-gnu -c dynobj.c -o dynobj.o
int xyz(const char *);
int abc(const char *);

int bar(void)
{
  return 1;
}

int foo(void)
{
  bar();
  return xyz("str1") + abc("str2");
}
