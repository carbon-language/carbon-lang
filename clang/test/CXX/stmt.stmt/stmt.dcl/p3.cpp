// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++0x -fsyntax-only -verify %s

// PR10034
struct X {};

void exx(X) {}

int main(int argc, char **argv)
{
 if (argc > 3)
   goto end;

 X x;
 X xs[16];
 exx(x);

 end:
   if (argc > 1) {
   for (int i = 0; i < argc; ++i)
   {

   }
   }
   return 0;
}
