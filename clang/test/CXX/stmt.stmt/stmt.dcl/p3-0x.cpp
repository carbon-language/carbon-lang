// RUN: %clang_cc1 -std=c++0x -fsyntax-only -verify %s

// PR10034
struct X {};

void exx(X) {}

int test_ptr10034(int argc, char **argv)
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

struct Y {
  ~Y();
};

void f();
void test_Y() {
  goto end;
  Y y;
 end:
  f();
  goto inner;
  {
    Y y2;
  inner:
    f();    
  }
  return;
}

struct Z {
  Z operator=(const Z&);
};

void test_Z() {
  goto end;
  Z z;
 end:
  return;
}
