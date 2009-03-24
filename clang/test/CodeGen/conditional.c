// RUN: clang-cc -emit-llvm %s -o %t

float test1(int cond, float a, float b)
{
  return cond ? a : b;
}
double test2(int cond, float a, double b)
{
  return cond ? a : b;
}

void f();

void test3(){
   1 ? f() : (void)0;
}

void test4() {
int i; short j;
float* k = 1 ? &i : &j;
}

void test5() {
  const int* cip;
  void* vp;
  cip = 0 ? vp : cip;
}

void test6();
void test7(int);
void* test8() {return 1 ? test6 : test7;}


void _efree(void *ptr);

void _php_stream_free3()
{
	(1 ? free(0) : _efree(0));
}

void _php_stream_free4()
{
	1 ? _efree(0) : free(0);
}
