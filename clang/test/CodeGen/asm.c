// RUN: clang -emit-llvm %s -o %t -arch=i386 
void t1(int len)
{
  __asm__ volatile("" : "=&r"(len), "+&r"(len));
}

void t2(unsigned long long t) 
{
  __asm__ volatile("" : "+m"(t));
}

void t3(unsigned char *src, unsigned long long temp)
{
  __asm__ volatile("" : "+m"(temp), "+r"(src));
}

void t4()
{
  unsigned long long a;
  struct reg { unsigned long long a, b; } b;

	__asm__ volatile ("":: "m"(a), "m"(b));
}





