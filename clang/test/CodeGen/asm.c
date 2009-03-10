// RUN: clang -emit-llvm %s -o %t -arch=i386 &&
void t1(int len) {
  __asm__ volatile("" : "=&r"(len), "+&r"(len));
}

void t2(unsigned long long t)  {
  __asm__ volatile("" : "+m"(t));
}

void t3(unsigned char *src, unsigned long long temp) {
  __asm__ volatile("" : "+m"(temp), "+r"(src));
}

void t4() {
  unsigned long long a;
  struct reg { unsigned long long a, b; } b;

	__asm__ volatile ("":: "m"(a), "m"(b));
}

// PR3417
void t5(int i) {
  asm("nop" : "=r"(i) : "0"(t5));
}

// PR3641
void t6(void) {
  __asm__ volatile("" : : "i" (t6));
}

// RUN: grep "T7 NAMED: \$2" %t
void t7(int a) {
  __asm__ volatile("T7 NAMED: %[input]" : "+r"(a): [input] "i" (4));
}