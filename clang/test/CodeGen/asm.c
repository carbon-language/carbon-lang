// RUN: clang %s -arch=i386 -verify -fsyntax-only
void f(int len)
{
  __asm__ volatile("" :"=&r"(len), "+&r"(len));
}
