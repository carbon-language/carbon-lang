// RUN: %clang_cc1 %s -fsyntax-only -Wno-unused-value -Wmicrosoft -fms-extensions -verify 


void f(long long);
void f(int);
 
int main()
{
  // This is an ambiguous call in standard C++.
  // This calls f(long long) in Microsoft mode because LL is always signed.
  f(0xffffffffffffffffLL);
  f(0xffffffffffffffffi64);
}
