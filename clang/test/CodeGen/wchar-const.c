// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck %s
// This should pass for any endianness combination of host and target.

// This bit is taken from Sema/wchar.c so we can avoid the wchar.h include.
typedef __WCHAR_TYPE__ wchar_t;
#if defined(_WIN32) || defined(_M_IX86) || defined(__CYGWIN__) \
  || defined(_M_X64) || defined(SHORT_WCHAR)
  #define WCHAR_T_TYPE unsigned short
#elif defined(__sun) || defined(__AuroraUX__)
  #define WCHAR_T_TYPE long
#else /* Solaris or AuroraUX. */
  #define WCHAR_T_TYPE int
#endif


// CHECK: @.str = private unnamed_addr constant [72 x i8] c"
extern void foo(const wchar_t* p);
int main (int argc, const char * argv[])
{
 foo(L"This is some text");
 return 0;
}
