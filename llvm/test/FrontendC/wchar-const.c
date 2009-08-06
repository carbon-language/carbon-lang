// RUN: %llvmgcc -S %s -o - | grep {constant \\\[18 x} | grep { 84, }
// This should pass for any endianness combination of host and target.
#include <ctype.h>
extern void foo(const wchar_t* p);
int main (int argc, const char * argv[])
{
 foo(L"This is some text");
 return 0;
}
