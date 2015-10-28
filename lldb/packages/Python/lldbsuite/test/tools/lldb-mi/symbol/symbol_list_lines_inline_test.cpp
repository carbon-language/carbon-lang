// Skip lines so we can make sure we're not seeing any lines from
// symbol_list_lines_inline_test.h included in -symbol-list-lines
// symbol_list_lines_inline_test.cpp, by checking that all the lines
// are between 30 and 39.
// line 5
// line 6
// line 7
// line 8
// line 9
// line 10
// line 11
// line 12
// line 13
// line 14
// line 15
// line 16
// line 17
// line 18
// line 19
// line 20
// line 21
// line 22
// line 23
// line 24
// line 25
// line 26
// line 27
// line 28
// line 29
#include "symbol_list_lines_inline_test.h"
int
gfunc(int i)
{ // FUNC_gfunc
    return ns::ifunc(i);
}
namespace ns
{
S s; // STRUCT_s
}
