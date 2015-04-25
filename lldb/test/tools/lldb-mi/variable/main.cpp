//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <cstdint>

struct complex_type
{
    int i;
    struct { long l; } inner;
    complex_type *complex_ptr;
};

void
var_update_test(void)
{
    long l = 1;
    complex_type complx = { 3, { 3L }, &complx };
    complex_type complx_array[2] = { { 4, { 4L }, &complx_array[1] }, { 5, { 5 }, &complx_array[0] } };
    // BP_var_update_test_init

    l = 0;
    // BP_var_update_test_l

    complx.inner.l = 2;
    // BP_var_update_test_complx

    complx_array[1].inner.l = 4;
    // BP_var_update_test_complx_array
}

void
gdb_set_show_print_char_array_as_string_test(void)
{
    const char *string_ptr = "string - const char *";
    const char string_arr[] = "string - char []";

    // BP_gdb_set_show_print_char_array_as_string_test
}

int g_MyVar = 3;
static int s_MyVar = 4;

int
main(int argc, char const *argv[])
{
    int a = 10, b = 20;
    s_MyVar = a + b;
    var_update_test();
    gdb_set_show_print_char_array_as_string_test();
    return 0; // BP_return
}
