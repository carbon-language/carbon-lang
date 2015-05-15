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

struct pcomplex_type : complex_type
{
    pcomplex_type(const complex_type &complx_base, const complex_type &complx_member)
        : complex_type(complx_base), complx(complx_member) { }
    complex_type complx;
    static int si;
};

int pcomplex_type::si;

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
var_list_children_test(void)
{
    complex_type complx = { 3, { 3L }, &complx };
    complex_type complx_array[2] = { { 4, { 4L }, &complx_array[1] }, { 5, { 5 }, &complx_array[0] } };
    pcomplex_type pcomplx({ 6, { 6L }, &pcomplx}, { 7, { 7L }, &pcomplx});

    // BP_var_list_children_test
}

void
gdb_set_show_print_char_array_as_string_test(void)
{
    const char *cp = "hello";
    const char ca[] = "hello";
    const char16_t *u16p = u"hello";
    const char16_t u16a[] = u"hello";
    const char32_t *u32p = U"hello";
    const char32_t u32a[] = U"hello";

    // BP_gdb_set_show_print_char_array_as_string_test
}

void
gdb_set_show_print_expand_aggregates(void)
{
    complex_type complx = { 3, { 3L }, &complx };
    complex_type complx_array[2] = { { 4, { 4L }, &complx_array[1] }, { 5, { 5 }, &complx_array[0] } };

    // BP_gdb_set_show_print_expand_aggregates
}

void
gdb_set_show_print_aggregate_field_names(void)
{
    complex_type complx = { 3, { 3L }, &complx };
    complex_type complx_array[2] = { { 4, { 4L }, &complx_array[1] }, { 5, { 5 }, &complx_array[0] } };

    // BP_gdb_set_show_print_aggregate_field_names
}

int g_MyVar = 3;
static int s_MyVar = 4;

int
main(int argc, char const *argv[])
{
    int a = 10, b = 20;
    s_MyVar = a + b;
    var_update_test();
    var_list_children_test();
    gdb_set_show_print_char_array_as_string_test();
    gdb_set_show_print_expand_aggregates();
    gdb_set_show_print_aggregate_field_names();
    return 0; // BP_return
}
