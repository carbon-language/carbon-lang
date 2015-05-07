//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

const char g_CharArray[] = "\x10\x11\x12\x13";
static const char s_CharArray[] = "\x20\x21\x22\x23";

void
local_array_test_inner()
{
    char array[] = { 0x01, 0x02, 0x03, 0x04 };
    char *first_element_ptr = &array[0];
    // BP_local_array_test_inner
    return;
}

void
local_array_test()
{
    char array[] = { 0x05, 0x06, 0x07, 0x08 };
    // BP_local_array_test
    local_array_test_inner();
    return;
}

int
main(int argc, char const *argv[])
{ // FUNC_main
    local_array_test();
    return 0;
}
