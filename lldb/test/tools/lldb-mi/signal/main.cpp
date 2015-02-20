//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <cstddef>
#include <unistd.h>

int do_loop;
int do_segfault;

int
main(int argc, char const *argv[])
{
    if (do_loop)
    {
        do
            sleep(1);
        while (do_loop); // BP_loop_condition
    }

    if (do_segfault)
    {
        int *null_ptr = NULL;
        return *null_ptr;
    }

    return 0;
}
