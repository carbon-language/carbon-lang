//===-- trampoline_setup_test.c - Test __trampoline_setup -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <sys/mman.h>

//
// Tests nested functions
// The ppc compiler generates a call to __trampoline_setup
// The i386 and x86_64 compilers generate a call to ___enable_execute_stack
//

typedef int (*nested_func_t)(int x);

nested_func_t proc;

int main()
{
    // some locals
    int c = 10;
    int d = 7;
    
    // define a nested function
    int bar(int x) { return x*5 + c*d; };

    // assign global to point to nested function
    // (really points to trampoline)
    proc = bar;
    
    // invoke nested function
    c = 4;
    if ( (*proc)(3) != 43 )
        return 1;
    d = 5;
    if ( (*proc)(4) != 40 )
        return 1;

    // success
    return 0;
}
