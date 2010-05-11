//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// test <setjmp.h>

#include <setjmp.h>
#include <type_traits>

int main()
{
    jmp_buf jb;
    static_assert((std::is_same<__typeof__(longjmp(jb, 0)), void>::value),
                  "std::is_same<__typeof__(longjmp(jb, 0)), void>::value");
}
