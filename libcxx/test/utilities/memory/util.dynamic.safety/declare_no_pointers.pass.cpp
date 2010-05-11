//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <memory>

// void declare_no_pointers(char* p, size_t n);
// void undeclare_no_pointers(char* p, size_t n);

#include <memory>

int main()
{
    char* p = new char[10];
    std::declare_no_pointers(p, 10);
    std::undeclare_no_pointers(p, 10);
    delete [] p;
}
