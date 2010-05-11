//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <iostream>

// istream wclog;

#include <iostream>

int main()
{
#if 0
    std::wclog << L"Hello World!\n";
#else
    (void)std::wclog;
#endif
}
