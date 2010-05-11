//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <iostream>

// istream wcin;

#include <iostream>
#include <cassert>

int main()
{
#if 0
    std::wcout << L"Hello World!\n";
    int i;
    std::wcout << L"Enter a number: ";
    std::wcin >> i;
    std::wcout << L"The number is : " << i << L'\n';
#else
    assert(std::wcin.tie() == &std::wcout);
#endif
}
