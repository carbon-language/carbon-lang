//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <iostream>

// istream cout;

#include <iostream>

int main()
{
#if 0
    std::cout << "Hello World!\n";
    int i;
    std::cout << "Enter a number: ";
    std::cin >> i;
    std::cout << "The number is : " << i << '\n';
#else  // 0
    (void)std::cout;
#endif
}
