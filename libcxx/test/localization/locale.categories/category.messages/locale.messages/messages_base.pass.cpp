//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <locale>

// class messages_base
// {
// public:
//     typedef unspecified catalog;
// };

#include <locale>
#include <type_traits>

int main()
{
    std::messages_base mb;
}
