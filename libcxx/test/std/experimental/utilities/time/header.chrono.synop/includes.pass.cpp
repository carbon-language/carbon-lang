//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <experimental/chrono>

#include <experimental/chrono>

int main()
{
  // Check that <chrono> has been included.
  std::chrono::seconds s;
  ((void)s);
}
