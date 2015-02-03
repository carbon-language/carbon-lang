//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11

// <experimental/system_error>

#include <experimental/system_error>

int main()
{
  // Check that <system_error> has been included
  std::error_code ec;
  ((void)ec);
}
