//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <thread>

// class thread::id

// template<class charT, class traits>
// basic_ostream<charT, traits>&
// operator<<(basic_ostream<charT, traits>& out, thread::id id);

#include <thread>
#include <sstream>
#include <cassert>

int main()
{
    std::thread::id id0 = std::this_thread::get_id();
    std::ostringstream os;
    os << id0;
}
