//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <chrono>

// duration

// If a program instantiates duration with a duration type for the template
// argument Rep a diagnostic is required.

#include <chrono>

int main()
{
    typedef std::chrono::duration<std::chrono::milliseconds> D;
    D d;
}
