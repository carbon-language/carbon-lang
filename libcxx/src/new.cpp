//===--------------------------- new.cpp ----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "new"

namespace std
{

bad_array_new_length::~bad_array_new_length() throw()
{
}

const char*
bad_array_new_length::what() const throw()
{
    return "bad_array_new_length";
}

void
__throw_bad_alloc()
{
    throw bad_alloc();
}

}  // std
