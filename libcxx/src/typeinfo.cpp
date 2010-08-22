//===------------------------- typeinfo.cpp -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include <stdlib.h>

#include "typeinfo"

std::bad_cast::bad_cast() throw()
{
}

std::bad_cast::~bad_cast() throw()
{
}

const char*
std::bad_cast::what() const throw()
{
  return "std::bad_cast";
}

std::bad_typeid::bad_typeid() throw()
{
}

std::bad_typeid::~bad_typeid() throw()
{
}

const char*
std::bad_typeid::what() const throw()
{
  return "std::bad_typeid";
}
