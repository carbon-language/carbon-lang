// -*- C++ -*-
//===-------------------------- algorithm ---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// test <errno.h>

#include <errno.h>

#ifndef EDOM
#error EDOM not defined
#endif

#ifndef EILSEQ
#error EILSEQ not defined
#endif

#ifndef ERANGE
#error ERANGE not defined
#endif

#ifndef errno
#error errno not defined
#endif

int main()
{
}
