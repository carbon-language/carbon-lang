// -*- C++ -*-
//===--------------------------- support/win32/support.h --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

/*
   Functions and constants used in libc++ that are missing from the Windows C library.
  */

int vasprintf( char **sptr, const char *__restrict__ fmt , va_list ap );
