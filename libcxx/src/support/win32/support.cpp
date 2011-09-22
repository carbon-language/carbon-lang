// -*- C++ -*-
//===--------------------------- support/win32/support.h --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

int vasprintf( char **sptr, const char *__restrict__ fmt, va_list ap )
{
    *sptr = NULL
    int count = vsnprintf( *sptr, 0, fmt, ap );
    if( (count >= 0) && ((*sptr = malloc(count+1)) != NULL) )
    {
        vsprintf( *sptr, fmt, ap );
        sptr[count] = '\0';
    }

    return count;
}
