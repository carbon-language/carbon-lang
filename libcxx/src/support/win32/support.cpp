// -*- C++ -*-
//===----------------------- support/win32/support.h ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <support/win32/support.h>
#include <stdarg.h> // va_start, va_end
#include <stddef.h> // size_t
#include <stdlib.h> // malloc
#include <stdio.h>  // vsprintf, vsnprintf
#include <string.h> // strcpy, wcsncpy

int asprintf(char **sptr, const char *__restrict fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    int result = vasprintf(sptr, fmt, ap);
    va_end(ap);
    return result;
}

// Like sprintf, but when return value >= 0 it returns a pointer to a malloc'd string in *sptr.
// If return >= 0, use free to delete *sptr.
int vasprintf( char **sptr, const char *__restrict fmt, va_list ap )
{
    *sptr = NULL;
    int count = vsnprintf( NULL, 0, fmt, ap ); // Query the buffer size required.
    if( count >= 0 ) {
        char* p = static_cast<char*>(malloc(count+1)); // Allocate memory for it and the terminator.
        if ( p == NULL )
            return -1;
        if ( vsnprintf( p, count+1, fmt, ap ) == count ) // We should have used exactly what was required.
            *sptr = p;
        else { // Otherwise something is wrong, likely a bug in vsnprintf. If so free the memory and report the error.
            free(p);
            return -1;
        }
    }

    return count;
}

// FIXME: use wcrtomb and avoid copy
// use mbsrtowcs which is available, first copy first nwc elements of src
size_t mbsnrtowcs( wchar_t *__restrict dst, const char **__restrict src,
                   size_t nmc, size_t len, mbstate_t *__restrict ps )
{
    char* local_src = new char[nmc+1];
    char* nmcsrc = local_src;
    strncpy( nmcsrc, *src, nmc );
    nmcsrc[nmc] = '\0';
    const size_t result = mbsrtowcs( dst, const_cast<const char **>(&nmcsrc), len, ps );
    // propagate error
    if( nmcsrc == NULL )
        *src = NULL;
    delete[] local_src;
    return result;
}
// FIXME: use wcrtomb and avoid copy
// use wcsrtombs which is available, first copy first nwc elements of src
size_t wcsnrtombs( char *__restrict dst, const wchar_t **__restrict src,
                   size_t nwc, size_t len, mbstate_t *__restrict ps )
{
    wchar_t* local_src = new wchar_t[nwc];
    wchar_t* nwcsrc = local_src;
    wcsncpy(nwcsrc, *src, nwc);
    nwcsrc[nwc] = '\0';
    const size_t result = wcsrtombs( dst, const_cast<const wchar_t **>(&nwcsrc), len, ps );
    // propogate error
    if( nwcsrc == NULL )
        *src = NULL;
    delete[] nwcsrc;
    return result;
}
