//  (C) Copyright Boost.org 2001. Permission to copy, use, modify, sell and
//  distribute this software is granted provided this copyright notice appears
//  in all copies. This software is provided "as is" without express or implied
//  warranty, and with no claim as to its suitability for any purpose.

//  See http://www.boost.org for most recent version.

//  Win32 specific config options:

#define BOOST_PLATFORM "Win32"

#if defined BOOST_DECL_EXPORTS
#  if defined BOOST_DECL_IMPORTS
#     error Not valid to define both BOOST_DECL_EXPORTS and BOOST_DECL_IMPORTS
#  endif
#  define BOOST_DECL __declspec(dllexport)
#elif defined BOOST_DECL_IMPORTS
#  define BOOST_DECL __declspec(dllimport)
#else
#  define BOOST_DECL
#endif

#if defined(__GNUC__) && !defined(BOOST_NO_SWPRINTF)
#  define BOOST_NO_SWPRINTF
#endif

#ifndef BOOST_DISABLE_WIN32
//
// Win32 will normally be using native Win32 threads,
// but there is a pthread library avaliable as an option:
//
#ifndef BOOST_HAS_PTHREADS
#  define BOOST_HAS_WINTHREADS
#endif

// WEK: Added
#define BOOST_HAS_FTIME

#endif
