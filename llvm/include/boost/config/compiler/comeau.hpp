//  (C) Copyright Boost.org 2001. Permission to copy, use, modify, sell and
//  distribute this software is granted provided this copyright notice appears
//  in all copies. This software is provided "as is" without express or implied
//  warranty, and with no claim as to its suitability for any purpose.

//  See http://www.boost.org for most recent version.

//  Comeau C++ compiler setup:

#include "boost/config/compiler/common_edg.hpp"

#if (__COMO_VERSION__ <= 4245) || !defined(BOOST_STRICT_CONFIG)
#  if defined(_MSC_VER) && _MSC_VER <= 1300
#     define BOOST_NO_STDC_NAMESPACE
#     define BOOST_NO_SWPRINTF
#     if _MSC_VER > 100
         // only set this in non-strict mode:
#        define BOOST_NO_ARGUMENT_DEPENDENT_LOOKUP
#     endif
#  endif

// Void returns don't work when emulating VC 6 (Peter Dimov)

#  if defined(_MSC_VER) && (_MSC_VER == 1200)
#     define BOOST_NO_VOID_RETURNS
#  endif


#endif  // version 4245

//
// enable __int64 support in VC emulation mode
// we should also set BOOST_HAS_LONG_LONG when that is
// supported, but there is no way we can detect it:
//
#  if defined(_MSC_VER) && (_MSC_VER >= 1200)
#     define BOOST_HAS_MS_INT64
#  endif

//
// disable win32 support unless we are in VC emulation mode,
// (what does this do to Como on top of Borland?):
//
#if defined(_WIN32) && (_MSC_VER+0 < 1000)
#  define BOOST_DISABLE_WIN32
#endif

#define BOOST_COMPILER "Comeau compiler version " BOOST_STRINGIZE(__COMO_VERSION__)

//
// versions check:
// we don't know Comeau prior to version 4245:
#if __COMO_VERSION__ < 4245
#  error "Compiler not configured - please reconfigure"
#endif
//
// last known and checked version is 4245:
#if (__COMO_VERSION__ > 4245)
#  if defined(BOOST_ASSERT_CONFIG)
#     error "Unknown compiler version - please run the configure tests and report the results"
#  endif
#endif



