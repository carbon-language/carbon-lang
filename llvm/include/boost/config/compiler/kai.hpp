//  (C) Copyright Boost.org 2001. Permission to copy, use, modify, sell and
//  distribute this software is granted provided this copyright notice appears
//  in all copies. This software is provided "as is" without express or implied
//  warranty, and with no claim as to its suitability for any purpose.

//  See http://www.boost.org for most recent version.

//  Kai C++ compiler setup:

#include "boost/config/compiler/common_edg.hpp"

#   if (__KCC_VERSION <= 4001) || !defined(BOOST_STRICT_CONFIG)
      // at least on Sun, the contents of <cwchar> is not in namespace std
#     define BOOST_NO_STDC_NAMESPACE
#   endif

#define BOOST_COMPILER "Kai C++ version " BOOST_STRINGIZE(__KCC_VERSION)

//
// last known and checked version is 4001:
#if (__KCC_VERSION > 4001)
#  if defined(BOOST_ASSERT_CONFIG)
#     error "Unknown compiler version - please run the configure tests and report the results"
#  endif
#endif


