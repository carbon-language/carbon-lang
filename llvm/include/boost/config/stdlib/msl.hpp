//  (C) Copyright Boost.org 2001. Permission to copy, use, modify, sell and
//  distribute this software is granted provided this copyright notice appears
//  in all copies. This software is provided "as is" without express or implied
//  warranty, and with no claim as to its suitability for any purpose.

//  See http://www.boost.org for most recent version.

//  Metrowerks standard library:

#ifndef __MSL_CPP__
#  include <utility>
#  ifndef __MSL_CPP__
#     error This is not the MSL standard library!
#  endif
#endif

#if __MSL_CPP__ >= 0x6000  // Pro 6
#  define BOOST_HAS_HASH
#  define BOOST_STD_EXTENSION_NAMESPACE Metrowerks
#endif
#define BOOST_HAS_SLIST

#if __MSL_CPP__ < 0x6209
#  define BOOST_NO_STD_MESSAGES
#endif

// check C lib version for <stdint.h>
#include <cstddef>

#if defined(__MSL__) && (__MSL__ >= 0x5000)
#  define BOOST_HAS_STDINT_H
#  define BOOST_HAS_UNISTD_H
   // boilerplate code:
#  include <boost/config/posix_features.hpp>
#endif

#if _MWMT
#  define BOOST_HAS_THREADS
#endif


#define BOOST_STDLIB "Metrowerks Standard Library version " BOOST_STRINGIZE(__MSL_CPP__)








