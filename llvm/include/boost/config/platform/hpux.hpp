//  (C) Copyright Boost.org 2001. Permission to copy, use, modify, sell and
//  distribute this software is granted provided this copyright notice appears
//  in all copies. This software is provided "as is" without express or implied
//  warranty, and with no claim as to its suitability for any purpose.

//  See http://www.boost.org for most recent version.

//  hpux specific config options:

#define BOOST_PLATFORM "HP-UX"

// In principle, HP-UX has a nice <stdint.h> under the name <inttypes.h>
// However, it has the following problem:
// Use of UINT32_C(0) results in "0u l" for the preprocessed source
// (verifyable with gcc 2.95.3, assumed for HP aCC)
// #define BOOST_HAS_STDINT_H

#define BOOST_NO_SWPRINTF 
#define BOOST_NO_CWCTYPE

// boilerplate code:
#define BOOST_HAS_UNISTD_H
#include <boost/config/posix_features.hpp>

#ifndef BOOST_HAS_GETTIMEOFDAY
// gettimeofday is always available
#define BOOST_HAS_GETTIMEOFDAY
#endif
