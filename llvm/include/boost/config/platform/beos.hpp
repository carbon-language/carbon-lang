//  (C) Copyright Boost.org 2001. Permission to copy, use, modify, sell and
//  distribute this software is granted provided this copyright notice appears
//  in all copies. This software is provided "as is" without express or implied
//  warranty, and with no claim as to its suitability for any purpose.

//  See http://www.boost.org for most recent version.

//  BeOS specific config options:

#define BOOST_PLATFORM "BeOS"

#define BOOST_NO_CWCHAR
#define BOOST_NO_CWCTYPE
#define BOOST_HAS_UNISTD_H

#define BOOST_HAS_BETHREADS

#ifndef BOOST_DISABLE_THREADS
#  define BOOST_HAS_THREADS
#endif

// boilerplate code:
#include <boost/config/posix_features.hpp>
 

