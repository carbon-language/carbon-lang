//  (C) Copyright Boost.org 2001. Permission to copy, use, modify, sell and
//  distribute this software is granted provided this copyright notice appears
//  in all copies. This software is provided "as is" without express or implied
//  warranty, and with no claim as to its suitability for any purpose.

//  See http://www.boost.org for most recent version.

//  Intel compiler setup:

#include "boost/config/compiler/common_edg.hpp"

#ifdef __ICL
#  define BOOST_COMPILER "Intel C++ version " BOOST_STRINGIZE(__ICL)
#  define BOOST_INTEL_CXX_VERSION __ICL
#else
#  define BOOST_COMPILER "Intel C++ version " BOOST_STRINGIZE(__ICC)
#  define BOOST_INTEL_CXX_VERSION __ICC
#endif

#if (BOOST_INTEL_CXX_VERSION <= 500) && defined(_MSC_VER)
#  define BOOST_NO_EXPLICIT_FUNCTION_TEMPLATE_ARGUMENTS
#endif

#if (BOOST_INTEL_CXX_VERSION <= 600) || !defined(BOOST_STRICT_CONFIG)

#  if defined(_MSC_VER) && (_MSC_VER <= 1300) // added check for <= VC 7 (Peter Dimov)

      // Intel C++ 5.0.1 uses EDG 2.45, but fails to activate Koenig lookup
      // in the frontend even in "strict" mode, unless you use 
      // -Qoption,cpp,--arg_dep_lookup.  (reported by Kirk Klobe & Thomas Witt)
      // Similarly, -Qoption,cpp,--new_for_init enables new-style "for" loop
      // variable scoping. (reported by Thomas Witt)
      // Intel C++ 6.0 (currently in Beta test) doesn't have any front-end
      // changes at all.  (reported by Kirk Klobe)
#     ifndef BOOST_NO_ARGUMENT_DEPENDENT_LOOKUP
#        define BOOST_NO_ARGUMENT_DEPENDENT_LOOKUP
#     endif
#     define BOOST_NO_SWPRINTF
#     define BOOST_NO_INCLASS_MEMBER_INITIALIZATION
#  endif

// Void returns, 64 bit integrals don't work when emulating VC 6 (Peter Dimov)

#  if defined(_MSC_VER) && (_MSC_VER <= 1200)
#     define BOOST_NO_VOID_RETURNS
#     define BOOST_NO_INTEGRAL_INT64_T
#  endif

#endif

#if _MSC_VER+0 >= 1000
#  ifndef _NATIVE_WCHAR_T_DEFINED
#     define BOOST_NO_INTRINSIC_WCHAR_T
#  endif
#  if _MSC_VER >= 1200
#     define BOOST_HAS_MS_INT64
#  endif
#  define BOOST_NO_SWPRINTF
#elif defined(_WIN32)
#  define BOOST_DISABLE_WIN32
#endif


//
// versions check:
// we don't support Intel prior to version 5.0:
#if BOOST_INTEL_CXX_VERSION < 500
#  error "Compiler not supported or configured - please reconfigure"
#endif
//
// last known and checked version is 600:
#if (BOOST_INTEL_CXX_VERSION > 600)
#  if defined(BOOST_ASSERT_CONFIG)
#     error "Unknown compiler version - please run the configure tests and report the results"
#  elif defined(_MSC_VER)
#     pragma message("Unknown compiler version - please run the configure tests and report the results")
#  endif
#endif




