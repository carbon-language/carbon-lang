//  (C) Copyright Boost.org 2001. Permission to copy, use, modify, sell and
//  distribute this software is granted provided this copyright notice appears
//  in all copies. This software is provided "as is" without express or implied
//  warranty, and with no claim as to its suitability for any purpose.

//  See http://www.boost.org for most recent version.

//  Metrowerks C++ compiler setup:

// locale support is disabled when linking with the dynamic runtime
#   ifdef _MSL_NO_LOCALE
#     define BOOST_NO_STD_LOCALE
#   endif 

#   if __MWERKS__ <= 0x2301  // 5.3
#     define BOOST_NO_FUNCTION_TEMPLATE_ORDERING
#     define BOOST_NO_POINTER_TO_MEMBER_CONST
#     define BOOST_NO_DEPENDENT_TYPES_IN_TEMPLATE_VALUE_PARAMETERS
#     define BOOST_NO_MEMBER_TEMPLATE_KEYWORD
#   endif

#   if __MWERKS__ <= 0x2401  // 6.2
//#     define BOOST_NO_FUNCTION_TEMPLATE_ORDERING
#   endif

#   if(__MWERKS__ <= 0x2406) || !defined(BOOST_STRICT_CONFIG)  // 7.0 & 7.1
#     define BOOST_NO_MEMBER_TEMPLATE_FRIENDS
#   endif

#if !__option(wchar_type)
#   define BOOST_NO_INTRINSIC_WCHAR_T
#endif


#define BOOST_COMPILER "Metrowerks CodeWarrior C++ version " BOOST_STRINGIZE(__MWERKS__)

//
// versions check:
// we don't support Metrowerks prior to version 5.3:
#if __MWERKS__ < 0x2301
#  error "Compiler not supported or configured - please reconfigure"
#endif
//
// last known and checked version is 0x2406:
#if (__MWERKS__ > 0x2406)
#  if defined(BOOST_ASSERT_CONFIG)
#     error "Unknown compiler version - please run the configure tests and report the results"
#  endif
#endif






