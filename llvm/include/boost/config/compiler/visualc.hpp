//  (C) Copyright Boost.org 2001. Permission to copy, use, modify, sell and
//  distribute this software is granted provided this copyright notice appears
//  in all copies. This software is provided "as is" without express or implied
//  warranty, and with no claim as to its suitability for any purpose.

//  See http://www.boost.org for most recent version.

//  Microsoft Visual C++ compiler setup:

#define BOOST_MSVC _MSC_VER

// turn off the warnings before we #include anything
#pragma warning( disable : 4503 ) // warning: decorated name length exceeded

#if _MSC_VER <= 1200  // 1200 == VC++ 6.0
#pragma warning( disable : 4786 ) // ident trunc to '255' chars in debug info
#  define BOOST_NO_EXPLICIT_FUNCTION_TEMPLATE_ARGUMENTS
#  define BOOST_NO_DEPENDENT_TYPES_IN_TEMPLATE_VALUE_PARAMETERS
#  define BOOST_NO_VOID_RETURNS
   // disable min/max macro defines on vc6:
   //
#  ifndef NOMINMAX
#     define NOMINMAX
#  endif
#endif

#if (_MSC_VER <= 1300) // || !defined(BOOST_STRICT_CONFIG) // VC7 Beta 2 or later

#if !defined(_MSC_EXTENSIONS) && !defined(BOOST_NO_DEPENDENT_TYPES_IN_TEMPLATE_VALUE_PARAMETERS)      // VC7 bug with /Za
#  define BOOST_NO_DEPENDENT_TYPES_IN_TEMPLATE_VALUE_PARAMETERS
#endif

#  define BOOST_NO_INCLASS_MEMBER_INITIALIZATION
#  define BOOST_NO_PRIVATE_IN_AGGREGATE
#  define BOOST_NO_ARGUMENT_DEPENDENT_LOOKUP
#  define BOOST_NO_INTEGRAL_INT64_T

//    VC++ 6/7 has member templates but they have numerous problems including
//    cases of silent failure, so for safety we define:
#  define BOOST_NO_MEMBER_TEMPLATES
//    For VC++ experts wishing to attempt workarounds, we define:
#  define BOOST_MSVC6_MEMBER_TEMPLATES

#  define BOOST_NO_MEMBER_TEMPLATE_FRIENDS
#  define BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
#  define BOOST_NO_CV_VOID_SPECIALIZATIONS
#  define BOOST_NO_FUNCTION_TEMPLATE_ORDERING
#  define BOOST_NO_USING_TEMPLATE
#  define BOOST_NO_SWPRINTF
   //
   // disable min/max macros if defined:
   //
#  ifdef min
#     undef min
#  endif
#  ifdef max
#     undef max
#  endif

#endif

#if _MSC_VER <= 1301
#  define BOOST_NO_SWPRINTF
#endif

#ifndef _NATIVE_WCHAR_T_DEFINED
#  define BOOST_NO_INTRINSIC_WCHAR_T
#endif

//   
// check for exception handling support:   
#ifndef _CPPUNWIND   
#  define BOOST_NO_EXCEPTIONS   
#endif 

//
// __int64 support:
//
#if (_MSC_VER >= 1200) && defined(_MSC_EXTENSIONS)
#   define BOOST_HAS_MS_INT64
#endif
//
// disable Win32 API's if compiler extentions are
// turned off:
//
#ifndef _MSC_EXTENSIONS
#  define BOOST_DISABLE_WIN32
#endif



#define BOOST_COMPILER "Microsoft Visual C++ version " BOOST_STRINGIZE(_MSC_VER)

//
// versions check:
// we don't support Visual C++ prior to version 6:
#if _MSC_VER < 1200
#error "Compiler not supported or configured - please reconfigure"
#endif
//
// last known and checked version is 1301:
#if (_MSC_VER > 1301)
#  if defined(BOOST_ASSERT_CONFIG)
#     error "Unknown compiler version - please run the configure tests and report the results"
#  else
#     pragma message("Unknown compiler version - please run the configure tests and report the results")
#  endif
#endif



