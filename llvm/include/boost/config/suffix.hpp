//  Boost config.hpp configuration header file  ------------------------------//

//  (C) Copyright Boost.org 2001. Permission to copy, use, modify, sell and
//  distribute this software is granted provided this copyright notice appears
//  in all copies. This software is provided "as is" without express or implied
//  warranty, and with no claim as to its suitability for any purpose.

//  See http://www.boost.org for most recent version.

//  Boost config.hpp policy and rationale documentation has been moved to
//  http://www.boost.org/libs/config
//
//  This file is intended to be stable, and relatively unchanging.
//  It should contain boilerplate code only - no compiler specific
//  code unless it is unavoidable - no changes unless unavoidable.

#ifndef BOOST_CONFIG_SUFFIX_HPP
#define BOOST_CONFIG_SUFFIX_HPP

# ifndef BOOST_DECL
#   define BOOST_DECL  // default for compilers not needing this decoration.
# endif

//
// look for long long by looking for the appropriate macros in <limits.h>.
// Note that we use limits.h rather than climits for maximal portability,
// remember that since these just declare a bunch of macros, there should be
// no namespace issues from this.
//
#include <limits.h>
# if !defined(BOOST_MSVC) && !defined(__BORLANDC__) \
   && (defined(ULLONG_MAX) || defined(ULONG_LONG_MAX) || defined(ULONGLONG_MAX))
#  define BOOST_HAS_LONG_LONG
#endif
#if !defined(BOOST_HAS_LONG_LONG) && !defined(BOOST_NO_INTEGRAL_INT64_T)
#  define BOOST_NO_INTEGRAL_INT64_T
#endif

// GCC 3.x will clean up all of those nasty macro definitions that
// BOOST_NO_CTYPE_FUNCTIONS is intended to help work around, so undefine
// it under GCC 3.x.
#if defined(__GNUC__) && (__GNUC__ >= 3) && defined(BOOST_NO_CTYPE_FUNCTIONS)
#  undef BOOST_NO_CTYPE_FUNCTIONS
#endif


//
// Assume any extensions are in namespace std:: unless stated otherwise:
//
#  ifndef BOOST_STD_EXTENSION_NAMESPACE
#    define BOOST_STD_EXTENSION_NAMESPACE std
#  endif

//
// If cv-qualified specializations are not allowed, then neither are cv-void ones:
//
#  if defined(BOOST_NO_CV_SPECIALIZATIONS) \
      && !defined(BOOST_NO_CV_VOID_SPECIALIZATIONS)
#     define BOOST_NO_CV_VOID_SPECIALIZATIONS
#  endif

//
// If there is no numeric_limits template, then it can't have any compile time
// constants either!
//
#  if defined(BOOST_NO_LIMITS) \
      && !defined(BOOST_NO_LIMITS_COMPILE_TIME_CONSTANTS)
#     define BOOST_NO_LIMITS_COMPILE_TIME_CONSTANTS
#  endif

//
// if member templates are supported then so is the
// VC6 subset of member templates:
//
#  if !defined(BOOST_NO_MEMBER_TEMPLATES) \
       && !defined(BOOST_MSVC6_MEMBER_TEMPLATES)
#     define BOOST_MSVC6_MEMBER_TEMPLATES
#  endif

//
// Without partial specialization, std::iterator_traits can't work:
//
#  if defined(BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION) \
      && !defined(BOOST_NO_STD_ITERATOR_TRAITS)
#     define BOOST_NO_STD_ITERATOR_TRAITS
#  endif

//
// Without member template support, we can't have template constructors
// in the standard library either:
//
#  if defined(BOOST_NO_MEMBER_TEMPLATES) \
      && !defined(BOOST_MSVC6_MEMBER_TEMPLATES) \
      && !defined(BOOST_NO_TEMPLATED_ITERATOR_CONSTRUCTORS)
#     define BOOST_NO_TEMPLATED_ITERATOR_CONSTRUCTORS
#  endif

//
// Without member template support, we can't have a conforming
// std::allocator template either:
//
#  if defined(BOOST_NO_MEMBER_TEMPLATES) \
      && !defined(BOOST_MSVC6_MEMBER_TEMPLATES) \
      && !defined(BOOST_NO_STD_ALLOCATOR)
#     define BOOST_NO_STD_ALLOCATOR
#  endif

//
// We can't have a working std::use_facet if there is no std::locale:
//
#  if defined(BOOST_NO_STD_LOCALE) && !defined(BOOST_NO_STD_USE_FACET)
#     define BOOST_NO_STD_USE_FACET
#  endif

//
// We can't have a std::messages facet if there is no std::locale:
//
#  if defined(BOOST_NO_STD_LOCALE) && !defined(BOOST_NO_STD_MESSAGES)
#     define BOOST_NO_STD_MESSAGES
#  endif

//
// We can't have a <cwctype> if there is no <cwchar>:
//
#  if defined(BOOST_NO_CWCHAR) && !defined(BOOST_NO_CWCTYPE)
#     define BOOST_NO_CWCTYPE
#  endif

//
// We can't have a swprintf if there is no <cwchar>:
//
#  if defined(BOOST_NO_CWCHAR) && !defined(BOOST_NO_SWPRINTF)
#     define BOOST_NO_SWPRINTF
#  endif

//
// If Win32 support is turned off, then we must turn off
// threading support also, unless there is some other
// thread API enabled:
//
#if defined(BOOST_DISABLE_WIN32) && defined(_WIN32) \
   && !defined(BOOST_DISABLE_THREADS) && !defined(BOOST_HAS_PTHREADS)
#  define BOOST_DISABLE_THREADS
#endif

//
// Turn on threading support if the compiler thinks that it's in
// multithreaded mode.  We put this here because there are only a
// limited number of macros that identify this (if there's any missing
// from here then add to the appropriate compiler section):
//
#if (defined(__MT__) || defined(_MT) || defined(_REENTRANT) \
    || defined(_PTHREADS)) && !defined(BOOST_HAS_THREADS)
#  define BOOST_HAS_THREADS
#endif

//
// Turn threading support off if BOOST_DISABLE_THREADS is defined:
//
#if defined(BOOST_DISABLE_THREADS) && defined(BOOST_HAS_THREADS)
#  undef BOOST_HAS_THREADS
#endif

//
// Turn threading support off if we don't recognise the threading API:
//
#if defined(BOOST_HAS_THREADS) && !defined(BOOST_HAS_PTHREADS)\
      && !defined(BOOST_HAS_WINTHREADS) && !defined(BOOST_HAS_BETHREADS)\
      && !defined(BOOST_HAS_MPTASKS)
#  undef BOOST_HAS_THREADS
#endif

//
// If the compiler claims to be C99 conformant, then it had better
// have a <stdint.h>:
//
#  if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901)
#     define BOOST_HAS_STDINT_H
#  endif

//
// Define BOOST_NO_SLIST and BOOST_NO_HASH if required.
// Note that this is for backwards compatibility only.
//
#  ifndef BOOST_HAS_SLIST
#     define BOOST_NO_SLIST
#  endif

#  ifndef BOOST_HAS_HASH
#     define BOOST_NO_HASH
#  endif

//  BOOST_NO_STDC_NAMESPACE workaround  --------------------------------------//
//  Because std::size_t usage is so common, even in boost headers which do not
//  otherwise use the C library, the <cstddef> workaround is included here so
//  that ugly workaround code need not appear in many other boost headers.
//  NOTE WELL: This is a workaround for non-conforming compilers; <cstddef> 
//  must still be #included in the usual places so that <cstddef> inclusion
//  works as expected with standard conforming compilers.  The resulting
//  double inclusion of <cstddef> is harmless.

# ifdef BOOST_NO_STDC_NAMESPACE
#   include <cstddef>
    namespace std { using ::ptrdiff_t; using ::size_t; }
# endif

//  BOOST_NO_STD_MIN_MAX workaround  -----------------------------------------//

#  ifdef BOOST_NO_STD_MIN_MAX

namespace std {
  template <class _Tp>
  inline const _Tp& min(const _Tp& __a, const _Tp& __b) {
    return __b < __a ? __b : __a;
  }
  template <class _Tp>
  inline const _Tp& max(const _Tp& __a, const _Tp& __b) {
    return  __a < __b ? __b : __a;
  }
#     ifdef BOOST_MSVC
  // Apparently, something in the Microsoft libraries requires the "long"
  // overload, because it calls the min/max functions with arguments of
  // slightly different type.  (If this proves to be incorrect, this
  // whole "BOOST_MSVC" section can be removed.)
  inline long min(long __a, long __b) {
    return __b < __a ? __b : __a;
  }
  inline long max(long __a, long __b) {
    return  __a < __b ? __b : __a;
  }
  // The "long double" overload is required, otherwise user code calling
  // min/max for floating-point numbers will use the "long" overload.
  // (SourceForge bug #495495)
  inline long double min(long double __a, long double __b) {
    return __b < __a ? __b : __a;
  }
  inline long double max(long double __a, long double __b) {
    return  __a < __b ? __b : __a;
  }
#     endif
}

#  endif

// BOOST_STATIC_CONSTANT workaround --------------------------------------- //
// On compilers which don't allow in-class initialization of static integral
// constant members, we must use enums as a workaround if we want the constants
// to be available at compile-time. This macro gives us a convenient way to
// declare such constants.

#  ifdef BOOST_NO_INCLASS_MEMBER_INITIALIZATION
#       define BOOST_STATIC_CONSTANT(type, assignment) enum { assignment }
#  else
#     define BOOST_STATIC_CONSTANT(type, assignment) static const type assignment
#  endif

// BOOST_USE_FACET workaround ----------------------------------------------//
// When the standard library does not have a conforming std::use_facet there
// are various workarounds available, but they differ from library to library.
// This macro provides a consistent way to access a locale's facets.
// Usage:
//    replace
//       std::use_facet<Type>(loc);
//    with
//       BOOST_USE_FACET(Type, loc);
//    Note do not add a std:: prefix to the front of BOOST_USE_FACET!

#if defined(BOOST_NO_STD_USE_FACET)
#  ifdef BOOST_HAS_TWO_ARG_USE_FACET
#     define BOOST_USE_FACET(Type, loc) std::use_facet(loc, static_cast<Type*>(0))
#  elif defined(BOOST_HAS_MACRO_USE_FACET)
#     define BOOST_USE_FACET(Type, loc) std::_USE(loc, Type)
#  elif defined(BOOST_HAS_STLP_USE_FACET)
#     define BOOST_USE_FACET(Type, loc) (*std::_Use_facet<Type >(loc))
#  endif
#else
#  define BOOST_USE_FACET(Type, loc) std::use_facet< Type >(loc)
#endif

// BOOST_NESTED_TEMPLATE workaround ------------------------------------------//
// Member templates are supported by some compilers even though they can't use
// the A::template member<U> syntax, as a workaround replace:
//
// typedef typename A::template rebind<U> binder;
//
// with:
//
// typedef typename A::BOOST_NESTED_TEMPLATE rebind<U> binder;

#ifndef BOOST_NO_MEMBER_TEMPLATE_KEYWORD
#  define BOOST_NESTED_TEMPLATE template
#else
#  define BOOST_NESTED_TEMPLATE
#endif

// ---------------------------------------------------------------------------//

//
// Helper macro BOOST_STRINGIZE:
// Converts the parameter X to a string after macro replacement
// on X has been performed.
//
#define BOOST_STRINGIZE(X) BOOST_DO_STRINGIZE(X)
#define BOOST_DO_STRINGIZE(X) #X

//
// Helper macro BOOST_JOIN:
// The following piece of macro magic joins the two 
// arguments together, even when one of the arguments is
// itself a macro (see 16.3.1 in C++ standard).  The key
// is that macro expansion of macro arguments does not
// occur in BOOST_DO_JOIN2 but does in BOOST_DO_JOIN.
//
#define BOOST_JOIN( X, Y ) BOOST_DO_JOIN( X, Y )
#define BOOST_DO_JOIN( X, Y ) BOOST_DO_JOIN2(X,Y)
#define BOOST_DO_JOIN2( X, Y ) X##Y

//
// Set some default values for compiler/library/platform names.
// These are for debugging config setup only:
//
#  ifndef BOOST_COMPILER
#     define BOOST_COMPILER "Unknown ISO C++ Compiler"
#  endif
#  ifndef BOOST_STDLIB
#     define BOOST_STDLIB "Unknown ISO standard library"
#  endif
#  ifndef BOOST_PLATFORM
#     if defined(unix) || defined(__unix) || defined(_XOPEN_SOURCE) \
         || defined(_POSIX_SOURCE)
#        define BOOST_PLATFORM "Generic Unix"
#     else
#        define BOOST_PLATFORM "Unknown"
#     endif
#  endif

#endif

