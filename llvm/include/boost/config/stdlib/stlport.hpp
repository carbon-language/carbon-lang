//  (C) Copyright Boost.org 2001. Permission to copy, use, modify, sell and
//  distribute this software is granted provided this copyright notice appears
//  in all copies. This software is provided "as is" without express or implied
//  warranty, and with no claim as to its suitability for any purpose.

//  See http://www.boost.org for most recent version.

//  STLPort standard library config:

#if !defined(__SGI_STL_PORT) && !defined(_STLPORT_VERSION)
#  include <utility>
#  if !defined(__SGI_STL_PORT) && !defined(_STLPORT_VERSION)
#      error "This is not STLPort!"
#  endif
#endif

//
// __STL_STATIC_CONST_INIT_BUG implies BOOST_NO_LIMITS_COMPILE_TIME_CONSTANTS
// for versions prior to 4.1(beta)
//
#if (defined(__STL_STATIC_CONST_INIT_BUG) || defined(_STLP_STATIC_CONST_INIT_BUG)) && (__SGI_STL_PORT <= 0x400)
#  define BOOST_NO_LIMITS_COMPILE_TIME_CONSTANTS
#endif

//
// If STLport thinks that there is no partial specialisation, then there is no
// std::iterator traits:
//
#if !(defined(_STLP_CLASS_PARTIAL_SPECIALIZATION) || defined(__STL_CLASS_PARTIAL_SPECIALIZATION))
#  define BOOST_NO_STD_ITERATOR_TRAITS
#endif

//
// No new style iostreams on GCC without STLport's iostreams enabled:
//
#if (defined(__GNUC__) && (__GNUC__ < 3)) && !(defined(__SGI_STL_OWN_IOSTREAMS) || defined(_STLP_OWN_IOSTREAMS))
#  define BOOST_NO_STRINGSTREAM
#endif

//
// No new iostreams implies no std::locale, and no std::stringstream:
//
#if defined(__STL_NO_IOSTREAMS) || defined(__STL_NO_NEW_IOSTREAMS) || defined(_STLP_NO_IOSTREAMS) || defined(_STLP_NO_NEW_IOSTREAMS)
#  define BOOST_NO_STD_LOCALE
#  define BOOST_NO_STRINGSTREAM
#endif

//
// If the streams are not native, and we have a "using ::x" compiler bug
// then the io stream facets are not available in namespace std::
//
#if !defined(_STLP_OWN_IOSTREAMS) && defined(_STLP_USE_NAMESPACES) && defined(BOOST_NO_USING_TEMPLATE) && !defined(__BORLANDC__)
#  define BOOST_NO_STD_LOCALE
#endif
#if !defined(__SGI_STL_OWN_IOSTREAMS) && defined(__STL_USE_NAMESPACES) && defined(BOOST_NO_USING_TEMPLATE) && !defined(__BORLANDC__)
#  define BOOST_NO_STD_LOCALE
#endif

//
// Without member template support enabled, their are no template
// iterate constructors, and no std::allocator:
//
#if !(defined(__STL_MEMBER_TEMPLATES) || defined(_STLP_MEMBER_TEMPLATES))
#  define BOOST_NO_TEMPLATED_ITERATOR_CONSTRUCTORS
#  define BOOST_NO_STD_ALLOCATOR
#endif

#if !defined(_STLP_MEMBER_TEMPLATE_CLASSES)
#  define BOOST_NO_STD_ALLOCATOR
#endif

//
// We always have SGI style hash_set, hash_map, and slist:
//
#define BOOST_HAS_HASH
#define BOOST_HAS_SLIST

//
// STLport does a good job of importing names into namespace std::,
// but doesn't always get them all, define BOOST_NO_STDC_NAMESPACE, since our
// workaround does not conflict with STLports:
//
//
// Harold Howe says:
// Borland switched to STLport in BCB6. Defining BOOST_NO_STDC_NAMESPACE with
// BCB6 does cause problems. If we detect BCB6, then don't define 
// BOOST_NO_STDC_NAMESPACE
//
#if !defined(__BORLANDC__) || (__BORLANDC__ < 0x560)
#  if defined(__STL_IMPORT_VENDOR_CSTD) || defined(__STL_USE_OWN_NAMESPACE) || defined(_STLP_IMPORT_VENDOR_CSTD) || defined(_STLP_USE_OWN_NAMESPACE)
#     define BOOST_NO_STDC_NAMESPACE
#  endif
#endif
//
// std::reverse_iterate behaves like VC6's under some circumstances:
//
#if defined (_STLP_USE_OLD_HP_ITERATOR_QUERIES)  || defined (__STL_USE_OLD_HP_ITERATOR_QUERIES)\
 || (!defined ( _STLP_CLASS_PARTIAL_SPECIALIZATION ) && !defined ( __STL_CLASS_PARTIAL_SPECIALIZATION ))
// disable this for now, it causes too many problems, we need a better
// fix for broken reverse iterators:
// #  define BOOST_MSVC_STD_ITERATOR
#endif

//
// std::use_facet may be non-standard, uses a class instead:
//
#if defined(__STL_NO_EXPLICIT_FUNCTION_TMPL_ARGS) || defined(_STLP_NO_EXPLICIT_FUNCTION_TMPL_ARGS)
#  define BOOST_NO_STD_USE_FACET
#  define BOOST_HAS_STLP_USE_FACET
#endif

//
// If STLport thinks there are no wide functions, <cwchar> etc. is not working; but
// only if BOOST_NO_STDC_NAMESPACE is not defined (if it is then we do the import 
// into std:: ourselves).
//
#if defined(_STLP_NO_NATIVE_WIDE_FUNCTIONS) && !defined(BOOST_NO_STDC_NAMESPACE)
#  define BOOST_NO_CWCHAR
#  define BOOST_NO_CWTYPE
#endif

//
// Borland ships a version of STLport with C++ Builder 6 that lacks
// hashtables and the like:
//
#if defined(__BORLANDC__) && (__BORLANDC__ == 0x560)
#  undef BOOST_HAS_HASH
#endif


#define BOOST_STDLIB "STLPort standard library version " BOOST_STRINGIZE(__SGI_STL_PORT)




