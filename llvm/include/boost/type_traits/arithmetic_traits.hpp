//  (C) Copyright Steve Cleary, Beman Dawes, Howard Hinnant & John Maddock 2000.
//  Permission to copy, use, modify, sell and
//  distribute this software is granted provided this copyright notice appears
//  in all copies. This software is provided "as is" without express or implied
//  warranty, and with no claim as to its suitability for any purpose.
//
//  See http://www.boost.org for most recent version including documentation.
//
//  defines traits classes for arithmetic types:
//  is_void, is_integral, is_float, is_arithmetic, is_fundamental.
//

//  Revision History:
// Feb 19 2001 Added #include <climits> (David Abrahams)

#ifndef BOOST_ARITHMETIC_TYPE_TRAITS_HPP
#define BOOST_ARITHMETIC_TYPE_TRAITS_HPP

#ifndef BOOST_ICE_TYPE_TRAITS_HPP
#include <boost/type_traits/ice.hpp>
#endif
#ifndef BOOST_FWD_TYPE_TRAITS_HPP
#include <boost/type_traits/fwd.hpp>
#endif

#include <limits.h> // for ULLONG_MAX/ULONG_LONG_MAX

namespace boost{

//* is a type T void - is_void<T>
template <typename T> struct is_void{ BOOST_STATIC_CONSTANT(bool, value = false); };
template <> struct is_void<void>{ BOOST_STATIC_CONSTANT(bool, value = true); };

//* is a type T an [cv-qualified-] integral type described in the standard (3.9.1p3)
// as an extention we include long long, as this is likely to be added to the
// standard at a later date
template <typename T> struct is_integral
{ BOOST_STATIC_CONSTANT(bool, value = false); };
template <> struct is_integral<unsigned char>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <> struct is_integral<unsigned short>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <> struct is_integral<unsigned int>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <> struct is_integral<unsigned long>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <> struct is_integral<signed char>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <> struct is_integral<signed short>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <> struct is_integral<signed int>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <> struct is_integral<signed long>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <> struct is_integral<char>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
#ifndef BOOST_NO_INTRINSIC_WCHAR_T
template <> struct is_integral<wchar_t>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
#endif
template <> struct is_integral<bool>
{ BOOST_STATIC_CONSTANT(bool, value = true); };

# if defined(BOOST_HAS_LONG_LONG)
template <> struct is_integral<unsigned long long>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <> struct is_integral<long long>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
#elif defined(BOOST_HAS_MS_INT64)
template <> struct is_integral<unsigned __int64>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <> struct is_integral<__int64>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
#endif

//* is a type T a floating-point type described in the standard (3.9.1p8)
template <typename T> struct is_float
{ BOOST_STATIC_CONSTANT(bool, value = false); };
template <> struct is_float<float>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <> struct is_float<double>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <> struct is_float<long double>
{ BOOST_STATIC_CONSTANT(bool, value = true); };

//
// declare cv-qualified specialisations of these templates only
// if BOOST_NO_CV_SPECIALIZATIONS is not defined:
#ifndef BOOST_NO_CV_VOID_SPECIALIZATIONS
template <> struct is_void<const void>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <> struct is_void<volatile void>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <> struct is_void<const volatile void>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
#endif
#ifndef BOOST_NO_CV_SPECIALIZATIONS
// const-variations:
template <> struct is_integral<const unsigned char>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <> struct is_integral<const unsigned short>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <> struct is_integral<const unsigned int>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <> struct is_integral<const unsigned long>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <> struct is_integral<const signed char>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <> struct is_integral<const signed short>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <> struct is_integral<const signed int>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <> struct is_integral<const signed long>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <> struct is_integral<const char>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
#ifndef BOOST_NO_INTRINSIC_WCHAR_T
template <> struct is_integral<const wchar_t>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
#endif
template <> struct is_integral<const bool>
{ BOOST_STATIC_CONSTANT(bool, value = true); };

# if defined(BOOST_HAS_LONG_LONG)
template <> struct is_integral<const unsigned long long>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <> struct is_integral<const long long>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
#elif defined(BOOST_HAS_MS_INT64)
template <> struct is_integral<const unsigned __int64>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <> struct is_integral<const __int64>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
#endif //__int64

template <> struct is_float<const float>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <> struct is_float<const double>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <> struct is_float<const long double>
{ BOOST_STATIC_CONSTANT(bool, value = true); };

// volatile-variations:
template <> struct is_integral<volatile  unsigned char>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <> struct is_integral<volatile  unsigned short>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <> struct is_integral<volatile  unsigned int>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <> struct is_integral<volatile  unsigned long>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <> struct is_integral<volatile  signed char>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <> struct is_integral<volatile  signed short>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <> struct is_integral<volatile  signed int>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <> struct is_integral<volatile  signed long>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <> struct is_integral<volatile  char>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
#ifndef BOOST_NO_INTRINSIC_WCHAR_T
template <> struct is_integral<volatile  wchar_t>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
#endif
template <> struct is_integral<volatile  bool>
{ BOOST_STATIC_CONSTANT(bool, value = true); };

# if defined(BOOST_HAS_LONG_LONG)
template <> struct is_integral<volatile  unsigned long long>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <> struct is_integral<volatile  long long>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
#elif defined(BOOST_HAS_MS_INT64)
template <> struct is_integral<volatile  unsigned __int64>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <> struct is_integral<volatile  __int64>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
#endif //__int64

template <> struct is_float<volatile  float>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <> struct is_float<volatile  double>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <> struct is_float<volatile  long double>
{ BOOST_STATIC_CONSTANT(bool, value = true); };

// const-volatile-variations:
template <> struct is_integral<const volatile unsigned char>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <> struct is_integral<const volatile unsigned short>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <> struct is_integral<const volatile unsigned int>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <> struct is_integral<const volatile unsigned long>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <> struct is_integral<const volatile signed char>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <> struct is_integral<const volatile signed short>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <> struct is_integral<const volatile signed int>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <> struct is_integral<const volatile signed long>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <> struct is_integral<const volatile char>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
#ifndef BOOST_NO_INTRINSIC_WCHAR_T
template <> struct is_integral<const volatile wchar_t>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
#endif
template <> struct is_integral<const volatile bool>
{ BOOST_STATIC_CONSTANT(bool, value = true); };

# if defined(BOOST_HAS_LONG_LONG)
template <> struct is_integral<const volatile unsigned long long>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <> struct is_integral<const volatile long long>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
#elif defined(BOOST_HAS_MS_INT64)
template <> struct is_integral<const volatile unsigned __int64>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <> struct is_integral<const volatile __int64>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
#endif //__int64

template <> struct is_float<const volatile float>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <> struct is_float<const volatile double>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <> struct is_float<const volatile long double>
{ BOOST_STATIC_CONSTANT(bool, value = true); };

#endif // BOOST_NO_CV_SPECIALIZATIONS

//* is a type T an arithmetic type described in the standard (3.9.1p8)
template <typename T> 
struct is_arithmetic
{ 
   BOOST_STATIC_CONSTANT(bool, value = 
      (::boost::type_traits::ice_or< 
         ::boost::is_integral<T>::value,
         ::boost::is_float<T>::value
      >::value)); 
};

//* is a type T a fundamental type described in the standard (3.9.1)
template <typename T> 
struct is_fundamental
{ 
   BOOST_STATIC_CONSTANT(bool, value = 
      (::boost::type_traits::ice_or< 
         ::boost::is_arithmetic<T>::value, 
         ::boost::is_void<T>::value
      >::value)); 
};

} // namespace boost

#endif







