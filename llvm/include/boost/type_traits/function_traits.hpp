
// Copyright (C) 2000 John Maddock (john_maddock@compuserve.com)
//
// Permission to copy and use this software is granted, 
// provided this copyright notice appears in all copies. 
// Permission to modify the code and to distribute modified code is granted, 
// provided this copyright notice appears in all copies, and a notice 
// that the code was modified is included with the copyright notice.
//
// This software is provided "as is" without express or implied warranty, 
// and with no claim as to its suitability for any purpose.

#ifndef BOOST_FUNCTION_TYPE_TRAITS_HPP
#define BOOST_FUNCTION_TYPE_TRAITS_HPP

#ifndef BOOST_ICE_TYPE_TRAITS_HPP
#include <boost/type_traits/ice.hpp>
#endif
#ifndef BOOST_FWD_TYPE_TRAITS_HPP
#include <boost/type_traits/fwd.hpp>
#endif
#ifndef BOOST_COMPOSITE_TYPE_TRAITS_HPP
#include <boost/type_traits/composite_traits.hpp>
#endif
//
// is a type a function?
// Please note that this implementation is unnecessarily complex:
// we could just use !is_convertible<T*, const volatile void*>::value,
// except that some compilers erroneously allow conversions from
// function pointers to void*.
//
namespace boost{
namespace detail{

#ifndef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION

template <class R>
struct is_function_helper_base{ BOOST_STATIC_CONSTANT(bool, value = false); };
template <class R>
struct is_function_helper_base<R (*)(void)>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class A0>
struct is_function_helper_base<R (*)(A0)>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class A0, class A1>
struct is_function_helper_base<R (*)(A0, A1)>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class A0, class A1, class A2>
struct is_function_helper_base<R (*)(A0, A1, A2)>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class A0, class A1, class A2, class A3>
struct is_function_helper_base<R (*)(A0, A1, A2, A3)>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class A0, class A1, class A2, class A3, class A4>
struct is_function_helper_base<R (*)(A0, A1, A2, A3, A4)>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class A0, class A1, class A2, class A3, class A4, class A5>
struct is_function_helper_base<R (*)(A0, A1, A2, A3, A4, A5)>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class A0, class A1, class A2, class A3, class A4, class A5, class A6>
struct is_function_helper_base<R (*)(A0, A1, A2, A3, A4, A5, A6)>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7>
struct is_function_helper_base<R (*)(A0, A1, A2, A3, A4, A5, A6, A7)>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8>
struct is_function_helper_base<R (*)(A0, A1, A2, A3, A4, A5, A6, A7, A8)>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9>
struct is_function_helper_base<R (*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9)>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10>
struct is_function_helper_base<R (*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10)>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11>
struct is_function_helper_base<R (*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11)>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12>
struct is_function_helper_base<R (*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12)>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13>
struct is_function_helper_base<R (*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13)>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14>
struct is_function_helper_base<R (*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14)>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15>
struct is_function_helper_base<R (*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15)>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16>
struct is_function_helper_base<R (*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16)>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17>
struct is_function_helper_base<R (*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17)>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18>
struct is_function_helper_base<R (*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18)>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19>
struct is_function_helper_base<R (*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19)>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20>
struct is_function_helper_base<R (*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20)>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21>
struct is_function_helper_base<R (*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21)>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22>
struct is_function_helper_base<R (*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22)>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22, class A23>
struct is_function_helper_base<R (*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23)>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22, class A23, class A24>
struct is_function_helper_base<R (*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24)>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22, class A23, class A24, class A25>
struct is_function_helper_base<R (*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, A25)>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22, class A23, class A24, class A25, class A26>
struct is_function_helper_base<R (*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, A25, A26)>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22, class A23, class A24, class A25, class A26, class A27>
struct is_function_helper_base<R (*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, A25, A26, A27)>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22, class A23, class A24, class A25, class A26, class A27, class A28>
struct is_function_helper_base<R (*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, A25, A26, A27, A28)>{ BOOST_STATIC_CONSTANT(bool, value = true); };

template <class T>
struct is_function_helper : is_function_helper_base<T*>{};

#else

template <class T>
struct is_function_helper
{
   static T* t;
   BOOST_STATIC_CONSTANT(bool, value = sizeof(is_function_tester(t)) == sizeof(::boost::type_traits::yes_type));
   //BOOST_STATIC_CONSTANT(bool, value = (!::boost::is_convertible<T*, const volatile void*>::value));
};

#endif

template <class T>
struct is_function_ref_helper
{
   BOOST_STATIC_CONSTANT(bool, value = false);
};

#if !defined(BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION)
template <class T, bool is_ref>
struct is_function_chooser
{
   typedef is_function_helper<T> type;
};
template <class T>
struct is_function_chooser<T, true>
{
   typedef is_function_ref_helper<T> type;
};
#endif
} // namespace detail

template <class T>
struct is_function
{
private:
#if !defined(BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION)
   typedef typename detail::is_function_chooser<T, ::boost::is_reference<T>::value>::type m_type;
#else
   // without partial specialistaion we can't use is_reference on
   // function types, that leaves this template broken in the case that
   // T is a reference:
   typedef detail::is_function_helper<T> m_type;
#endif
public:
   BOOST_STATIC_CONSTANT(bool, value = m_type::value);
};

} // boost

#endif // BOOST_FUNCTION_TYPE_TRAITS_HPP
