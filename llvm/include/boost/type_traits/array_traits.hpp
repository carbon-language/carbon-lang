//  (C) Copyright Dave Abrahams, Steve Cleary, Beman Dawes, Howard
//  Hinnant & John Maddock 2000.  Permission to copy, use, modify,
//  sell and distribute this software is granted provided this
//  copyright notice appears in all copies. This software is provided
//  "as is" without express or implied warranty, and with no claim as
//  to its suitability for any purpose.
//
//  See http://www.boost.org for most recent version including documentation.
//
#ifndef BOOST_TT_ARRAY_TRAITS_HPP
# define BOOST_TT_ARRAY_TRAITS_HPP
# include <boost/type_traits/utility.hpp>

namespace boost { 

/**********************************************
 *
 * is_array
 *
 **********************************************/
#ifndef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
template <typename T> struct is_array
{ BOOST_STATIC_CONSTANT(bool, value = false); };
template <typename T, std::size_t N> struct is_array<T[N]>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <typename T, std::size_t N> struct is_array<const T[N]>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <typename T, std::size_t N> struct is_array<volatile T[N]>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <typename T, std::size_t N> struct is_array<const volatile T[N]>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
#else // BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
namespace detail
{
  using ::boost::type_traits::yes_type;
  using ::boost::type_traits::no_type;
  using ::boost::type_traits::wrap;
  
  template <class T> T(* is_array_helper1(wrap<T>) )(wrap<T>);
  char is_array_helper1(...);

  template <class T> no_type is_array_helper2(T(*)(wrap<T>));
  yes_type is_array_helper2(...);
}

template <typename T> 
struct is_array
{ 
public:
   BOOST_STATIC_CONSTANT(
       bool, value = sizeof(
           ::boost::detail::is_array_helper2(
               ::boost::detail::is_array_helper1(
                   ::boost::type_traits::wrap<T>()))) == 1
       );
};

template <> 
struct is_array<void>
{ 
   BOOST_STATIC_CONSTANT(bool, value = false);
};

# ifndef BOOST_NO_CV_VOID_SPECIALIZATIONS
template <> 
struct is_array<const void>
{ 
   BOOST_STATIC_CONSTANT(bool, value = false);
};
template <> 
struct is_array<volatile void>
{ 
   BOOST_STATIC_CONSTANT(bool, value = false);
};
template <> 
struct is_array<const volatile void>
{ 
   BOOST_STATIC_CONSTANT(bool, value = false);
};
# endif
#endif // BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION

} // namespace boost

#endif // BOOST_TT_ARRAY_TRAITS_HPP
