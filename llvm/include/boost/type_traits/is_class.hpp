//  (C) Copyright Dave Abrahams, Steve Cleary, Beman Dawes, Howard
//  Hinnant & John Maddock 2000-2002.  Permission to copy, use,
//  modify, sell and distribute this software is granted provided this
//  copyright notice appears in all copies. This software is provided
//  "as is" without express or implied warranty, and with no claim as
//  to its suitability for any purpose.
//
//  See http://www.boost.org for most recent version including documentation.
//
#ifndef BOOST_TYPE_TRAITS_IS_CLASS_HPP
# define BOOST_TYPE_TRAITS_IS_CLASS_HPP

# if (defined(__MWERKS__) && __MWERKS__ >= 0x3000) || defined(BOOST_MSVC) && _MSC_FULL_VER > 13012108 || defined(BOOST_NO_COMPILER_CONFIG)
# ifndef BOOST_ICE_TYPE_TRAITS_HPP
#  include <boost/type_traits/ice.hpp>
# endif

# define BOOST_TYPE_TRAITS_IS_CLASS_DEFINED
namespace boost {

template <typename T>
struct is_class
{
    // This is actually the conforming implementation which works with
    // abstract classes.  However, enough compilers have trouble with
    // it that most will use the one in
    // boost/type_traits/object_traits.hpp. This implementation
    // actually works with VC7.0, but other interactions seem to fail
    // when we use it.

// is_class<> metafunction due to Paul Mensonides
// (leavings@attbi.com). For more details:
// http://groups.google.com/groups?hl=en&selm=000001c1cc83%24e154d5e0%247772e50c%40c161550a&rnum=1
 private:
    template <class U> static ::boost::type_traits::yes_type is_class_helper(void(U::*)(void));
    template <class U> static ::boost::type_traits::no_type is_class_helper(...);
 public:
    BOOST_STATIC_CONSTANT(
        bool, value = sizeof(
            is_class_helper<T>(0)
            ) == sizeof(::boost::type_traits::yes_type));
};
}

# else // nonconforming compilers will use a different impelementation, in object_traits.hpp

# endif // nonconforming implementations

#endif // BOOST_TYPE_TRAITS_IS_CLASS_HPP
