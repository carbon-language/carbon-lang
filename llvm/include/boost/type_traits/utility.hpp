// Copyright David Abrahams 2002. Permission to copy, use,
// modify, sell and distribute this software is granted provided this
// copyright notice appears in all copies. This software is provided
// "as is" without express or implied warranty, and with no claim as
// to its suitability for any purpose.
#ifndef BOOST_TT_UTILITY_HPP
# define BOOST_TT_UTILITY_HPP

namespace boost { namespace type_traits
{
  // Utility metafunction class which always returns false
  struct false_unary_metafunction
  {
      template <class T>
      struct apply
      {
          BOOST_STATIC_CONSTANT(bool, value = false);
      };
  };

  template <class T> struct wrap {};
}} // namespace boost::type_traits

#endif // BOOST_TT_UTILITY_HPP
