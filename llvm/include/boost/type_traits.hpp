//  (C) Copyright Boost.org 2000. Permission to copy, use, modify, sell and
//  distribute this software is granted provided this copyright notice appears
//  in all copies. This software is provided "as is" without express or implied
//  warranty, and with no claim as to its suitability for any purpose.

//  See http://www.boost.org for most recent version including documentation.
//  See boost/detail/type_traits.hpp and boost/detail/ob_type_traits.hpp
//  for full copyright notices.

#ifndef BOOST_TYPE_TRAITS_HPP
#define BOOST_TYPE_TRAITS_HPP

#include <boost/type_traits/fwd.hpp>
#include <boost/type_traits/ice.hpp>
#include <boost/type_traits/conversion_traits.hpp>
#include <boost/type_traits/arithmetic_traits.hpp>
#include <boost/type_traits/cv_traits.hpp>
#include <boost/type_traits/composite_traits.hpp>
#include <boost/type_traits/alignment_traits.hpp>
#include <boost/type_traits/object_traits.hpp>
#include <boost/type_traits/transform_traits.hpp>
#include <boost/type_traits/same_traits.hpp>
#include <boost/type_traits/function_traits.hpp>

/**************************************************************************/

//
// undefine helper macro's:
//
#undef BOOST_IS_CLASS
#undef BOOST_IS_ENUM
#undef BOOST_IS_UNION
#undef BOOST_IS_POD
#undef BOOST_IS_EMPTY
#undef BOOST_HAS_TRIVIAL_CONSTRUCTOR
#undef BOOST_HAS_TRIVIAL_COPY
#undef BOOST_HAS_TRIVIAL_ASSIGN
#undef BOOST_HAS_TRIVIAL_DESTRUCTOR

#endif // BOOST_TYPE_TRAITS_HPP



