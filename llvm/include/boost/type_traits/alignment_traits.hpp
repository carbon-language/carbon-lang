
//  (C) Copyright John Maddock 2000.
//  Permission to copy, use, modify, sell and
//  distribute this software is granted provided this copyright notice appears
//  in all copies. This software is provided "as is" without express or implied
//  warranty, and with no claim as to its suitability for any purpose.

//
// defines alignment_of:

#ifndef ALIGNMENT_TYPE_TRAITS_HPP
#define ALIGNMENT_TYPE_TRAITS_HPP

#include <cstdlib>
#include <cstddef>
#ifndef BOOST_ICE_TYPE_TRAITS_HPP
#include <boost/type_traits/ice.hpp>
#endif
#include <boost/preprocessor/list/for_each_i.hpp>
#include <boost/preprocessor/tuple/to_list.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/type_traits/transform_traits.hpp>
#include <boost/static_assert.hpp>

#ifdef BOOST_MSVC
# pragma warning(push)
# pragma warning(disable: 4121) // alignment is sensitive to packing
#endif

namespace boost{
template <class T> struct alignment_of;

//
// get the alignment of some arbitrary type:
namespace detail{

template <class T>
struct alignment_of_hack
{
   char c;
   T t;
   alignment_of_hack();
};


template <unsigned A, unsigned S>
struct alignment_logic
{
   BOOST_STATIC_CONSTANT(std::size_t, value = A < S ? A : S);
};

} // namespace detail

template <class T>
struct alignment_of
{
   BOOST_STATIC_CONSTANT(std::size_t, value =
      (::boost::detail::alignment_logic<
         sizeof(detail::alignment_of_hack<T>) - sizeof(T),
         sizeof(T)
      >::value));
};

//
// references have to be treated specially, assume
// that a reference is just a special pointer:
#ifndef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
template <class T>
struct alignment_of<T&>
{
public:
   BOOST_STATIC_CONSTANT(std::size_t, value = ::boost::alignment_of<T*>::value);
};
#endif
//
// void has to be treated specially:
template <>
struct alignment_of<void>
{ BOOST_STATIC_CONSTANT(std::size_t, value = 0); };
#ifndef BOOST_NO_CV_VOID_SPECIALIZATIONS
template <>
struct alignment_of<const void>
{ BOOST_STATIC_CONSTANT(std::size_t, value = 0); };
template <>
struct alignment_of<volatile void>
{ BOOST_STATIC_CONSTANT(std::size_t, value = 0); };
template <>
struct alignment_of<const volatile void>
{ BOOST_STATIC_CONSTANT(std::size_t, value = 0); };
#endif

namespace detail {
class alignment_dummy;
typedef void (*function_ptr)();
typedef int (alignment_dummy::*member_ptr);
typedef int (alignment_dummy::*member_function_ptr)();

/*
 * The ct_if implementation is temporary code. It will be replaced with MPL
 * in the future...
 */
struct select_then
{
  template<typename Then, typename Else>
  struct result
  {
    typedef Then type;
  };
};
 
struct select_else
{
  template<typename Then, typename Else>
  struct result
  { 
    typedef Else type;
  };
};
 
template<bool Condition>
struct ct_if_selector
{
  typedef select_then type;
};
 
template<>
struct ct_if_selector<false>
{
  typedef select_else type;
};
 
template<bool Condition, typename Then, typename Else>
struct ct_if
{
  typedef typename ct_if_selector<Condition>::type select;
  typedef typename select::template result<Then,Else>::type type;
};

#define BOOST_TT_ALIGNMENT_TYPES BOOST_PP_TUPLE_TO_LIST( \
        11, ( \
        char, short, int, long, float, double, long double \
        , void*, function_ptr, member_ptr, member_function_ptr))

#define BOOST_TT_CHOOSE_LOWER_ALIGNMENT(R,P,I,T) \
        typename ct_if< \
           alignment_of<T>::value <= target, T, char>::type BOOST_PP_CAT(t,I);

#define BOOST_TT_CHOOSE_T(R,P,I,T) T BOOST_PP_CAT(t,I);
           
template <std::size_t target>
union lower_alignment
{
  BOOST_PP_LIST_FOR_EACH_I(
      BOOST_TT_CHOOSE_LOWER_ALIGNMENT
      , ignored, BOOST_TT_ALIGNMENT_TYPES)
};

union max_align
{
  BOOST_PP_LIST_FOR_EACH_I(
      BOOST_TT_CHOOSE_T
      , ignored, BOOST_TT_ALIGNMENT_TYPES)
};

#undef BOOST_TT_ALIGNMENT_TYPES
#undef BOOST_TT_CHOOSE_LOWER_ALIGNMENT
#undef BOOST_TT_CHOOSE_T

template<int TAlign, int Align>
struct is_aligned 
{
  BOOST_STATIC_CONSTANT(bool, 
                        value = (TAlign >= Align) & (TAlign % Align == 0));
};

}

// This alignment method originally due to Brian Parker, implemented by David
// Abrahams, and then ported here by Doug Gregor. 
template <int Align>
class type_with_alignment
{
  typedef detail::lower_alignment<Align> t1;

  typedef type_with_alignment<Align> this_type;

  typedef typename detail::ct_if<
              (detail::is_aligned<(alignment_of<t1>::value), Align>::value)
            , t1
            , detail::max_align
          >::type align_t;

  BOOST_STATIC_CONSTANT(std::size_t, found = alignment_of<align_t>::value);
  
  BOOST_STATIC_ASSERT(found >= Align);
  BOOST_STATIC_ASSERT(found % Align == 0);
  
public:
  typedef align_t type;
};

} // namespace boost

#ifdef BOOST_MSVC
# pragma warning(pop)
#endif

#endif // ALIGNMENT_TYPE_TRAITS_HPP
