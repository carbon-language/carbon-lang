// RUN: %llvmgxx -S %s -o - -O2 | FileCheck %s
namespace boost {
  namespace detail {
    template <typename T> struct cv_traits_imp {};
    template <typename T> struct cv_traits_imp<T*> {typedef T unqualified_type;};
  }
}
namespace mpl_ {}
namespace boost {
  namespace mpl {using namespace mpl_;}
  template< typename T > struct remove_cv {typedef typename boost::detail::cv_traits_imp<T*>::unqualified_type type;};
  namespace type_traits {
    typedef char yes_type;
    struct no_type {char padding[8];};
  }
}
namespace mpl_ {
  template< bool C_ > struct bool_;
  typedef bool_<true> true_;
  typedef bool_<false> false_;
  template< bool C_ > struct bool_ {static const bool value = C_;};
  template< typename T, T N > struct integral_c;
}
namespace boost{
  template <class T, T val>   struct integral_constant :
    public mpl::integral_c<T, val> {};
  template<> struct integral_constant<bool,true> : public mpl::true_ {};
  template<> struct integral_constant<bool,false> : public mpl::false_ {};
  namespace type_traits {
    template <bool b1, bool b2, bool b3 = false, bool b4 = false,
              bool b5 = false, bool b6 = false, bool b7 = false> struct ice_or;
    template <bool b1, bool b2, bool b3, bool b4, bool b5, bool b6, bool b7>
    struct ice_or {static const bool value = true; };
    template <> struct ice_or<false, false, false, false, false, false, false>
    {static const bool value = false;};
    template <bool b1, bool b2, bool b3 = true, bool b4 = true, bool b5 = true,
              bool b6 = true, bool b7 = true> struct ice_and;
    template <bool b1, bool b2, bool b3, bool b4, bool b5, bool b6, bool b7>
    struct ice_and {static const bool value = false;};
    template <> struct ice_and<true, true, true, true, true, true, true>
    {static const bool value = true;};
    template <bool b> struct ice_not {static const bool value = true;};
  };
  namespace detail {
    template <typename T> struct is_union_impl {static const bool value = false;};
  }
  template< typename T > struct is_union :
  ::boost::integral_constant<bool, ::boost::detail::is_union_impl<T>::value> {};
  namespace detail {
    template <class U> ::boost::type_traits::yes_type is_class_tester(void(U::*)(void));
    template <class U> ::boost::type_traits::no_type is_class_tester(...);
    template <typename T> struct is_class_impl {
      static const bool value = (::boost::type_traits::ice_and< sizeof(is_class_tester<T>(0))
                                 == sizeof(::boost::type_traits::yes_type),
                                 ::boost::type_traits::ice_not< ::boost::is_union<T>::value >::value >::value);};
}
  template<typename T> struct is_class:
  ::boost::integral_constant<bool,::boost::detail::is_class_impl<T>::value> {  };
namespace detail {
  template <typename T> struct empty_helper_t1: public T {int i[256];};
  struct empty_helper_t2 {int i[256];};
  template <typename T, bool is_a_class = false> struct empty_helper
  {static const bool value = false;};
  template <typename T> struct empty_helper<T, true>
  {static const bool value = (sizeof(empty_helper_t1<T>) == sizeof(empty_helper_t2));};
  template <typename T> struct is_empty_impl {
    typedef typename remove_cv<T>::type cvt;
    static const bool value = (::boost::type_traits::ice_or< ::boost::detail::empty_helper
                               <cvt,::boost::is_class<T>::value>::value, false>::value);
  };
}
template<typename T> struct is_empty:
::boost::integral_constant<bool,::boost::detail::is_empty_impl<T>::value> {};
template<typename T, typename U > struct is_same:
::boost::integral_constant<bool,false> {};
template<typename T> struct call_traits {typedef T& reference;};
namespace details {
  template <class T1, class T2, bool IsSame, bool FirstEmpty, bool SecondEmpty>
  struct compressed_pair_switch;
  template <class T1, class T2>
  struct compressed_pair_switch<T1, T2, false, true, false>
  {static const int value = 1;};
  template <class T1, class T2, int Version> class compressed_pair_imp;
  template <class T1, class T2> class compressed_pair_imp<T1, T2, 1>:
  protected ::boost::remove_cv<T1>::type {
  public:
    typedef T1 first_type;
    typedef T2 second_type;
    typedef typename call_traits<first_type>::reference first_reference;
    typedef typename call_traits<second_type>::reference second_reference;
    first_reference first() {return *this;}
    second_reference second() {return second_;}
    second_type second_;
  };
}
template <class T1, class T2> class compressed_pair:
  private ::boost::details::compressed_pair_imp<T1, T2, ::boost::details::compressed_pair_switch<
                                                          T1, T2, ::boost::is_same<typename remove_cv<T1>::type,
                                                                                   typename remove_cv<T2>::type>::value,
                                                          ::boost::is_empty<T1>::value, ::boost::is_empty<T2>::value>::value>
  {
  private:
    typedef details::compressed_pair_imp<T1, T2, ::boost::details::compressed_pair_switch<
                                                   T1, T2, ::boost::is_same<typename remove_cv<T1>::type,
                                                                            typename remove_cv<T2>::type>::value,
                                                                              ::boost::is_empty<T1>::value, ::boost::is_empty<T2>::value>::value> base;
  public:
    typedef T1 first_type;
    typedef T2 second_type;
    typedef typename call_traits<first_type>::reference first_reference;
    typedef typename call_traits<second_type>::reference second_reference;
    first_reference first() {return base::first();}
    second_reference second() {return base::second();}
  };
}
struct empty_base_t {};
struct empty_t : empty_base_t {};
typedef boost::compressed_pair<empty_t, int> data_t;
extern "C" {int printf(const char * , ...);}
extern "C" {void abort(void);}
int main (int argc, char * const argv[]) {
  data_t x;
  x.second() = -3;
  // This store should be elided:
  x.first() = empty_t();
  // If x.second() has been clobbered by the elided store, fail.
  if (x.second() != -3) {
    printf("x.second() was clobbered\n");
    // CHECK-NOT: x.second() was clobbered
    abort();
  }
  return 0;
}
// CHECK: ret i32
