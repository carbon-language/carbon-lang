//  (C) Copyright John Maddock 2000. Permission to copy, use, modify, sell and   
//  distribute this software is granted provided this copyright notice appears   
//  in all copies. This software is provided "as is" without express or implied   
//  warranty, and with no claim as to its suitability for any purpose.   

//  common test code for type-traits tests
//  WARNING: contains code as well as declarations!


#ifndef BOOST_TYPE_TRAITS_TEST_HPP
#define BOOST_TYPE_TRAITS_TEST_HPP
#include <iostream>
#include <typeinfo>
#include <boost/config.hpp>
#include <boost/utility.hpp>
#include <boost/type_traits/alignment_traits.hpp>
//
// define tests here
unsigned failures = 0;
unsigned test_count = 0;
//
// This must get defined within the test file.
// All compilers have bugs, set this to the number of
// regressions *expected* from a given compiler,
// if there are no workarounds for the bugs, *and*
// the regressions have been investigated.
//
extern unsigned int expected_failures;
//
// proc check_result()
// Checks that there were no regressions:
//
int check_result(int argc, char** argv)
{
   std::cout << test_count << " tests completed, "
      << failures << " failures found, "
      << expected_failures << " failures expected from this compiler." << std::endl;
   if((argc == 2) 
      && (argv[1][0] == '-')
      && (argv[1][1] == 'a')
      && (argv[1][2] == 0))
   {
      std::cout << "Press any key to continue...";
      std::cin.get();
   }
   return (failures == expected_failures)
       ? 0
       : (failures != 0) ? static_cast<int>(failures) : -1;
}


//
// this one is to verify that a constant is indeed a
// constant-integral-expression:
//
// HP aCC cannot deal with missing names for template value parameters
template <bool b>
struct checker
{
   static void check(bool, bool, const char*, bool){ ++test_count; }
};

template <>
struct checker<false>
{
   static void check(bool o, bool n, const char* name, bool soft)
   {
      ++test_count;
      ++failures;
      // if this is a soft test, then failure is expected,
      // or may depend upon factors outside our control
      // (like compiler options)...
      if(soft)++expected_failures;
      std::cout << "checking value of " << name << "...failed" << std::endl;
      std::cout << "\tfound: " << n << " expected " << o << std::endl;
   }
};

template <class T>
struct typify{};

template <class T, class U>
struct type_checker
{
   static void check(const char* TT, const char*, const char* expression)
   {
      ++test_count;
      if(typeid(typify<T>) != typeid(typify<U>))
      {
         ++failures;
         std::cout << "checking type of " << expression << "...failed" << std::endl;
         std::cout << "   evaluating:  type_checker<" << TT << "," << expression << ">" << std::endl;
         std::cout << "   expected:    type_checker<" << TT << "," << TT << ">" << std::endl;
         std::cout << "   but got:     " << typeid(type_checker<T,U>).name() << std::endl;
      }
   }
};

#ifndef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
template <class T>
struct type_checker<T,T>
{
   static void check(const char*, const char*, const char*)
   {
      ++test_count;
   }
};
#endif


#define value_test(v, x) checker<(v == x)>::check(v, x, #x, false);
#define soft_value_test(v, x) checker<(v == x)>::check(v, x, #x, true);

#define value_fail(v, x) \
      ++test_count; \
      ++failures; \
      ++expected_failures;\
      std::cout << "checking value of " << #x << "...failed" << std::endl; \
      std::cout << "   " #x " does not compile on this compiler" << std::endl;


#define type_test(v, x) type_checker<v,x>::check(#v, #x, #x);
#define type_test3(v, x, z) type_checker<v,x,z>::check(#v, #x "," #z, #x "," #z);
#ifndef SHORT_TRANSFORM_TEST
#define transform_check(name, from_suffix, to_suffix)\
   type_test(bool to_suffix, name<bool from_suffix>::type);\
   type_test(char to_suffix, name<char from_suffix>::type);\
   type_test(wchar_t to_suffix, name<wchar_t from_suffix>::type);\
   type_test(signed char to_suffix, name<signed char from_suffix>::type);\
   type_test(unsigned char to_suffix, name<unsigned char from_suffix>::type);\
   type_test(short to_suffix, name<short from_suffix>::type);\
   type_test(unsigned short to_suffix, name<unsigned short from_suffix>::type);\
   type_test(int to_suffix, name<int from_suffix>::type);\
   type_test(unsigned int to_suffix, name<unsigned int from_suffix>::type);\
   type_test(long to_suffix, name<long from_suffix>::type);\
   type_test(unsigned long to_suffix, name<unsigned long from_suffix>::type);\
   type_test(float to_suffix, name<float from_suffix>::type);\
   type_test(long double to_suffix, name<long double from_suffix>::type);\
   type_test(double to_suffix, name<double from_suffix>::type);\
   type_test(UDT to_suffix, name<UDT from_suffix>::type);\
   type_test(enum1 to_suffix, name<enum1 from_suffix>::type);
#else
#define transform_check(name, from_suffix, to_suffix)\
   type_test(int to_suffix, name<int from_suffix>::type);\
   type_test(UDT to_suffix, name<UDT from_suffix>::type);\
   type_test(enum1 to_suffix, name<enum1 from_suffix>::type);
#endif

#define boost_dummy_macro_param

template <class T>
struct test_align
{
   struct padded
   {
      char c;
      T t;
   };
   static void do_it()
   {
      padded p;
      unsigned a = reinterpret_cast<char*>(&(p.t)) - reinterpret_cast<char*>(&p);
      ++test_count;
      // only fail if we do not have a multiple of the actual value:
      if((a > ::boost::alignment_of<T>::value) || (a % ::boost::alignment_of<T>::value))
      {
         ++failures;
         std::cout << "checking value of " << typeid(boost::alignment_of<T>).name() << "...failed" << std::endl;
         std::cout << "\tfound: " << boost::alignment_of<T>::value << " expected " << a << std::endl;
      }
      // suppress warnings about unused variables:
      (void)p;
      (void)a;
   }
};
#ifndef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
template <class T>
struct test_align<T&>
{
   static void do_it()
   {
      //
      // we can't do the usual test because we can't take the address
      // of a reference, so check that the result is the same as for a
      // pointer type instead:
      unsigned a = boost::alignment_of<T*>::value;
      ++test_count;
      if(a != boost::alignment_of<T&>::value)
      {
         ++failures;
         std::cout << "checking value of " << typeid(boost::alignment_of<T&>).name() << "...failed" << std::endl;
         std::cout << "\tfound: " << boost::alignment_of<T&>::value << " expected " << a << std::endl;
      }
   }
};
#endif

#define align_test(T) test_align<T>::do_it()

template<class T>
struct test_type_with_align 
{
  typedef typename boost::type_with_alignment<
                     (boost::alignment_of<T>::value)>::type 
    align_t;

  static void do_it()
  {
    int align = boost::alignment_of<T>::value;
    int new_align = boost::alignment_of<align_t>::value;
    ++test_count;
    if (new_align % align != 0) {
      ++failures;
      std::cerr << "checking for an object with same alignment as " 
      << typeid(T).name() << "...failed" << std::endl;
      std::cerr << "\tfound: " << typeid(align_t).name() << std::endl;
    }
  }
};

#define type_with_align_test(T) test_type_with_align<T>::do_it()

//
// the following code allows us to test that a particular
// template functions correctly when instanciated inside another template
// (some bugs only show up in that situation).  For each template
// we declare one NESTED_DECL(classname) that sets up the template class
// and multiple NESTED_TEST(classname, template-arg) declarations, to carry
// the actual tests:
template <bool b>
struct nested_test
{
   typedef nested_test type;
   bool run_time_value;
   const char* what;
   nested_test(bool b2, const char* w) : run_time_value(b2), what(w) { check(); }
   void check()
   {
      ++test_count;
      if(b != run_time_value)
      {
         ++failures;
         std::cerr << "Mismatch between runtime and compile time values in " << what << std::endl;
      }
   }
};

#ifndef __SUNPRO_CC
#define NESTED_DECL(what)\
template <class T> \
struct BOOST_TT_JOIN(nested_tester_,what){\
   nested_test< (::boost::type_traits::ice_ne<0, ::boost::what<T>::value>::value)> tester;\
   BOOST_TT_JOIN(nested_tester_,what)(const char* s) : tester(::boost::what<T>::value, s){}\
};
#define NESTED_TEST(what, with)\
{BOOST_TT_JOIN(nested_tester_,what)<with> check(#what "<" #with ">"); (void)check;}
#else
#define NESTED_DECL(what)
#define NESTED_TEST(what, with)
#endif

#define BOOST_TT_JOIN( X, Y ) BOOST_DO_TT_JOIN( X, Y )
#define BOOST_DO_TT_JOIN( X, Y ) X##Y



//
// define some types to test with:
//
enum enum_UDT{ one, two, three };
struct UDT
{
   UDT(){};
   ~UDT(){};
   UDT(const UDT&);
   UDT& operator=(const UDT&);
   int i;

   void f1();
   int f2();
   int f3(int);
   int f4(int, float);
};

typedef void(*f1)();
typedef int(*f2)(int);
typedef int(*f3)(int, bool);
typedef void (UDT::*mf1)();
typedef int (UDT::*mf2)();
typedef int (UDT::*mf3)(int);
typedef int (UDT::*mf4)(int, float);
typedef int (UDT::*mp);
typedef int (UDT::*cmf)(int) const;

// cv-qualifiers applied to reference types should have no effect
// declare these here for later use with is_reference and remove_reference:
# ifdef BOOST_MSVC
#  pragma warning(push)
#  pragma warning(disable: 4181)
# elif defined(__ICL)
#  pragma warning(push)
#  pragma warning(disable: 21)
# endif
//
// This is intentional:
// r_type and cr_type should be the same type
// but some compilers wrongly apply cv-qualifiers
// to reference types (this may generate a warning
// on some compilers):
//
typedef int& r_type;
typedef const r_type cr_type;
# ifdef BOOST_MSVC
#  pragma warning(pop)
# elif defined(__ICL)
#  pragma warning(pop)
#  pragma warning(disable: 985) // identifier truncated in debug information
# endif

struct POD_UDT { int x; };
struct empty_UDT
{
   ~empty_UDT(){};
   empty_UDT& operator=(const empty_UDT&){ return *this; }
   bool operator==(const empty_UDT&)const
   { return true; }
};
struct empty_POD_UDT
{
   empty_POD_UDT& operator=(const empty_POD_UDT&){ return *this; }
   bool operator==(const empty_POD_UDT&)const
   { return true; }
};
union union_UDT
{
  int x;
  double y;
  ~union_UDT();
};
union POD_union_UDT
{
  int x;
  double y;
};
union empty_union_UDT
{
  ~empty_union_UDT();
};
union empty_POD_union_UDT{};

class Base { };

class Derived : public Base { };

class NonDerived { };

enum enum1
{
   one_,two_
};

enum enum2
{
   three_,four_
};

struct VB
{
   virtual ~VB(){};
};

struct VD : VB
{
   ~VD(){};
};
//
// struct non_pointer:
// used to verify that is_pointer does not return
// true for class types that implement operator void*()
//
struct non_pointer
{
   operator void*(){return this;}
};
struct non_int_pointer
{
   int i;
   operator int*(){return &i;}
};
struct int_constructible
{
   int_constructible(int);
};
struct int_convertible
{
   operator int();
};
//
// struct non_empty:
// used to verify that is_empty does not emit
// spurious warnings or errors.
//
struct non_empty : private boost::noncopyable
{
   int i;
};
//
// abstract base classes:
struct test_abc1
{
   virtual void foo() = 0;
   virtual void foo2() = 0;
};

struct test_abc2
{
   virtual void foo() = 0;
   virtual void foo2() = 0;
};

struct incomplete_type;


#endif // BOOST_TYPE_TRAITS_TEST_HPP










