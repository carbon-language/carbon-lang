// RUN: %clang_cc1 -fsyntax-only -verify %s

// <rdar://problem/8124080>
template<typename _Alloc> class allocator;
template<class _CharT> struct char_traits;
template<typename _CharT, typename _Traits = char_traits<_CharT>,            
         typename _Alloc = allocator<_CharT> >
class basic_string;
template<typename _CharT, typename _Traits, typename _Alloc>
const typename basic_string<_CharT, _Traits, _Alloc>::size_type   
basic_string<_CharT, _Traits, _Alloc>::_Rep::_S_max_size // expected-error{{no member named '_Rep' in 'basic_string<_CharT, _Traits, _Alloc>'}}
  = (((npos - sizeof(_Rep_base))/sizeof(_CharT)) - 1) / 4; 

// PR7118
template<typename T>
class Foo {
  class Bar;
  void f() {
    Bar i;
  }
};

// PR7625
template<typename T> struct a : T {
 struct x : T {
   int aa() { return p; } // expected-error{{use of undeclared identifier 'p'}}
 };
};

// rdar://8605381
namespace rdar8605381 {
struct X {};

struct Y { // expected-note{{candidate}}
  Y();
};

struct {
  Y obj;
} objs[] = {
  new Y // expected-error{{no viable conversion}}
};
}

// http://llvm.org/PR8234
namespace PR8234 {
template<typename Signature>
class callback
{
};

template<typename R , typename ARG_TYPE0>
class callback<R( ARG_TYPE0)>
{
    public:
        callback() {}
};

template< typename ARG_TYPE0>
class callback<void( ARG_TYPE0)>
{
    public:
        callback() {}
};

void f()
{
    callback<void(const int&)> op;
}
}
