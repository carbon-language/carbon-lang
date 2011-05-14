// RUN: %clang_cc1 -std=c++0x -fsyntax-only -verify %s

template<typename S>
struct A {
  typedef S B;
  template<typename T> using C = typename T::B;
  template<typename T> struct D {
    template<typename U> using E = typename A<U>::template C<A<T>>;
    template<typename U> using F = A<E<U>>;
    template<typename U> using G = C<F<U>>;
    G<T> g;
  };
  typedef decltype(D<B>().g) H;
  D<H> h;
  template<typename T> using I = A<decltype(h.g)>;
  template<typename T> using J = typename A<decltype(h.g)>::template C<I<T>>;
};

A<int> a;
A<char>::D<double> b;

template<typename T> T make();

namespace X {
  template<typename T> struct traits {
    typedef T thing;
    typedef decltype(val(make<thing>())) inner_ptr;

    template<typename U> using rebind_thing = typename thing::template rebind<U>;
    template<typename U> using rebind = traits<rebind_thing<U>>;

    inner_ptr &&alloc();
    void free(inner_ptr&&);
  };

  template<typename T> struct ptr_traits {
    typedef T *type;
  };
  template<typename T> using ptr = typename ptr_traits<T>::type;

  template<typename T> struct thing {
    typedef T inner;
    typedef ptr<inner> inner_ptr;
    typedef traits<thing<inner>> traits_type;

    template<typename U> using rebind = thing<U>;

    thing(traits_type &traits) : traits(traits), val(traits.alloc()) {}
    ~thing() { traits.free(static_cast<inner_ptr&&>(val)); }

    traits_type &traits;
    inner_ptr val;

    friend inner_ptr val(const thing &t) { return t.val; }
  };

  template<> struct ptr_traits<bool> {
    typedef bool &type;
  };
  template<> bool &traits<thing<bool>>::alloc() { static bool b; return b; }
  template<> void traits<thing<bool>>::free(bool&) {}
}

typedef X::traits<X::thing<int>> itt;

itt::thing::traits_type itr;
itt::thing ith(itr);

itt::rebind<bool> btr;
itt::rebind_thing<bool> btt(btr);
