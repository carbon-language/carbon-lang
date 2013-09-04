//===-----------------------------------------------------------*- C++ -*--===//
//
// This file contains a shell implementation of the 'auto_ptr' type from the
// standard library. This shell aims to support the variations between standard
// library implementations.
//
// Variations for how 'auto_ptr' is presented:
// 1. Defined directly in namespace std
// 2. Use a versioned inline namespace in std (default on libc++).
//
// Use the preprocessor to define USE_INLINE_NAMESPACE=1 and use the second
// variation.
//
//===----------------------------------------------------------------------===//

namespace std {

#if USE_INLINE_NAMESPACE
inline namespace _1 {
#endif

template <class Y> struct auto_ptr_ref {
  Y *y_;
};

template <class X> class auto_ptr {
public:
  typedef X element_type;
  // D.10.1.1 construct/copy/destroy:
  explicit auto_ptr(X *p = 0) throw() {}
  auto_ptr(auto_ptr &) throw() {}
  template <class Y> auto_ptr(auto_ptr<Y> &) throw() {}
  auto_ptr &operator=(auto_ptr &) throw() { return *this; }
  template <class Y> auto_ptr &operator=(auto_ptr<Y> &) throw() {
    return *this;
  }
  auto_ptr &operator=(auto_ptr_ref<X> r) throw() { return *this; }
  ~auto_ptr() throw() {}
  // D.10.1.3 conversions:
  auto_ptr(auto_ptr_ref<X> r) throw() : x_(r.y_) {}
  template <class Y> operator auto_ptr_ref<Y>() throw() {
    auto_ptr_ref<Y> r;
    r.y_ = x_;
    return r;
  }
  template <class Y> operator auto_ptr<Y>() throw() { return auto_ptr<Y>(x_); }

private:
  X *x_;
};

template <> class auto_ptr<void> {
public:
  typedef void element_type;
};

#if USE_INLINE_NAMESPACE
} // namespace _1
#endif

} // end namespace std
