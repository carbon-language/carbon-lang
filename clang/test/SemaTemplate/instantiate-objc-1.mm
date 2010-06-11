// RUN: %clang_cc1 -fsyntax-only -verify %s

// Obj-C string literal expressions
template <typename T> struct StringTest {
  void f() {
    (void)@"Hello";
  }
};

template struct StringTest<int>;
template struct StringTest<double>;

// @selector expressions
template <typename T> struct SelectorTest {
  SEL f() {
    return @selector(multiple:arguments:);
  }
  SEL f2() {
    return @selector(multiple:arguments:);
  }
};

template struct SelectorTest<int>;
template struct SelectorTest<double>;

// @protocol expressions
@protocol P
@end

template <typename T> struct ProtocolTest {
  void f() {
    (void)@protocol(P);
  }
};

template struct ProtocolTest<int>;
template struct ProtocolTest<double>;

// @encode expressions
template <typename T> struct EncodeTest {
  static const char *encode(T t) { 
    return @encode(T);
  }
};

template struct EncodeTest<int>;
template struct EncodeTest<double>;
template struct EncodeTest<wchar_t>;
