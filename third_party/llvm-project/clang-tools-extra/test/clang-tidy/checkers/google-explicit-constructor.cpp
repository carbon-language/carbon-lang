// RUN: %check_clang_tidy %s google-explicit-constructor %t

namespace std {
  typedef decltype(sizeof(int)) size_t;

  // libc++'s implementation
  template <class _E>
  class initializer_list
  {
    const _E* __begin_;
    size_t    __size_;

    initializer_list(const _E* __b, size_t __s)
      : __begin_(__b),
        __size_(__s)
    {}

  public:
    typedef _E        value_type;
    typedef const _E& reference;
    typedef const _E& const_reference;
    typedef size_t    size_type;

    typedef const _E* iterator;
    typedef const _E* const_iterator;

    initializer_list() : __begin_(nullptr), __size_(0) {}

    size_t    size()  const {return __size_;}
    const _E* begin() const {return __begin_;}
    const _E* end()   const {return __begin_ + __size_;}
  };
}

struct A {
  A() {}
  A(int x, int y) {}

  explicit A(void *x) {}
  explicit A(void *x, void *y) {}
  explicit operator bool() const { return true; }

  operator double() const = delete;

  explicit A(const A& a) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: copy constructor should not be declared explicit [google-explicit-constructor]
  // CHECK-FIXES: {{^  }}A(const A& a) {}

  A(int x1);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: single-argument constructors must be marked explicit to avoid unintentional implicit conversions [google-explicit-constructor]
  // CHECK-FIXES: {{^  }}explicit A(int x1);

  A(double x2, double y = 3.14) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: constructors that are callable with a single argument must be marked explicit to avoid unintentional implicit conversions [google-explicit-constructor]
  // CHECK-FIXES: {{^  }}explicit A(double x2, double y = 3.14) {}

  template <typename... T>
  A(T&&... args);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: constructors that are callable with a single argument
  // CHECK-FIXES: {{^  }}explicit A(T&&... args);
};

inline A::A(int x1) {}

struct B {
  B(std::initializer_list<int> list1) {}
  B(const std::initializer_list<unsigned> &list2) {}
  B(std::initializer_list<unsigned> &&list3) {}

  operator bool() const { return true; }
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'operator bool' must be marked explicit to avoid unintentional implicit conversions [google-explicit-constructor]
  // CHECK-FIXES: {{^  }}explicit operator bool() const { return true; }

  operator double() const;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'operator double' must be marked explicit to avoid unintentional implicit conversions [google-explicit-constructor]
  // CHECK-FIXES: {{^  }}explicit operator double() const;

  explicit B(::std::initializer_list<double> list4) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: initializer-list constructor should not be declared explicit [google-explicit-constructor]
  // CHECK-FIXES: {{^  }}B(::std::initializer_list<double> list4) {}

  explicit B(const ::std::initializer_list<char> &list5) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: initializer-list constructor
  // CHECK-FIXES: {{^  }}B(const ::std::initializer_list<char> &list5) {}

  explicit B(::std::initializer_list<char> &&list6) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: initializer-list constructor
  // CHECK-FIXES: {{^  }}B(::std::initializer_list<char> &&list6) {}
};

inline B::operator double() const { return 0.0; }

struct StructWithFnPointer {
  void (*f)();
} struct_with_fn_pointer = {[] {}};

using namespace std;

struct C {
  C(initializer_list<int> list1) {}
  C(const initializer_list<unsigned> &list2) {}
  C(initializer_list<unsigned> &&list3) {}
};

template <typename T>
struct C2 {
  C2(initializer_list<int> list1) {}
  C2(const initializer_list<unsigned> &list2) {}
  C2(initializer_list<unsigned> &&list3) {}

  explicit C2(initializer_list<double> list4) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: initializer-list constructor
  // CHECK-FIXES: {{^  }}C2(initializer_list<double> list4) {}
};

template <typename T>
struct C3 {
  C3(initializer_list<T> list1) {}
  C3(const std::initializer_list<T*> &list2) {}
  C3(::std::initializer_list<T**> &&list3) {}

  template <typename U>
  C3(initializer_list<U> list3) {}
};

struct D {
  template <typename T>
  explicit D(T t) {}
};

template <typename T>
struct E {
  E(T *pt) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: single-argument constructors
  // CHECK-FIXES: {{^  }}explicit E(T *pt) {}
  template <typename U>
  E(U *pu) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: single-argument constructors
  // CHECK-FIXES: {{^  }}explicit E(U *pu) {}

  explicit E(T t) {}
  template <typename U>
  explicit E(U u) {}
};

void f(std::initializer_list<int> list) {
  D d(list);
  E<decltype(list)> e(list);
  E<int> e2(list);
}

template <typename T>
struct F {};

template<typename T>
struct G {
  operator bool() const;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'operator bool' must be marked
  // CHECK-FIXES: {{^}}  explicit operator bool() const;
  operator F<T>() const;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'operator F<type-parameter-0-0>' must be marked
  // CHECK-FIXES: {{^}}  explicit operator F<T>() const;
  template<typename U>
  operator F<U>*() const;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'operator F<type-parameter-1-0> *' must be marked
  // CHECK-FIXES: {{^}}  explicit operator F<U>*() const;
};

void f2() {
  G<int> a;
  (void)(F<int>)a;
  if (a) {}
  (void)(F<int>*)a;
  (void)(F<int*>*)a;

  G<double> b;
  (void)(F<double>)b;
  if (b) {}
  (void)(F<double>*)b;
  (void)(F<double*>*)b;
}

#define DEFINE_STRUCT_WITH_OPERATOR_BOOL(name) \
  struct name {                                \
    operator bool() const;                     \
  }

DEFINE_STRUCT_WITH_OPERATOR_BOOL(H);
