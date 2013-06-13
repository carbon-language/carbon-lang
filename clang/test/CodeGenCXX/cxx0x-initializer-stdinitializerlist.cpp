// RUN: %clang_cc1 -std=c++11 -S -triple x86_64-none-linux-gnu -emit-llvm -o - %s | FileCheck %s

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

struct destroyme1 {
  ~destroyme1();
};
struct destroyme2 {
  ~destroyme2();
};
struct witharg1 {
  witharg1(const destroyme1&);
  ~witharg1();
};
struct wantslist1 {
  wantslist1(std::initializer_list<destroyme1>);
  ~wantslist1();
};

// CHECK: @_ZGR15globalInitList1 = private constant [3 x i32] [i32 1, i32 2, i32 3]
// CHECK: @globalInitList1 = global %{{[^ ]+}} { i32* getelementptr inbounds ([3 x i32]* @_ZGR15globalInitList1, i32 0, i32 0), i{{32|64}} 3 }
std::initializer_list<int> globalInitList1 = {1, 2, 3};

namespace thread_local_global_array {
  // FIXME: We should be able to constant-evaluate this even though the
  // initializer is not a constant expression (pointers to thread_local
  // objects aren't really a problem).
  //
  // CHECK: @_ZN25thread_local_global_array1xE = thread_local global
  // CHECK: @_ZGRN25thread_local_global_array1xE = private thread_local global [4 x i32]
  std::initializer_list<int> thread_local x = { 1, 2, 3, 4 };
}

// CHECK: @globalInitList2 = global %{{[^ ]+}} zeroinitializer
// CHECK: @_ZGR15globalInitList2 = private global [2 x %[[WITHARG:[^ ]*]]] zeroinitializer
// CHECK: appending global


// thread_local initializer:
// CHECK: define internal void
// CHECK: store i32 1, i32* getelementptr inbounds ([4 x i32]* @_ZGRN25thread_local_global_array1xE, i64 0, i64 0)
// CHECK: store i32 2, i32* getelementptr inbounds ([4 x i32]* @_ZGRN25thread_local_global_array1xE, i64 0, i64 1)
// CHECK: store i32 3, i32* getelementptr inbounds ([4 x i32]* @_ZGRN25thread_local_global_array1xE, i64 0, i64 2)
// CHECK: store i32 4, i32* getelementptr inbounds ([4 x i32]* @_ZGRN25thread_local_global_array1xE, i64 0, i64 3)
// CHECK: store i32* getelementptr inbounds ([4 x i32]* @_ZGRN25thread_local_global_array1xE, i64 0, i64 0),
// CHECK:       i32** getelementptr inbounds ({{.*}}* @_ZN25thread_local_global_array1xE, i32 0, i32 0), align 8
// CHECK: store i64 4, i64* getelementptr inbounds ({{.*}}* @_ZN25thread_local_global_array1xE, i32 0, i32 1), align 8


// CHECK: define internal void
// CHECK: call void @_ZN8witharg1C1ERK10destroyme1(%[[WITHARG]]* getelementptr inbounds ([2 x %[[WITHARG]]]* @_ZGR15globalInitList2, i{{32|64}} 0, i{{32|64}} 0
// CHECK: call void @_ZN8witharg1C1ERK10destroyme1(%[[WITHARG]]* getelementptr inbounds ([2 x %[[WITHARG]]]* @_ZGR15globalInitList2, i{{32|64}} 0, i{{32|64}} 1
// CHECK: __cxa_atexit
// CHECK: store %[[WITHARG]]* getelementptr inbounds ([2 x %[[WITHARG]]]* @_ZGR15globalInitList2, i64 0, i64 0),
// CHECK:       %[[WITHARG]]** getelementptr inbounds (%{{.*}}* @globalInitList2, i32 0, i32 0), align 8
// CHECK: store i64 2, i64* getelementptr inbounds (%{{.*}}* @globalInitList2, i32 0, i32 1), align 8
// CHECK: call void @_ZN10destroyme1D1Ev
// CHECK: call void @_ZN10destroyme1D1Ev
std::initializer_list<witharg1> globalInitList2 = {
  witharg1(destroyme1()), witharg1(destroyme1())
};

void fn1(int i) {
  // CHECK: define void @_Z3fn1i
  // temporary array
  // CHECK: [[array:%[^ ]+]] = alloca [3 x i32]
  // CHECK: getelementptr inbounds [3 x i32]* [[array]], i{{32|64}} 0
  // CHECK-NEXT: store i32 1, i32*
  // CHECK-NEXT: getelementptr
  // CHECK-NEXT: store
  // CHECK-NEXT: getelementptr
  // CHECK-NEXT: load
  // CHECK-NEXT: store
  // init the list
  // CHECK-NEXT: getelementptr
  // CHECK-NEXT: getelementptr inbounds [3 x i32]*
  // CHECK-NEXT: store i32*
  // CHECK-NEXT: getelementptr
  // CHECK-NEXT: store i{{32|64}} 3
  std::initializer_list<int> intlist{1, 2, i};
}

void fn2() {
  // CHECK: define void @_Z3fn2v
  void target(std::initializer_list<destroyme1>);
  // objects should be destroyed before dm2, after call returns
  // CHECK: call void @_Z6targetSt16initializer_listI10destroyme1E
  target({ destroyme1(), destroyme1() });
  // CHECK: call void @_ZN10destroyme1D1Ev
  destroyme2 dm2;
  // CHECK: call void @_ZN10destroyme2D1Ev
}

void fn3() {
  // CHECK: define void @_Z3fn3v
  // objects should be destroyed after dm2
  auto list = { destroyme1(), destroyme1() };
  destroyme2 dm2;
  // CHECK: call void @_ZN10destroyme2D1Ev
  // CHECK: call void @_ZN10destroyme1D1Ev
}

void fn4() {
  // CHECK: define void @_Z3fn4v
  void target(std::initializer_list<witharg1>);
  // objects should be destroyed before dm2, after call returns
  // CHECK: call void @_ZN8witharg1C1ERK10destroyme1
  // CHECK: call void @_Z6targetSt16initializer_listI8witharg1E
  target({ witharg1(destroyme1()), witharg1(destroyme1()) });
  // CHECK: call void @_ZN8witharg1D1Ev
  // CHECK: call void @_ZN10destroyme1D1Ev
  destroyme2 dm2;
  // CHECK: call void @_ZN10destroyme2D1Ev
}

void fn5() {
  // CHECK: define void @_Z3fn5v
  // temps should be destroyed before dm2
  // objects should be destroyed after dm2
  // CHECK: call void @_ZN8witharg1C1ERK10destroyme1
  auto list = { witharg1(destroyme1()), witharg1(destroyme1()) };
  // CHECK: call void @_ZN10destroyme1D1Ev
  destroyme2 dm2;
  // CHECK: call void @_ZN10destroyme2D1Ev
  // CHECK: call void @_ZN8witharg1D1Ev
}

void fn6() {
  // CHECK: define void @_Z3fn6v
  void target(const wantslist1&);
  // objects should be destroyed before dm2, after call returns
  // CHECK: call void @_ZN10wantslist1C1ESt16initializer_listI10destroyme1E
  // CHECK: call void @_Z6targetRK10wantslist1
  target({ destroyme1(), destroyme1() });
  // CHECK: call void @_ZN10wantslist1D1Ev
  // CHECK: call void @_ZN10destroyme1D1Ev
  destroyme2 dm2;
  // CHECK: call void @_ZN10destroyme2D1Ev
}

void fn7() {
  // CHECK: define void @_Z3fn7v
  // temps should be destroyed before dm2
  // object should be destroyed after dm2
  // CHECK: call void @_ZN10wantslist1C1ESt16initializer_listI10destroyme1E
  wantslist1 wl = { destroyme1(), destroyme1() };
  // CHECK: call void @_ZN10destroyme1D1Ev
  destroyme2 dm2;
  // CHECK: call void @_ZN10destroyme2D1Ev
  // CHECK: call void @_ZN10wantslist1D1Ev
}

void fn8() {
  // CHECK: define void @_Z3fn8v
  void target(std::initializer_list<std::initializer_list<destroyme1>>);
  // objects should be destroyed before dm2, after call returns
  // CHECK: call void @_Z6targetSt16initializer_listIS_I10destroyme1EE
  std::initializer_list<destroyme1> inner;
  target({ inner, { destroyme1() } });
  // CHECK: call void @_ZN10destroyme1D1Ev
  // Only one destroy loop, since only one inner init list is directly inited.
  // CHECK-NOT: call void @_ZN10destroyme1D1Ev
  destroyme2 dm2;
  // CHECK: call void @_ZN10destroyme2D1Ev
}

void fn9() {
  // CHECK: define void @_Z3fn9v
  // objects should be destroyed after dm2
  std::initializer_list<destroyme1> inner;
  std::initializer_list<std::initializer_list<destroyme1>> list =
      { inner, { destroyme1() } };
  destroyme2 dm2;
  // CHECK: call void @_ZN10destroyme2D1Ev
  // CHECK: call void @_ZN10destroyme1D1Ev
  // Only one destroy loop, since only one inner init list is directly inited.
  // CHECK-NOT: call void @_ZN10destroyme1D1Ev
  // CHECK: ret void
}

struct haslist1 {
  std::initializer_list<int> il;
  haslist1();
};

// CHECK: define void @_ZN8haslist1C2Ev
haslist1::haslist1()
// CHECK: alloca [3 x i32]
// CHECK: store i32 1
// CHECK: store i32 2
// CHECK: store i32 3
// CHECK: store i{{32|64}} 3
  : il{1, 2, 3}
{
  destroyme2 dm2;
}

struct haslist2 {
  std::initializer_list<destroyme1> il;
  haslist2();
};

// CHECK: define void @_ZN8haslist2C2Ev
haslist2::haslist2()
  : il{destroyme1(), destroyme1()}
{
  destroyme2 dm2;
  // CHECK: call void @_ZN10destroyme2D1Ev
  // CHECK: call void @_ZN10destroyme1D1Ev
}

void fn10() {
  // CHECK: define void @_Z4fn10v
  // CHECK: alloca [3 x i32]
  // CHECK: call noalias i8* @_Znw{{[jm]}}
  // CHECK: store i32 1
  // CHECK: store i32 2
  // CHECK: store i32 3
  // CHECK: store i32*
  // CHECK: store i{{32|64}} 3
  (void) new std::initializer_list<int> {1, 2, 3};
}

void fn11() {
  // CHECK: define void @_Z4fn11v
  (void) new std::initializer_list<destroyme1> {destroyme1(), destroyme1()};
  // CHECK: call void @_ZN10destroyme1D1Ev
  destroyme2 dm2;
  // CHECK: call void @_ZN10destroyme2D1Ev
}

namespace PR12178 {
  struct string {
    string(int);
    ~string();
  };

  struct pair {
    string a;
    int b;
  };

  struct map {
    map(std::initializer_list<pair>);
  };

  map m{ {1, 2}, {3, 4} };
}

namespace rdar13325066 {
  struct X { ~X(); };

  // CHECK: define void @_ZN12rdar133250664loopERNS_1XES1_
  void loop(X &x1, X &x2) {
    // CHECK: br label
    // CHECK: br i1
    // CHECK: br label
    // CHECK call void @_ZN12rdar133250661XD1Ev
    // CHECK: br label
    // CHECK: br label
    // CHECK: call void @_ZN12rdar133250661XD1Ev
    // CHECK: br i1
    // CHECK: br label
    // CHECK: ret void
    for (X x : { x1, x2 }) { }
  }
}

namespace dtors {
  struct S {
    S();
    ~S();
  };
  void z();

  // CHECK: define void @_ZN5dtors1fEv(
  void f() {
    // CHECK: call void @_ZN5dtors1SC1Ev(
    // CHECK: call void @_ZN5dtors1SC1Ev(
    std::initializer_list<S>{ S(), S() };

    // Destruction loop for underlying array.
    // CHECK: br label
    // CHECK: call void @_ZN5dtors1SD1Ev(
    // CHECK: br i1

    // CHECK: call void @_ZN5dtors1zEv(
    z();

    // CHECK-NOT: call void @_ZN5dtors1SD1Ev(
  }

  // CHECK: define void @_ZN5dtors1gEv(
  void g() {
    // CHECK: call void @_ZN5dtors1SC1Ev(
    // CHECK: call void @_ZN5dtors1SC1Ev(
    auto x = std::initializer_list<S>{ S(), S() };

    // Destruction loop for underlying array.
    // CHECK: br label
    // CHECK: call void @_ZN5dtors1SD1Ev(
    // CHECK: br i1

    // CHECK: call void @_ZN5dtors1zEv(
    z();

    // CHECK-NOT: call void @_ZN5dtors1SD1Ev(
  }

  // CHECK: define void @_ZN5dtors1hEv(
  void h() {
    // CHECK: call void @_ZN5dtors1SC1Ev(
    // CHECK: call void @_ZN5dtors1SC1Ev(
    std::initializer_list<S> x = { S(), S() };

    // CHECK-NOT: call void @_ZN5dtors1SD1Ev(

    // CHECK: call void @_ZN5dtors1zEv(
    z();

    // Destruction loop for underlying array.
    // CHECK: br label
    // CHECK: call void @_ZN5dtors1SD1Ev(
    // CHECK: br i1
  }
}
