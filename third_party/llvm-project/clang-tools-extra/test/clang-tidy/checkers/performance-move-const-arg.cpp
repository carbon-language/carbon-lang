// RUN: %check_clang_tidy %s performance-move-const-arg %t

namespace std {
template <typename>
struct remove_reference;

template <typename _Tp>
struct remove_reference {
  typedef _Tp type;
};

template <typename _Tp>
struct remove_reference<_Tp &> {
  typedef _Tp type;
};

template <typename _Tp>
struct remove_reference<_Tp &&> {
  typedef _Tp type;
};

template <typename _Tp>
constexpr typename std::remove_reference<_Tp>::type &&move(_Tp &&__t) {
  return static_cast<typename std::remove_reference<_Tp>::type &&>(__t);
}

template <typename _Tp>
constexpr _Tp &&
forward(typename remove_reference<_Tp>::type &__t) noexcept {
  return static_cast<_Tp &&>(__t);
}

} // namespace std

class A {
public:
  A() {}
  A(const A &rhs) {}
  A(A &&rhs) {}
};

struct TriviallyCopyable {
  int i;
};

void f(TriviallyCopyable) {}

void g() {
  TriviallyCopyable obj;
  f(std::move(obj));
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: std::move of the variable 'obj' of the trivially-copyable type 'TriviallyCopyable' has no effect; remove std::move() [performance-move-const-arg]
  // CHECK-FIXES: f(obj);
}

int f1() {
  return std::move(42);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: std::move of the expression of the trivially-copyable type 'int' has no effect; remove std::move() [performance-move-const-arg]
  // CHECK-FIXES: return 42;
}

int f2(int x2) {
  return std::move(x2);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: std::move of the variable 'x2' of the trivially-copyable type 'int'
  // CHECK-FIXES: return x2;
}

int *f3(int *x3) {
  return std::move(x3);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: std::move of the variable 'x3' of the trivially-copyable type 'int *'
  // CHECK-FIXES: return x3;
}

A f4(A x4) { return std::move(x4); }

A f5(const A x5) {
  return std::move(x5);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: std::move of the const variable 'x5' has no effect; remove std::move() or make the variable non-const [performance-move-const-arg]
  // CHECK-FIXES: return x5;
}

template <typename T>
T f6(const T x6) {
  return std::move(x6);
}

void f7() { int a = f6(10); }

#define M1(x) x
void f8() {
  const A a;
  M1(A b = std::move(a);)
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: std::move of the const variable 'a' has no effect; remove std::move() or make the variable non-const
  // CHECK-FIXES: M1(A b = a;)
}

#define M2(x) std::move(x)
int f9() { return M2(1); }

template <typename T>
T f10(const int x10) {
  return std::move(x10);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: std::move of the const variable 'x10' of the trivially-copyable type 'const int' has no effect; remove std::move() [performance-move-const-arg]
  // CHECK-FIXES: return x10;
}
void f11() {
  f10<int>(1);
  f10<double>(1);
}

class NoMoveSemantics {
public:
  NoMoveSemantics();
  NoMoveSemantics(const NoMoveSemantics &);

  NoMoveSemantics &operator=(const NoMoveSemantics &);
};

void callByConstRef(const NoMoveSemantics &);
void callByConstRef(int i, const NoMoveSemantics &);

void moveToConstReferencePositives() {
  NoMoveSemantics obj;

  // Basic case.
  callByConstRef(std::move(obj));
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: passing result of std::move() as
  // CHECK-FIXES: callByConstRef(obj);

  // Also works for second argument.
  callByConstRef(1, std::move(obj));
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: passing result of std::move() as
  // CHECK-FIXES: callByConstRef(1, obj);

  // Works if std::move() applied to a temporary.
  callByConstRef(std::move(NoMoveSemantics()));
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: passing result of std::move() as
  // CHECK-FIXES: callByConstRef(NoMoveSemantics());

  // Works if calling a copy constructor.
  NoMoveSemantics other(std::move(obj));
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: passing result of std::move() as
  // CHECK-FIXES: NoMoveSemantics other(obj);

  // Works if calling assignment operator.
  other = std::move(obj);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: passing result of std::move() as
  // CHECK-FIXES: other = obj;
}

class MoveSemantics {
public:
  MoveSemantics();
  MoveSemantics(MoveSemantics &&);

  MoveSemantics &operator=(MoveSemantics &&);
};

void callByValue(MoveSemantics);

void callByRValueRef(MoveSemantics &&);

template <class T>
void templateFunction(T obj) {
  T other = std::move(obj);
}

#define M3(T, obj)            \
  do {                        \
    T other = std::move(obj); \
  } while (true)

#define CALL(func) (func)()

void moveToConstReferenceNegatives() {
  // No warning when actual move takes place.
  MoveSemantics move_semantics;
  callByValue(std::move(move_semantics));
  callByRValueRef(std::move(move_semantics));
  MoveSemantics other(std::move(move_semantics));
  other = std::move(move_semantics);

  // No warning if std::move() not used.
  NoMoveSemantics no_move_semantics;
  callByConstRef(no_move_semantics);

  // No warning if instantiating a template.
  templateFunction(no_move_semantics);

  // No warning inside of macro expansions.
  M3(NoMoveSemantics, no_move_semantics);

  // No warning inside of macro expansion, even if the macro expansion is inside
  // a lambda that is, in turn, an argument to a macro.
  CALL([no_move_semantics] { M3(NoMoveSemantics, no_move_semantics); });

  auto lambda = [] {};
  auto lambda2 = std::move(lambda);
}

class MoveOnly {
public:
  MoveOnly(const MoveOnly &other) = delete;
  MoveOnly &operator=(const MoveOnly &other) = delete;
  MoveOnly(MoveOnly &&other) = default;
  MoveOnly &operator=(MoveOnly &&other) = default;
};
template <class T>
void Q(T);
void moveOnlyNegatives(MoveOnly val) {
  Q(std::move(val));
}

void fmovable(MoveSemantics);

void lambda1() {
  auto f = [](MoveSemantics m) {
    fmovable(std::move(m));
  };
  f(MoveSemantics());
}

template<class T> struct function {};

template<typename Result, typename... Args>
class function<Result(Args...)> {
public:
  function() = default;
  void operator()(Args... args) const {
    fmovable(std::forward<Args>(args)...);
  }
};

void functionInvocation() {
  function<void(MoveSemantics)> callback;
  MoveSemantics m;
  callback(std::move(m));
}

void lambda2() {
  function<void(MoveSemantics)> callback;

  auto f = [callback = std::move(callback)](MoveSemantics m) mutable {
    // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: std::move of the variable 'callback' of the trivially-copyable type 'function<void (MoveSemantics)>' has no effect; remove std::move()
    // CHECK-FIXES: auto f = [callback = callback](MoveSemantics m) mutable {
    callback(std::move(m));
  };
  f(MoveSemantics());
}

void showInt(int &&v);
void showInt(int v1, int &&v2);
void showPointer(const char *&&s);
void showPointer2(const char *const &&s);
void showTriviallyCopyable(TriviallyCopyable &&obj);
void showTriviallyCopyablePointer(const TriviallyCopyable *&&obj);
void testFunctions() {
  int a = 10;
  showInt(std::move(a));
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: std::move of the variable 'a' of the trivially-copyable type 'int' has no effect [performance-move-const-arg]
  // CHECK-MESSAGES: :[[@LINE-10]]:20: note: consider changing the 1st parameter of 'showInt' from 'int &&' to 'const int &'
  showInt(int());
  showInt(a, std::move(a));
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: std::move of the variable 'a' of the trivially-copyable type 'int' has no effect [performance-move-const-arg]
  // CHECK-MESSAGES: :[[@LINE-13]]:28: note: consider changing the 2nd parameter of 'showInt' from 'int &&' to 'const int &'
  const char* s = "";
  showPointer(std::move(s));
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: std::move of the variable 's' of the trivially-copyable type 'const char *' has no effect [performance-move-const-arg]
  // CHECK-MESSAGES: :[[@LINE-16]]:32: note: consider changing the 1st parameter of 'showPointer' from 'const char *&&' to 'const char *'
  showPointer2(std::move(s)); 
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: std::move of the variable 's' of the trivially-copyable type 'const char *' has no effect [performance-move-const-arg]
  // CHECK-MESSAGES: :[[@LINE-18]]:39: note: consider changing the 1st parameter of 'showPointer2' from 'const char *const &&' to 'const char *const'
  TriviallyCopyable *obj = new TriviallyCopyable();
  showTriviallyCopyable(std::move(*obj));
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: std::move of the expression of the trivially-copyable type 'TriviallyCopyable' has no effect [performance-move-const-arg]
  // CHECK-MESSAGES: :[[@LINE-21]]:48: note: consider changing the 1st parameter of 'showTriviallyCopyable' from 'TriviallyCopyable &&' to 'const TriviallyCopyable &'
  showTriviallyCopyablePointer(std::move(obj));
  // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: std::move of the variable 'obj' of the trivially-copyable type 'TriviallyCopyable *' has no effect [performance-move-const-arg]
  // CHECK-MESSAGES: :[[@LINE-23]]:62: note: consider changing the 1st parameter of 'showTriviallyCopyablePointer' from 'const TriviallyCopyable *&&' to 'const TriviallyCopyable *'
}
template <class T>
void forwardToShowInt(T && t) {
  showInt(static_cast<T &&>(t));
}
void testTemplate() {
  int a = 10;
  forwardToShowInt(std::move(a));
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: std::move of the variable 'a' of the trivially-copyable type 'int' has no effect [performance-move-const-arg]
}

struct Tmp {
  Tmp();
  Tmp(int &&a);
  Tmp(int v1, int &&a);
  Tmp(const char *&&s);
  Tmp(TriviallyCopyable&& obj);
  Tmp(const TriviallyCopyable *&&obj);
  void showTmp(TriviallyCopyable&& t);
  static void showTmpStatic(TriviallyCopyable&& t);
};
void testMethods() {
  Tmp t;
  int a = 10;
  Tmp t1(std::move(a));
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: std::move of the variable 'a' of the trivially-copyable type 'int' has no effect [performance-move-const-arg]
  // CHECK-MESSAGES: :[[@LINE-13]]:13: note: consider changing the 1st parameter of 'Tmp' from 'int &&' to 'const int &'
  Tmp t2(a, std::move(a));
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: std::move of the variable 'a' of the trivially-copyable type 'int' has no effect [performance-move-const-arg]
  // CHECK-MESSAGES: :[[@LINE-15]]:21: note: consider changing the 2nd parameter of 'Tmp' from 'int &&' to 'const int &'
  const char* s = "";
  Tmp t3(std::move(s));
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: std::move of the variable 's' of the trivially-copyable type 'const char *' has no effect [performance-move-const-arg]
  // CHECK-MESSAGES: :[[@LINE-18]]:21: note: consider changing the 1st parameter of 'Tmp' from 'const char *&&' to 'const char *'
  TriviallyCopyable *obj = new TriviallyCopyable();
  Tmp t4(std::move(*obj));
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: std::move of the expression of the trivially-copyable type 'TriviallyCopyable' has no effect [performance-move-const-arg]
  // CHECK-MESSAGES: :[[@LINE-21]]:27: note: consider changing the 1st parameter of 'Tmp' from 'TriviallyCopyable &&' to 'const TriviallyCopyable &'
  Tmp t5(std::move(obj));
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: std::move of the variable 'obj' of the trivially-copyable type 'TriviallyCopyable *' has no effect [performance-move-const-arg]
  // CHECK-MESSAGES: :[[@LINE-23]]:34: note: consider changing the 1st parameter of 'Tmp' from 'const TriviallyCopyable *&&' to 'const TriviallyCopyable *'
  t.showTmp(std::move(*obj));
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: std::move of the expression of the trivially-copyable type 'TriviallyCopyable' has no effect [performance-move-const-arg]
  // CHECK-MESSAGES: :[[@LINE-25]]:36: note: consider changing the 1st parameter of 'showTmp' from 'TriviallyCopyable &&' to 'const TriviallyCopyable &'
  Tmp::showTmpStatic(std::move(*obj));
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: std::move of the expression of the trivially-copyable type 'TriviallyCopyable' has no effect [performance-move-const-arg]
  // CHECK-MESSAGES: :[[@LINE-27]]:49: note: consider changing the 1st parameter of 'showTmpStatic' from 'TriviallyCopyable &&' to 'const TriviallyCopyable &'
}

void showA(A &&v) {}
void testA() {
  A a;
  showA(std::move(a));
}

void testFuncPointer() {
  int a = 10;
  void (*choice)(int, int &&);
  choice = showInt;
  choice(std::move(a), std::move(a));
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: std::move of the variable 'a' of the trivially-copyable type 'int' has no effect; remove std::move() [performance-move-const-arg]
  // CHECK-FIXES: choice(a, std::move(a));
  // CHECK-MESSAGES: :[[@LINE-3]]:24: warning: std::move of the variable 'a' of the trivially-copyable type 'int' has no effect [performance-move-const-arg]
}
