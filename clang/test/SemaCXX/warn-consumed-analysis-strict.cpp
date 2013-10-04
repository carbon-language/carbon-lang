// RUN: %clang_cc1 -fsyntax-only -verify -Wconsumed-strict -std=c++11 %s

#define CALLABLE_WHEN(...)      __attribute__ ((callable_when(__VA_ARGS__)))
#define CONSUMABLE(state)       __attribute__ ((consumable(state)))
#define CONSUMES                __attribute__ ((consumes))
#define RETURN_TYPESTATE(state) __attribute__ ((return_typestate(state)))
#define TESTS_UNCONSUMED        __attribute__ ((tests_unconsumed))

#define TEST_VAR(Var) Var.isValid()

typedef decltype(nullptr) nullptr_t;

template <typename T>
class CONSUMABLE(unconsumed) ConsumableClass {
  T var;
  
  public:
  ConsumableClass();
  ConsumableClass(nullptr_t p) RETURN_TYPESTATE(consumed);
  ConsumableClass(T val);
  ConsumableClass(ConsumableClass<T> &other);
  ConsumableClass(ConsumableClass<T> &&other);
  
  ConsumableClass<T>& operator=(ConsumableClass<T>  &other);
  ConsumableClass<T>& operator=(ConsumableClass<T> &&other);
  ConsumableClass<T>& operator=(nullptr_t) CONSUMES;
  
  template <typename U>
  ConsumableClass<T>& operator=(ConsumableClass<U>  &other);
  
  template <typename U>
  ConsumableClass<T>& operator=(ConsumableClass<U> &&other);
  
  void operator()(int a) CONSUMES;
  void operator*() const CALLABLE_WHEN("unconsumed");
  void unconsumedCall() const CALLABLE_WHEN("unconsumed");
  
  bool isValid() const TESTS_UNCONSUMED;
  operator bool() const TESTS_UNCONSUMED;
  bool operator!=(nullptr_t) const TESTS_UNCONSUMED;
  
  void constCall() const;
  void nonconstCall();
  
  void consume() CONSUMES;
};

void testIfStmt() {
  ConsumableClass<int> var;
  
  if (var.isValid()) { // expected-warning {{unnecessary test. Variable 'var' is known to be in the 'consumed' state}}
    
    // Empty
    
  } else {
    // Empty
  }
}

void testNoWarnTestFromMacroExpansion() {
  ConsumableClass<int> var(42);
  
  if (TEST_VAR(var)) {
    *var;
  }
}
