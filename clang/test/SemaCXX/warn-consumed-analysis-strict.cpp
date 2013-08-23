// RUN: %clang_cc1 -fsyntax-only -verify -Wconsumed-strict -std=c++11 %s

#define CALLABLE_WHEN_UNCONSUMED __attribute__ ((callable_when_unconsumed))
#define CONSUMES __attribute__ ((consumes))
#define TESTS_UNCONSUMED __attribute__ ((tests_unconsumed))

typedef decltype(nullptr) nullptr_t;

template <typename T>
class ConsumableClass {
  T var;
  
  public:
  ConsumableClass(void);
  ConsumableClass(T val);
  ConsumableClass(ConsumableClass<T> &other);
  ConsumableClass(ConsumableClass<T> &&other);
  
  ConsumableClass<T>& operator=(ConsumableClass<T>  &other);
  ConsumableClass<T>& operator=(ConsumableClass<T> &&other);
  ConsumableClass<T>& operator=(nullptr_t);
  
  template <typename U>
  ConsumableClass<T>& operator=(ConsumableClass<U>  &other);
  
  template <typename U>
  ConsumableClass<T>& operator=(ConsumableClass<U> &&other);
  
  void operator*(void) const CALLABLE_WHEN_UNCONSUMED;
  
  bool isValid(void) const TESTS_UNCONSUMED;
  
  void constCall(void) const;
  void nonconstCall(void);
  
  void consume(void) CONSUMES;
};

void baf0(ConsumableClass<int>  &var);
void baf1(ConsumableClass<int>  *var);

void testIfStmt(void) {
  ConsumableClass<int> var;
  
  if (var.isValid()) { // expected-warning {{unnecessary test. Variable 'var' is known to be in the 'consumed' state}}
    
    // Empty
    
  } else {
    // Empty
  }
}

void testConditionalMerge(void) {
  ConsumableClass<int> var;
  
  if (var.isValid()) {// expected-warning {{unnecessary test. Variable 'var' is known to be in the 'consumed' state}}
    
    // Empty
  }
  
  *var; // expected-warning {{invocation of method 'operator*' on object 'var' while it is in an unknown state}}
  
  if (var.isValid()) {
    // Empty
    
  } else {
    // Empty
  }
  
  *var; // expected-warning {{invocation of method 'operator*' on object 'var' while it is in an unknown state}}
}

void testCallingConventions(void) {
  ConsumableClass<int> var(42);
  
  baf0(var);  
  *var; // expected-warning {{invocation of method 'operator*' on object 'var' while it is in an unknown state}}
  
  var = ConsumableClass<int>(42);
  baf1(&var);  
  *var; // expected-warning {{invocation of method 'operator*' on object 'var' while it is in an unknown state}}
}

void testMoveAsignmentish(void) {
  ConsumableClass<int> var;
  
  var = nullptr;
  *var; // expected-warning {{invocation of method 'operator*' on object 'var' while it is in an unknown state}}
}

void testConstAndNonConstMemberFunctions(void) {
  ConsumableClass<int> var(42);
  
  var.constCall();
  *var;
  
  var.nonconstCall();
  *var; // expected-warning {{invocation of method 'operator*' on object 'var' while it is in an unknown state}}
}

void testSimpleForLoop(void) {
  ConsumableClass<int> var;
  
  for (int i = 0; i < 10; ++i) {
    *var; // expected-warning {{invocation of method 'operator*' on object 'var' while it is in an unknown state}}
  }
  
  *var; // expected-warning {{invocation of method 'operator*' on object 'var' while it is in an unknown state}}
}

void testSimpleWhileLoop(void) {
  int i = 0;
  
  ConsumableClass<int> var;
  
  while (i < 10) {
    *var; // expected-warning {{invocation of method 'operator*' on object 'var' while it is in an unknown state}}
    ++i;
  }
  
  *var; // expected-warning {{invocation of method 'operator*' on object 'var' while it is in an unknown state}}
}
