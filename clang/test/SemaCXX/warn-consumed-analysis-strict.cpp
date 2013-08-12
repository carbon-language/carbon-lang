// RUN: %clang_cc1 -fsyntax-only -verify -Wconsumed-strict -std=c++11 %s

#define CALLABLE_WHEN_UNCONSUMED __attribute__ ((callable_when_unconsumed))
#define CONSUMES __attribute__ ((consumes))
#define TESTS_UNCONSUMED __attribute__ ((tests_unconsumed))

typedef decltype(nullptr) nullptr_t;

template <typename T>
class Bar {
  T var;
  
  public:
  Bar(void);
  Bar(T val);
  Bar(Bar<T> &other);
  Bar(Bar<T> &&other);
  
  Bar<T>& operator=(Bar<T>  &other);
  Bar<T>& operator=(Bar<T> &&other);
  Bar<T>& operator=(nullptr_t);
  
  template <typename U>
  Bar<T>& operator=(Bar<U>  &other);
  
  template <typename U>
  Bar<T>& operator=(Bar<U> &&other);
  
  void operator*(void) const CALLABLE_WHEN_UNCONSUMED;
  
  bool isValid(void) const TESTS_UNCONSUMED;
  
  void constCall(void) const;
  void nonconstCall(void);
  
  void consume(void) CONSUMES;
};

void baf0(Bar<int>  &var);
void baf1(Bar<int>  *var);

void testIfStmt(void) {
  Bar<int> var;
  
  if (var.isValid()) { // expected-warning {{unnecessary test. Variable 'var' is known to be in the 'consumed' state}}
    
    // Empty
    
  } else {
    // Empty
  }
}

void testConditionalMerge(void) {
  Bar<int> var;
  
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
  Bar<int> var(42);
  
  baf0(var);  
  *var; // expected-warning {{invocation of method 'operator*' on object 'var' while it is in an unknown state}}
  
  var = Bar<int>(42);
  baf1(&var);  
  *var; // expected-warning {{invocation of method 'operator*' on object 'var' while it is in an unknown state}}
}

void testMoveAsignmentish(void) {
  Bar<int> var;
  
  var = nullptr;
  *var; // expected-warning {{invocation of method 'operator*' on object 'var' while it is in an unknown state}}
}

void testConstAndNonConstMemberFunctions(void) {
  Bar<int> var(42);
  
  var.constCall();
  *var;
  
  var.nonconstCall();
  *var; // expected-warning {{invocation of method 'operator*' on object 'var' while it is in an unknown state}}
}

void testSimpleForLoop(void) {
  Bar<int> var;
  
  for (int i = 0; i < 10; ++i) {
    *var; // expected-warning {{invocation of method 'operator*' on object 'var' while it is in an unknown state}}
  }
  
  *var; // expected-warning {{invocation of method 'operator*' on object 'var' while it is in an unknown state}}
}

void testSimpleWhileLoop(void) {
  int i = 0;
  
  Bar<int> var;
  
  while (i < 10) {
    *var; // expected-warning {{invocation of method 'operator*' on object 'var' while it is in an unknown state}}
    ++i;
  }
  
  *var; // expected-warning {{invocation of method 'operator*' on object 'var' while it is in an unknown state}}
}
