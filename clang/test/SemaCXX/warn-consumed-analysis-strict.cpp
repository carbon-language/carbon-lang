// RUN: %clang_cc1 -fsyntax-only -verify -Wconsumed-strict -std=c++11 %s

#define CALLABLE_WHEN_UNCONSUMED __attribute__ ((callable_when_unconsumed))
#define CONSUMES __attribute__ ((consumes))
#define TESTS_UNCONSUMED __attribute__ ((tests_unconsumed))

#define TEST_VAR(Var) Var.isValid()

typedef decltype(nullptr) nullptr_t;

template <typename T>
class ConsumableClass {
  T var;
  
  public:
  ConsumableClass();
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
  void operator*() const CALLABLE_WHEN_UNCONSUMED;
  void unconsumedCall() const CALLABLE_WHEN_UNCONSUMED;
  
  bool isValid() const TESTS_UNCONSUMED;
  operator bool() const TESTS_UNCONSUMED;
  bool operator!=(nullptr_t) const TESTS_UNCONSUMED;
  
  void constCall() const;
  void nonconstCall();
  
  void consume() CONSUMES;
};

void baf0(ConsumableClass<int>  &var);
void baf1(ConsumableClass<int>  *var);

void testIfStmt() {
  ConsumableClass<int> var;
  
  if (var.isValid()) { // expected-warning {{unnecessary test. Variable 'var' is known to be in the 'consumed' state}}
    
    // Empty
    
  } else {
    // Empty
  }
}

void testComplexConditionals() {
  ConsumableClass<int> var0, var1, var2;
  
  // Coerce all variables into the unknown state.
  baf0(var0);
  baf0(var1);
  baf0(var2);
  
  if (var0 && var1) {
    *var0;
    *var1;
    
  } else {
    *var0; // expected-warning {{invocation of method 'operator*' on object 'var0' while it is in an unknown state}}
    *var1; // expected-warning {{invocation of method 'operator*' on object 'var1' while it is in an unknown state}}
  }
  
  if (var0 || var1) {
    *var0; // expected-warning {{invocation of method 'operator*' on object 'var0' while it is in an unknown state}}
    *var1; // expected-warning {{invocation of method 'operator*' on object 'var1' while it is in an unknown state}}
    
  } else {
    *var0; // expected-warning {{invocation of method 'operator*' on object 'var0' while it is in the 'consumed' state}}
    *var1; // expected-warning {{invocation of method 'operator*' on object 'var1' while it is in the 'consumed' state}}
  }
  
  if (var0 && !var1) {
    *var0;
    *var1; // expected-warning {{invocation of method 'operator*' on object 'var1' while it is in the 'consumed' state}}
    
  } else {
    *var0; // expected-warning {{invocation of method 'operator*' on object 'var0' while it is in an unknown state}}
    *var1; // expected-warning {{invocation of method 'operator*' on object 'var1' while it is in an unknown state}}
  }
  
  if (var0 || !var1) {
    *var0; // expected-warning {{invocation of method 'operator*' on object 'var0' while it is in an unknown state}}
    *var1; // expected-warning {{invocation of method 'operator*' on object 'var1' while it is in an unknown state}}
    
  } else {
    *var0; // expected-warning {{invocation of method 'operator*' on object 'var0' while it is in the 'consumed' state}}
    *var1;
  }
  
  if (!var0 && !var1) {
    *var0; // expected-warning {{invocation of method 'operator*' on object 'var0' while it is in the 'consumed' state}}
    *var1; // expected-warning {{invocation of method 'operator*' on object 'var1' while it is in the 'consumed' state}}
    
  } else {
    *var0; // expected-warning {{invocation of method 'operator*' on object 'var0' while it is in an unknown state}}
    *var1; // expected-warning {{invocation of method 'operator*' on object 'var1' while it is in an unknown state}}
  }
  
  if (!(var0 || var1)) {
    *var0; // expected-warning {{invocation of method 'operator*' on object 'var0' while it is in the 'consumed' state}}
    *var1; // expected-warning {{invocation of method 'operator*' on object 'var1' while it is in the 'consumed' state}}
    
  } else {
    *var0; // expected-warning {{invocation of method 'operator*' on object 'var0' while it is in an unknown state}}
    *var1; // expected-warning {{invocation of method 'operator*' on object 'var1' while it is in an unknown state}}
  }
  
  if (!var0 || !var1) {
    *var0; // expected-warning {{invocation of method 'operator*' on object 'var0' while it is in an unknown state}}
    *var1; // expected-warning {{invocation of method 'operator*' on object 'var1' while it is in an unknown state}}
    
  } else {
    *var0;
    *var1;
  }
  
  if (!(var0 && var1)) {
    *var0; // expected-warning {{invocation of method 'operator*' on object 'var0' while it is in an unknown state}}
    *var1; // expected-warning {{invocation of method 'operator*' on object 'var1' while it is in an unknown state}}
    
  } else {
    *var0;
    *var1;
  }
  
  if (var0 && var1 && var2) {
    *var0;
    *var1;
    *var2;
    
  } else {
    *var0; // expected-warning {{invocation of method 'operator*' on object 'var0' while it is in an unknown state}}
    *var1; // expected-warning {{invocation of method 'operator*' on object 'var1' while it is in an unknown state}}
    *var2; // expected-warning {{invocation of method 'operator*' on object 'var2' while it is in an unknown state}}
  }
  
#if 0
  // FIXME: Get this test to pass.
  if (var0 || var1 || var2) {
    *var0; // expected-warning {{invocation of method 'operator*' on object 'var0' while it is in an unknown state}}
    *var1; // expected-warning {{invocation of method 'operator*' on object 'var1' while it is in an unknown state}}
    *var2; // expected-warning {{invocation of method 'operator*' on object 'var2' while it is in an unknown state}}
    
  } else {
    *var0; // expected-warning {{invocation of method 'operator*' on object 'var0' while it is in the 'consumed' state}}
    *var1; // expected-warning {{invocation of method 'operator*' on object 'var1' while it is in the 'consumed' state}}
    *var2; // expected-warning {{invocation of method 'operator*' on object 'var2' while it is in the 'consumed' state}}
  }
#endif
}

void testCallingConventions() {
  ConsumableClass<int> var(42);
  
  baf0(var);  
  *var; // expected-warning {{invocation of method 'operator*' on object 'var' while it is in an unknown state}}
  
  var = ConsumableClass<int>(42);
  baf1(&var);  
  *var; // expected-warning {{invocation of method 'operator*' on object 'var' while it is in an unknown state}}
}

void testConstAndNonConstMemberFunctions() {
  ConsumableClass<int> var(42);
  
  var.constCall();
  *var;
  
  var.nonconstCall();
  *var;
}

void testNoWarnTestFromMacroExpansion() {
  ConsumableClass<int> var(42);
  
  if (TEST_VAR(var)) {
    *var;
  }
}

void testFunctionParam(ConsumableClass<int> param) {
  *param; // expected-warning {{invocation of method 'operator*' on object 'param' while it is in an unknown state}}
}

void testSimpleForLoop() {
  ConsumableClass<int> var;
  
  for (int i = 0; i < 10; ++i) {
    *var; // expected-warning {{invocation of method 'operator*' on object 'var' while it is in an unknown state}}
  }
  
  *var; // expected-warning {{invocation of method 'operator*' on object 'var' while it is in an unknown state}}
}

void testSimpleWhileLoop() {
  int i = 0;
  
  ConsumableClass<int> var;
  
  while (i < 10) {
    *var; // expected-warning {{invocation of method 'operator*' on object 'var' while it is in an unknown state}}
    ++i;
  }
  
  *var; // expected-warning {{invocation of method 'operator*' on object 'var' while it is in an unknown state}}
}
