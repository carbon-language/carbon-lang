// RUN: %clang_cc1 -fsyntax-only -verify -Wconsumed -std=c++11 %s

#define CALLABLE_WHEN_UNCONSUMED __attribute__ ((callable_when_unconsumed))
#define CONSUMES __attribute__ ((consumes))
#define TESTS_UNCONSUMED __attribute__ ((tests_unconsumed))

typedef decltype(nullptr) nullptr_t;

template <typename T>
class Bar {
  T var;
  
  public:
  Bar(void);
  Bar(nullptr_t p) CONSUMES;
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
  
  void operator()(int a) CONSUMES;
  void operator*(void) const CALLABLE_WHEN_UNCONSUMED;
  void unconsumedCall(void) const CALLABLE_WHEN_UNCONSUMED;
  
  bool isValid(void) const TESTS_UNCONSUMED;
  operator bool() const TESTS_UNCONSUMED;
  bool operator!=(nullptr_t) const TESTS_UNCONSUMED;
  
  void constCall(void) const;
  void nonconstCall(void);
  
  void consume(void) CONSUMES;
};

void baf0(const Bar<int>  var);
void baf1(const Bar<int> &var);
void baf2(const Bar<int> *var);

void baf3(Bar<int> &&var);

void testInitialization(void) {
  Bar<int> var0;
  Bar<int> var1 = Bar<int>();
  
  var0 = Bar<int>();
  
  *var0; // expected-warning {{invocation of method 'operator*' on object 'var0' while it is in the 'consumed' state}}
  *var1; // expected-warning {{invocation of method 'operator*' on object 'var1' while it is in the 'consumed' state}}
  
  if (var0.isValid()) {
    *var0;
    *var1;  // expected-warning {{invocation of method 'operator*' on object 'var1' while it is in the 'consumed' state}}
    
  } else {
    *var0;  // expected-warning {{invocation of method 'operator*' on object 'var0' while it is in the 'consumed' state}}
  }
}

void testTempValue(void) {
  *Bar<int>(); // expected-warning {{invocation of method 'operator*' on a temporary object while it is in the 'consumed' state}}
}

void testSimpleRValueRefs(void) {
  Bar<int> var0;
  Bar<int> var1(42);
  
  *var0; // expected-warning {{invocation of method 'operator*' on object 'var0' while it is in the 'consumed' state}}
  *var1;
  
  var0 = static_cast<Bar<int>&&>(var1);
  
  *var0;
  *var1; // expected-warning {{invocation of method 'operator*' on object 'var1' while it is in the 'consumed' state}}
}

void testIfStmt(void) {
  Bar<int> var;
  
  if (var.isValid()) {
    // Empty
    
  } else {
    *var; // expected-warning {{invocation of method 'operator*' on object 'var' while it is in the 'consumed' state}}
  }
  
  if (!var.isValid()) {
    *var; // expected-warning {{invocation of method 'operator*' on object 'var' while it is in the 'consumed' state}}
    
  } else {
    *var;
  }
  
  if (var) {
    // Empty
    
  } else {
    *var; // expected-warning {{invocation of method 'operator*' on object 'var' while it is in the 'consumed' state}}
  }
  
  if (var != nullptr) {
    // Empty
    
  } else {
    *var; // expected-warning {{invocation of method 'operator*' on object 'var' while it is in the 'consumed' state}}
  }
}

void testCallingConventions(void) {
  Bar<int> var(42);
  
  baf0(var);  
  *var;
  
  baf1(var);  
  *var;
  
  baf2(&var);  
  *var;
  
  baf3(static_cast<Bar<int>&&>(var));  
  *var; // expected-warning {{invocation of method 'operator*' on object 'var' while it is in the 'consumed' state}}
}

void testMoveAsignmentish(void) {
  Bar<int>  var0;
  Bar<long> var1(42);
  
  *var0; // expected-warning {{invocation of method 'operator*' on object 'var0' while it is in the 'consumed' state}}
  *var1;
  
  var0 = static_cast<Bar<long>&&>(var1);
  
  *var0;
  *var1; // expected-warning {{invocation of method 'operator*' on object 'var1' while it is in the 'consumed' state}}
}

void testConditionalMerge(void) {
  Bar<int> var;
  
  if (var.isValid()) {
    // Empty
  }
  
  *var;
  
  if (var.isValid()) {
    // Empty
    
  } else {
    // Empty
  }
  
  *var;
}

void testConsumes0(void) {
  Bar<int> var(42);
  
  *var;
  
  var.consume();
  
  *var; // expected-warning {{invocation of method 'operator*' on object 'var' while it is in the 'consumed' state}}
}

void testConsumes1(void) {
  Bar<int> var(nullptr);
  
  *var; // expected-warning {{invocation of method 'operator*' on object 'var' while it is in the 'consumed' state}}
}

void testConsumes2(void) {
  Bar<int> var(42);
  
  var.unconsumedCall();
  var(6);
  
  var.unconsumedCall(); // expected-warning {{invocation of method 'unconsumedCall' on object 'var' while it is in the 'consumed' state}}
}

void testSimpleForLoop(void) {
  Bar<int> var;
  
  for (int i = 0; i < 10; ++i) {
    *var;
  }
  
  *var;
}

void testSimpleWhileLoop(void) {
  int i = 0;
  
  Bar<int> var;
  
  while (i < 10) {
    *var;
    ++i;
  }
  
  *var;
}
