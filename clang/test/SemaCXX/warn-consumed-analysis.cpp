// RUN: %clang_cc1 -fsyntax-only -verify -Wconsumed -std=c++11 %s

#define CALLABLE_WHEN_UNCONSUMED __attribute__ ((callable_when_unconsumed))
#define CONSUMES __attribute__ ((consumes))
#define TESTS_UNCONSUMED __attribute__ ((tests_unconsumed))

typedef decltype(nullptr) nullptr_t;

template <typename T>
class ConsumableClass {
  T var;
  
  public:
  ConsumableClass(void);
  ConsumableClass(nullptr_t p) CONSUMES;
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

void baf0(const ConsumableClass<int>  var);
void baf1(const ConsumableClass<int> &var);
void baf2(const ConsumableClass<int> *var);

void baf3(ConsumableClass<int> &&var);

void testInitialization(void) {
  ConsumableClass<int> var0;
  ConsumableClass<int> var1 = ConsumableClass<int>();
  
  var0 = ConsumableClass<int>();
  
  *var0; // expected-warning {{invocation of method 'operator*' on object 'var0' while it is in the 'consumed' state}}
  *var1; // expected-warning {{invocation of method 'operator*' on object 'var1' while it is in the 'consumed' state}}
  
  if (var0.isValid()) {
    *var0;
    *var1;  // expected-warning {{invocation of method 'operator*' on object 'var1' while it is in the 'consumed' state}}
    
  } else {
    *var0;  // expected-warning {{invocation of method 'operator*' on object 'var0' while it is in the 'consumed' state}}
  }
}

void testTempValue() {
  *ConsumableClass<int>(); // expected-warning {{invocation of method 'operator*' on a temporary object while it is in the 'consumed' state}}
}

void testSimpleRValueRefs(void) {
  ConsumableClass<int> var0;
  ConsumableClass<int> var1(42);
  
  *var0; // expected-warning {{invocation of method 'operator*' on object 'var0' while it is in the 'consumed' state}}
  *var1;
  
  var0 = static_cast<ConsumableClass<int>&&>(var1);
  
  *var0;
  *var1; // expected-warning {{invocation of method 'operator*' on object 'var1' while it is in the 'consumed' state}}
}

void testIfStmt(void) {
  ConsumableClass<int> var;
  
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
  ConsumableClass<int> var(42);
  
  baf0(var);  
  *var;
  
  baf1(var);  
  *var;
  
  baf2(&var);  
  *var;
  
  baf3(static_cast<ConsumableClass<int>&&>(var));  
  *var; // expected-warning {{invocation of method 'operator*' on object 'var' while it is in the 'consumed' state}}
}

void testMoveAsignmentish(void) {
  ConsumableClass<int>  var0;
  ConsumableClass<long> var1(42);
  
  *var0; // expected-warning {{invocation of method 'operator*' on object 'var0' while it is in the 'consumed' state}}
  *var1;
  
  var0 = static_cast<ConsumableClass<long>&&>(var1);
  
  *var0;
  *var1; // expected-warning {{invocation of method 'operator*' on object 'var1' while it is in the 'consumed' state}}
}

void testConditionalMerge(void) {
  ConsumableClass<int> var;
  
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
  ConsumableClass<int> var(42);
  
  *var;
  
  var.consume();
  
  *var; // expected-warning {{invocation of method 'operator*' on object 'var' while it is in the 'consumed' state}}
}

void testConsumes1(void) {
  ConsumableClass<int> var(nullptr);
  
  *var; // expected-warning {{invocation of method 'operator*' on object 'var' while it is in the 'consumed' state}}
}

void testConsumes2(void) {
  ConsumableClass<int> var(42);
  
  var.unconsumedCall();
  var(6);
  
  var.unconsumedCall(); // expected-warning {{invocation of method 'unconsumedCall' on object 'var' while it is in the 'consumed' state}}
}

void testSimpleForLoop(void) {
  ConsumableClass<int> var;
  
  for (int i = 0; i < 10; ++i) {
    *var;
  }
  
  *var;
}

void testSimpleWhileLoop(void) {
  int i = 0;
  
  ConsumableClass<int> var;
  
  while (i < 10) {
    *var;
    ++i;
  }
  
  *var;
}
