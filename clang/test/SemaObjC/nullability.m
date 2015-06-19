// RUN: %clang_cc1 -fsyntax-only -fblocks -Woverriding-method-mismatch -Wno-nullability-declspec %s -verify

__attribute__((objc_root_class))
@interface NSFoo
- (void)methodTakingIntPtr:(__nonnull int *)ptr;
- (__nonnull int *)methodReturningIntPtr;
@end

// Nullability applies to all pointer types.
typedef NSFoo * __nonnull nonnull_NSFoo_ptr;
typedef id __nonnull nonnull_id;
typedef SEL __nonnull nonnull_SEL;

// Nullability can move into Objective-C pointer types.
typedef __nonnull NSFoo * nonnull_NSFoo_ptr_2;

// Conflicts from nullability moving into Objective-C pointer type.
typedef __nonnull NSFoo * __nullable conflict_NSFoo_ptr_2; // expected-error{{'__nonnull' cannot be applied to non-pointer type 'NSFoo'}}

void testBlocksPrinting(NSFoo * __nullable (^bp)(int)) {
  int *ip = bp; // expected-error{{'NSFoo * __nullable (^)(int)'}}
}

// Check returning nil from a __nonnull-returning method.
@implementation NSFoo
- (void)methodTakingIntPtr:(__nonnull int *)ptr { }
- (__nonnull int *)methodReturningIntPtr {
  return 0; // no warning
}
@end

// Context-sensitive keywords and property attributes for nullability.
__attribute__((objc_root_class))
@interface NSBar
- (nonnull NSFoo *)methodWithFoo:(nonnull NSFoo *)foo;

- (nonnull NSFoo **)invalidMethod1; // expected-error{{nullability keyword 'nonnull' cannot be applied to multi-level pointer type 'NSFoo **'}}
// expected-note@-1{{use nullability type specifier '__nonnull' to affect the innermost pointer type of 'NSFoo **'}}
- (nonnull NSFoo * __nullable)conflictingMethod1; // expected-error{{nullability specifier '__nullable' conflicts with existing specifier '__nonnull'}}
- (nonnull NSFoo * __nonnull)redundantMethod1; // expected-warning{{duplicate nullability specifier '__nonnull'}}

@property(nonnull,retain) NSFoo *property1;
@property(nullable,assign) NSFoo ** invalidProperty1; // expected-error{{nullability keyword 'nullable' cannot be applied to multi-level pointer type 'NSFoo **'}}
// expected-note@-1{{use nullability type specifier '__nullable' to affect the innermost pointer type of 'NSFoo **'}}
@property(null_unspecified,retain) NSFoo * __nullable conflictingProperty1; // expected-error{{nullability specifier '__nullable' conflicts with existing specifier '__null_unspecified'}}
@property(retain,nonnull) NSFoo * __nonnull redundantProperty1; // expected-warning{{duplicate nullability specifier '__nonnull'}}

@property(null_unspecified,retain,nullable) NSFoo *conflictingProperty3; // expected-error{{nullability specifier 'nullable' conflicts with existing specifier 'null_unspecified'}}
@property(nullable,retain,nullable) NSFoo *redundantProperty3; // expected-warning{{duplicate nullability specifier 'nullable'}}
@end

@interface NSBar ()
@property(nonnull,retain) NSFoo *property2;
@property(nullable,assign) NSFoo ** invalidProperty2; // expected-error{{nullability keyword 'nullable' cannot be applied to multi-level pointer type 'NSFoo **'}}
// expected-note@-1{{use nullability type specifier '__nullable' to affect the innermost pointer type of 'NSFoo **'}}
@property(null_unspecified,retain) NSFoo * __nullable conflictingProperty2; // expected-error{{nullability specifier '__nullable' conflicts with existing specifier '__null_unspecified'}}
@property(retain,nonnull) NSFoo * __nonnull redundantProperty2; // expected-warning{{duplicate nullability specifier '__nonnull'}}
@end

void test_accepts_nonnull_null_pointer_literal(NSFoo *foo, __nonnull NSBar *bar) {
  [foo methodTakingIntPtr: 0]; // expected-warning{{null passed to a callee that requires a non-null argument}}
  [bar methodWithFoo: 0]; // expected-warning{{null passed to a callee that requires a non-null argument}}
  bar.property1 = 0; // expected-warning{{null passed to a callee that requires a non-null argument}}
  bar.property2 = 0; // expected-warning{{null passed to a callee that requires a non-null argument}}
  [bar setProperty1: 0]; // expected-warning{{null passed to a callee that requires a non-null argument}}
  [bar setProperty2: 0]; // expected-warning{{null passed to a callee that requires a non-null argument}}
  int *ptr = bar.property1; // expected-warning{{incompatible pointer types initializing 'int *' with an expression of type 'NSFoo * __nonnull'}}
}

// Check returning nil from a nonnull-returning method.
@implementation NSBar
- (nonnull NSFoo *)methodWithFoo:(nonnull NSFoo *)foo {
  return 0; // no warning
}

- (NSFoo **)invalidMethod1 { 
  return 0; 
}

- (NSFoo *)conflictingMethod1 { 
  return 0; // no warning
}
- (NSFoo *)redundantMethod1 {
  int *ip = 0;
  return ip; // expected-warning{{result type 'NSFoo * __nonnull'}}
}
@end

__attribute__((objc_root_class))
@interface NSMerge
- (nonnull NSFoo *)methodA:(nonnull NSFoo*)foo;
- (nonnull NSFoo *)methodB:(nonnull NSFoo*)foo;
- (NSFoo *)methodC:(NSFoo*)foo;
@end

@implementation NSMerge
- (NSFoo *)methodA:(NSFoo*)foo {
  int *ptr = foo; // expected-warning{{incompatible pointer types initializing 'int *' with an expression of type 'NSFoo * __nonnull'}}
  return ptr; // expected-warning{{result type 'NSFoo * __nonnull'}}
}

- (nullable NSFoo *)methodB:(null_unspecified NSFoo*)foo { // expected-error{{nullability specifier 'nullable' conflicts with existing specifier 'nonnull'}} \
  // expected-error{{nullability specifier 'null_unspecified' conflicts with existing specifier 'nonnull'}}
  return 0;
}

- (nonnull NSFoo *)methodC:(nullable NSFoo*)foo {
  int *ip = 0;
  return ip; // expected-warning{{result type 'NSFoo * __nonnull'}}
}
@end

// Checking merging of nullability when sending a message.
@interface NSMergeReceiver
- (id)returnsNone;
- (nonnull id)returnsNonNull;
- (nullable id)returnsNullable;
- (null_unspecified id)returnsNullUnspecified;
@end

void test_receiver_merge(NSMergeReceiver *none,
                         __nonnull NSMergeReceiver *nonnull,
                         __nullable NSMergeReceiver *nullable,
                         __null_unspecified NSMergeReceiver *null_unspecified) {
  int *ptr;

  ptr = [nullable returnsNullable]; // expected-warning{{'id __nullable'}}
  ptr = [nullable returnsNullUnspecified]; // expected-warning{{'id __nullable'}}
  ptr = [nullable returnsNonNull]; // expected-warning{{'id __nullable'}}
  ptr = [nullable returnsNone]; // expected-warning{{'id __nullable'}}

  ptr = [null_unspecified returnsNullable]; // expected-warning{{'id __nullable'}}
  ptr = [null_unspecified returnsNullUnspecified]; // expected-warning{{'id __null_unspecified'}}
  ptr = [null_unspecified returnsNonNull]; // expected-warning{{'id __null_unspecified'}}
  ptr = [null_unspecified returnsNone]; // expected-warning{{'id'}}

  ptr = [nonnull returnsNullable]; // expected-warning{{'id __nullable'}}
  ptr = [nonnull returnsNullUnspecified]; // expected-warning{{'id __null_unspecified'}}
  ptr = [nonnull returnsNonNull]; // expected-warning{{'id __nonnull'}}
  ptr = [nonnull returnsNone]; // expected-warning{{'id'}}

  ptr = [none returnsNullable]; // expected-warning{{'id __nullable'}}
  ptr = [none returnsNullUnspecified]; // expected-warning{{'id'}}
  ptr = [none returnsNonNull]; // expected-warning{{'id'}}
  ptr = [none returnsNone]; // expected-warning{{'id'}}
  
}

// instancetype
@protocol Initializable
- (instancetype)initWithBlah:(id)blah;
@end

__attribute__((objc_root_class))
@interface InitializableClass <Initializable>
- (nonnull instancetype)initWithBlah:(nonnull id)blah;
- (nullable instancetype)returnMe;
+ (nullable instancetype)returnInstanceOfMe;
@end

void test_instancetype(InitializableClass * __nonnull ic, id __nonnull object) {
  int *ip = [ic returnMe]; // expected-warning{{incompatible pointer types initializing 'int *' with an expression of type 'InitializableClass * __nullable'}}
  ip = [InitializableClass returnMe]; // expected-warning{{incompatible pointer types assigning to 'int *' from 'id __nullable'}}
  ip = [InitializableClass returnInstanceOfMe]; // expected-warning{{incompatible pointer types assigning to 'int *' from 'InitializableClass * __nullable'}}
  ip = [object returnMe]; // expected-warning{{incompatible pointer types assigning to 'int *' from 'id __nullable'}}
}

// Check null_resettable getters/setters.
__attribute__((objc_root_class))
@interface NSResettable
@property(null_resettable,retain) NSResettable *resettable1; // expected-note{{passing argument to parameter 'resettable1' here}}
@property(null_resettable,retain,nonatomic) NSResettable *resettable2;
@property(null_resettable,retain,nonatomic) NSResettable *resettable3;
@property(null_resettable,retain,nonatomic) NSResettable *resettable4;
@property(null_resettable,retain,nonatomic) NSResettable *resettable5;
@property(null_resettable,retain,nonatomic) NSResettable *resettable6;
@end

void test_null_resettable(NSResettable *r, int *ip) {
  [r setResettable1:ip]; // expected-warning{{incompatible pointer types sending 'int *' to parameter of type 'NSResettable * __nullable'}}
  r.resettable1 = ip; // expected-warning{{incompatible pointer types assigning to 'NSResettable * __nullable' from 'int *'}}
}

@implementation NSResettable // expected-warning{{synthesized setter 'setResettable4:' for null_resettable property 'resettable4' does not handle nil}}
- (NSResettable *)resettable1 {
  int *ip = 0;
  return ip; // expected-warning{{result type 'NSResettable * __nonnull'}}
}

- (void)setResettable1:(NSResettable *)param {
}

@synthesize resettable2; // no warning; not synthesized
@synthesize resettable3; // expected-warning{{synthesized setter 'setResettable3:' for null_resettable property 'resettable3' does not handle nil}}

- (void)setResettable2:(NSResettable *)param {
}

@dynamic resettable5;

- (NSResettable *)resettable6 {
  return 0; // no warning
}
@end

// rdar://problem/19814852
@interface MultiProp
@property (nullable, copy) id a, b, c;
@property (nullable, copy) MultiProp *d, *(^e)(int);
@end

void testMultiProp(MultiProp *foo) {
  int *ip;
  ip = foo.a; // expected-warning{{from 'id __nullable'}}
  ip = foo.d; // expected-warning{{from 'MultiProp * __nullable'}}
  ip = foo.e; // expected-error{{incompatible type 'MultiProp *(^ __nullable)(int)'}}
}
