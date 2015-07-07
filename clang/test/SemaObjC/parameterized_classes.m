// RUN: %clang_cc1 %s -verify

@protocol NSObject
@end

__attribute__((objc_root_class))
@interface NSObject <NSObject> // expected-note{{'NSObject' defined here}}
@end

@interface NSString : NSObject
@end

// --------------------------------------------------------------------------
// Parsing parameterized classes.
// --------------------------------------------------------------------------

// Parse type parameters with a bound
@interface PC1<T, U : NSObject*> : NSObject
// expected-note@-1{{type parameter 'T' declared here}}
// expected-note@-2{{type parameter 'U' declared here}}
@end

// Parse a type parameter with a bound that terminates in '>>'.
@interface PC2<T : id<NSObject>> : NSObject // expected-error{{a space is required between consecutive right angle brackets (use '> >')}}
@end

// Parse multiple type parameters.
@interface PC3<T, U : id> : NSObject
@end

// Parse multiple type parameters--grammatically ambiguous with protocol refs.
@interface PC4<T, U, V> : NSObject
@end

// Parse a type parameter list without a superclass.
@interface PC5<T : id> // expected-error{{parameterized Objective-C class 'PC5' must have a superclass}}
@end

// Parse a type parameter with name conflicts.
@interface PC6<T, U, 
               T> : NSObject // expected-error{{redeclaration of type parameter 'T'}}
@end

// Parse Objective-C protocol references.
@interface PC7<T> // expected-error{{cannot find protocol declaration for 'T'}}
@end

// Parse both type parameters and protocol references.
@interface PC8<T> : NSObject <NSObject>
@end

// Type parameters with improper bounds.
@interface PC9<T : int, // expected-error{{type bound 'int' for type parameter 'T' is not an Objective-C pointer type}}
               U : NSString> : NSObject // expected-error{{missing '*' in type bound 'NSString' for type parameter 'U'}}
@end

// --------------------------------------------------------------------------
// Parsing parameterized forward declarations classes.
// --------------------------------------------------------------------------

// Okay: forward declaration without type parameters.
@class PC10;

// Okay: forward declarations with type parameters.
@class PC10<T, U : NSObject *>, PC11<T : NSObject *, U : id>; // expected-note{{type parameter 'T' declared here}}

// Okay: forward declaration without type parameters following ones
// with type parameters.
@class PC10, PC11;

// Okay: definition of class with type parameters that was formerly
// declared with the same type parameters.
@interface PC10<T, U : NSObject *> : NSObject
@end

// Mismatched parameters in declaration of @interface following @class.
@interface PC11<T, U> : NSObject // expected-error{{missing type bound 'NSObject *' for type parameter 'T' in @interface}}
@end

@interface PC12<T : NSObject *> : NSObject  // expected-note{{type parameter 'T' declared here}}
@end

@class PC12;

// Mismatched parameters in subsequent forward declarations.
@class PC13<T : NSObject *>; // expected-note{{type parameter 'T' declared here}}
@class PC13;
@class PC13<U>; // expected-error{{missing type bound 'NSObject *' for type parameter 'U' in @class}}

// Mismatch parameters in declaration of @class following @interface.
@class PC12<T>; // expected-error{{missing type bound 'NSObject *' for type parameter 'T' in @class}}

// Parameterized forward declaration a class that is not parameterized.
@class NSObject<T>; // expected-error{{forward declaration of non-parameterized class 'NSObject' cannot have type parameters}}

// Parameterized forward declaration preceding the definition (that is
// not parameterized).
@class NSNumber<T : NSObject *>; // expected-note{{'NSNumber' declared here}}
@interface NSNumber : NSObject // expected-error{{class 'NSNumber' previously declared with type parameters}}
@end

@class PC14;

// Okay: definition of class with type parameters that was formerly
// declared without type parameters.
@interface PC14<T, U : NSObject *> : NSObject
@end

// --------------------------------------------------------------------------
// Parsing parameterized categories and extensions.
// --------------------------------------------------------------------------

// Inferring type bounds
@interface PC1<T, U> (Cat1) <NSObject>
@end

// Matching type bounds
@interface PC1<T : id, U : NSObject *> (Cat2) <NSObject>
@end

// Inferring type bounds
@interface PC1<T, U> () <NSObject>
@end

// Matching type bounds
@interface PC1<T : id, U : NSObject *> () <NSObject>
@end

// Missing type parameters.
@interface PC1<T> () // expected-error{{extension has too few type parameters (expected 2, have 1)}}
@end

// Extra type parameters.
@interface PC1<T, U, V> (Cat3) // expected-error{{category has too many type parameters (expected 2, have 3)}}
@end

// Mismatched bounds.
@interface PC1<T : NSObject *, // expected-error{{type bound 'NSObject *' for type parameter 'T' conflicts with implicit bound 'id'}}
               X : id> () // expected-error{{type bound 'id' for type parameter 'X' conflicts with previous bound 'NSObject *'for type parameter 'U'}}
@end

// Parameterized category/extension of non-parameterized class.
@interface NSObject<T> (Cat1) // expected-error{{category of non-parameterized class 'NSObject' cannot have type parameters}}
@end

@interface NSObject<T> () // expected-error{{extension of non-parameterized class 'NSObject' cannot have type parameters}}
@end

// --------------------------------------------------------------------------
// @implementations cannot have type parameters
// --------------------------------------------------------------------------
@implementation PC1<T : id> // expected-error{{@implementation cannot have type parameters}}
@end

@implementation PC2<T> // expected-error{{@implementation declaration cannot be protocol qualified}}
@end

@implementation PC1<T> (Cat1) // expected-error{{@implementation cannot have type parameters}}
@end

@implementation PC1<T : id> (Cat2) // expected-error{{@implementation cannot have type parameters}}
@end

// --------------------------------------------------------------------------
// Interfaces involving type parameters
// --------------------------------------------------------------------------
@interface PC20<T : id, U : NSObject *, V : NSString *> : NSObject {
  T object;
}

- (U)method:(V)param; // expected-note{{passing argument to parameter 'param' here}}
@end

@interface PC20<T, U, V> (Cat1)
- (U)catMethod:(V)param; // expected-note{{passing argument to parameter 'param' here}}
@end

@interface PC20<X, Y, Z>()
- (X)extMethod:(Y)param; // expected-note{{passing argument to parameter 'param' here}}
@end

void test_PC20_unspecialized(PC20 *pc20) {
  // FIXME: replace type parameters with underlying types?
  int *ip = [pc20 method: 0]; // expected-warning{{incompatible pointer types initializing 'int *' with an expression of type 'U' (aka 'NSObject *')}}
  [pc20 method: ip]; // expected-warning{{incompatible pointer types sending 'int *' to parameter of type 'V' (aka 'NSString *')}}

  ip = [pc20 catMethod: 0]; // expected-warning{{incompatible pointer types assigning to 'int *' from 'U' (aka 'NSObject *')}}
  [pc20 catMethod: ip]; // expected-warning{{incompatible pointer types sending 'int *' to parameter of type 'V' (aka 'NSString *')}}

  ip = [pc20 extMethod: 0]; // expected-warning{{incompatible pointer types assigning to 'int *' from 'X' (aka 'id')}}
  [pc20 extMethod: ip]; // expected-warning{{incompatible pointer types sending 'int *' to parameter of type 'Y' (aka 'NSObject *')}}
}
