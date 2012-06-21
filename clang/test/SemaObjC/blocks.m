// RUN: %clang_cc1 -fsyntax-only -verify -fblocks %s

#define bool _Bool
@protocol NSObject;

void bar(id(^)(void));
void foo(id <NSObject>(^objectCreationBlock)(void)) {
    return bar(objectCreationBlock);
}

void bar2(id(*)(void));
void foo2(id <NSObject>(*objectCreationBlock)(void)) {
    return bar2(objectCreationBlock);
}

void bar3(id(*)());
void foo3(id (*objectCreationBlock)(int)) {
    return bar3(objectCreationBlock);
}

void bar4(id(^)());
void foo4(id (^objectCreationBlock)(int)) {
    return bar4(objectCreationBlock);
}

void bar5(id(^)(void)); // expected-note 3{{passing argument to parameter here}}
void foo5(id (^objectCreationBlock)(bool)) {
    bar5(objectCreationBlock); // expected-error {{incompatible block pointer types passing 'id (^)(bool)' to parameter of type 'id (^)(void)'}}
#undef bool
    bar5(objectCreationBlock); // expected-error {{incompatible block pointer types passing 'id (^)(_Bool)' to parameter of type 'id (^)(void)'}}
#define bool int
    bar5(objectCreationBlock); // expected-error {{incompatible block pointer types passing 'id (^)(_Bool)' to parameter of type 'id (^)(void)'}}
}

void bar6(id(^)(int));
void foo6(id (^objectCreationBlock)()) {
    return bar6(objectCreationBlock);
}

void foo7(id (^x)(int)) {
  if (x) { }
}

@interface itf
@end

void foo8() {
  void *P = ^(itf x) {};  // expected-error {{interface type 'itf' cannot be passed by value; did you forget * in 'itf'}}
  P = ^itf(int x) {};     // expected-error {{interface type 'itf' cannot be returned by value; did you forget * in 'itf'}}
  P = ^itf() {};          // expected-error {{interface type 'itf' cannot be returned by value; did you forget * in 'itf'}}
  P = ^itf{};             // expected-error {{interface type 'itf' cannot be returned by value; did you forget * in 'itf'}}
}


int foo9() {
  typedef void (^DVTOperationGroupScheduler)();
  id _suboperationSchedulers;

  for (DVTOperationGroupScheduler scheduler in _suboperationSchedulers) {
            ;
        }

}

// rdar 7725203
@class NSString;

extern void NSLog(NSString *format, ...) __attribute__((format(__NSString__, 1, 2)));

void foo10() {
    void(^myBlock)(void) = ^{
    };
    NSLog(@"%@", myBlock);
}


// In C, enum constants have the type of the underlying integer type, not the
// enumeration they are part of. We pretend the constants have enum type when
// inferring block return types, so that they can be mixed-and-matched with
// other expressions of enum type.
enum CStyleEnum {
  CSE_Value = 1
};
enum CStyleEnum getCSE();
typedef enum CStyleEnum (^cse_block_t)();

void testCStyleEnumInference(bool arg) {
  cse_block_t a;

  // No warnings here.
  a = ^{ return CSE_Value; };
  a = ^{ return getCSE(); };

  a = ^{ // expected-error {{incompatible block pointer types assigning to 'cse_block_t' (aka 'enum CStyleEnum (^)()') from 'int (^)(void)'}}
    return 1;
  };

  // No warnings here.
  a = ^{ if (arg) return CSE_Value; else return CSE_Value; };
  a = ^{ if (arg) return getCSE();  else return getCSE();  };
  a = ^{ if (arg) return CSE_Value; else return getCSE();  };
  a = ^{ if (arg) return getCSE();  else return CSE_Value; };

  // Technically these two blocks should return 'int'.
  // The first case is easy to handle -- just don't cast the enum constant
  // to the enum type. However, the second guess would require going back
  // and REMOVING the cast from the first return statement, which isn't really
  // feasible (there may be more than one previous return statement with enum
  // type). For symmetry, we just treat them the same way.
  a = ^{ // expected-error {{incompatible block pointer types assigning to 'cse_block_t' (aka 'enum CStyleEnum (^)()') from 'int (^)(void)'}}
    if (arg)
      return 1;
    else
      return CSE_Value; // expected-error {{return type 'enum CStyleEnum' must match previous return type 'int'}}
  };

  a = ^{
    if (arg)
      return CSE_Value;
    else
      return 1; // expected-error {{return type 'int' must match previous return type 'enum CStyleEnum'}}
  };
}


enum FixedTypeEnum : unsigned {
  FTE_Value = 1U
};
enum FixedTypeEnum getFTE();
typedef enum FixedTypeEnum (^fte_block_t)();

void testFixedTypeEnumInference(bool arg) {
  fte_block_t a;
  
  // No warnings here.
  a = ^{ return FTE_Value; };
  a = ^{ return getFTE(); };

  // Since we fixed the underlying type of the enum, this is considered a
  // compatible block type.
  a = ^{
    return 1U;
  };
  
  // No warnings here.
  a = ^{ if (arg) return FTE_Value; else return FTE_Value; };
  a = ^{ if (arg) return getFTE();  else return getFTE();  };
  a = ^{ if (arg) return FTE_Value; else return getFTE();  };
  a = ^{ if (arg) return getFTE();  else return FTE_Value; };
  
  // Technically these two blocks should return 'unsigned'.
  // The first case is easy to handle -- just don't cast the enum constant
  // to the enum type. However, the second guess would require going back
  // and REMOVING the cast from the first return statement, which isn't really
  // feasible (there may be more than one previous return statement with enum
  // type). For symmetry, we just treat them the same way.
  a = ^{
    if (arg)
      return 1U;
    else
      return FTE_Value; // expected-error{{return type 'enum FixedTypeEnum' must match previous return type 'unsigned int'}}
  };
  
  a = ^{
    if (arg)
      return FTE_Value;
    else
      return 1U; // expected-error{{return type 'unsigned int' must match previous return type 'enum FixedTypeEnum'}}
  };
}


enum {
  AnonymousValue = 1
};

enum : short {
  FixedAnonymousValue = 1
};

typedef enum {
  TDE_Value
} TypeDefEnum;

typedef enum : short {
  TDFTE_Value
} TypeDefFixedTypeEnum;


typedef int (^int_block_t)();
typedef short (^short_block_t)();
void testAnonymousEnumTypes() {
  int_block_t IB;
  IB = ^{ return AnonymousValue; };
  IB = ^{ return TDE_Value; }; // expected-error {{incompatible block pointer types assigning to 'int_block_t' (aka 'int (^)()') from 'TypeDefEnum (^)(void)'}}
  IB = ^{ return CSE_Value; }; // expected-error {{incompatible block pointer types assigning to 'int_block_t' (aka 'int (^)()') from 'enum CStyleEnum (^)(void)'}}

  short_block_t SB;
  SB = ^{ return FixedAnonymousValue; };
  // This is not an error anyway since the enum has a fixed underlying type.
  SB = ^{ return TDFTE_Value; };
}
