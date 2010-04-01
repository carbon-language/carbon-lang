// RUN: %clang_cc1 -fsyntax-only %s -verify -fblocks
// test for block type safety.

@interface Super  @end
@interface Sub : Super @end

void f2(void(^f)(Super *)) {
    Super *o;
    f(o);
}

void f3(void(^f)(Sub *)) {
    Sub *o;
    f(o);
}

void r0(Super* (^f)()) {
     Super *o = f();
}

void r1(Sub* (^f)()) {
    Sub *o = f();
}

@protocol NSObject;

void r2 (id<NSObject> (^f) (void)) {
  id o = f();
}

void test1() {
    f2(^(Sub *o) { });    // expected-error {{incompatible block pointer types passing 'void (^)(Sub *)', expected 'void (^)(Super *)'}}
    f3(^(Super *o) { });  // OK, block taking Super* may be called with a Sub*

    r0(^Super* () { return 0; });  // OK
    r0(^Sub* () { return 0; });    // OK, variable of type Super* gets return value of type Sub*
    r0(^id () { return 0; });

    r1(^Super* () { return 0; });  // expected-error {{incompatible block pointer types passing 'Super *(^)(void)', expected 'Sub *(^)()'}}
    r1(^Sub* () { return 0; });    // OK
    r1(^id () { return 0; }); 
     
    r2(^id<NSObject>() { return 0; });
}


@interface A @end
@interface B @end

void f0(void (^f)(A* x)) {
  A* a;
  f(a);
}

void f1(void (^f)(id x)) {
  B *b;
  f(b);
}

void test2(void) 
{ 
  f0(^(id a) { }); // OK
  f1(^(A* a) { });
   f1(^(id<NSObject> a) { });	// OK
}

@interface NSArray
   // Calls block() with every object in the array
   -enumerateObjectsWithBlock:(void (^)(id obj))block;
@end

@interface MyThing
-(void) printThing;
@end

@implementation MyThing
    static NSArray* myThings;  // array of MyThing*

   -(void) printThing {  }

// programmer wants to write this:
   -printMyThings1 {
       [myThings enumerateObjectsWithBlock: ^(MyThing *obj) {
           [obj printThing];
       }];
   }

// strict type safety requires this:
   -printMyThings {
       [myThings enumerateObjectsWithBlock: ^(id obj) {
           MyThing *obj2 = (MyThing *)obj;
           [obj2 printThing];
       }];
   }
@end

