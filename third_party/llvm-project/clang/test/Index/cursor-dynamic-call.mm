
struct SB {
  virtual void meth();
};

struct SS : public SB {
  void submeth() {
    this->meth();
    SB::meth();
  }
};

@interface IB
-(void)meth;
+(void)ClsMeth;
@end

@interface IS : IB
-(void)submeth;
+(void)ClsMeth;
@end

@implementation IS
-(void)submeth {
  [self meth];
  [super meth];
}
+(void)ClsMeth {
  [super ClsMeth];
}
@end

void foo(SS *ss, IS* is, Class cls) {
  ss->meth();
  [is meth];
  [IB ClsMeth];
  [cls ClsMeth];
}

@interface NSObject
+(id)alloc;
-(id)init;
@end

@interface Test : NSObject
@end

void test2() {
  id o = [[Test alloc] init];
}

@interface Test2 : NSObject
@property (assign) id someProp;
@end

void test3(Test2 *o) {
  id v = o.someProp;
}

// RUN: c-index-test -cursor-at=%s:8:11 \
// RUN:              -cursor-at=%s:9:11 \
// RUN:              -cursor-at=%s:25:11 \
// RUN:              -cursor-at=%s:26:11 \
// RUN:              -cursor-at=%s:29:11 \
// RUN:              -cursor-at=%s:34:9 \
// RUN:              -cursor-at=%s:35:9 \
// RUN:              -cursor-at=%s:36:9 \
// RUN:              -cursor-at=%s:37:9 \
// RUN:              -cursor-at=%s:49:26 \
// RUN:              -cursor-at=%s:57:12 \
// RUN:       %s | FileCheck %s

// CHECK:     8:11 MemberRefExpr=meth:3:16 {{.*}} Dynamic-call
// CHECK-NOT: 9:9 {{.*}} Dynamic-call
// CHECK:     25:3 ObjCMessageExpr=meth:14:8 {{.*}} Dynamic-call Receiver-type=ObjCObjectPointer
// CHECK-NOT: 26:3 {{.*}} Dynamic-call
// CHECK-NOT: 29:3 {{.*}} Dynamic-call
// CHECK:     29:3 {{.*}} Receiver-type=ObjCInterface
// CHECK:     34:7 MemberRefExpr=meth:3:16 {{.*}} Dynamic-call Receiver-type=Pointer
// CHECK:     35:3 ObjCMessageExpr=meth:14:8 {{.*}} Dynamic-call Receiver-type=ObjCObjectPointer
// CHECK-NOT: 36:3 {{.*}} Dynamic-call
// CHECK:     36:3 {{.*}} Receiver-type=ObjCInterface
// CHECK:     37:3 ObjCMessageExpr=ClsMeth:15:8 {{.*}} Dynamic-call Receiver-type=ObjCClass
// CHECK-NOT: 49:10 {{.*}} Dynamic-call
// CHECK:     57:12 MemberRefExpr=someProp:53:23 {{.*}} Dynamic-call  Receiver-type=ObjCObjectPointer
