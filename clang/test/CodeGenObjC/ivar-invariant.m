// RUN: %clang_cc1 -triple x86_64-apple-darwin -x objective-c %s -o - | FileCheck %s

@interface NSObject
+ (id) new;
- (id) init;
@end

@interface Base : NSObject @end

// @implementation Base
// {
//     int dummy;
// }
// @end

@interface Derived : Base
{
    @public int member;
}
@end

@implementation Derived
- (id) init
{
    self = [super init];
    member = 42;
    return self;
}
@end

// CHECK: [[IVAR:%.*]] = load i64* @"OBJC_IVAR_$_Derived.member", !invariant.load

void * variant_load_1(int i) {
    void *ptr;
    while (i--) {
        Derived *d = [Derived new];
        ptr = &d->member;
    }
    return ptr;
}

// CHECK: [[IVAR:%.*]] = load i64* @"OBJC_IVAR_$_Derived.member"{{$}}

@interface Container : Derived @end
@implementation Container
- (void *) invariant_load_1
{
    return &self->member;
}
@end

// CHECK: [[IVAR:%.*]] = load i64* @"OBJC_IVAR_$_Derived.member", !invariant.load

