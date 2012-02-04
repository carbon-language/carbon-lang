// RUN: %clang_cc1 -rewrite-objc -fobjc-fragile-abi  %s -o -

typedef struct _NSPoint {
    float x;
    float y;
} NSPoint;

@interface Intf
- (void) MyMeth : (NSPoint) Arg1;
@end

@implementation Intf
- (void) MyMeth : (NSPoint) Arg1{}
@end

