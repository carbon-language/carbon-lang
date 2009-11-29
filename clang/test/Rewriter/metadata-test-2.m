// RUN: clang-cc -rewrite-objc %s -o -

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

