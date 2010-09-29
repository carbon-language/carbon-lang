// RUN: %clang -fverbose-asm -g -S %s -o - | grep DW_AT_artificial | count 3
// self and _cmd are marked as DW_AT_artificial. 
// abbrev code emits another DT_artificial comment.
// myarg is not marked as DW_AT_artificial.

@interface MyClass {
}
- (id)init:(int) myarg;
@end

@implementation MyClass
- (id) init:(int) myarg
{
    return self;
}
@end
