// FIXME: Check IR rather than asm, then triple is not needed.
// RUN: %clang_cc1 -triple %itanium_abi_triple -S -debug-info-kind=limited %s -o - | FileCheck %s

// CHECK: AT_APPLE_property_name
// CHECK: AT_APPLE_property_getter
// CHECK: AT_APPLE_property_setter
// CHECK: AT_APPLE_property_attribute
// CHECK: AT_APPLE_property

@interface BaseClass2 
{
	int _baseInt;
}
- (int) myGetBaseInt;
- (void) mySetBaseInt: (int) in_int;
@property(getter=myGetBaseInt,setter=mySetBaseInt:) int baseInt;
@end

@implementation BaseClass2

- (int) myGetBaseInt
{
        return _baseInt;
}

- (void) mySetBaseInt: (int) in_int
{
    _baseInt = 2 * in_int;
}
@end


void foo(BaseClass2 *ptr) {}
