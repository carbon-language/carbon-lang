// RUN: %llvmgcc -x objective-c -S %s -g --emit-llvm -o - | grep "dbg.compile_unit =" | grep "null, i32"
// Last parameter represent i32 runtime version id. The previous paramenter
// encodes command line flags when certain env. variables are set. In this
// example it is the only compile_unit parameter that is null. This test case
// tests existence of new additional compile_unit parameter to encode 
// Objective-C runtime version number.

@interface foo
@end
@implementation foo
@end

void fn(foo *f) {}
