// RUN: %llvmgcc -S %s -fobjc-abi-version=2 -emit-llvm -o %t
// RUN: grep {OBJC_CLASS_\\\$_A.*section.*__DATA, __objc_data.*align} %t
// XTARGETS: darwin

@interface A
@end

@implementation A
@end
