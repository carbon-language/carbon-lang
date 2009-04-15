// RUN: %llvmgcc -S %s -fobjc-abi-version=2 -emit-llvm -o - | grep {OBJC_CLASS_\$_A.*section.*__DATA, __objc_data}
// XTARGETS: darwin

@interface A
@end

@implementation A
@end
