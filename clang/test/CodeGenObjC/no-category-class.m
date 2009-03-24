// RUN: clang-cc -triple x86_64-unknown-unknown -emit-llvm -o %t %s

@interface NSObject
@end

@implementation NSObject(IBXLIFFIntegration)
@end

