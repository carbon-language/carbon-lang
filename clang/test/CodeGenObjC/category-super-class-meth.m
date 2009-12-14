// RUN: clang -cc1 -emit-llvm -o %t %s

@interface BASE
+ (int) BaseMeth;
@end

@interface Child: BASE
@end

@interface Child (Categ)
+ (int) flushCache2;
@end

@implementation Child  @end

@implementation Child (Categ)
+ (int) flushCache2 { [super BaseMeth]; }
@end

