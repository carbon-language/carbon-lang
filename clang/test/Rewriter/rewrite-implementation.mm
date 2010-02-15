// RUN: %clang_cc1 -x objective-c++ -fblocks -fms-extensions -rewrite-objc %s -o %t-rw.cpp
// RUN: %clang_cc1 -DSEL="void *" -S %t-rw.cpp
// radar 7649577

@interface a
@end

@interface b : a
@end

@implementation b
@end

