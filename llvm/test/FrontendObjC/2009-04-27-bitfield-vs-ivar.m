// RUN: %llvmgcc -S -x objective-c -m64 -fobjc-abi-version=2 %s -o %t
// RUN: grep {OBJC_CLASS_RO_\\\$_I4} %t | grep {i32 0, i32 1, i32 2, i32 0}
// RUN: grep {OBJC_CLASS_RO_\\\$_I2} %t | grep {i32 0, i32 1, i32 1, i32 0}
// RUN: grep {OBJC_CLASS_RO_\\\$_I5} %t | grep {i32 0, i32 0, i32 0, i32 0}
// XTARGETS: darwin

// Test instance variable sizing when base class ends in bitfield
@interface I3 {
  unsigned int _iv2 :1;
}
@end

@interface I4 : I3 {
  char _iv4;
}
@end

// Test case with no instance variables in derived class
@interface I1 {
  unsigned int _iv2 :1;
}
@end

@interface I2 : I1 {
}
@end

// Test case with no instance variables anywhere
@interface I6 {
}
@end

@interface I5 : I6 {
}
@end

@implementation I4
@end

@implementation I2
@end

@implementation I5
@end
