// RUN: clang-cc -triple x86_64-apple-darwin9 -emit-llvm -o %t %s &&
// RUNX: llvm-gcc -m64 -emit-llvm -S -o %t %s &&

// RUN: grep '@"OBJC_IVAR_$_I3._iv2" = global i64 8, section "__DATA, __objc_const", align 8' %t &&
// RUN: grep '@"OBJC_IVAR_$_I3._iv3" = global i64 12, section "__DATA, __objc_const", align 8' %t &&
// RUN: grep '@"OBJC_IVAR_$_I4._iv4" = global i64 16, section "__DATA, __objc_const", align 8' %t &&
// RUN: grep '@"OBJC_IVAR_$_I5._iv5" = global i64 24, section "__DATA, __objc_const", align 8' %t &&
// RUN: grep '@"OBJC_IVAR_$_I5._iv6_synth" = global i64 28, section "__DATA, __objc_const", align 8' %t &&
// RUN: grep '@"OBJC_IVAR_$_I5._iv7_synth" = global i64 32, section "__DATA, __objc_const", align 8' %t &&
// RUN: grep '_OBJC_CLASS_RO_$_I6" = internal global .* { i32 2, i32 0, i32 1, .*' %t &&
// RUN: grep '_OBJC_CLASS_RO_$_I8" = internal global .* { i32 0, i32 8, i32 16, .*' %t &&

// RUN: true

struct s0 {
  double x;
};

@interface I2 {
  struct s0 _iv1;
}
@end

@interface I3 : I2 {
  unsigned int _iv2 :1;
  unsigned : 0;
  unsigned int _iv3 : 3;
}
@end

@interface I4 : I3 {
 char _iv4;
}
@end

@interface I5 : I4 {
 char _iv5;
}

@property int prop0;
@end

@implementation I3
@end

@implementation I4 
@end

@interface I5 ()
@property int prop1;
@property char prop2;
@end

@implementation I5
@synthesize prop0 = _iv6_synth;
@synthesize prop1 = _iv7_synth;
@synthesize prop2 = _iv5;
@end

// The size rounds up to the next available byte.
@interface I6 {
  unsigned iv0 : 2;
}
@end
@implementation I6
@end

// The start of the subclass includes padding for its own alignment.
@interface I7 {
  char a;
}
@end
@interface I8 : I7 {
  double b;
}
@end
@implementation I8
@end

