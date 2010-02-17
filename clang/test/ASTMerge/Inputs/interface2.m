// Matches
@interface I1 {
  int ivar1;
}
@end

// Matches
@interface I2 : I1 {
  float ivar2;
}
@end

// Ivar mismatch
@interface I3 {
  int ivar1;
  float ivar2;
}
@end

// Superclass mismatch
@interface I4 : I1 {
}
@end

// Methods match
@interface I5
+ (float)bar;
- (int)foo;
@end

// Method mismatch
@interface I6
+ (float)foo;
@end

// Method mismatch
@interface I7
- (int)foo;
+ (int)bar:(float)x;
@end

// Method mismatch
@interface I8
- (int)foo;
+ (int)bar:(float)x, ...;
@end
