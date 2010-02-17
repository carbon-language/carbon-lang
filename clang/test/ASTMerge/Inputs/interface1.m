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
  int ivar2;
}
@end

// Superclass mismatch
@interface I4 : I2 {
}
@end

// Methods match
@interface I5
- (int)foo;
+ (float)bar;
@end

// Method mismatch
@interface I6
- (int)foo;
+ (int)foo;
@end

// Method mismatch
@interface I7
- (int)foo;
+ (int)bar:(int)x;
@end

// Method mismatch
@interface I8
- (int)foo;
+ (int)bar:(float)x;
@end
