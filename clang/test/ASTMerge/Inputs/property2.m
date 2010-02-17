// Matching properties
@interface I1 {
}
- (int)getProp2;
- (void)setProp2:(int)value;
@property (readonly) int Prop1;
@property (getter = getProp2, setter = setProp2:) int Prop2;
@end

// Mismatched property
@interface I2
@property (readonly) int Prop1;
@end
