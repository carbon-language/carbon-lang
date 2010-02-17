// Matching properties
@interface I1 {
}
- (int)getProp2;
- (void)setProp2:(int)value;
@end

// Mismatched property
@interface I2
@property (readonly) float Prop1;
@end

