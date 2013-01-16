@interface Foo(RightSub) <P2>
@property id right_sub_prop;
@end

@interface Foo() {
@public
  int right_sub_ivar;
}
@end
