__import_module__ category_left;

@interface Foo(Bottom)
-(void)bottom;
@end

__import_module__ category_right;

@interface LeftFoo(Bottom)
-(void)bottom;
@end
