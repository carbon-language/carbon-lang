@__experimental_modules_import category_top;

@interface Foo(Right1)
-(void)right1;
@end

@interface Foo(Right2)
-(void)right2;
@end

@interface Foo(Duplicate) // expected-warning {{duplicate definition of category}}
@end
