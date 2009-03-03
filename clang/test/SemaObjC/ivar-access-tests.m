// RUN: clang -fsyntax-only -verify %s

@interface MySuperClass
{
@private
  int private;

@protected
  int protected;

@public
  int public;
}
@end

@implementation MySuperClass
- (void) test {
    int access;
    MySuperClass *s = 0;
    access = s->private;   
    access = s->protected;
}
@end


@interface MyClass : MySuperClass 
@end

@implementation MyClass
- (void) test {
    int access;
    MySuperClass *s = 0;
    access = s->private; // expected-error {{instance variable 'private' is private}}
    access = s->protected;
    MyClass *m=0;
    access = m->private; // expected-error {{instance variable 'private' is private}}
    access = m->protected;
}
@end


@interface Deeper : MyClass
@end

@implementation Deeper 
- (void) test {
    int access;
    MySuperClass *s = 0;
    access = s->private; // expected-error {{instance variable 'private' is private}}
    access = s->protected;
    MyClass *m=0;
    access = m->private; // expected-error {{instance variable 'private' is private}}
    access = m->protected;
}
@end

@interface Unrelated
@end

@implementation Unrelated 
- (void) test {
    int access;
    MySuperClass *s = 0;
    access = s->private; // expected-error {{instance variable 'private' is private}}
    access = s->protected; // expected-error {{instance variable 'protected' is protected}}
    MyClass *m=0;
    access = m->private; // expected-error {{instance variable 'private' is private}}
    access = m->protected; // expected-error {{instance variable 'protected' is protected}}
}
@end

int main (void)
{
  MySuperClass *s = 0;
  int access;
  access = s->private;   // expected-error {{instance variable 'private' is private}}
  access = s->protected; // expected-error {{instance variable 'protected' is protected}}
  return 0;
}

