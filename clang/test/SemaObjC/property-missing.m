// RUN: clang-cc -fsyntax-only -verify %s

// PR3234

@protocol NSCopying @end
@interface NSObject @end

void f1(NSObject *o)
{
  o.foo; // expected-error{{property 'foo' not found on object of type 'NSObject *'}}
}

void f2(id<NSCopying> o)
{
  o.foo; // expected-error{{property 'foo' not found on object of type 'id<NSCopying>'}}
}

void f3(id o)
{
  o.foo; // expected-error{{property 'foo' not found on object of type 'id'}}
}

