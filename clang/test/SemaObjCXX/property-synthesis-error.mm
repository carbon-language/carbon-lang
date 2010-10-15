// RUN: %clang_cc1 -fsyntax-only -verify %s
// rdar: //8550657

@interface NSArray @end

@interface NSMutableArray : NSArray @end

@interface MyClass
{
  NSMutableArray * _array;
}

@property (readonly) NSArray * array;

@end

@interface MyClass ()

@property (readwrite) NSMutableArray * array;

@end

@implementation MyClass

@synthesize array=_array;

@end

int main(void)
{
  return 0;
}
