#import <Foundation/Foundation.h>

@interface Foo : NSObject
+(void) doSomethingWithString: (NSString *) string;

@end

@implementation Foo
+(void) doSomethingWithString: (NSString *) string
{
  NSLog (@"String is: %@.", string); // Set breakpoint here.

}
@end

int main()
{
  [Foo doSomethingWithString: @"Some string I have in mind."];
  return 0;
}
