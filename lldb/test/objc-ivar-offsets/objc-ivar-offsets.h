#import <Foundation/Foundation.h>

@interface BaseClass : NSObject
{
  int _backed_int;
#if !__OBJC2__
  int _unbacked_int;
#endif
}
@property int backed_int;
@property int unbacked_int;
@end

@interface DerivedClass : BaseClass
{
  int _derived_backed_int;
#if !__OBJC2__
  int _unbacked_int;
#endif
}
@property int derived_backed_int;
@property int derived_unbacked_int;
@end
