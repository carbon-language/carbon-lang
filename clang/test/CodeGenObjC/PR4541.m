// RUN: clang-cc -o %t -w  -g %s


@class NSString;
@interface NSAttributedString 
- (NSString *)string;
@end 
@interface NSMutableAttributedString : NSAttributedString 
@end 
@class NSImage;
@implementation CYObjectsController 
+ (void)initialize {
}
+ (NSAttributedString *)attributedStringWithString:(id)string image:(NSImage *)image  {
  NSMutableAttributedString *attrStr;
}
@end


