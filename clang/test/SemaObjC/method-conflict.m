// RUN: clang -cc1 -fsyntax-only -verify %s

typedef signed char BOOL;
typedef unsigned int NSUInteger;
typedef struct _NSZone NSZone;
@class NSInvocation, NSMethodSignature, NSCoder, NSString, NSEnumerator;
@protocol NSObject  - (BOOL)isEqual:(id)object;
@end  @protocol NSCopying  - (id)copyWithZone:(NSZone *)zone;
@end  @protocol NSMutableCopying  - (id)mutableCopyWithZone:(NSZone *)zone;
@end  @protocol NSCoding  - (void)encodeWithCoder:(NSCoder *)aCoder;
@end    @interface NSObject <NSObject> {
}
@end    extern id NSAllocateObject(Class aClass, NSUInteger extraBytes, NSZone *zone);
@interface NSValue : NSObject <NSCopying, NSCoding>  - (void)getValue:(void *)value;
@end        @class NSString;
typedef struct _NSRange {
}
  NSRange;
@interface NSValue (NSValueRangeExtensions)  + (NSValue *)valueWithRange:(NSRange)range;
@end  @interface NSAttributedString : NSObject <NSCopying, NSMutableCopying, NSCoding>  - (NSString *)string;
@end  @interface NSMutableAttributedString : NSAttributedString  - (void)replaceCharactersInRange:(NSRange)range withString:(NSString *)str;
@end       @class NSArray, NSDictionary, NSString, NSError;
@interface NSScanner : NSObject <NSCopying>  - (NSString *)string;
@end        typedef struct {
}
  CSSM_FIELDGROUP, *CSSM_FIELDGROUP_PTR;
@protocol XDUMLClassifier;
@protocol XDUMLClassInterfaceCommons <XDUMLClassifier> 
@end  @protocol XDUMLImplementation;
@protocol XDUMLElement <NSObject> - (NSArray *) ownedElements;
@end @protocol XDUMLDataType;
@protocol XDUMLNamedElement <XDUMLElement>     - (NSString *) name;
@end enum _XDSourceLanguage {
XDSourceUnknown=0,     XDSourceJava,     XDSourceC,     XDSourceCPP,     XDSourceObjectiveC };
typedef NSUInteger XDSourceLanguage;
@protocol XDSCClassifier <XDUMLClassInterfaceCommons> - (XDSourceLanguage)language;
@end  @class XDSCDocController;
@interface XDSCDisplaySpecification : NSObject <NSCoding>{
}
@end  @class XDSCOperation;
@interface XDSCClassFormatter : NSObject {
}
+ (NSUInteger) compartmentsForClassifier: (id <XDUMLClassifier>) classifier withSpecification: (XDSCDisplaySpecification *) displaySpec; 
@end  
@class NSString;
@implementation XDSCClassFormatter       

+ appendVisibility: (id <XDUMLNamedElement>) element withSpecification: (XDSCDisplaySpecification *) displaySpec to: (NSMutableAttributedString *) attributedString
{
  return 0;
}
+ (NSUInteger) compartmentsForClassifier: (id <XDSCClassifier>) classifier withSpecification: (XDSCDisplaySpecification *) displaySpec {
  return 0;
}
@end 
