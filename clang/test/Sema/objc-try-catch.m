// RUN: clang -fsyntax-only -verify %s
typedef signed char BOOL;
typedef struct _NSZone NSZone;

@class NSInvocation, NSMethodSignature, NSCoder, NSString, NSEnumerator;

@protocol NSObject
- (BOOL)isEqual:(id)object;
@end

@protocol NSCopying
- (id)copyWithZone:(NSZone *)zone;
@end

@protocol NSCoding
- (void)encodeWithCoder:(NSCoder *)aCoder;
@end

@interface NSObject <NSObject> {}
@end

@class NSData, NSArray, NSDictionary, NSCharacterSet, NSData, NSURL, NSError, NSLocale;

@interface NSException : NSObject <NSCopying, NSCoding> {}
@end

@class ASTNode, XCRefactoringParser, Transform, TransformInstance, XCRefactoringSelectionInfo;

@interface XCRefactoringTransformation : NSObject {}
@end

@implementation XCRefactoringTransformation
- (NSDictionary *)setUpInfoForTransformKey:(NSString *)transformKey outError:(NSError **)outError; {
    @try {}
    // the exception name is optional (weird)
    @catch (NSException *) {}
}
