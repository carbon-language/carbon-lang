// RUN: clang -fsyntax-only -verify %s

typedef signed char BOOL;
@protocol NSObject  - (BOOL)isEqual:(id)object; @end

@interface NSObject <NSObject> {} @end

@interface _NSServicesInContextMenu : NSObject {
    id _requestor;
    NSObject *_appleEventDescriptor;
}

@property (retain, nonatomic) id requestor;
@property (retain, nonatomic) id appleEventDescriptor;

@end

@implementation _NSServicesInContextMenu

@synthesize requestor = _requestor, appleEventDescriptor = _appleEventDescriptor;

@end

@class NSString;

@protocol MyProtocol
- (NSString *)stringValue;
@end

@interface MyClass : NSObject {
  id  _myIvar;
}
@property (readwrite, retain) id<MyProtocol> myIvar;
@end

@implementation MyClass
@synthesize myIvar = _myIvar;
@end
