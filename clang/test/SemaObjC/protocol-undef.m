// RUN: clang-cc -fsyntax-only -verify %s

typedef signed char BOOL;
@class NSInvocation, NSMethodSignature, NSCoder, NSString, NSEnumerator;
@protocol NSObject
- (BOOL)isEqual:(id)object;
@end
@protocol NSCoding
- (void)encodeWithCoder:(NSCoder *)aCoder;
@end 
@interface NSObject <NSObject> {} @end
@class NSArray, NSAttributedString, NSEvent, NSInputServer, NSImage;
@interface NSController : NSObject <NSCoding> {} @end
@class OzzyView;
typedef struct _OzzyInset {} OzzyInset;
@protocol OzzyP;
typedef NSObject <OzzyP> Ozzy;
@protocol OzzyAnchorP;
typedef NSObject <OzzyAnchorP> OzzyAnchor;
@protocol OzzyAnchorDelegateP
- (BOOL)anchor:(OzzyAnchor *)anchor confirmRepresentedObject:(id)newObject;
@end
typedef NSObject <OzzyAnchorDelegateP> OzzyAnchorDelegate;

@protocol OzzyAnchorP <OzzyP>
  @property(nonatomic,retain) id representedObject;
  @property(nonatomic,retain) Ozzy * contentGroup;
@end
@interface XXX : NSObject <OzzyP> // expected-warning{{cannot find protocol definition for 'OzzyP'}}
@end
@protocol OzzyActionDelegateP
  @optional - (BOOL)elementView:(OzzyView *)elementView shouldDragElement:(Ozzy *)element;
@end
typedef NSObject <OzzyActionDelegateP> OzzyActionDelegate;
@interface OzzyUnit : OzzyAnchorDelegate <OzzyAnchorDelegateP> {}
@end
@interface OzzyArrayUnit : OzzyUnit {} @end
@implementation OzzyArrayUnit
- (BOOL)willChangeLayoutForObjects:(NSArray *)objects fromObjects:(NSArray *)oldObjects {}
- (void)_recalculateStoredArraysForAnchor:(OzzyAnchor *)anchor {
  Ozzy * contentGroup = anchor.contentGroup;
  if (contentGroup == ((void *)0)) {
    contentGroup = anchor;
  }
}
@end
