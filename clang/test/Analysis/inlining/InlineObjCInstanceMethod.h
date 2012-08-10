
// Define a public header for the ObjC methods that are "visible" externally
// and, thus, could be sub-classed. We should explore the path on which these
// are sub-classed with unknown class by not inlining them.

typedef signed char BOOL;
typedef struct objc_class *Class;
typedef struct objc_object {
    Class isa;
} *id;
@protocol NSObject  - (BOOL)isEqual:(id)object; @end
@interface NSObject <NSObject> {}
+(id)alloc;
-(id)init;
-(id)autorelease;
-(id)copy;
- (Class)class;
-(id)retain;
@end

@interface PublicClass : NSObject
- (int)getZeroPublic;
@end

@interface PublicSubClass : PublicClass
@end
