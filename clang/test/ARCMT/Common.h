#if __has_feature(objc_arr)
#define NS_AUTOMATED_REFCOUNT_UNAVAILABLE __attribute__((unavailable("not available in automatic reference counting mode")))
#else
#define NS_AUTOMATED_REFCOUNT_UNAVAILABLE
#endif

typedef int BOOL;
typedef unsigned NSUInteger;
typedef int int32_t;
typedef unsigned char uint8_t;
typedef int32_t UChar32;
typedef unsigned char UChar;

@protocol NSObject
- (BOOL)isEqual:(id)object;
- (id)retain NS_AUTOMATED_REFCOUNT_UNAVAILABLE;
- (NSUInteger)retainCount NS_AUTOMATED_REFCOUNT_UNAVAILABLE;
- (oneway void)release NS_AUTOMATED_REFCOUNT_UNAVAILABLE;
- (id)autorelease NS_AUTOMATED_REFCOUNT_UNAVAILABLE;
@end

@interface NSObject <NSObject> {}
- (id)init;

+ (id)new;
+ (id)alloc;
- (void)dealloc;

- (void)finalize;

- (id)copy;
- (id)mutableCopy;
@end

NS_AUTOMATED_REFCOUNT_UNAVAILABLE
@interface NSAutoreleasePool : NSObject {
@private
    void    *_token;
    void    *_reserved3;
    void    *_reserved2;
    void    *_reserved;
}

+ (void)addObject:(id)anObject;

- (void)addObject:(id)anObject;

- (void)drain;

@end

typedef const void* objc_objectptr_t; 
extern __attribute__((ns_returns_retained)) id objc_retainedObject(objc_objectptr_t __attribute__((cf_consumed)) pointer);
extern __attribute__((ns_returns_not_retained)) id objc_unretainedObject(objc_objectptr_t pointer);
extern objc_objectptr_t objc_unretainedPointer(id object);
