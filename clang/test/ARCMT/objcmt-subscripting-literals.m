// RUN: rm -rf %t
// RUN: %clang_cc1 -objcmt-migrate-literals -objcmt-migrate-subscripting -mt-migrate-directory %t %s -x objective-c -triple x86_64-apple-darwin11 
// RUN: c-arcmt-test -mt-migrate-directory %t | arcmt-test -verify-transformed-files %s.result
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsyntax-only -x objective-c %s.result

typedef signed char BOOL;
#define nil ((void*) 0)

typedef const struct __CFString * CFStringRef;

@interface NSObject
+ (id)alloc;
@end

@interface NSString : NSObject
+ (id)stringWithString:(NSString *)string;
- (id)initWithString:(NSString *)aString;
@end

@interface NSArray : NSObject
- (id)objectAtIndex:(unsigned long)index;
- (id)objectAtIndexedSubscript:(int)index;
@end

@interface NSArray (NSArrayCreation)
+ (id)array;
+ (id)arrayWithObject:(id)anObject;
+ (id)arrayWithObjects:(const id [])objects count:(unsigned long)cnt;
+ (id)arrayWithObjects:(id)firstObj, ...;
+ (id)arrayWithArray:(NSArray *)array;

- (id)initWithObjects:(const id [])objects count:(unsigned long)cnt;
- (id)initWithObjects:(id)firstObj, ...;
- (id)initWithArray:(NSArray *)array;

- (id)objectAtIndex:(unsigned long)index;
@end

@interface NSMutableArray : NSArray
- (void)replaceObjectAtIndex:(unsigned long)index withObject:(id)anObject;
- (void)setObject:(id)object atIndexedSubscript:(int)index;
@end

@interface NSDictionary : NSObject
- (id)objectForKeyedSubscript:(id)key;
@end

@interface NSDictionary (NSDictionaryCreation)
+ (id)dictionary;
+ (id)dictionaryWithObject:(id)object forKey:(id)key;
+ (id)dictionaryWithObjects:(const id [])objects forKeys:(const id [])keys count:(unsigned long)cnt;
+ (id)dictionaryWithObjectsAndKeys:(id)firstObject, ...;
+ (id)dictionaryWithDictionary:(NSDictionary *)dict;
+ (id)dictionaryWithObjects:(NSArray *)objects forKeys:(NSArray *)keys;

- (id)initWithObjects:(const id [])objects forKeys:(const id [])keys count:(unsigned long)cnt;
- (id)initWithObjectsAndKeys:(id)firstObject, ...;
- (id)initWithDictionary:(NSDictionary *)otherDictionary;
- (id)initWithObjects:(NSArray *)objects forKeys:(NSArray *)keys;

- (id)objectForKey:(id)aKey;
@end

@interface NSMutableDictionary : NSDictionary
- (void)setObject:(id)anObject forKey:(id)aKey;
- (void)setObject:(id)object forKeyedSubscript:(id)key;
@end

@interface NSNumber : NSObject
@end

@interface NSNumber (NSNumberCreation)
+ (NSNumber *)numberWithInt:(int)value;
@end

#define M(x) (x)
#define PAIR(x) @#x, [NSNumber numberWithInt:(x)]
#define TWO(x) ((x), (x))

@interface I
@end
@implementation I
-(void) foo {
  NSString *str;
  NSArray *arr;
  NSDictionary *dict;

  arr = [NSArray array];
  arr = [NSArray arrayWithObject:str];
  arr = [NSArray arrayWithObjects:str, str, nil];
  dict = [NSDictionary dictionary];
  dict = [NSDictionary dictionaryWithObject:arr forKey:str];
  dict = [NSDictionary dictionaryWithObjectsAndKeys: @"value1", @"key1", @"value2", @"key2", nil];
  dict = [NSDictionary dictionaryWithObjectsAndKeys: PAIR(1), PAIR(2), nil];
  dict = [NSDictionary dictionaryWithObjectsAndKeys:
                                               @"value1", @"key1",
#ifdef BLAH
                                               @"value2", @"key2",
#else
                                               @"value3", @"key3",
#endif
                                               nil ];

  id o = [arr objectAtIndex:2];
  o = [dict objectForKey:@"key"];
  o = TWO([dict objectForKey:@"key"]);
  o = [NSDictionary dictionaryWithObject:[NSDictionary dictionary] forKey:@"key"];
  NSMutableArray *marr = 0;
  NSMutableDictionary *mdict = 0;
  [marr replaceObjectAtIndex:2 withObject:@"val"];
  [mdict setObject:@"value" forKey:@"key"];
  [marr replaceObjectAtIndex:2 withObject:[arr objectAtIndex:4]];
  [mdict setObject:[dict objectForKey:@"key2"] forKey:@"key"];
  [mdict setObject:[dict objectForKey:@"key2"] forKey:
#if 1
                     @"key1"
#else
                     @"key2"
#endif
                    ];
  [mdict setObject:[dict objectForKey:
#if 2
                     @"key3"
#else
                     @"key4"
#endif
                   ] forKey:@"key"];
  [mdict setObject:@"value" forKey:[dict objectForKey:
#if 3
                     @"key5"
#else
                     @"key6"
#endif
                   ] ];
  [mdict setObject:@"val" forKey:[dict objectForKey:@"key2"]];
  [mdict setObject:[dict objectForKey:@"key1"] forKey:[dict objectForKey:[NSArray arrayWithObject:@"arrkey"]]];
  __strong NSArray **parr = 0;
  o = [*parr objectAtIndex:2];
  void *hd;
  o = [(NSArray*)hd objectAtIndex:2];
}
@end

extern const CFStringRef globStr;

void test1(NSString *str) {
  NSDictionary *dict = [NSDictionary dictionaryWithObjectsAndKeys: str, globStr, nil];
  dict = [NSDictionary dictionaryWithObjectsAndKeys: globStr, str, nil];
  dict = [NSDictionary dictionaryWithObject:str forKey:globStr];
  dict = [NSDictionary dictionaryWithObject:globStr forKey:str];

  NSArray *arr = [NSArray arrayWithObjects: globStr, globStr, nil];
  arr = [NSArray arrayWithObjects: str, globStr, nil];
  arr = [NSArray arrayWithObjects: globStr, str, nil];
  arr = [NSArray arrayWithObject:globStr];
}
