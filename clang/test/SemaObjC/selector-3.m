// RUN: %clang_cc1  -fsyntax-only -Wselector -verify -Wno-objc-root-class %s
// rdar://8851684

@interface Foo
- (void) foo;
- (void) bar;
@end

@implementation Foo
- (void) bar
{
}

- (void) foo
{
  SEL a,b,c;
  a = @selector(b1ar);  // expected-warning {{creating selector for nonexistent method 'b1ar'}}
  b = @selector(bar);
}
@end

@interface I
- length;
@end

SEL func()
{
    return  @selector(length);  // expected-warning {{creating selector for nonexistent method 'length'}}
}

// rdar://9545564
@class MSPauseManager;

@protocol MSPauseManagerDelegate 
@optional
- (void)pauseManagerDidPause:(MSPauseManager *)manager;
- (int)respondsToSelector:(SEL)aSelector;
@end

@interface MSPauseManager
{
  id<MSPauseManagerDelegate> _delegate;
}
@end


@implementation MSPauseManager
- (id) Meth {
  if ([_delegate respondsToSelector:@selector(pauseManagerDidPause:)])
    return 0;
  return 0;
}
@end

// rdar://12938616
@class NSXPCConnection;

@interface NSObject
@end

@interface INTF : NSObject
{
  NSXPCConnection *cnx; // Comes in as a parameter.
}
- (void) Meth;
@end

extern SEL MySelector(SEL s);

@implementation INTF
- (void) Meth {
  if( [cnx respondsToSelector:MySelector(@selector( _setQueue: ))] ) // expected-warning {{creating selector for nonexistent method '_setQueue:'}} 
  {
  }

  if( [cnx respondsToSelector:@selector( _setQueueXX: )] ) // No warning here.
  {
  }
  if( [cnx respondsToSelector:(@selector( _setQueueXX: ))] ) // No warning here.
  {
  }
}
@end

// rdar://14007194
@interface UxTechTest : NSObject
- (int) invalidate : (id)Arg; // expected-warning {{multiple selectors named 'invalidate:' found}}
+ (int) C_invalidate : (int)arg; // expected-warning {{multiple selectors named 'C_invalidate:' found}}
@end

@interface UxTechTest(CAT)
- (char) invalidate : (int)arg; // expected-note {{also found}}
+ (int) C_invalidate : (char)arg; // expected-note {{also found}}
@end

@interface NSPort : NSObject
- (double) invalidate : (void*)Arg1; // expected-note {{also found}}
+ (int) C_invalidate : (id*)arg; // expected-note {{also found}}
@end


@interface USEText : NSPort
- (int) invalidate : (int)arg; // expected-note {{also found}}
@end

@implementation USEText
- (int) invalidate :(int) arg { return 0; }
@end

@interface USETextSub : USEText
- (int) invalidate : (id)arg;
@end
