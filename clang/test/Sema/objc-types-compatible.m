// RUN: clang -fsyntax-only -verify -pedantic %s
typedef signed char BOOL;
typedef int NSInteger;

@class NSString;

@protocol PBXCompletionItem
- (NSString *) name;
- (NSInteger)priority;
@end

extern NSInteger codeAssistantCaseCompareItems(id a, id b, void *context);

NSInteger codeAssistantCaseCompareItems(id<PBXCompletionItem> a, id<PBXCompletionItem> b, void *context)
{
}

#if 0
FIXME: clang needs to compare each method prototype with its definition (see below).

GCC produces the following correct warnning:
[snaroff:llvm/tools/clang] snarofflocal% cc -c test/Sema/objc-types-compatible.m 
test/Sema/objc-types-compatible.m: In function ‘-[TedWantsToVerifyObjCDoesTheRightThing compareThis:withThat:]’:
test/Sema/objc-types-compatible.m:26: warning: conflicting types for ‘-(id)compareThis:(id <PBXCompletionItem>)a withThat:(id <PBXCompletionItem>)b’
test/Sema/objc-types-compatible.m:20: warning: previous declaration of ‘-(id)compareThis:(int)a withThat:(id)b’
#endif

@interface TedWantsToVerifyObjCDoesTheRightThing

- compareThis:(int)a withThat:(id)b;

@end

@implementation TedWantsToVerifyObjCDoesTheRightThing

- compareThis:(id<PBXCompletionItem>)a withThat:(id<PBXCompletionItem>)b {
  return self;
}

@end
