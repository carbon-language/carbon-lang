// RUN: clang-cc -fsyntax-only -verify -pedantic %s
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

@interface TedWantsToVerifyObjCDoesTheRightThing

- compareThis:(int)a withThat:(id)b;  // expected-note {{previous definition is here}}

@end

@implementation TedWantsToVerifyObjCDoesTheRightThing

- compareThis:(id<PBXCompletionItem>)
    a // expected-warning {{conflicting parameter types in implementation of 'compareThis:withThat:': 'int' vs 'id<PBXCompletionItem>'}}
     withThat:(id<PBXCompletionItem>)b {
  return self;
}

@end
