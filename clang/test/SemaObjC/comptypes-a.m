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

@interface TedWantsToVerifyObjCDoesTheRightThing

- compareThis:(int)a withThat:(id)b;  // expected-note {{previous definition is here}}

@end

@implementation TedWantsToVerifyObjCDoesTheRightThing

- compareThis:(id<PBXCompletionItem>)a withThat:(id<PBXCompletionItem>)b { // expected-warning {{conflicting types for 'compareThis:withThat:'}}
  return self;
}

@end
