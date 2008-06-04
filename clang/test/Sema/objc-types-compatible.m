// RUN: clang -fsyntax-only -verify %s
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
