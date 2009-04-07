// RUN: clang-cc  -fsyntax-only -verify %s

@interface Subtask
{
  id _delegate;
}
@property(nonatomic,readwrite,assign)   id __weak       delegate;
@end

@implementation Subtask
@synthesize delegate = _delegate;
@end

 
