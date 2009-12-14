// RUN: clang -cc1 -triple i386-apple-darwin9 -fsyntax-only -verify %s

@interface Subtask
{
  id _delegate;
}
@property(nonatomic,readwrite,assign)   id __weak       delegate;
@end

@implementation Subtask
@synthesize delegate = _delegate;
@end

 
@interface PVSelectionOverlayView2 
{
 id __weak _selectionRect;
}

@property(assign) id selectionRect;

@end

@implementation PVSelectionOverlayView2

@synthesize selectionRect = _selectionRect;
@end

