// RUN: clang -cc1 -fobjc-nonfragile-abi -verify -fsyntax-only %s

@interface NSView 
  - (id)initWithView:(id)realView;
@end

@implementation NSView
 - (id)initWithView:(id)realView {
     *(NSView *)self = *(NSView *)realView;	// expected-error {{indirection cannot be to an interface in non-fragile ABI}}
 }
@end

