// RUN: clang -cc1 -fsyntax-only -verify %s

@class FOO, BAR;
@class FOO, BAR;

@interface INTF : FOO	// expected-error {{cannot find interface declaration for 'FOO', superclass of 'INTF'}}
@end

@interface FOO 
- (BAR*) Meth1;
- (FOO*) Meth2;
@end

@interface INTF1 : FOO	
@end

@interface INTF2 : INTF1 // expected-note {{previous definition is here}}
@end


@class INTF1, INTF2;

@interface INTF2 : INTF1 // expected-error {{duplicate interface definition for class 'INTF2'}}
@end

// 2nd test of a forward class declaration matching a typedef name
// referring to class object.
// FIXME. This may become a negative test should we decide to make this an error.
//
@interface NSObject @end

@protocol XCElementP @end

typedef NSObject <XCElementP> XCElement;

@interface XCElementMainImp  {
  XCElement * _editingElement;
}
@end

@class XCElement;

@implementation XCElementMainImp
- (XCElement *)editingElement  { return _editingElement;  }
@end


