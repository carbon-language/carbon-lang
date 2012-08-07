// RUN: %clang_cc1  -fsyntax-only -Wdirect-ivar-access -verify -Wno-objc-root-class %s
// rdar://6505197

__attribute__((objc_root_class)) @interface MyObject {
@public
    id _myMaster;
    id _isTickledPink;
}
@property(retain) id myMaster;
@property(assign) id isTickledPink;
@end

@implementation MyObject

@synthesize myMaster = _myMaster;
@synthesize isTickledPink = _isTickledPink;

- (void) doSomething {
    _myMaster = _isTickledPink; // expected-warning {{instance variable '_myMaster' is being directly accessed}} \
    // expected-warning {{instance variable '_isTickledPink' is being directly accessed}}
}

- (id) init {
    _myMaster=0;
    return _myMaster;
}
- (void) dealloc { _myMaster = 0; }
@end

MyObject * foo ()
{
	MyObject* p=0;
        p.isTickledPink = p.myMaster;	// ok
	p->_isTickledPink = (*p)._myMaster; // expected-warning {{instance variable '_isTickledPink' is being directly accessed}} \
        // expected-warning {{instance variable '_myMaster' is being directly accessed}}
	return p->_isTickledPink; // expected-warning {{instance variable '_isTickledPink' is being directly accessed}}
}

