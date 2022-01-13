// RUN: %clang_cc1 -E %s -o %t.mm
// RUN: %clang_cc1 -fms-extensions -rewrite-objc -debug-info-kind=limited %t.mm -o %t-rw.cpp
// RUN: FileCheck  -check-prefix CHECK-LINE --input-file=%t-rw.cpp %s
// RUN: %clang_cc1 -fms-extensions -rewrite-objc %t.mm -o %t-rwnog.cpp
// RUN: FileCheck  -check-prefix CHECK-NOLINE --input-file=%t-rwnog.cpp %s
// rdar://13138170

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
    _myMaster = _isTickledPink;
}

@end

MyObject * foo ()
{
	MyObject* p;
        p.isTickledPink = p.myMaster;	// ok
	p->_isTickledPink = p->_myMaster;
	return p->_isTickledPink;
}

// CHECK-LINE: #line 22
// CHECK-LINE: #line 28
// CHECK-NOLINE-NOT: #line 22
// CHECK-NOLINE-NOT: #line 28

