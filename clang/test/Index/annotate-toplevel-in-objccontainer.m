@interface Foo
void func1(int);
void func2(int);

-(void)meth1;
-(void)meth2;
@end

@implementation Foo
void func(int);
static int glob1;
static int glob2;

-(void)meth1 {}
-(void)meth2 {}
@end

// RUN: c-index-test -write-pch %t.h.pch -x objective-c-header %s.h

// RUN: c-index-test -test-annotate-tokens=%s:5:1:7:1 %s -include %t.h \
// RUN:     | FileCheck -check-prefix=INTER %s
// CHECK-INTER: Identifier: "meth1" [5:8 - 5:13] ObjCInstanceMethodDecl=meth1:5:1
// CHECK-INTER: Identifier: "meth2" [6:8 - 6:13] ObjCInstanceMethodDecl=meth2:6:1

// RUN: c-index-test -test-annotate-tokens=%s:14:1:16:1 %s -include %t.h \
// RUN:     | FileCheck -check-prefix=IMPL %s
// CHECK-IMPL: Identifier: "meth1" [14:8 - 14:13] ObjCInstanceMethodDecl=meth1:14:1 (Definition)
// CHECK-IMPL: Identifier: "meth2" [15:8 - 15:13] ObjCInstanceMethodDecl=meth2:15:1 (Definition)

// RUN: c-index-test -test-annotate-tokens=%s.h:5:1:7:1 %s -include %t.h \
// RUN:     | FileCheck -check-prefix=PCH %s
// CHECK-PCH: Identifier: "meth1" [5:8 - 5:13] ObjCInstanceMethodDecl=meth1:5:1
// CHECK-PCH: Identifier: "meth2" [6:8 - 6:13] ObjCInstanceMethodDecl=meth2:6:1
