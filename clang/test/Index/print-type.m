@interface Foo
@property (readonly) id x;
-(int) mymethod;
-(const id) mymethod2:(id)x blah:(Class)y boo:(SEL)z;
@end

// RUN: c-index-test -test-print-type %s | FileCheck %s
// CHECK: ObjCPropertyDecl=x:2:25 [readonly,] [type=id] [typekind=ObjCId] [canonicaltype=id] [canonicaltypekind=ObjCObjectPointer] [isPOD=1]
// CHECK: ObjCInstanceMethodDecl=mymethod:3:8 [type=] [typekind=Invalid] [resulttype=int] [resulttypekind=Int] [isPOD=0]
// CHECK: ObjCInstanceMethodDecl=mymethod2:blah:boo::4:13 [type=] [typekind=Invalid] [resulttype=const id] [resulttypekind=ObjCId] [args= [id] [ObjCId] [Class] [ObjCClass] [SEL] [ObjCSel]] [isPOD=0]
