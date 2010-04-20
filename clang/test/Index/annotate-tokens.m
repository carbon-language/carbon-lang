@interface Foo
- (int)compare:(Foo*)other;
@end

@implementation Foo
- (int)compare:(Foo*)other {
  return 0;
  (void)@encode(Foo);
}
@end

// RUN: c-index-test -test-annotate-tokens=%s:1:1:10:5 %s | FileCheck %s
// CHECK: Punctuation: "@" [1:1 - 1:2]
// CHECK: Identifier: "interface" [1:2 - 1:11]
// CHECK: Identifier: "Foo" [1:12 - 1:15] ObjCInterfaceDecl=Foo:1:12
// CHECK: Punctuation: "-" [2:1 - 2:2] ObjCInstanceMethodDecl=compare::2:1
// CHECK: Punctuation: "(" [2:3 - 2:4]
// CHECK: Keyword: "int" [2:4 - 2:7]
// CHECK: Punctuation: ")" [2:7 - 2:8]
// CHECK: Identifier: "compare" [2:8 - 2:15]
// CHECK: Punctuation: ":" [2:15 - 2:16]
// CHECK: Punctuation: "(" [2:16 - 2:17]
// CHECK: Identifier: "Foo" [2:17 - 2:20] ObjCClassRef=Foo:1:12
// CHECK: Punctuation: "*" [2:20 - 2:21]
// CHECK: Punctuation: ")" [2:21 - 2:22]
// CHECK: Identifier: "other" [2:22 - 2:27] ParmDecl=other:2:22 (Definition)
// CHECK: Punctuation: ";" [2:27 - 2:28]
// CHECK: Punctuation: "@" [3:1 - 3:2]
// CHECK: Identifier: "end" [3:2 - 3:5]
// CHECK: Punctuation: "@" [5:1 - 5:2] ObjCImplementationDecl=Foo:5:1 (Definition)
// CHECK: Identifier: "implementation" [5:2 - 5:16]
// CHECK: Identifier: "Foo" [5:17 - 5:20]
// CHECK: Punctuation: "-" [6:1 - 6:2] ObjCInstanceMethodDecl=compare::6:1 (Definition)
// CHECK: Punctuation: "(" [6:3 - 6:4]
// CHECK: Keyword: "int" [6:4 - 6:7]
// CHECK: Punctuation: ")" [6:7 - 6:8]
// CHECK: Identifier: "compare" [6:8 - 6:15]
// CHECK: Punctuation: ":" [6:15 - 6:16]
// CHECK: Punctuation: "(" [6:16 - 6:17]
// CHECK: Identifier: "Foo" [6:17 - 6:20] ObjCClassRef=Foo:1:12
// CHECK: Punctuation: "*" [6:20 - 6:21]
// CHECK: Punctuation: ")" [6:21 - 6:22]
// CHECK: Identifier: "other" [6:22 - 6:27] ParmDecl=other:6:22 (Definition)
// CHECK: Punctuation: "{" [6:28 - 6:29]
// CHECK: Keyword: "return" [7:3 - 7:9]
// CHECK: Literal: "0" [7:10 - 7:11]
// CHECK: Punctuation: ";" [7:11 - 7:12]
// CHECK: Punctuation: "(" [8:3 - 8:4]
// CHECK: Keyword: "void" [8:4 - 8:8]
// CHECK: Punctuation: ")" [8:8 - 8:9]
// CHECK: Punctuation: "@" [8:9 - 8:10]
// CHECK: Identifier: "encode" [8:10 - 8:16]
// CHECK: Punctuation: "(" [8:16 - 8:17]
// CHECK: Identifier: "Foo" [8:17 - 8:20] ObjCClassRef=Foo:1:12
// CHECK: Punctuation: ")" [8:20 - 8:21]
// CHECK: Punctuation: ";" [8:21 - 8:22]
// CHECK: Punctuation: "}" [9:1 - 9:2]
// CHECK: Punctuation: "@" [10:1 - 10:2]
// CHECK: Identifier: "end" [10:2 - 10:5]
