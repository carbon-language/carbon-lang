@interface Foo
- (int)compare:(Foo*)other;
@end

// RUN: c-index-test -test-annotate-tokens=%s:1:1:3:5 %s | FileCheck %s
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
