// RUN: c-index-test -test-load-source all %s | FileCheck %s

struct __attribute__((packed)) Test2 {
  char a;
};

// CHECK: attributes.c:3:32: StructDecl=Test2:3:32 (Definition) Extent=[3:1 - 5:2]
// CHECK: attributes.c:3:23: attribute(packed)=packed Extent=[3:23 - 3:29]
// CHECK: attributes.c:4:8: FieldDecl=a:4:8 (Definition) Extent=[4:3 - 4:9] [access=public]

