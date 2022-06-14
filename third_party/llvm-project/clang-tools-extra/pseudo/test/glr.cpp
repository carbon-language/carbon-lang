// RUN: clang-pseudo -grammar=%cxx-bnf-file -source=%s --print-forest | FileCheck %s

void foo() {
  T* a; // a multiply expression or a pointer declaration?
// CHECK:      statement-seq~statement := <ambiguous>
// CHECK-NEXT: ├─statement~expression-statement := expression ;
// CHECK-NEXT: │ ├─expression~multiplicative-expression := multiplicative-expression * pm-expression
// CHECK-NEXT: │ │ ├─multiplicative-expression~IDENTIFIER := tok[5]
// CHECK-NEXT: │ │ ├─* := tok[6]
// CHECK-NEXT: │ │ └─pm-expression~IDENTIFIER := tok[7]
// CHECK-NEXT: │ └─; := tok[8]
// CHECK-NEXT: └─statement~simple-declaration := decl-specifier-seq init-declarator-list ;
// CHECK-NEXT:   ├─decl-specifier-seq~simple-type-specifier := <ambiguous>
// CHECK-NEXT:   │ ├─simple-type-specifier~type-name := <ambiguous>
// CHECK-NEXT:   │ │ ├─type-name~IDENTIFIER := tok[5]
// CHECK-NEXT:   │ │ ├─type-name~IDENTIFIER := tok[5]
// CHECK-NEXT:   │ │ └─type-name~IDENTIFIER := tok[5]
// CHECK-NEXT:   │ └─simple-type-specifier~IDENTIFIER := tok[5]
// CHECK-NEXT:   ├─init-declarator-list~ptr-declarator := ptr-operator ptr-declarator
// CHECK-NEXT:   │ ├─ptr-operator~* := tok[6]
// CHECK-NEXT:   │ └─ptr-declarator~IDENTIFIER := tok[7]
// CHECK-NEXT:   └─; := tok[8]
}
