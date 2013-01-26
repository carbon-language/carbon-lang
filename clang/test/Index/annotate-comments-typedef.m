// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: c-index-test -test-load-source all -comments-xml-schema=%S/../../bindings/xml/comment-xml-schema.rng %s > %t/out
// RUN: FileCheck %s < %t/out
// rdar://13067629

// Ensure that XML we generate is not invalid.
// RUN: FileCheck %s -check-prefix=WRONG < %t/out
// WRONG-NOT: CommentXMLInvalid

/** Documentation for NSUInteger */
typedef unsigned int NSUInteger;

/** Documentation for MyEnum */
typedef enum : NSUInteger {
        MyEnumFoo, /**< value Foo */
        MyEnumBar, /**< value Bar */
        MyEnumBaz, /**< value Baz */
} MyEnum;
// CHECK: TypedefDecl=MyEnum:[[@LINE-1]]:3 (Definition) FullCommentAsHTML=[<p class="para-brief"> Documentation for MyEnum </p>] FullCommentAsXML=[<Typedef file="{{[^"]+}}annotate-comments-typedef.m" line="[[@LINE-1]]" column="3"><Name>&lt;anonymous&gt;</Name><USR>c:@EA@MyEnum</USR><Declaration>typedef enum MyEnum MyEnum</Declaration><Abstract><Para> Documentation for MyEnum </Para></Abstract></Typedef>]


/** Documentation for E */
enum E {
        E_MyEnumFoo, /**< value Foo */
        E_MyEnumBar, /**< value Bar */
        E_MyEnumBaz, /**< value Baz */
};
typedef enum E E_T;
// CHECK: TypedefDecl=E_T:[[@LINE-1]]:16 (Definition) FullCommentAsHTML=[<p class="para-brief"> Documentation for E </p>] FullCommentAsXML=[<Typedef file="{{[^"]+}}annotate-comments-typedef.m" line="[[@LINE-1]]" column="16"><Name>E</Name><USR>c:@E@E</USR><Declaration>typedef enum E E_T</Declaration><Abstract><Para> Documentation for E </Para></Abstract></Typedef>]

/** Comment about Foo */
typedef struct {
	  int iii;
        } Foo;
// CHECK: TypedefDecl=Foo:[[@LINE-1]]:11 (Definition) FullCommentAsHTML=[<p class="para-brief"> Comment about Foo </p>] FullCommentAsXML=[<Typedef file="{{[^"]+}}annotate-comments-typedef.m" line="[[@LINE-1]]" column="11"><Name>&lt;anonymous&gt;</Name><USR>c:@SA@Foo</USR><Declaration>typedef struct Foo Foo</Declaration><Abstract><Para> Comment about Foo </Para></Abstract></Typedef>]
