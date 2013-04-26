// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: c-index-test -test-load-source all -comments-xml-schema=%S/../../bindings/xml/comment-xml-schema.rng %s > %t/out
// RUN: FileCheck %s < %t/out
// rdar://13647476

//! NSObject is root of all.
@interface NSObject
@end
// CHECK:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph
// CHECK-NEXT:         (CXComment_Text Text=[ NSObject is root of all.])))]

//! An umbrella class for super classes.
@interface SuperClass 
@end
// CHECK:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph
// CHECK-NEXT:         (CXComment_Text Text=[ An umbrella class for super classes.])))]

@interface SubClass : SuperClass
@end
// CHECK:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph
// CHECK-NEXT:         (CXComment_Text Text=[ An umbrella class for super classes.])))]

@interface SubSubClass : SubClass
@end
// CHECK:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph
// CHECK-NEXT:         (CXComment_Text Text=[ An umbrella class for super classes.])))]

@interface SubSubClass (Private)
@end
// CHECK:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph
// CHECK-NEXT:         (CXComment_Text Text=[ An umbrella class for super classes.])))]

//! Something valuable to the organization.
class Asset {
};
// CHECK:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph
// CHECK-NEXT:         (CXComment_Text Text=[ Something valuable to the organization.])))]

//! An individual human or human individual.
class Person : public Asset {
};
// CHECK:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph
// CHECK-NEXT:         (CXComment_Text Text=[ An individual human or human individual.])))]

class Student : public Person {
};
// CHECK:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph
// CHECK-NEXT:         (CXComment_Text Text=[ An individual human or human individual.])))]

//! Every thing is a part
class Parts {
};
// CHECK:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph
// CHECK-NEXT:         (CXComment_Text Text=[ Every thing is a part])))]

class Window : public virtual Parts {
};
// CHECK:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph
// CHECK-NEXT:         (CXComment_Text Text=[ Every thing is a part])))]

class Door : public virtual Parts {
};
// CHECK:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph
// CHECK-NEXT:         (CXComment_Text Text=[ Every thing is a part])))]

class House : public Window, Door {
};
// CHECK:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph
// CHECK-NEXT:         (CXComment_Text Text=[ Every thing is a part])))]

//! Any Material
class Material : virtual Parts {
};

class Building : Window, public Material {
};
// CHECK:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph
// CHECK-NEXT:         (CXComment_Text Text=[ Any Material])))]


