// RUN: %clang_cc1 -x objective-c++ -Wdocumentation -ast-dump %s | FileCheck %s
// rdar://13647476

//! NSObject is root of all.
@interface NSObject
@end
// CHECK: ObjCInterfaceDecl{{.*}}NSObject
// CHECK-NEXT:   FullComment 0x{{[^ ]*}} <line:[[@LINE-4]]:4, col:28>
// CHECK-NEXT:     ParagraphComment{{.*}} <col:4, col:28>
// CHECK-NEXT:       TextComment{{.*}} <col:4, col:28> Text=" NSObject is root of all."

//! An umbrella class for super classes.
@interface SuperClass 
@end
// CHECK: ObjCInterfaceDecl{{.*}}SuperClass
// CHECK-NEXT: FullComment 0x{{[^ ]*}}  <line:[[@LINE-4]]:4, col:40>
// CHECK-NEXT:   ParagraphComment{{.*}} <col:4, col:40>
// CHECK-NEXT:     TextComment{{.*}} <col:4, col:40> Text=" An umbrella class for super classes."

@interface SubClass : SuperClass
@end
// CHECK: ObjCInterfaceDecl 0x{{[^ ]*}} <line:[[@LINE-2]]:1, line:[[@LINE-1]]:2> SubClass
// CHECK-NEXT: ObjCInterface 0x{{[^ ]*}} 'SuperClass'
// CHECK-NEXT: FullComment
// CHECK-NEXT: ParagraphComment{{.*}} <col:4, col:40>
// CHECK-NEXT: TextComment{{.*}} <col:4, col:40> Text=" An umbrella class for super classes."

@interface SubSubClass : SubClass
@end
// CHECK: ObjCInterfaceDecl 0x{{[^ ]*}}  <line:[[@LINE-2]]:1, line:[[@LINE-1]]:2> SubSubClass
// CHECK-NEXT: ObjCInterface{{.*}} 'SubClass'
// CHECK-NEXT: FullComment
// CHECK-NEXT: ParagraphComment{{.*}} <col:4, col:40>
// CHECK-NEXT: TextComment{{.*}} <col:4, col:40> Text=" An umbrella class for super classes."

@interface SubSubClass (Private)
@end
// CHECK: ObjCCategoryDecl 0x{{[^ ]*}} <line:[[@LINE-2]]:1, line:[[@LINE-1]]:2> Private
// CHECK-NEXT: ObjCInterface{{.*}} 'SubSubClass'
// CHECK-NEXT: FullComment
// CHECK-NEXT: ParagraphComment{{.*}} <col:4, col:40>
// CHECK-TEXT: TextComment{{.*}} <col:4, col:40> Text=" An umbrella class for super classes."

//! Something valuable to the organization.
class Asset {

};
// CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-3]]:1, line:[[@LINE-1]]:1> class Asset
// CHECK-NEXT: FullComment
// CHECK-NEXT: ParagraphComment{{.*}} <col:4, col:43>
// CHECK-NEXT: TextComment{{.*}} <col:4, col:43> Text=" Something valuable to the organization."

//! An individual human or human individual.
class Person : public Asset {
};
// CHECK: CXXRecordDecl 0x{{[^ ]*}}  <line:[[@LINE-2]]:1, line:[[@LINE-1]]:1> class Person
// CHECK-NEXT: public 'class Asset'
// CHECK-NEXT: FullComment
// CHECK-NEXT: ParagraphComment{{.*}} <col:4, col:44>
// CHECK-NEXT: TextComment{{.*}} <col:4, col:44> Text=" An individual human or human individual."

class Student : public Person {
};
// CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-2]]:1, line:[[@LINE-1]]:1> class Student
// CHECK-NEXT: public 'class Person'
// CHECK-NEXT: FullComment
// CHECK-NEXT: ParagraphComment{{.*}} <col:4, col:44>
// CHECK-NEXT: TextComment{{.*}} <col:4, col:44> Text=" An individual human or human individual."

//! Every thing is a part
class Parts {
};
// CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-2]]:1, line:[[@LINE-1]]:1> class Parts
// CHECK-NEXT: FullComment
// CHECK-NEXT: ParagraphComment{{.*}} <col:4, col:25>
// CHECK-NEXT: TextComment{{.*}} <col:4, col:25> Text=" Every thing is a part"

class Window : virtual Parts {
};
// CHECK: CXXRecordDecl  0x{{[^ ]*}} <line:[[@LINE-2]]:1, line:[[@LINE-1]]:1> class Window
// CHECK-NEXT: virtual private 'class Parts'
// CHECK-NEXT: FullComment
// CHECK-NEXT: ParagraphComment{{.*}} <col:4, col:25>
// CHECK-NEXT: TextComment{{.*}} <col:4, col:25> Text=" Every thing is a part"

class Door : virtual Parts {
};
// CHECK: CXXRecordDecl  0x{{[^ ]*}} <line:[[@LINE-2]]:1, line:[[@LINE-1]]:1> class Door
// CHECK-NEXT: virtual private 'class Parts'
// CHECK-NEXT: FullComment
// CHECK-NEXT: ParagraphComment{{.*}} <col:4, col:25>
// CHECK-NEXT: TextComment{{.*}} <col:4, col:25> Text=" Every thing is a part"

class House : Window, Door {
};
// CHECK: CXXRecordDecl  0x{{[^ ]*}} <line:[[@LINE-2]]:1, line:[[@LINE-1]]:1> class House
// CHECK-NEXT: private 'class Window'
// CHECK-NEXT: private 'class Door'
// CHECK-NEXT: FullComment
// CHECK-NEXT: ParagraphComment{{.*}} <col:4, col:25>
// CHECK-NEXT: TextComment{{.*}} <col:4, col:25> Text=" Every thing is a part"
