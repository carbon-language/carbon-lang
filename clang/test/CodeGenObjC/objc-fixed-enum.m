// RUN: %clang_cc1 -g -emit-llvm -o - %s | FileCheck %s

// The DWARF standard says the underlying data type of an enum may be
// stored in an DW_AT_type entry in the enum DIE. This is useful to have
// so the debugger knows about the signedness of the underlying type.

typedef long NSInteger;
#define NS_ENUM(_type, _name) enum _name : _type _name; enum _name : _type

// Enum with no specified underlying type
typedef enum {
  Enum0One,
  Enum0Two
} Enum0;

// Enum declared with the NS_ENUM macro
typedef NS_ENUM(NSInteger, Enum1) {
  Enum1One = -1,
  Enum1Two
};

// Enum declared with a fixed underlying type
typedef enum : NSInteger {
  Enum2One = -1,
  Enum2Two
} Enum2;

// Typedef and declaration separately
enum : NSInteger
{
  Enum3One = -1,
  Enum3Two
};
typedef NSInteger Enum3;

int main() {
  Enum0 e0 = Enum0One;
  // CHECK: call void @llvm.dbg.declare(metadata !{{.*}}, metadata ![[ENUM0:[0-9]+]])
  Enum1 e1 = Enum1One;
  // CHECK: call void @llvm.dbg.declare(metadata !{{.*}}, metadata ![[ENUM1:[0-9]+]])
  Enum2 e2 = Enum2One;
  // CHECK: call void @llvm.dbg.declare(metadata !{{.*}}, metadata ![[ENUM2:[0-9]+]])
  Enum3 e3 = Enum3One;
  // CHECK: call void @llvm.dbg.declare(metadata !{{.*}}, metadata ![[ENUM3:[0-9]+]])

  // -Werror and the following line ensures that these enums are not
  // -treated as C++11 strongly typed enums.
  return e0 != e1 && e1 == e2 && e2 == e3;
}
// CHECK: ![[ENUMERATOR0:[0-9]+]] = {{.*}}; [ DW_TAG_enumeration_type ] [line 10
// CHECK: ![[ENUMERATOR1:[0-9]+]] = {{.*}}; [ DW_TAG_enumeration_type ] [Enum1] [line 16{{.*}}] [from NSInteger]
// CHECK: ![[ENUMERATOR3:[0-9]+]] = {{.*}}; [ DW_TAG_typedef ] [NSInteger] [line 6{{.*}}] [from long int]
// CHECK: ![[ENUMERATOR2:[0-9]+]] = {{.*}}; [ DW_TAG_enumeration_type ] [line 22{{.*}}] [from NSInteger]

// CHECK: ![[ENUM0]] = metadata !{{{.*}}!"e0", metadata !{{[0-9]+}}, i32 {{[0-9]+}}, metadata ![[TYPE0:[0-9]+]]
// CHECK: ![[TYPE0]] = metadata !{{{.*}}!"Enum0", {{.*}} metadata ![[ENUMERATOR0]]} ; [ DW_TAG_typedef ] [Enum0]

// CHECK: ![[ENUM1]] = metadata !{{{.*}}!"e1", metadata !{{[0-9]+}}, i32 {{[0-9]+}}, metadata ![[TYPE1:[0-9]+]]
// CHECK: ![[TYPE1]] = metadata !{{{.*}}!"Enum1", {{.*}} metadata ![[ENUMERATOR1]]} ; [ DW_TAG_typedef ] [Enum1]

// CHECK: ![[ENUM2]] = metadata !{{{.*}}!"e2", metadata !{{[0-9]+}}, i32 {{[0-9]+}}, metadata ![[TYPE2:[0-9]+]]
// CHECK: ![[TYPE2]] = metadata !{{{.*}}!"Enum2", {{.*}} metadata ![[ENUMERATOR2]]} ; [ DW_TAG_typedef ] [Enum2]

// CHECK: ![[ENUM3]] = metadata !{{{.*}}!"e3", metadata !{{[0-9]+}}, i32 {{[0-9]+}}, metadata ![[TYPE3:[0-9]+]]
// CHECK: ![[TYPE3]] = metadata !{{{.*}}!"Enum3", {{.*}} metadata ![[ENUMERATOR3]]} ; [ DW_TAG_typedef ] [Enum3]
