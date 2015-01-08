// RUN: %clang_cc1 -emit-llvm -gdwarf-2 -x c++ -o - %s | FileCheck %s
//
// PR21941: crasher for self-referencing DW_TAG_structure_type node.  If we get
// rid of self-referenceing structure_types (PR21902), then it should be safe
// to just kill this test.
//
// CHECK: ![[SELF:[0-9]+]] = distinct !{!"0x13\00B\00{{[^"]*}}", {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, ![[SELF]], {{[^}]+}}} ; [ DW_TAG_structure_type ] [B]

void foo() {
  struct V {
    int vi;
  };
  struct B : virtual V {};
  B b;
}
