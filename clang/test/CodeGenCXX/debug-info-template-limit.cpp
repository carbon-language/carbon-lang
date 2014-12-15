// RUN: %clang_cc1 -emit-llvm -fno-standalone-debug -triple %itanium_abi_triple -g %s -o - | FileCheck %s

// Check that this pointer type is TC<int>
// CHECK: ![[LINE:[0-9]+]] = !{!"0x2\00TC<int>\00{{.*}}", {{.*}} !"_ZTS2TCIiE"} ; [ DW_TAG_class_type ]
// CHECK: !"_ZTS2TCIiE"} ; [ DW_TAG_pointer_type ]{{.*}}[from _ZTS2TCIiE]

template<typename T>
class TC {
public:
  TC(const TC &) {}
  TC() {}
};

TC<int> tci;
