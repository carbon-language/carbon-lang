// Test -fsanitize-address-field-padding
// RUN: echo 'type:SomeNamespace::BlacklistedByName=field-padding' > %t.type.blacklist
// RUN: echo 'src:*sanitize-address-field-padding.cpp=field-padding' > %t.file.blacklist
// RUN: %clang_cc1 -fsanitize=address -fsanitize-address-field-padding=1 -fsanitize-blacklist=%t.type.blacklist -Rsanitize-address -emit-llvm -o - %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fsanitize=address -fsanitize-address-field-padding=1 -fsanitize-blacklist=%t.file.blacklist -Rsanitize-address -emit-llvm -o - %s 2>&1 | FileCheck %s --check-prefix=FILE_BLACKLIST
// RUN: %clang_cc1 -fsanitize=address -emit-llvm -o - %s 2>&1 | FileCheck %s --check-prefix=NO_PADDING
// REQUIRES: shell
//

// The reasons to ignore a particular class are not set in stone and will change.
//
// CHECK: -fsanitize-address-field-padding applied to Positive1
// CHECK: -fsanitize-address-field-padding ignored for Negative1 because it is trivially copyable
// CHECK: -fsanitize-address-field-padding ignored for Negative2 because it is trivially copyable
// CHECK: -fsanitize-address-field-padding ignored for Negative3 because it is a union
// CHECK: -fsanitize-address-field-padding ignored for Negative4 because it is trivially copyable
// CHECK: -fsanitize-address-field-padding ignored for Negative5 because it is packed
// CHECK: -fsanitize-address-field-padding ignored for SomeNamespace::BlacklistedByName because it is blacklisted
// CHECK: -fsanitize-address-field-padding ignored for ExternCStruct because it is not C++
//
// FILE_BLACKLIST: -fsanitize-address-field-padding ignored for Positive1 because it is in a blacklisted file
// FILE_BLACKLIST-NOT: __asan_poison_intra_object_redzone
// NO_PADDING-NOT: __asan_poison_intra_object_redzone


class Positive1 {
 public:
  Positive1() {}
  ~Positive1() {}
  int make_it_non_standard_layout;
 private:
  char private1;
  int private2;
  short private_array[6];
  long long private3;
};

Positive1 positive1;
// Positive1 with extra paddings
// CHECK: type { i32, [12 x i8], i8, [15 x i8], i32, [12 x i8], [6 x i16], [12 x i8], i64, [8 x i8] }

class Negative1 {
 public:
  Negative1() {}
  int public1, public2;
};
Negative1 negative1;
// CHECK: type { i32, i32 }

class Negative2 {
 public:
  Negative2() {}
 private:
  int private1, private2;
};
Negative2 negative2;
// CHECK: type { i32, i32 }

union Negative3 {
  char m1[8];
  long long m2;
};

Negative3 negative3;
// CHECK: type { i64 }

class Negative4 {
 public:
  Negative4() {}
  // No DTOR
  int make_it_non_standard_layout;
 private:
  char private1;
  int private2;
};

Negative4 negative4;
// CHECK: type { i32, i8, i32 }

class __attribute__((packed)) Negative5 {
 public:
  Negative5() {}
  ~Negative5() {}
  int make_it_non_standard_layout;
 private:
  char private1;
  int private2;
};

Negative5 negative5;
// CHECK: type <{ i32, i8, i32 }>


namespace SomeNamespace {
class BlacklistedByName {
 public:
  BlacklistedByName() {}
  ~BlacklistedByName() {}
  int make_it_non_standard_layout;
 private:
  char private1;
  int private2;
};
}  // SomeNamespace

SomeNamespace::BlacklistedByName blacklisted_by_name;

extern "C" {
class ExternCStruct {
 public:
  ExternCStruct() {}
  ~ExternCStruct() {}
  int make_it_non_standard_layout;
 private:
  char private1;
  int private2;
};
}  // extern "C"

ExternCStruct extern_C_struct;

// CTOR
// CHECK-LABEL: define linkonce_odr void {{.*}}Positive1
// CHECK: call void @__asan_poison_intra_object_redzone({{.*}}12)
// CHECK: call void @__asan_poison_intra_object_redzone({{.*}}15)
// CHECK: call void @__asan_poison_intra_object_redzone({{.*}}12)
// CHECK: call void @__asan_poison_intra_object_redzone({{.*}}12)
// CHECK: call void @__asan_poison_intra_object_redzone({{.*}}8)
// CHECK-NOT: __asan_poison_intra_object_redzone
// CHECK: ret void
//
// DTOR
// CHECK: call void @__asan_unpoison_intra_object_redzone({{.*}}12)
// CHECK: call void @__asan_unpoison_intra_object_redzone({{.*}}15)
// CHECK: call void @__asan_unpoison_intra_object_redzone({{.*}}12)
// CHECK: call void @__asan_unpoison_intra_object_redzone({{.*}}12)
// CHECK: call void @__asan_unpoison_intra_object_redzone({{.*}}8)
// CHECK-NOT: __asan_unpoison_intra_object_redzone
// CHECK: ret void
