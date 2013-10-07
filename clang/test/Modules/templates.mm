// RUN: rm -rf %t
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++11 -x objective-c++ -fmodules -fmodules-cache-path=%t -I %S/Inputs -verify %s -Wno-objc-root-class
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++11 -x objective-c++ -fmodules -fmodules-cache-path=%t -I %S/Inputs -emit-llvm %s -o - -Wno-objc-root-class | FileCheck %s
// expected-no-diagnostics

@import templates_left;
@import templates_right;

// CHECK: @list_left = global { %{{.*}}*, i32, [4 x i8] } { %{{.*}}* null, i32 8,
// CHECK: @list_right = global { %{{.*}}*, i32, [4 x i8] } { %{{.*}}* null, i32 12,
// CHECK: @_ZZ15testMixedStructvE1l = {{.*}} constant { %{{.*}}*, i32, [4 x i8] } { %{{.*}}* null, i32 1,
// CHECK: @_ZZ15testMixedStructvE1r = {{.*}} constant { %{{.*}}*, i32, [4 x i8] } { %{{.*}}* null, i32 2,

void testTemplateClasses() {
  Vector<int> vec_int;
  vec_int.push_back(0);

  List<bool> list_bool;
  list_bool.push_back(false);

  N::Set<char> set_char;
  set_char.insert('A');
}

void testPendingInstantiations() {
  // CHECK: call {{.*pendingInstantiationEmit}}
  // CHECK: call {{.*pendingInstantiationEmit}}
  // CHECK: define {{.*pendingInstantiationEmit.*[(]i}}
  // CHECK: define {{.*pendingInstantiationEmit.*[(]double}}
  triggerPendingInstantiation();
  triggerPendingInstantiationToo();
}

void testRedeclDefinition() {
  // CHECK: define {{.*redeclDefinitionEmit}}
  redeclDefinitionEmit();
}

// These three are all the same type.
typedef OuterIntInner_left OuterIntInner;
typedef OuterIntInner_right OuterIntInner;
typedef Outer<int>::Inner OuterIntInner;

// CHECK: call {{.*pendingInstantiation}}
// CHECK: call {{.*redeclDefinitionEmit}}

static_assert(size_left == size_right, "same field both ways");
void useListInt(List<int> &);

// CHECK-LABEL: define i32 @_Z15testMixedStructv(
unsigned testMixedStruct() {
  // CHECK: %[[l:.*]] = alloca %[[ListInt:[^ ]*]], align 8
  // CHECK: %[[r:.*]] = alloca %[[ListInt]], align 8

  // CHECK: call {{.*}}memcpy{{.*}}(i8* %{{.*}}, i8* bitcast ({{.*}}* @_ZZ15testMixedStructvE1l to i8*), i64 16,
  ListInt_left l{0, 1};

  // CHECK: call {{.*}}memcpy{{.*}}(i8* %{{.*}}, i8* bitcast ({{.*}}* @_ZZ15testMixedStructvE1r to i8*), i64 16,
  ListInt_right r{0, 2};

  // CHECK: call void @_Z10useListIntR4ListIiE(%[[ListInt]]* %[[l]])
  useListInt(l);
  // CHECK: call void @_Z10useListIntR4ListIiE(%[[ListInt]]* %[[r]])
  useListInt(r);

  // CHECK: load i32* bitcast (i8* getelementptr inbounds (i8* bitcast ({{.*}}* @list_left to i8*), i64 8) to i32*)
  // CHECK: load i32* bitcast (i8* getelementptr inbounds (i8* bitcast ({{.*}}* @list_right to i8*), i64 8) to i32*)
  return list_left.*size_right + list_right.*size_left;
}
