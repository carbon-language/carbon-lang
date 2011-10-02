// RUN: %clang_cc1 -fsyntax-only -fobjc-arc -Warc-abi -verify -fblocks -triple x86_64-apple-darwin10.0.0 %s

// Classes that have an Objective-C object pointer.
struct HasObjectMember0 { // expected-warning{{'HasObjectMember0' cannot be shared between ARC and non-ARC code; add a copy constructor, a copy assignment operator, and a destructor to make it ABI-compatible}}
  id x;
};

struct HasObjectMember1 { // expected-warning{{'HasObjectMember1' cannot be shared between ARC and non-ARC code; add a copy constructor, a copy assignment operator, and a destructor to make it ABI-compatible}}
  id x[3];
};

struct HasObjectMember2 { // expected-warning{{'HasObjectMember2' cannot be shared between ARC and non-ARC code; add a copy constructor, a copy assignment operator, and a destructor to make it ABI-compatible}}
  id x[3][2];
};

// Don't complain if the type has non-external linkage
namespace {
  struct HasObjectMember3 {
    id x[3][2];
  };
}

// Don't complain if the Objective-C pointer type was explicitly given
// no ownership.
struct HasObjectMember3 { 
  __unsafe_unretained id x[3][2];
};

struct HasBlockPointerMember0 { // expected-warning{{'HasBlockPointerMember0' cannot be shared between ARC and non-ARC code; add a copy constructor, a copy assignment operator, and a destructor to make it ABI-compatible}}
  int (^bp)(int);
};

struct HasBlockPointerMember1 { // expected-warning{{'HasBlockPointerMember1' cannot be shared between ARC and non-ARC code; add a copy constructor, a copy assignment operator, and a destructor to make it ABI-compatible}}
  int (^bp[2][3])(int);
};

struct NonPOD {
  NonPOD(const NonPOD&);
};

struct HasObjectMemberAndNonPOD0 { // expected-warning{{'HasObjectMemberAndNonPOD0' cannot be shared between ARC and non-ARC code; add a non-trivial copy assignment operator to make it ABI-compatible}} \
  // expected-warning{{'HasObjectMemberAndNonPOD0' cannot be shared between ARC and non-ARC code; add a non-trivial destructor to make it ABI-compatible}}
  id x;
  NonPOD np;
};

struct HasObjectMemberAndNonPOD1 { // expected-warning{{'HasObjectMemberAndNonPOD1' cannot be shared between ARC and non-ARC code; add a non-trivial copy assignment operator to make it ABI-compatible}} \
  // expected-warning{{'HasObjectMemberAndNonPOD1' cannot be shared between ARC and non-ARC code; add a non-trivial destructor to make it ABI-compatible}}
  NonPOD np;
  id x[3];
};

struct HasObjectMemberAndNonPOD2 { // expected-warning{{'HasObjectMemberAndNonPOD2' cannot be shared between ARC and non-ARC code; add a non-trivial copy assignment operator to make it ABI-compatible}} \
  // expected-warning{{'HasObjectMemberAndNonPOD2' cannot be shared between ARC and non-ARC code; add a non-trivial destructor to make it ABI-compatible}}
  NonPOD np;
  id x[3][2];
};

struct HasObjectMemberAndNonPOD3 {
  HasObjectMemberAndNonPOD3 &operator=(const HasObjectMemberAndNonPOD3&);
  ~HasObjectMemberAndNonPOD3();
  NonPOD np;
  id x[3][2];
};

struct HasBlockPointerMemberAndNonPOD0 { // expected-warning{{'HasBlockPointerMemberAndNonPOD0' cannot be shared between ARC and non-ARC code; add a non-trivial copy assignment operator to make it ABI-compatible}} \
// expected-warning{{'HasBlockPointerMemberAndNonPOD0' cannot be shared between ARC and non-ARC code; add a non-trivial destructor to make it ABI-compatible}}
  NonPOD np;
  int (^bp)(int);
};

struct HasBlockPointerMemberAndNonPOD1 { // expected-warning{{'HasBlockPointerMemberAndNonPOD1' cannot be shared between ARC and non-ARC code; add a non-trivial copy assignment operator to make it ABI-compatible}} \
// expected-warning{{'HasBlockPointerMemberAndNonPOD1' cannot be shared between ARC and non-ARC code; add a non-trivial destructor to make it ABI-compatible}}
  NonPOD np;
  int (^bp[2][3])(int);
};

int check_non_pod_objc_pointer0[__is_pod(id)? 1 : -1];
int check_non_pod_objc_pointer1[__is_pod(__strong id)? -1 : 1];
int check_non_pod_objc_pointer2[__is_pod(__unsafe_unretained id)? 1 : -1];
int check_non_pod_objc_pointer3[__is_pod(id[2][3])? 1 : -1];
int check_non_pod_objc_pointer4[__is_pod(__unsafe_unretained id[2][3])? 1 : -1];
int check_non_pod_block0[__is_pod(int (^)(int))? 1 : -1];
int check_non_pod_block1[__is_pod(int (^ __unsafe_unretained)(int))? 1 : -1];
int check_non_pod_block2[__is_pod(int (^ __strong)(int))? -1 : 1];

struct FlexibleArrayMember0 {
  int length;
  id array[]; // expected-error{{flexible array member 'array' of non-POD element type 'id __strong[]'}}
};

struct FlexibleArrayMember1 {
  int length;
  __unsafe_unretained id array[];
};

// It's okay to pass a retainable type through an ellipsis.
void variadic(...);
void test_variadic() {
  variadic(1, 17, @"Foo");
}

// It's okay to create a VLA of retainable types.
void vla(int n) {
  id vla[n];
}

@interface Crufty {
  union {
    struct {
      id object; // expected-note{{has __strong ownership}}
    } an_object; // expected-error{{union member 'an_object' has a non-trivial copy constructor}}
    void *ptr;
  } storage;
}
@end
