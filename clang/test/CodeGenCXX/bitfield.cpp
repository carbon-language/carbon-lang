// RUN: %clang_cc1 -triple x86_64-unknown-unknown -verify -emit-llvm -o - %s | FileCheck %s
//
// Tests for bitfield access patterns in C++ with special attention to
// conformance to C++11 memory model requirements.

namespace N1 {
  // Ensure that neither loads nor stores to bitfields are not widened into
  // other memory locations. (PR13691)
  //
  // NOTE: We could potentially widen loads based on their alignment if we are
  // comfortable requiring that subsequent memory locations within the
  // alignment-widened load are not volatile.
  struct S {
    char a;
    unsigned b : 1;
    char c;
  };
  unsigned read(S* s) {
    // CHECK: define i32 @_ZN2N14read
    // CHECK:   %[[ptr:.*]] = getelementptr inbounds %{{.*}}* %{{.*}}, i32 0, i32 1
    // CHECK:   %[[val:.*]] = load i8* %[[ptr]]
    // CHECK:   %[[and:.*]] = and i8 %[[val]], 1
    // CHECK:   %[[ext:.*]] = zext i8 %[[and]] to i32
    // CHECK:                 ret i32 %[[ext]]
    return s->b;
  }
  void write(S* s, unsigned x) {
    // CHECK: define void @_ZN2N15write
    // CHECK:   %[[ptr:.*]]     = getelementptr inbounds %{{.*}}* %{{.*}}, i32 0, i32 1
    // CHECK:   %[[x_trunc:.*]] = trunc i32 %{{.*}} to i8
    // CHECK:   %[[old:.*]]     = load i8* %[[ptr]]
    // CHECK:   %[[x_and:.*]]   = and i8 %[[x_trunc]], 1
    // CHECK:   %[[old_and:.*]] = and i8 %[[old]], -2
    // CHECK:   %[[new:.*]]     = or i8 %[[old_and]], %[[x_and]]
    // CHECK:                     store i8 %[[new]], i8* %[[ptr]]
    s->b = x;
  }
}

namespace N2 {
  // Do widen loads and stores to bitfields when those bitfields have padding
  // within the struct following them.
  struct S {
    unsigned b : 24;
    void *p;
  };
  unsigned read(S* s) {
    // CHECK: define i32 @_ZN2N24read
    // CHECK:   %[[ptr:.*]] = bitcast %{{.*}}* %{{.*}} to i32*
    // CHECK:   %[[val:.*]] = load i32* %[[ptr]]
    // CHECK:   %[[and:.*]] = and i32 %[[val]], 16777215
    // CHECK:                 ret i32 %[[and]]
    return s->b;
  }
  void write(S* s, unsigned x) {
    // CHECK: define void @_ZN2N25write
    // CHECK:   %[[ptr:.*]]     = bitcast %{{.*}}* %{{.*}} to i32*
    // CHECK:   %[[old:.*]]     = load i32* %[[ptr]]
    // CHECK:   %[[x_and:.*]]   = and i32 %{{.*}}, 16777215
    // CHECK:   %[[old_and:.*]] = and i32 %[[old]], -16777216
    // CHECK:   %[[new:.*]]     = or i32 %[[old_and]], %[[x_and]]
    // CHECK:                     store i32 %[[new]], i32* %[[ptr]]
    s->b = x;
  }
}

namespace N3 {
  // Do widen loads and stores to bitfields through the trailing padding at the
  // end of a struct.
  struct S {
    unsigned b : 24;
  };
  unsigned read(S* s) {
    // CHECK: define i32 @_ZN2N34read
    // CHECK:   %[[ptr:.*]] = bitcast %{{.*}}* %{{.*}} to i32*
    // CHECK:   %[[val:.*]] = load i32* %[[ptr]]
    // CHECK:   %[[and:.*]] = and i32 %[[val]], 16777215
    // CHECK:                 ret i32 %[[and]]
    return s->b;
  }
  void write(S* s, unsigned x) {
    // CHECK: define void @_ZN2N35write
    // CHECK:   %[[ptr:.*]]     = bitcast %{{.*}}* %{{.*}} to i32*
    // CHECK:   %[[old:.*]]     = load i32* %[[ptr]]
    // CHECK:   %[[x_and:.*]]   = and i32 %{{.*}}, 16777215
    // CHECK:   %[[old_and:.*]] = and i32 %[[old]], -16777216
    // CHECK:   %[[new:.*]]     = or i32 %[[old_and]], %[[x_and]]
    // CHECK:                     store i32 %[[new]], i32* %[[ptr]]
    s->b = x;
  }
}

namespace N4 {
  // Do NOT widen loads and stores to bitfields into padding at the end of
  // a class which might end up with members inside of it when inside a derived
  // class.
  struct Base {
    virtual ~Base() {}

    unsigned b : 24;
  };
  // Imagine some other translation unit introduces:
#if 0
  struct Derived : public Base {
    char c;
  };
#endif
  unsigned read(Base* s) {
    // FIXME: We should widen this load as long as the function isn't being
    // instrumented by thread-sanitizer.
    //
    // CHECK: define i32 @_ZN2N44read
    // CHECK:   %[[ptr:.*]] = bitcast {{.*}}* %{{.*}} to i24*
    // CHECK:   %[[val:.*]] = load i24* %[[ptr]]
    // CHECK:   %[[ext:.*]] = zext i24 %[[val]] to i32
    // CHECK:                 ret i32 %[[ext]]
    return s->b;
  }
  void write(Base* s, unsigned x) {
    // CHECK: define void @_ZN2N45write
    // CHECK:   %[[ptr:.*]] = bitcast {{.*}}* %{{.*}} to i24*
    // CHECK:   %[[new:.*]] = trunc i32 %{{.*}} to i24
    // CHECK:                 store i24 %[[new]], i24* %[[ptr]]
    s->b = x;
  }
}

namespace N5 {
  // Widen through padding at the end of a struct even if that struct
  // participates in a union with another struct which has a separate field in
  // that location. The reasoning is that if the operation is storing to that
  // member of the union, it must be the active member, and thus we can write
  // through the padding. If it is a load, it might be a load of a common
  // prefix through a non-active member, but in such a case the extra bits
  // loaded are masked off anyways.
  union U {
    struct X { unsigned b : 24; char c; } x;
    struct Y { unsigned b : 24; } y;
  };
  unsigned read(U* u) {
    // CHECK: define i32 @_ZN2N54read
    // CHECK:   %[[ptr:.*]] = bitcast %{{.*}}* %{{.*}} to i32*
    // CHECK:   %[[val:.*]] = load i32* %[[ptr]]
    // CHECK:   %[[and:.*]] = and i32 %[[val]], 16777215
    // CHECK:                 ret i32 %[[and]]
    return u->y.b;
  }
  void write(U* u, unsigned x) {
    // CHECK: define void @_ZN2N55write
    // CHECK:   %[[ptr:.*]]     = bitcast %{{.*}}* %{{.*}} to i32*
    // CHECK:   %[[old:.*]]     = load i32* %[[ptr]]
    // CHECK:   %[[x_and:.*]]   = and i32 %{{.*}}, 16777215
    // CHECK:   %[[old_and:.*]] = and i32 %[[old]], -16777216
    // CHECK:   %[[new:.*]]     = or i32 %[[old_and]], %[[x_and]]
    // CHECK:                     store i32 %[[new]], i32* %[[ptr]]
    u->y.b = x;
  }
}

namespace N6 {
  // Zero-length bitfields partition the memory locations of bitfields for the
  // purposes of the memory model. That means stores must not span zero-length
  // bitfields and loads may only span them when we are not instrumenting with
  // thread sanitizer.
  // FIXME: We currently don't widen loads even without thread sanitizer, even
  // though we could.
  struct S {
    unsigned b1 : 24;
    unsigned char : 0;
    unsigned char b2 : 8;
  };
  unsigned read(S* s) {
    // CHECK: define i32 @_ZN2N64read
    // CHECK:   %[[ptr1:.*]] = bitcast {{.*}}* %{{.*}} to i24*
    // CHECK:   %[[val1:.*]] = load i24* %[[ptr1]]
    // CHECK:   %[[ext1:.*]] = zext i24 %[[val1]] to i32
    // CHECK:   %[[ptr2:.*]] = getelementptr inbounds {{.*}}* %{{.*}}, i32 0, i32 1
    // CHECK:   %[[val2:.*]] = load i8* %[[ptr2]]
    // CHECK:   %[[ext2:.*]] = zext i8 %[[val2]] to i32
    // CHECK:   %[[add:.*]]  = add nsw i32 %[[ext1]], %[[ext2]]
    // CHECK:                  ret i32 %[[add]]
    return s->b1 + s->b2;
  }
  void write(S* s, unsigned x) {
    // CHECK: define void @_ZN2N65write
    // CHECK:   %[[ptr1:.*]] = bitcast {{.*}}* %{{.*}} to i24*
    // CHECK:   %[[new1:.*]] = trunc i32 %{{.*}} to i24
    // CHECK:                  store i24 %[[new1]], i24* %[[ptr1]]
    // CHECK:   %[[new2:.*]] = trunc i32 %{{.*}} to i8
    // CHECK:   %[[ptr2:.*]] = getelementptr inbounds {{.*}}* %{{.*}}, i32 0, i32 1
    // CHECK:                  store i8 %[[new2]], i8* %[[ptr2]]
    s->b1 = x;
    s->b2 = x;
  }
}
