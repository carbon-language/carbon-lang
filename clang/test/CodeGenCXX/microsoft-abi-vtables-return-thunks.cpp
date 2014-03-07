// RUN: %clang_cc1 -fno-rtti %s -emit-llvm -o %t -triple=i386-pc-win32 -fdump-vtable-layouts 2>&1 | FileCheck --check-prefix=VFTABLES %s
// RUN: FileCheck --check-prefix=GLOBALS %s < %t
// RUN: FileCheck --check-prefix=CODEGEN %s < %t

namespace test1 {

// Some covariant types.
struct A { int a; };
struct B { int b; };
struct C : A, B { int c; };
struct D : C { int d; };
struct E : D { int e; };

// One base class and two overrides, all with covariant return types.
struct H     { virtual B *foo(); };
struct I : H { virtual C *foo(); };
struct J : I { virtual D *foo(); J(); };
struct K : J { virtual E *foo(); K(); };

J::J() {}

// VFTABLES-LABEL: VFTable for 'test1::H' in 'test1::I' in 'test1::J' (3 entries).
// VFTABLES-NEXT:   0 | test1::D *test1::J::foo()
// VFTABLES-NEXT:       [return adjustment (to type 'struct test1::B *'): 4 non-virtual]
// VFTABLES-NEXT:   1 | test1::D *test1::J::foo()
// VFTABLES-NEXT:       [return adjustment (to type 'struct test1::C *'): 0 non-virtual]
// VFTABLES-NEXT:   2 | test1::D *test1::J::foo()

// GLOBALS-LABEL: @"\01??_7J@test1@@6B@" = linkonce_odr unnamed_addr constant [3 x i8*]
// GLOBALS: @"\01?foo@J@test1@@QAEPAUB@2@XZ"
// GLOBALS: @"\01?foo@J@test1@@QAEPAUC@2@XZ"
// GLOBALS: @"\01?foo@J@test1@@UAEPAUD@2@XZ"

K::K() {}

// VFTABLES-LABEL: VFTable for 'test1::H' in 'test1::I' in 'test1::J' in 'test1::K' (4 entries).
// VFTABLES-NEXT:   0 | test1::E *test1::K::foo()
// VFTABLES-NEXT:       [return adjustment (to type 'struct test1::B *'): 4 non-virtual]
// VFTABLES-NEXT:   1 | test1::E *test1::K::foo()
// VFTABLES-NEXT:       [return adjustment (to type 'struct test1::C *'): 0 non-virtual]
// VFTABLES-NEXT:   2 | test1::E *test1::K::foo()
// VFTABLES-NEXT:       [return adjustment (to type 'struct test1::D *'): 0 non-virtual]
// VFTABLES-NEXT:   3 | test1::E *test1::K::foo()

// Only B to C requires adjustment, but we get 3 thunks in K's vftable, two of
// which are trivial.
// GLOBALS-LABEL: @"\01??_7K@test1@@6B@" = linkonce_odr unnamed_addr constant [4 x i8*]
// GLOBALS: @"\01?foo@K@test1@@QAEPAUB@2@XZ"
// GLOBALS: @"\01?foo@K@test1@@QAEPAUC@2@XZ"
// GLOBALS: @"\01?foo@K@test1@@QAEPAUD@2@XZ"
// GLOBALS: @"\01?foo@K@test1@@UAEPAUE@2@XZ"

//  This thunk has a return adjustment.
// CODEGEN-LABEL: define {{.*}} @"\01?foo@K@test1@@QAEPAUB@2@XZ"
// CODEGEN: call {{.*}} @"\01?foo@K@test1@@UAEPAUE@2@XZ"
// CODEGEN: icmp {{.*}}, null
// CODEGEN: getelementptr
// CODEGEN: ret

//  These two don't.
// CODEGEN-LABEL: define {{.*}} @"\01?foo@K@test1@@QAEPAUC@2@XZ"
// CODEGEN: call {{.*}} @"\01?foo@K@test1@@UAEPAUE@2@XZ"
// CODEGEN-NEXT: ret

// CODEGEN-LABEL: define {{.*}} @"\01?foo@K@test1@@QAEPAUD@2@XZ"
// CODEGEN: call {{.*}} @"\01?foo@K@test1@@UAEPAUE@2@XZ"
// CODEGEN-NEXT: ret

}

namespace test2 {

// Covariant types.  D* is not trivially convertible to C*.
struct A { int a; };
struct B { int b; };
struct C : B { int c; };
struct D : A, C { int d; };
struct E : D { int e; };

// J's foo will require an adjusting thunk, and K will require a trivial thunk.
struct H     { virtual B *foo(); };
struct I : H { virtual C *foo(); };
struct J : I { virtual D *foo(); J(); };
struct K : J { virtual E *foo(); K(); };

J::J() {}

// VFTABLES-LABEL: VFTable for 'test2::H' in 'test2::I' in 'test2::J' (2 entries).
// VFTABLES-NEXT:    0 | test2::D *test2::J::foo()
// VFTABLES-NEXT:         [return adjustment (to type 'struct test2::B *'): 4 non-virtual]
// VFTABLES-NEXT:    1 | test2::D *test2::J::foo()

// GLOBALS-LABEL: @"\01??_7J@test2@@6B@" = linkonce_odr unnamed_addr constant [2 x i8*]

K::K() {}

// VFTABLES-LABEL: VFTable for 'test2::H' in 'test2::I' in 'test2::J' in 'test2::K' (3 entries).
// VFTABLES-NEXT:    0 | test2::E *test2::K::foo()
// VFTABLES-NEXT:         [return adjustment (to type 'struct test2::B *'): 4 non-virtual]
// VFTABLES-NEXT:    1 | test2::E *test2::K::foo()
// VFTABLES-NEXT:         [return adjustment (to type 'struct test2::D *'): 0 non-virtual]
// VFTABLES-NEXT:    2 | test2::E *test2::K::foo()

// GLOBALS-LABEL: @"\01??_7K@test2@@6B@" = linkonce_odr unnamed_addr constant [3 x i8*]

}
