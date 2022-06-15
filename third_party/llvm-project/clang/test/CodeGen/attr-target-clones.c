// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefixes=LINUX,CHECK
// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-windows-pc -emit-llvm %s -o - | FileCheck %s --check-prefixes=WINDOWS,CHECK

// LINUX: $foo.resolver = comdat any
// LINUX: $foo_dupes.resolver = comdat any
// LINUX: $unused.resolver = comdat any
// LINUX: $foo_inline.resolver = comdat any
// LINUX: $foo_inline2.resolver = comdat any

// WINDOWS: $foo = comdat any
// WINDOWS: $foo_dupes = comdat any
// WINDOWS: $unused = comdat any
// WINDOWS: $foo_inline = comdat any
// WINDOWS: $foo_inline2 = comdat any

// LINUX: @foo.ifunc = weak_odr ifunc i32 (), i32 ()* ()* @foo.resolver
// LINUX: @foo_dupes.ifunc = weak_odr ifunc void (), void ()* ()* @foo_dupes.resolver
// LINUX: @unused.ifunc = weak_odr ifunc void (), void ()* ()* @unused.resolver
// LINUX: @foo_inline.ifunc = weak_odr ifunc i32 (), i32 ()* ()* @foo_inline.resolver
// LINUX: @foo_inline2.ifunc = weak_odr ifunc i32 (), i32 ()* ()* @foo_inline2.resolver
// LINUX: @foo_used_no_defn.ifunc = weak_odr ifunc i32 (), i32 ()* ()* @foo_used_no_defn.resolver

int __attribute__((target_clones("sse4.2, default"))) foo(void) { return 0; }
// LINUX: define {{.*}}i32 @foo.sse4.2.0()
// LINUX: define {{.*}}i32 @foo.default.1()
// LINUX: define weak_odr i32 ()* @foo.resolver() comdat
// LINUX: ret i32 ()* @foo.sse4.2.0
// LINUX: ret i32 ()* @foo.default.1

// WINDOWS: define dso_local i32 @foo.sse4.2.0()
// WINDOWS: define dso_local i32 @foo.default.1()
// WINDOWS: define weak_odr dso_local i32 @foo() comdat
// WINDOWS: musttail call i32 @foo.sse4.2.0
// WINDOWS: musttail call i32 @foo.default.1

__attribute__((target_clones("default,default ,sse4.2"))) void foo_dupes(void) {}
// LINUX: define {{.*}}void @foo_dupes.default.1()
// LINUX: define {{.*}}void @foo_dupes.sse4.2.0()
// LINUX: define weak_odr void ()* @foo_dupes.resolver() comdat
// LINUX: ret void ()* @foo_dupes.sse4.2.0
// LINUX: ret void ()* @foo_dupes.default.1

// WINDOWS: define dso_local void @foo_dupes.default.1()
// WINDOWS: define dso_local void @foo_dupes.sse4.2.0()
// WINDOWS: define weak_odr dso_local void @foo_dupes() comdat
// WINDOWS: musttail call void @foo_dupes.sse4.2.0
// WINDOWS: musttail call void @foo_dupes.default.1

void bar2(void) {
  // LINUX: define {{.*}}void @bar2()
  // WINDOWS: define dso_local void @bar2()
  foo_dupes();
  // LINUX: call void @foo_dupes.ifunc()
  // WINDOWS: call void @foo_dupes()
}

int bar(void) {
  // LINUX: define {{.*}}i32 @bar() #[[DEF:[0-9]+]]
  // WINDOWS: define dso_local i32 @bar() #[[DEF:[0-9]+]]
  return foo();
  // LINUX: call i32 @foo.ifunc()
  // WINDOWS: call i32 @foo()
}

void __attribute__((target_clones("default, arch=ivybridge"))) unused(void) {}
// LINUX: define {{.*}}void @unused.default.1()
// LINUX: define {{.*}}void @unused.arch_ivybridge.0()
// LINUX: define weak_odr void ()* @unused.resolver() comdat
// LINUX: ret void ()* @unused.arch_ivybridge.0
// LINUX: ret void ()* @unused.default.1

// WINDOWS: define dso_local void @unused.default.1()
// WINDOWS: define dso_local void @unused.arch_ivybridge.0()
// WINDOWS: define weak_odr dso_local void @unused() comdat
// WINDOWS: musttail call void @unused.arch_ivybridge.0
// WINDOWS: musttail call void @unused.default.1


inline int __attribute__((target_clones("arch=sandybridge,default,sse4.2")))
foo_inline(void) { return 0; }
inline int __attribute__((target_clones("arch=sandybridge,default,sse4.2")))
foo_inline2(void);

int bar3(void) {
  // LINUX: define {{.*}}i32 @bar3()
  // WINDOWS: define dso_local i32 @bar3()
  return foo_inline() + foo_inline2();
  // LINUX: call i32 @foo_inline.ifunc()
  // LINUX: call i32 @foo_inline2.ifunc()
  // WINDOWS: call i32 @foo_inline()
  // WINDOWS: call i32 @foo_inline2()
}

// LINUX: define weak_odr i32 ()* @foo_inline.resolver() comdat
// LINUX: ret i32 ()* @foo_inline.arch_sandybridge.0
// LINUX: ret i32 ()* @foo_inline.sse4.2.1
// LINUX: ret i32 ()* @foo_inline.default.2

// WINDOWS: define weak_odr dso_local i32 @foo_inline() comdat
// WINDOWS: musttail call i32 @foo_inline.arch_sandybridge.0
// WINDOWS: musttail call i32 @foo_inline.sse4.2.1
// WINDOWS: musttail call i32 @foo_inline.default.2

inline int __attribute__((target_clones("arch=sandybridge,default,sse4.2")))
foo_inline2(void){ return 0; }
// LINUX: define weak_odr i32 ()* @foo_inline2.resolver() comdat
// LINUX: ret i32 ()* @foo_inline2.arch_sandybridge.0
// LINUX: ret i32 ()* @foo_inline2.sse4.2.1
// LINUX: ret i32 ()* @foo_inline2.default.2

// WINDOWS: define weak_odr dso_local i32 @foo_inline2() comdat
// WINDOWS: musttail call i32 @foo_inline2.arch_sandybridge.0
// WINDOWS: musttail call i32 @foo_inline2.sse4.2.1
// WINDOWS: musttail call i32 @foo_inline2.default.2


int __attribute__((target_clones("default", "sse4.2")))
foo_unused_no_defn(void);

int __attribute__((target_clones("default", "sse4.2")))
foo_used_no_defn(void);

int test_foo_used_no_defn(void) {
  // LINUX: define {{.*}}i32 @test_foo_used_no_defn()
  // WINDOWS: define dso_local i32 @test_foo_used_no_defn()
  return foo_used_no_defn();
  // LINUX: call i32 @foo_used_no_defn.ifunc()
  // WINDOWS: call i32 @foo_used_no_defn()
}


// LINUX: define weak_odr i32 ()* @foo_used_no_defn.resolver() comdat
// LINUX: ret i32 ()* @foo_used_no_defn.sse4.2.0
// LINUX: ret i32 ()* @foo_used_no_defn.default.1

// WINDOWS: define weak_odr dso_local i32 @foo_used_no_defn() comdat
// WINDOWS: musttail call i32 @foo_used_no_defn.sse4.2.0
// WINDOWS: musttail call i32 @foo_used_no_defn.default.1


// Deferred emission of inline definitions.

// LINUX: define linkonce i32 @foo_inline.arch_sandybridge.0() #[[SB:[0-9]+]]
// LINUX: define linkonce i32 @foo_inline.default.2() #[[DEF:[0-9]+]]
// LINUX: define linkonce i32 @foo_inline.sse4.2.1() #[[SSE42:[0-9]+]]

// WINDOWS: define linkonce_odr dso_local i32 @foo_inline.arch_sandybridge.0() #[[SB:[0-9]+]]
// WINDOWS: define linkonce_odr dso_local i32 @foo_inline.default.2() #[[DEF]]
// WINDOWS: define linkonce_odr dso_local i32 @foo_inline.sse4.2.1() #[[SSE42:[0-9]+]]


// LINUX: define linkonce i32 @foo_inline2.arch_sandybridge.0() #[[SB]]
// LINUX: define linkonce i32 @foo_inline2.default.2() #[[DEF]]
// LINUX: define linkonce i32 @foo_inline2.sse4.2.1() #[[SSE42]]

// WINDOWS: define linkonce_odr dso_local i32 @foo_inline2.arch_sandybridge.0() #[[SB]]
// WINDOWS: define linkonce_odr dso_local i32 @foo_inline2.default.2() #[[DEF]]
// WINDOWS: define linkonce_odr dso_local i32 @foo_inline2.sse4.2.1() #[[SSE42]]


// LINUX: declare i32 @foo_used_no_defn.default.1()
// LINUX: declare i32 @foo_used_no_defn.sse4.2.0()

// WINDOWS: declare dso_local i32 @foo_used_no_defn.default.1()
// WINDOWS: declare dso_local i32 @foo_used_no_defn.sse4.2.0()


// CHECK: attributes #[[SSE42]] =
// CHECK-SAME: "target-features"="+crc32,+cx8,+mmx,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87"
// CHECK: attributes #[[DEF]] =
// Don't bother checking features, we verified it is the same as a normal function.
// CHECK: attributes #[[SB]] =
// CHECK-SAME: "target-features"="+avx,+crc32,+cx16,+cx8,+fxsr,+mmx,+pclmul,+popcnt,+sahf,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt"
