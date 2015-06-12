// RUN: %clang_cc1 -triple x86_64-linux-gnu -target-cpu x86-64 -emit-llvm %s -o - | FileCheck %s

int baz(int a) { return 4; }

int __attribute__((target("avx,sse4.2,arch=ivybridge"))) foo(int a) { return 4; }

int __attribute__((target("tune=sandybridge"))) walrus(int a) { return 4; }
int __attribute__((target("fpmath=387"))) koala(int a) { return 4; }

int __attribute__((target("mno-sse2"))) echidna(int a) { return 4; }

int bar(int a) { return baz(a) + foo(a); }

// Check that we emit the additional subtarget and cpu features for foo and not for baz or bar.
// CHECK: baz{{.*}} #0
// CHECK: foo{{.*}} #1
// We ignore the tune attribute so walrus should be identical to baz and bar.
// CHECK: walrus{{.*}} #0
// We're currently ignoring the fpmath attribute so koala should be identical to baz and bar.
// CHECK: koala{{.*}} #0
// CHECK: echidna{{.*}} #2
// CHECK: bar{{.*}} #0
// CHECK: #0 = {{.*}}"target-cpu"="x86-64" "target-features"="+sse,+sse2"
// CHECK: #1 = {{.*}}"target-cpu"="ivybridge" "target-features"="+sse,+sse2,+avx,+sse4.2"
// CHECK: #2 = {{.*}}"target-cpu"="x86-64" "target-features"="+sse,+sse2,-sse2"
