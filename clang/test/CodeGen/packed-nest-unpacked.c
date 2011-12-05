// RUN: %clang_cc1 %s -triple x86_64-apple-macosx10.7.2 -emit-llvm -o - | FileCheck %s
// <rdar://problem/10463337>

struct X { int x[6]; };
struct Y { char x[13]; struct X y; } __attribute((packed));
struct Y g;
void f(struct X);

struct X test1() {
  // CHECK: @test1
  // CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* {{.*}}, i8* bitcast (%struct.X* getelementptr inbounds (%struct.Y* @g, i32 0, i32 1) to i8*), i64 24, i32 1, i1 false)
  return g.y;
}
struct X test2() {
  // CHECK: @test2
  // CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* {{.*}}, i8* bitcast (%struct.X* getelementptr inbounds (%struct.Y* @g, i32 0, i32 1) to i8*), i64 24, i32 1, i1 false)
  struct X a = g.y;
  return a;
}

void test3(struct X a) {
  // CHECK: @test3
  // CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* bitcast (%struct.X* getelementptr inbounds (%struct.Y* @g, i32 0, i32 1) to i8*), i8* {{.*}}, i64 24, i32 1, i1 false)
  g.y = a;
}

void test4() {
  // CHECK: @test4
  // FIXME: call void @llvm.memcpy.p0i8.p0i8.i64(i8* {{.*}}, i8* bitcast (%struct.X* getelementptr inbounds (%struct.Y* @g, i32 0, i32 1) to i8*), i64 24, i32 1, i1 false)
  f(g.y);
}
