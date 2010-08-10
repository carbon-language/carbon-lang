// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s
// PR3800
int *foo(void);

// CHECK: @test1
void test1() {
  // CHECK: [[REGCALLRESULT:%[a-zA-Z0-9\.]+]] = call i32* @foo()
  // CHECK: call void asm "foobar", "=*m,*m,~{dirflag},~{fpsr},~{flags}"(i32* [[REGCALLRESULT]], i32* [[REGCALLRESULT]])
  asm ("foobar" : "+m"(*foo()));
}

// CHECK: @test2
void test2() {
  // CHECK: [[REGCALLRESULT:%[a-zA-Z0-9\.]+]] = call i32* @foo()
  // CHECK: load i32* [[REGCALLRESULT]]
  // CHECK: call i32 asm
  // CHECK: store i32 {{%[a-zA-Z0-9\.]+}}, i32* [[REGCALLRESULT]]
  asm ("foobar" : "+r"(*foo()));
}

// PR7338
// CHECK: @test3
void test3(int *vout, int vin)
{
  // CHECK: entry:
  // CHECK: [[REGCALLRESULT1:%[a-zA-Z0-9\.]+]] = alloca i32*, align 4               ; <i32**> [#uses=2]
  // CHECK: [[REGCALLRESULT2:%[a-zA-Z0-9\.]+]] = alloca i32, align 4                 ; <i32*> [#uses=2]
  // CHECK: store i32* [[REGCALLRESULT5:%[a-zA-Z0-9\.]+]], i32** [[REGCALLRESULT1]]
  // CHECK: store i32 [[REGCALLRESULT6:%[a-zA-Z0-9\.]+]], i32* [[REGCALLRESULT2]]
  // CHECK: [[REGCALLRESULT3:%[a-zA-Z0-9\.]+]] = load i32** [[REGCALLRESULT1]]                    ; <i32*> [#uses=1]
  // CHECK: [[REGCALLRESULT4:%[a-zA-Z0-9\.]+]] = load i32* [[REGCALLRESULT2]]                     ; <i32> [#uses=1]
  //  The following is disabled until mult-alt constraint support is enabled.
  //  call void asm "opr $0,$1", "=*rm,rm,~{di},~{dirflag},~{fpsr},~{flags}"(i32* [[REGCALLRESULT3]], i32 [[REGCALLRESULT4]]) nounwind,
  //  Delete the following line when mult-alt constraint support is enabled.
  // CHECK: call void asm "opr $0,$1", "=*r,r,~{di},~{dirflag},~{fpsr},~{flags}"(i32* [[REGCALLRESULT3]], i32 [[REGCALLRESULT4]]) nounwind,
asm(
		"opr %[vout],%[vin]"
		: [vout] "=r,=m,=r" (*vout)
		: [vin] "r,m,r" (vin)
		: "edi"
		);
}
