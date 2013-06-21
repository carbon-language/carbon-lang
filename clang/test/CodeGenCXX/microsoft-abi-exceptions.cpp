// RUN: %clang_cc1 -emit-llvm %s -o - -triple=i386-pc-win32 -cxx-abi microsoft -fexceptions | FileCheck -check-prefix WIN32 %s

struct A {
  A();
  ~A();
  int a;
};

A getA();

int TakesTwo(A a, A b);
void HasEHCleanup() {
  TakesTwo(getA(), getA());
}

// With exceptions, we need to clean up at least one of these temporaries.
// WIN32: define void @"\01?HasEHCleanup@@YAXXZ"() {{.*}} {
//    First one doesn't have any cleanups, no need for invoke.
// WIN32:   call void @"\01?getA@@YA?AUA@@XZ"(%struct.A* sret %{{.*}})
//    If this call throws, we have to cleanup the first temporary.
// WIN32:   invoke void @"\01?getA@@YA?AUA@@XZ"(%struct.A* sret %{{.*}})
//    If this call throws, we already popped our cleanups
// WIN32:   call i32 @"\01?TakesTwo@@YAHUA@@0@Z"
// WIN32:   ret void
//
//    There should be one dtor call for unwinding from the second getA.
// WIN32:   invoke x86_thiscallcc void @"\01??1A@@QAE@XZ"
// WIN32: }

void TakeRef(const A &a);
int HasDeactivatedCleanups() {
  return TakesTwo((TakeRef(A()), A()), (TakeRef(A()), A()));
}

// WIN32: define i32 @"\01?HasDeactivatedCleanups@@YAHXZ"() {{.*}} {
// WIN32:   %[[isactive:.*]] = alloca i1
// WIN32:   call x86_thiscallcc %struct.A* @"\01??0A@@QAE@XZ"
// WIN32:   invoke void @"\01?TakeRef@@YAXABUA@@@Z"
// WIN32:   invoke x86_thiscallcc %struct.A* @"\01??0A@@QAE@XZ"(%struct.A* %[[arg1:.*]])
// WIN32:   store i1 true, i1* %[[isactive]]
// WIN32:   invoke x86_thiscallcc %struct.A* @"\01??0A@@QAE@XZ"
// WIN32:   invoke void @"\01?TakeRef@@YAXABUA@@@Z"
// WIN32:   invoke x86_thiscallcc %struct.A* @"\01??0A@@QAE@XZ"
// WIN32:   store i1 false, i1* %[[isactive]]
// WIN32:   invoke i32 @"\01?TakesTwo@@YAHUA@@0@Z"
//        Destroy the two const ref temporaries.
// WIN32:   invoke x86_thiscallcc void @"\01??1A@@QAE@XZ"
// WIN32:   call x86_thiscallcc void @"\01??1A@@QAE@XZ"
// WIN32:   ret i32
//
//        Conditionally destroy arg1.
// WIN32:   %[[cond:.*]] = load i1* %[[isactive]]
// WIN32:   br i1 %[[cond]]
// WIN32:   invoke x86_thiscallcc void @"\01??1A@@QAE@XZ"(%struct.A* %[[arg1]])
// WIN32: }

// Test putting the cleanups inside a conditional.
int CouldThrow();
int HasConditionalCleanup(bool cond) {
  return (cond ? TakesTwo(A(), A()) : CouldThrow());
}

// WIN32: define i32 @"\01?HasConditionalCleanup@@YAH_N@Z"(i1 zeroext %{{.*}}) {{.*}} {
// WIN32:   store i1 false
// WIN32:   br i1
//        No cleanups, so we call and then activate a cleanup if it succeeds.
// WIN32:   call x86_thiscallcc %struct.A* @"\01??0A@@QAE@XZ"(%struct.A* %[[arg1:.*]])
// WIN32:   store i1 true
//        Now we have a cleanup for the first aggregate, so we invoke.
// WIN32:   invoke x86_thiscallcc %struct.A* @"\01??0A@@QAE@XZ"(%struct.A* %{{.*}})
//        Now we have no cleanups because TakeTwo will destruct both args.
// WIN32:   call i32 @"\01?TakesTwo@@YAHUA@@0@Z"
//        Still no cleanups, so call.
// WIN32:   call i32 @"\01?CouldThrow@@YAHXZ"()
//        Somewhere in the landing pad for our single invoke, call the dtor.
// WIN32:   invoke x86_thiscallcc void @"\01??1A@@QAE@XZ"(%struct.A* %[[arg1]])
// WIN32: }

// Now test both.
int HasConditionalDeactivatedCleanups(bool cond) {
  return (cond ? TakesTwo((TakeRef(A()), A()), (TakeRef(A()), A())) : CouldThrow());
}

// WIN32: define i32 @"\01?HasConditionalDeactivatedCleanups@@YAH_N@Z"{{.*}} {
// WIN32:   %[[arg1:.*]] = alloca %struct.A, align 4
// WIN32:   alloca i1
// WIN32:   %[[arg1_cond:.*]] = alloca i1
//        Start all four cleanups as deactivated.
// WIN32:   store i1 false
// WIN32:   store i1 false
// WIN32:   store i1 false
// WIN32:   store i1 false
// WIN32:   br i1
//        True condition.
// WIN32:   call x86_thiscallcc %struct.A* @"\01??0A@@QAE@XZ"
// WIN32:   store i1 true
// WIN32:   invoke void @"\01?TakeRef@@YAXABUA@@@Z"
// WIN32:   invoke x86_thiscallcc %struct.A* @"\01??0A@@QAE@XZ"(%struct.A* %[[arg1]])
// WIN32:   store i1 true, i1* %[[arg1_cond]]
// WIN32:   invoke x86_thiscallcc %struct.A* @"\01??0A@@QAE@XZ"
// WIN32:   store i1 true
// WIN32:   invoke void @"\01?TakeRef@@YAXABUA@@@Z"
// WIN32:   invoke x86_thiscallcc %struct.A* @"\01??0A@@QAE@XZ"
// WIN32:   store i1 true
// WIN32:   store i1 false, i1* %[[arg1_cond]]
// WIN32:   invoke i32 @"\01?TakesTwo@@YAHUA@@0@Z"
//        False condition.
// WIN32:   invoke i32 @"\01?CouldThrow@@YAHXZ"()
//        Two normal cleanups for TakeRef args.
// WIN32:   invoke x86_thiscallcc void @"\01??1A@@QAE@XZ"
// WIN32:   call x86_thiscallcc void @"\01??1A@@QAE@XZ"
// WIN32:   ret i32
//
//        Somewhere in the landing pad soup, we conditionally destroy arg1.
// WIN32:   %[[isactive:.*]] = load i1* %[[arg1_cond]]
// WIN32:   br i1 %[[isactive]]
// WIN32:   invoke x86_thiscallcc void @"\01??1A@@QAE@XZ"(%struct.A* %[[arg1]])
// WIN32: }
