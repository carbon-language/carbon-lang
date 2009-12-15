// RUN: %clang_cc1 -I%S -triple x86_64-apple-darwin -std=c++0x -emit-llvm %s -o %t.ll
// RUN: FileCheck -check-prefix LL --input-file=%t.ll %s

#include <typeinfo>

class test1_A { virtual void f() { } };
class test1_B { virtual void g() { } };
class test1_D : public virtual test1_A, private test1_B {};
class test1_E : public test1_D, public test1_B {};
class test1_F : public test1_E, public test1_D {};

extern test1_D test1_d;
extern test1_F test1_f;

extern "C" int printf(const char *str...);

#define S(V, N) if (V) printf("PASS: %d\n", N); else printf("FAIL: %d\n", N)

void test1() {
  test1_B* bp = (test1_B*)&test1_d;
  test1_A* ap = &test1_d;
  // This throws
  //  test1_D&  dr = dynamic_cast<D&>(*bp);
  test1_D* dp = dynamic_cast<test1_D*>(bp);
  S(dp == 0, 1);
  ap = dynamic_cast<test1_A*>(bp);
  S(ap == 0, 2);
  bp = dynamic_cast<test1_B*>(ap);
  S(bp == 0, 3);
  ap = dynamic_cast<test1_A*>(&test1_d);
  S(ap != 0, 4);
  // FIXME: Doesn't work yet, gcc fails this at compile time.  We'd need access
  // control for this to work.
  // bp = dynamic_cast<test1_B*>(&test1_d);
  // S(bp == 0, 5);
  {
    test1_A*  ap  = &test1_f;
    S(ap != 0, 6);
    test1_D*  dp  = dynamic_cast<test1_D*>(ap);
    S(dp == 0, 7);
    // cast from virtual base
    test1_E*  ep1 = dynamic_cast<test1_E*>(ap);
    S(ep1 != 0, 8);
  }
  dp = dynamic_cast<test1_D*>(&test1_d);
  S(dp == &test1_d, 9);
  const test1_D *cdp = dynamic_cast<const test1_D*>(&test1_d);
  S(cdp == &test1_d, 10);
  dp = dynamic_cast<test1_D*>((test1_A*)0);
  S(dp == 0, 11);
  ap = dynamic_cast<test1_A*>(&test1_d);
  S(ap == (test1_A*)&test1_d, 12);
  test1_E* ep = dynamic_cast<test1_E*>(&test1_f);
  S(ep == (test1_E*)&test1_f, 13);
  void *vp = dynamic_cast<void*>(ap);
  S(vp == &test1_d, 14);
  const void *cvp = dynamic_cast<const void*>(ap);
  S(cvp == &test1_d, 15);
}

// CHECK-LL:     define void @_Z5test1v() nounwind {
// CHECK-LL-NEXT:entry:
// CHECK-LL-NEXT:  %bp = alloca %class.test1_A*, align 8
// CHECK-LL-NEXT:  %ap = alloca %class.test1_A*, align 8
// CHECK-LL-NEXT:  %dp = alloca %class.test1_D*, align 8
// CHECK-LL-NEXT:  %ap37 = alloca %class.test1_A*, align 8
// CHECK-LL-NEXT:  %dp53 = alloca %class.test1_D*, align 8
// CHECK-LL-NEXT:  %ep1 = alloca %class.test1_E*, align 8
// CHECK-LL-NEXT:  %cdp = alloca %class.test1_D*, align 8
// CHECK-LL-NEXT:  %ep = alloca %class.test1_E*, align 8
// CHECK-LL-NEXT:  %vp = alloca i8*, align 8
// CHECK-LL-NEXT:  %cvp = alloca i8*, align 8
// CHECK-LL-NEXT:  br i1 false, label %cast.null, label %cast.notnull
// CHECK-LL:       cast.notnull:
// CHECK-LL-NEXT:  br label %cast.end
// CHECK-LL:       cast.null:
// CHECK-LL-NEXT:  br label %cast.end
// CHECK-LL:       cast.end:
// CHECK-LL-NEXT:  %0 = phi %class.test1_A* [ bitcast (%class.test1_D* @test1_d to %class.test1_A*), %cast.notnull ], [ null, %cast.null ]
// CHECK-LL-NEXT:  store %class.test1_A* %0, %class.test1_A** %bp
// CHECK-LL-NEXT:  br i1 false, label %cast.null2, label %cast.notnull1
// CHECK-LL:       cast.notnull1:
// CHECK-LL-NEXT:  %vtable = load i8** bitcast (%class.test1_D* @test1_d to i8**)
// CHECK-LL-NEXT:  %vbase.offset.ptr = getelementptr i8* %vtable, i64 -24
// CHECK-LL-NEXT:  %1 = bitcast i8* %vbase.offset.ptr to i64*
// CHECK-LL-NEXT:  %vbase.offset = load i64* %1
// CHECK-LL-NEXT:  %add.ptr = getelementptr i8* getelementptr inbounds (%class.test1_D* @test1_d, i32 0, i32 0, i32 0), i64 %vbase.offset
// CHECK-LL-NEXT:  %2 = bitcast i8* %add.ptr to %class.test1_A*
// CHECK-LL-NEXT:  br label %cast.end3
// CHECK-LL:       cast.null2:
// CHECK-LL-NEXT:  br label %cast.end3
// CHECK-LL:       cast.end3:
// CHECK-LL-NEXT:  %3 = phi %class.test1_A* [ %2, %cast.notnull1 ], [ null, %cast.null2 ]
// CHECK-LL-NEXT:  store %class.test1_A* %3, %class.test1_A** %ap
// CHECK-LL-NEXT:  %tmp = load %class.test1_A** %bp
// CHECK-LL-NEXT:  %4 = icmp ne %class.test1_A* %tmp, null
// CHECK-LL-NEXT:  br i1 %4, label %5, label %9
// CHECK-LL:       ; <label>:5
// CHECK-LL-NEXT:  %6 = bitcast %class.test1_A* %tmp to i8*
// CHECK-LL-NEXT:  %7 = call i8* @__dynamic_cast(i8* %6, i8* bitcast ({{.*}} @_ZTI7test1_B to i8*), i8* bitcast (i8** @_ZTI7test1_D to i8*), i64 -1)
// CHECK-LL-NEXT:  %8 = bitcast i8* %7 to %class.test1_D*
// CHECK-LL-NEXT:  br label %10
// CHECK-LL:       ; <label>:9
// CHECK-LL-NEXT:  br label %10
// CHECK-LL:       ; <label>:10
// CHECK-LL-NEXT:  %11 = phi %class.test1_D* [ %8, %5 ], [ null, %9 ]
// CHECK-LL-NEXT:  store %class.test1_D* %11, %class.test1_D** %dp
// CHECK-LL-NEXT:  %tmp4 = load %class.test1_D** %dp
// CHECK-LL-NEXT:  %cmp = icmp eq %class.test1_D* %tmp4, null
// CHECK-LL-NEXT:  br i1 %cmp, label %if.then, label %if.else
// CHECK-LL:       if.then:
// CHECK-LL-NEXT:  %call = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([10 x i8]* @.str, i32 0, i32 0), i32 1)
// CHECK-LL-NEXT:  br label %if.end
// CHECK-LL:       if.else:
// CHECK-LL-NEXT:  %call5 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([10 x i8]* @.str1, i32 0, i32 0), i32 1)
// CHECK-LL-NEXT:  br label %if.end
// CHECK-LL:       if.end:
// CHECK-LL-NEXT:  %tmp6 = load %class.test1_A** %bp
// CHECK-LL-NEXT:  %12 = icmp ne %class.test1_A* %tmp6, null
// CHECK-LL-NEXT:  br i1 %12, label %13, label %17
// CHECK-LL:       ; <label>:13
// CHECK-LL-NEXT:  %14 = bitcast %class.test1_A* %tmp6 to i8*
// CHECK-LL-NEXT:  %15 = call i8* @__dynamic_cast(i8* %14, i8* bitcast ({{.*}} @_ZTI7test1_B to i8*), i8* bitcast ({{.*}} @_ZTI7test1_A to i8*), i64 -1)
// CHECK-LL-NEXT:  %16 = bitcast i8* %15 to %class.test1_A*
// CHECK-LL-NEXT:  br label %18
// CHECK-LL:       ; <label>:17
// CHECK-LL-NEXT:  br label %18
// CHECK-LL:       ; <label>:18
// CHECK-LL-NEXT:  %19 = phi %class.test1_A* [ %16, %13 ], [ null, %17 ]
// CHECK-LL-NEXT:  store %class.test1_A* %19, %class.test1_A** %ap
// CHECK-LL-NEXT:  %tmp7 = load %class.test1_A** %ap
// CHECK-LL-NEXT:  %cmp8 = icmp eq %class.test1_A* %tmp7, null
// CHECK-LL-NEXT:  br i1 %cmp8, label %if.then9, label %if.else11
// CHECK-LL:       if.then9:
// CHECK-LL-NEXT:  %call10 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([10 x i8]* @.str, i32 0, i32 0), i32 2)
// CHECK-LL-NEXT:  br label %if.end13
// CHECK-LL:       if.else11:
// CHECK-LL-NEXT:  %call12 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([10 x i8]* @.str1, i32 0, i32 0), i32 2)
// CHECK-LL-NEXT:  br label %if.end13
// CHECK-LL:       if.end13:
// CHECK-LL-NEXT:  %tmp14 = load %class.test1_A** %ap
// CHECK-LL-NEXT:  %20 = icmp ne %class.test1_A* %tmp14, null
// CHECK-LL-NEXT:  br i1 %20, label %21, label %25
// CHECK-LL:       ; <label>:21
// CHECK-LL-NEXT:  %22 = bitcast %class.test1_A* %tmp14 to i8*
// CHECK-LL-NEXT:  %23 = call i8* @__dynamic_cast({{.*}} %22, i8* bitcast ({{.*}} @_ZTI7test1_A to i8*), i8* bitcast ({{.*}} @_ZTI7test1_B to i8*), i64 -1)
// CHECK-LL-NEXT:  %24 = bitcast i8* %23 to %class.test1_A*
// CHECK-LL-NEXT:  br label %26
// CHECK-LL:       ; <label>:25
// CHECK-LL-NEXT:  br label %26
// CHECK-LL:       ; <label>:26
// CHECK-LL-NEXT:  %27 = phi %class.test1_A* [ %24, %21 ], [ null, %25 ]
// CHECK-LL-NEXT:  store %class.test1_A* %27, %class.test1_A** %bp
// CHECK-LL-NEXT:  %tmp15 = load %class.test1_A** %bp
// CHECK-LL-NEXT:  %cmp16 = icmp eq %class.test1_A* %tmp15, null
// CHECK-LL-NEXT:  br i1 %cmp16, label %if.then17, label %if.else19
// CHECK-LL:       if.then17:
// CHECK-LL-NEXT:  %call18 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([10 x i8]* @.str, i32 0, i32 0), i32 3)
// CHECK-LL-NEXT:  br label %if.end21
// CHECK-LL:       if.else19:
// CHECK-LL-NEXT:  %call20 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([10 x i8]* @.str1, i32 0, i32 0), i32 3)
// CHECK-LL-NEXT:  br label %if.end21
// CHECK-LL:       if.end21:
// CHECK-LL-NEXT:  br i1 false, label %cast.null27, label %cast.notnull22
// CHECK-LL:       cast.notnull22:
// CHECK-LL-NEXT:  %vtable23 = load i8** bitcast (%class.test1_D* @test1_d to i8**)
// CHECK-LL-NEXT:  %vbase.offset.ptr24 = getelementptr i8* %vtable23, i64 -24
// CHECK-LL-NEXT:  %28 = bitcast i8* %vbase.offset.ptr24 to i64*
// CHECK-LL-NEXT:  %vbase.offset25 = load i64* %28
// CHECK-LL-NEXT:  %add.ptr26 = getelementptr i8* getelementptr inbounds (%class.test1_D* @test1_d, i32 0, i32 0, i32 0), i64 %vbase.offset25
// CHECK-LL-NEXT:  %29 = bitcast i8* %add.ptr26 to %class.test1_A*
// CHECK-LL-NEXT:  br label %cast.end28
// CHECK-LL:       cast.null27:
// CHECK-LL-NEXT:  br label %cast.end28
// CHECK-LL:       cast.end28:
// CHECK-LL-NEXT:  %30 = phi %class.test1_A* [ %29, %cast.notnull22 ], [ null, %cast.null27 ]
// CHECK-LL-NEXT:  store %class.test1_A* %30, %class.test1_A** %ap
// CHECK-LL-NEXT:  %tmp29 = load %class.test1_A** %ap
// CHECK-LL-NEXT:  %cmp30 = icmp ne %class.test1_A* %tmp29, null
// CHECK-LL-NEXT:  br i1 %cmp30, label %if.then31, label %if.else33
// CHECK-LL:       if.then31:
// CHECK-LL-NEXT:  %call32 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([10 x i8]* @.str, i32 0, i32 0), i32 4)
// CHECK-LL-NEXT:  br label %if.end35
// CHECK-LL:       if.else33:
// CHECK-LL-NEXT:  %call34 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([10 x i8]* @.str1, i32 0, i32 0), i32 4)
// CHECK-LL-NEXT:  br label %if.end35
// CHECK-LL:       if.end35:
// CHECK-LL-NEXT:  br i1 false, label %cast.null43, label %cast.notnull38
// CHECK-LL:       cast.notnull38:
// CHECK-LL-NEXT:  %vtable39 = load i8** bitcast (%class.test1_F* @test1_f to i8**)
// CHECK-LL-NEXT:  %vbase.offset.ptr40 = getelementptr i8* %vtable39, i64 -24
// CHECK-LL-NEXT:  %31 = bitcast i8* %vbase.offset.ptr40 to i64*
// CHECK-LL-NEXT:  %vbase.offset41 = load i64* %31
// CHECK-LL-NEXT:  %add.ptr42 = getelementptr i8* getelementptr inbounds (%class.test1_F* @test1_f, i32 0, i32 0, i32 0), i64 %vbase.offset41
// CHECK-LL-NEXT:  %32 = bitcast i8* %add.ptr42 to %class.test1_A*
// CHECK-LL-NEXT:  br label %cast.end44
// CHECK-LL:       cast.null43:
// CHECK-LL-NEXT:  br label %cast.end44
// CHECK-LL:       cast.end44:
// CHECK-LL-NEXT:  %33 = phi %class.test1_A* [ %32, %cast.notnull38 ], [ null, %cast.null43 ]
// CHECK-LL-NEXT:  store %class.test1_A* %33, %class.test1_A** %ap37
// CHECK-LL-NEXT:  %tmp45 = load %class.test1_A** %ap37
// CHECK-LL-NEXT:  %cmp46 = icmp ne %class.test1_A* %tmp45, null
// CHECK-LL-NEXT:  br i1 %cmp46, label %if.then47, label %if.else49
// CHECK-LL:       if.then47:
// CHECK-LL-NEXT:  %call48 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([10 x i8]* @.str, i32 0, i32 0), i32 6)
// CHECK-LL-NEXT:  br label %if.end51
// CHECK-LL:       if.else49:
// CHECK-LL-NEXT:  %call50 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([10 x i8]* @.str1, i32 0, i32 0), i32 6)
// CHECK-LL-NEXT:  br label %if.end51
// CHECK-LL:       if.end51:
// CHECK-LL-NEXT:  %tmp54 = load %class.test1_A** %ap37
// CHECK-LL-NEXT:  %34 = icmp ne %class.test1_A* %tmp54, null
// CHECK-LL-NEXT:  br i1 %34, label %35, label %39
// CHECK-LL:       ; <label>:35
// CHECK-LL-NEXT:  %36 = bitcast %class.test1_A* %tmp54 to i8*
// CHECK-LL-NEXT:  %37 = call i8* @__dynamic_cast(i8* %36, i8* bitcast ({{.*}} @_ZTI7test1_A to i8*), i8* bitcast ({{.*}} @_ZTI7test1_D to i8*), i64 -1)
// CHECK-LL-NEXT:  %38 = bitcast i8* %37 to %class.test1_D*
// CHECK-LL-NEXT:  br label %40
// CHECK-LL:       ; <label>:39
// CHECK-LL-NEXT:  br label %40
// CHECK-LL:       ; <label>:40
// CHECK-LL-NEXT:  %41 = phi %class.test1_D* [ %38, %35 ], [ null, %39 ]
// CHECK-LL-NEXT:  store %class.test1_D* %41, %class.test1_D** %dp53
// CHECK-LL-NEXT:  %tmp55 = load %class.test1_D** %dp53
// CHECK-LL-NEXT:  %cmp56 = icmp eq %class.test1_D* %tmp55, null
// CHECK-LL-NEXT:  br i1 %cmp56, label %if.then57, label %if.else59
// CHECK-LL:       if.then57:
// CHECK-LL-NEXT:  %call58 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([10 x i8]* @.str, i32 0, i32 0), i32 7)
// CHECK-LL-NEXT:  br label %if.end61
// CHECK-LL:       if.else59:
// CHECK-LL-NEXT:  %call60 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([10 x i8]* @.str1, i32 0, i32 0), i32 7)
// CHECK-LL-NEXT:  br label %if.end61
// CHECK-LL:       if.end61:
// CHECK-LL-NEXT:  %tmp63 = load %class.test1_A** %ap37
// CHECK-LL-NEXT:  %42 = icmp ne %class.test1_A* %tmp63, null
// CHECK-LL-NEXT:  br i1 %42, label %43, label %47
// CHECK-LL:       ; <label>:43
// CHECK-LL-NEXT:  %44 = bitcast %class.test1_A* %tmp63 to i8*
// CHECK-LL-NEXT:  %45 = call i8* @__dynamic_cast(i8* %44, i8* bitcast ({{.*}} @_ZTI7test1_A to i8*), i8* bitcast ({{.*}} @_ZTI7test1_E to i8*), i64 -1)
// CHECK-LL-NEXT:  %46 = bitcast i8* %45 to %class.test1_E*
// CHECK-LL-NEXT:  br label %48
// CHECK-LL:       ; <label>:47
// CHECK-LL-NEXT:  br label %48
// CHECK-LL:       ; <label>:48
// CHECK-LL-NEXT:  %49 = phi %class.test1_E* [ %46, %43 ], [ null, %47 ]
// CHECK-LL-NEXT:  store %class.test1_E* %49, %class.test1_E** %ep1
// CHECK-LL-NEXT:  %tmp64 = load %class.test1_E** %ep1
// CHECK-LL-NEXT:  %cmp65 = icmp ne %class.test1_E* %tmp64, null
// CHECK-LL-NEXT:  br i1 %cmp65, label %if.then66, label %if.else68
// CHECK-LL:       if.then66:
// CHECK-LL-NEXT:  %call67 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([10 x i8]* @.str, i32 0, i32 0), i32 8)
// CHECK-LL-NEXT:  br label %if.end70
// CHECK-LL:       if.else68:
// CHECK-LL-NEXT:  %call69 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([10 x i8]* @.str1, i32 0, i32 0), i32 8)
// CHECK-LL-NEXT:  br label %if.end70
// CHECK-LL:       if.end70:
// CHECK-LL-NEXT:  store %class.test1_D* @test1_d, %class.test1_D** %dp
// CHECK-LL-NEXT:  %tmp71 = load %class.test1_D** %dp
// CHECK-LL-NEXT:  %cmp72 = icmp eq %class.test1_D* %tmp71, @test1_d
// CHECK-LL-NEXT:  br i1 %cmp72, label %if.then73, label %if.else75
// CHECK-LL:       if.then73:
// CHECK-LL-NEXT:  %call74 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([10 x i8]* @.str, i32 0, i32 0), i32 9)
// CHECK-LL-NEXT:  br label %if.end77
// CHECK-LL:       if.else75:
// CHECK-LL-NEXT:  %call76 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([10 x i8]* @.str1, i32 0, i32 0), i32 9)
// CHECK-LL-NEXT:  br label %if.end77
// CHECK-LL:       if.end77:
// CHECK-LL-NEXT:  store %class.test1_D* @test1_d, %class.test1_D** %cdp
// CHECK-LL-NEXT:  %tmp79 = load %class.test1_D** %cdp
// CHECK-LL-NEXT:  %cmp80 = icmp eq %class.test1_D* %tmp79, @test1_d
// CHECK-LL-NEXT:  br i1 %cmp80, label %if.then81, label %if.else83
// CHECK-LL:       if.then81:
// CHECK-LL-NEXT:  %call82 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([10 x i8]* @.str, i32 0, i32 0), i32 10)
// CHECK-LL-NEXT:  br label %if.end85
// CHECK-LL:       if.else83:
// CHECK-LL-NEXT:  %call84 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([10 x i8]* @.str1, i32 0, i32 0), i32 10)
// CHECK-LL-NEXT:  br label %if.end85
// CHECK-LL:       if.end85:
// CHECK-LL-NEXT:  br i1 false, label %50, label %53
// CHECK-LL:       ; <label>:50
// CHECK-LL-NEXT:  %51 = call i8* @__dynamic_cast(i8* null, i8* bitcast ({{.*}}* @_ZTI7test1_A to i8*), i8* bitcast ({{.*}} @_ZTI7test1_D to i8*), i64 -1)
// CHECK-LL-NEXT:  %52 = bitcast i8* %51 to %class.test1_D*
// CHECK-LL-NEXT:  br label %54
// CHECK-LL:       ; <label>:53
// CHECK-LL-NEXT:  br label %54
// CHECK-LL:       ; <label>:54
// CHECK-LL-NEXT:  %55 = phi %class.test1_D* [ %52, %50 ], [ null, %53 ]
// CHECK-LL-NEXT:  store %class.test1_D* %55, %class.test1_D** %dp
// CHECK-LL-NEXT:  %tmp86 = load %class.test1_D** %dp
// CHECK-LL-NEXT:  %cmp87 = icmp eq %class.test1_D* %tmp86, null
// CHECK-LL-NEXT:  br i1 %cmp87, label %if.then88, label %if.else90
// CHECK-LL:       if.then88:
// CHECK-LL-NEXT:  %call89 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([10 x i8]* @.str, i32 0, i32 0), i32 11)
// CHECK-LL-NEXT:  br label %if.end92
// CHECK-LL:       if.else90:
// CHECK-LL-NEXT:  %call91 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([10 x i8]* @.str1, i32 0, i32 0), i32 11)
// CHECK-LL-NEXT:  br label %if.end92
// CHECK-LL:       if.end92:
// CHECK-LL-NEXT:  br i1 false, label %cast.null98, label %cast.notnull93
// CHECK-LL:       cast.notnull93:
// CHECK-LL-NEXT:  %vtable94 = load i8** bitcast (%class.test1_D* @test1_d to i8**)
// CHECK-LL-NEXT:  %vbase.offset.ptr95 = getelementptr i8* %vtable94, i64 -24
// CHECK-LL-NEXT:  %56 = bitcast i8* %vbase.offset.ptr95 to i64*
// CHECK-LL-NEXT:  %vbase.offset96 = load i64* %56
// CHECK-LL-NEXT:  %add.ptr97 = getelementptr i8* getelementptr inbounds (%class.test1_D* @test1_d, i32 0, i32 0, i32 0), i64 %vbase.offset96
// CHECK-LL-NEXT:  %57 = bitcast i8* %add.ptr97 to %class.test1_A*
// CHECK-LL-NEXT:  br label %cast.end99
// CHECK-LL:       cast.null98:
// CHECK-LL-NEXT:  br label %cast.end99
// CHECK-LL:       cast.end99:
// CHECK-LL-NEXT:  %58 = phi %class.test1_A* [ %57, %cast.notnull93 ], [ null, %cast.null98 ]
// CHECK-LL-NEXT:  store %class.test1_A* %58, %class.test1_A** %ap
// CHECK-LL-NEXT:  %tmp100 = load %class.test1_A** %ap
// CHECK-LL-NEXT:  br i1 false, label %cast.null106, label %cast.notnull101
// CHECK-LL:       cast.notnull101:
// CHECK-LL-NEXT:  %vtable102 = load i8** bitcast (%class.test1_D* @test1_d to i8**)
// CHECK-LL-NEXT:  %vbase.offset.ptr103 = getelementptr i8* %vtable102, i64 -24
// CHECK-LL-NEXT:  %59 = bitcast i8* %vbase.offset.ptr103 to i64*
// CHECK-LL-NEXT:  %vbase.offset104 = load i64* %59
// CHECK-LL-NEXT:  %add.ptr105 = getelementptr i8* getelementptr inbounds (%class.test1_D* @test1_d, i32 0, i32 0, i32 0), i64 %vbase.offset104
// CHECK-LL-NEXT:  %60 = bitcast i8* %add.ptr105 to %class.test1_A*
// CHECK-LL-NEXT:  br label %cast.end107
// CHECK-LL:       cast.null106:
// CHECK-LL-NEXT:  br label %cast.end107
// CHECK-LL:       cast.end107:
// CHECK-LL-NEXT:  %61 = phi %class.test1_A* [ %60, %cast.notnull101 ], [ null, %cast.null106 ]
// CHECK-LL-NEXT:  %cmp108 = icmp eq %class.test1_A* %tmp100, %61
// CHECK-LL-NEXT:  br i1 %cmp108, label %if.then109, label %if.else111
// CHECK-LL:       if.then109:
// CHECK-LL-NEXT:  %call110 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([10 x i8]* @.str, i32 0, i32 0), i32 12)
// CHECK-LL-NEXT:  br label %if.end113
// CHECK-LL:       if.else111:
// CHECK-LL-NEXT:  %call112 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([10 x i8]* @.str1, i32 0, i32 0), i32 12)
// CHECK-LL-NEXT:  br label %if.end113
// CHECK-LL:       if.end113:
// CHECK-LL-NEXT:  br i1 false, label %cast.null116, label %cast.notnull115
// CHECK-LL:       cast.notnull115:
// CHECK-LL-NEXT:  br label %cast.end117
// CHECK-LL:       cast.null116:
// CHECK-LL-NEXT:  br label %cast.end117
// CHECK-LL:       cast.end117:
// CHECK-LL-NEXT:  %62 = phi %class.test1_E* [ bitcast (%class.test1_F* @test1_f to %class.test1_E*), %cast.notnull115 ], [ null, %cast.null116 ]
// CHECK-LL-NEXT:  store %class.test1_E* %62, %class.test1_E** %ep
// CHECK-LL-NEXT:  %tmp118 = load %class.test1_E** %ep
// CHECK-LL-NEXT:  br i1 false, label %cast.null120, label %cast.notnull119
// CHECK-LL:       cast.notnull119:
// CHECK-LL-NEXT:  br label %cast.end121
// CHECK-LL:       cast.null120:
// CHECK-LL-NEXT:  br label %cast.end121
// CHECK-LL:       cast.end121:
// CHECK-LL-NEXT:  %63 = phi %class.test1_E* [ bitcast (%class.test1_F* @test1_f to %class.test1_E*), %cast.notnull119 ], [ null, %cast.null120 ]
// CHECK-LL-NEXT:  %cmp122 = icmp eq %class.test1_E* %tmp118, %63
// CHECK-LL-NEXT:  br i1 %cmp122, label %if.then123, label %if.else125
// CHECK-LL:       if.then123:
// CHECK-LL-NEXT:  %call124 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([10 x i8]* @.str, i32 0, i32 0), i32 13)
// CHECK-LL-NEXT:  br label %if.end127
// CHECK-LL:       if.else125:
// CHECK-LL-NEXT:  %call126 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([10 x i8]* @.str1, i32 0, i32 0), i32 13)
// CHECK-LL-NEXT:  br label %if.end127
// CHECK-LL:       if.end127:
// CHECK-LL-NEXT:  %tmp129 = load %class.test1_A** %ap
// CHECK-LL-NEXT:  %64 = icmp ne %class.test1_A* %tmp129, null
// CHECK-LL-NEXT:  br i1 %64, label %65, label %70
// CHECK-LL:       ; <label>:65
// CHECK-LL-NEXT:  %66 = bitcast %class.test1_A* %tmp129 to i64**
// CHECK-LL-NEXT:  %vtable130 = load i64** %66
// CHECK-LL-NEXT:  %67 = getelementptr inbounds i64* %vtable130, i64 -2
// CHECK-LL-NEXT:  %"offset to top" = load i64* %67
// CHECK-LL-NEXT:  %68 = bitcast %class.test1_A* %tmp129 to i8*
// CHECK-LL-NEXT:  %69 = getelementptr inbounds i8* %68, i64 %"offset to top"
// CHECK-LL-NEXT:  br label %71
// CHECK-LL:       ; <label>:70
// CHECK-LL-NEXT:  br label %71
// CHECK-LL:       ; <label>:71
// CHECK-LL-NEXT:  %72 = phi i8* [ %69, %65 ], [ null, %70 ]
// CHECK-LL-NEXT:  store i8* %72, i8** %vp
// CHECK-LL-NEXT:  %tmp131 = load i8** %vp
// CHECK-LL-NEXT:  %cmp132 = icmp eq i8* %tmp131, getelementptr inbounds (%class.test1_D* @test1_d, i32 0, i32 0, i32 0)
// CHECK-LL-NEXT:  br i1 %cmp132, label %if.then133, label %if.else135
// CHECK-LL:       if.then133:
// CHECK-LL-NEXT:  %call134 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([10 x i8]* @.str, i32 0, i32 0), i32 14)
// CHECK-LL-NEXT:  br label %if.end137
// CHECK-LL:       if.else135:
// CHECK-LL-NEXT:  %call136 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([10 x i8]* @.str1, i32 0, i32 0), i32 14)
// CHECK-LL-NEXT:  br label %if.end137
// CHECK-LL:       if.end137:
// CHECK-LL-NEXT:  %tmp139 = load %class.test1_A** %ap
// CHECK-LL-NEXT:  %73 = icmp ne %class.test1_A* %tmp139, null
// CHECK-LL-NEXT:  br i1 %73, label %74, label %79
// CHECK-LL:       ; <label>:74
// CHECK-LL-NEXT:  %75 = bitcast %class.test1_A* %tmp139 to i64**
// CHECK-LL-NEXT:  %vtable140 = load i64** %75
// CHECK-LL-NEXT:  %76 = getelementptr inbounds i64* %vtable140, i64 -2
// CHECK-LL-NEXT:  %"offset to top141" = load i64* %76
// CHECK-LL-NEXT:  %77 = bitcast %class.test1_A* %tmp139 to i8*
// CHECK-LL-NEXT:  %78 = getelementptr inbounds i8* %77, i64 %"offset to top141"
// CHECK-LL-NEXT:  br label %80
// CHECK-LL:       ; <label>:79
// CHECK-LL-NEXT:  br label %80
// CHECK-LL:       ; <label>:80
// CHECK-LL-NEXT:  %81 = phi i8* [ %78, %74 ], [ null, %79 ]
// CHECK-LL-NEXT:  store i8* %81, i8** %cvp
// CHECK-LL-NEXT:  %tmp142 = load i8** %cvp
// CHECK-LL-NEXT:  %cmp143 = icmp eq i8* %tmp142, getelementptr inbounds (%class.test1_D* @test1_d, i32 0, i32 0, i32 0)
// CHECK-LL-NEXT:  br i1 %cmp143, label %if.then144, label %if.else146
// CHECK-LL:       if.then144:
// CHECK-LL-NEXT:  %call145 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([10 x i8]* @.str, i32 0, i32 0), i32 15)
// CHECK-LL-NEXT:  br label %if.end148
// CHECK-LL:       if.else146:
// CHECK-LL-NEXT:  %call147 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([10 x i8]* @.str1, i32 0, i32 0), i32 15)
// CHECK-LL-NEXT:  br label %if.end148
// CHECK-LL:       if.end148:
// CHECK-LL-NEXT:  ret void
