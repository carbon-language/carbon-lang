// RUN: %clang_cc1 -I%S -triple x86_64-apple-darwin -std=c++0x -emit-llvm %s -o %t.ll
// RUN: FileCheck -check-prefix LL --input-file=%t.ll %s
// XFAIL: win32

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
// CHECK-LL:       [[bp:%.*]] = alloca %class.test1_A*, align 8
// CHECK-LL-NEXT:  [[ap:%.*]] = alloca %class.test1_A*, align 8
// CHECK-LL-NEXT:  [[dp:%.*]] = alloca %class.test1_D*, align 8
// CHECK-LL-NEXT:  [[ap37:%.*]] = alloca %class.test1_A*, align 8
// CHECK-LL-NEXT:  [[dp53:%.*]] = alloca %class.test1_D*, align 8
// CHECK-LL-NEXT:  [[ep1:%.*]] = alloca %class.test1_E*, align 8
// CHECK-LL-NEXT:  [[cdp:%.*]] = alloca %class.test1_D*, align 8
// CHECK-LL-NEXT:  [[ep:%.*]] = alloca %class.test1_E*, align 8
// CHECK-LL-NEXT:  [[vp:%.*]] = alloca i8*, align 8
// CHECK-LL-NEXT:  [[cvp:%.*]] = alloca i8*, align 8
// CHECK-LL-NEXT:  store %class.test1_A* bitcast (%class.test1_D* @test1_d to %class.test1_A*), %class.test1_A** [[bp]]
// CHECK-LL-NEXT:  br i1 false, label %[[castnull2:.*]], label %[[castnotnull1:.*]]
// CHECK-LL:       [[castnotnull1]]
// CHECK-LL-NEXT:  [[vtable:%.*]] = load i8** bitcast (%class.test1_D* @test1_d to i8**)
// CHECK-LL-NEXT:  [[vbaseoffsetptr:%.*]] = getelementptr i8* [[vtable]], i64 -24
// CHECK-LL-NEXT:  [[v1:%.*]] = bitcast i8* [[vbaseoffsetptr]] to i64*
// CHECK-LL-NEXT:  [[vbaseoffset:%.*]] = load i64* [[v1]]
// CHECK-LL-NEXT:  [[addptr:%.*]] = getelementptr i8* getelementptr inbounds (%class.test1_D* @test1_d, i32 0, i32 0, i32 0), i64 [[vbaseoffset:.*]]
// CHECK-LL-NEXT:  [[v2:%.*]] = bitcast i8* [[addptr]] to %class.test1_A*
// CHECK-LL-NEXT:  br label %[[castend3:.*]]
// CHECK-LL:       [[castnull2]]
// CHECK-LL-NEXT:  br label %[[castend3]]
// CHECK-LL:       [[castend3]]
// CHECK-LL-NEXT:  [[v3:%.*]] = phi %class.test1_A* [ [[v2]], %[[castnotnull1]] ], [ null, %[[castnull2]] ]
// CHECK-LL-NEXT:  store %class.test1_A* [[v3]], %class.test1_A** [[ap]]
// CHECK-LL-NEXT:  [[tmp:%.*]] = load %class.test1_A** [[bp]]
// CHECK-LL-NEXT:  [[v4:%.*]] = icmp ne %class.test1_A* [[tmp]], null
// CHECK-LL-NEXT:  br i1 [[v4]], label %[[v5:.*]], label %[[v9:.*]]
// CHECK-LL:       [[v6:%.*]] = bitcast %class.test1_A* [[tmp]] to i8*
// CHECK-LL-NEXT:  [[v7:%.*]] = call i8* @__dynamic_cast(i8* [[v6]], i8* bitcast (%0* @_ZTI7test1_B to i8*), i8* bitcast (%1* @_ZTI7test1_D to i8*), i64 -1) 
// CHECK-LL-NEXT:  [[v8:%.*]] = bitcast i8* [[v7]] to %class.test1_D*
// CHECK-LL-NEXT:  br label %[[v10:.*]]
// CHECK-LL:  br label %[[v10]]
// CHECK-LL:  [[v11:%.*]] = phi %class.test1_D* [ [[v8]], %[[v5]] ], [ null, %[[v9]] ]
// CHECK-LL-NEXT:  store %class.test1_D* [[v11]], %class.test1_D** [[dp]]
// CHECK-LL-NEXT:  [[tmp4:%.*]] = load %class.test1_D** [[dp]]
// CHECK-LL-NEXT:  [[cmp:%.*]] = icmp eq %class.test1_D* [[tmp4]], null
// CHECK-LL-NEXT:  br i1 [[cmp]], label %[[ifthen:.*]], label %[[ifelse:.*]]
// CHECK-LL:       [[ifthen]]
// CHECK-LL-NEXT:  call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([10 x i8]* @.str, i32 0, i32 0), i32 1)
// CHECK-LL-NEXT:  br label %[[ifend:.*]]
// CHECK-LL:       [[ifelse]]
// CHECK-LL-NEXT:  call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([10 x i8]* @.str1, i32 0, i32 0), i32 1)
// CHECK-LL-NEXT:  br label %[[ifend]]
// CHECK-LL:       [[ifend]]
// CHECK-LL-NEXT:  [[tmp6:%.*]] = load %class.test1_A** [[bp]]
// CHECK-LL-NEXT:  [[v12:%.*]] = icmp ne %class.test1_A* [[tmp6]], null
// CHECK-LL-NEXT:  br i1 [[v12]], label %[[v13:.*]], label %[[v17:.*]]
// CHECK-LL:  [[v14:%.*]] = bitcast %class.test1_A* [[tmp6]] to i8*
// CHECK-LL-NEXT:  [[v15:%.*]] = call i8* @__dynamic_cast(i8* [[v14]], i8* bitcast ({{.*}} @_ZTI7test1_B to i8*), i8* bitcast ({{.*}} @_ZTI7test1_A to i8*), i64 -1)
// CHECK-LL-NEXT:  [[v16:%.*]] = bitcast i8* [[v15]] to %class.test1_A*
// CHECK-LL-NEXT:  br label %[[v18:.*]]
// CHECK-LL:  br label %[[v18]]
// CHECK-LL:  [[v19:%.*]] = phi %class.test1_A* [ [[v16]], %[[v13]] ], [ null, %[[v17]] ]
// CHECK-LL-NEXT:  store %class.test1_A* [[v19]], %class.test1_A** [[ap]]
// CHECK-LL-NEXT:  [[tmp7:%.*]] = load %class.test1_A** [[ap]]
// CHECK-LL-NEXT:  [[cmp8:%.*]] = icmp eq %class.test1_A* [[tmp7]], null
// CHECK-LL-NEXT:  br i1 [[cmp8]], label %[[ifthen9:.*]], label %[[ifelse11:.*]]
// CHECK-LL:       [[ifthen9]]
// CHECK-LL-NEXT:  call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([10 x i8]* @.str, i32 0, i32 0), i32 2)
// CHECK-LL-NEXT:  br label %[[ifend13:.*]]
// CHECK-LL:       [[ifelse11]]
// CHECK-LL-NEXT:  call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([10 x i8]* @.str1, i32 0, i32 0), i32 2)
// CHECK-LL-NEXT:  br label %[[ifend13]]
// CHECK-LL:       [[ifend13]]
// CHECK-LL-NEXT:  [[tmp14:%.*]] = load %class.test1_A** [[ap]]
// CHECK-LL-NEXT:  [[v20:%.*]] = icmp ne %class.test1_A* [[tmp14]], null
// CHECK-LL-NEXT:  br i1 [[v20]], label %[[v21:.*]], label %[[v25:.*]]
// CHECK-LL:  [[v22:%.*]] = bitcast %class.test1_A* [[tmp14]] to i8*
// CHECK-LL-NEXT:  [[v23:%.*]] = call i8* @__dynamic_cast({{.*}} [[v22]], i8* bitcast ({{.*}} @_ZTI7test1_A to i8*), i8* bitcast ({{.*}} @_ZTI7test1_B to i8*), i64 -1)
// CHECK-LL-NEXT:  [[v24:%.*]] = bitcast i8* [[v23]] to %class.test1_A*
// CHECK-LL-NEXT:  br label %[[v26:.*]]
// CHECK-LL:  br label %[[v26]]
// CHECK-LL:  [[v27:%.*]] = phi %class.test1_A* [ [[v24]], %[[v21]] ], [ null, %[[v25]] ]
// CHECK-LL-NEXT:  store %class.test1_A* [[v27]], %class.test1_A** [[bp]]
// CHECK-LL-NEXT:  [[tmp15:%.*]] = load %class.test1_A** [[bp]]
// CHECK-LL-NEXT:  [[cmp16:%.*]] = icmp eq %class.test1_A* [[tmp15]], null
// CHECK-LL-NEXT:  br i1 [[cmp16]], label %[[ifthen17:.*]], label %[[ifelse19:.*]]
// CHECK-LL:       [[ifthen17]]
// CHECK-LL-NEXT:  call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([10 x i8]* @.str, i32 0, i32 0), i32 3)
// CHECK-LL-NEXT:  br label %[[ifend21:.*]]
// CHECK-LL:       [[ifelse19]]
// CHECK-LL-NEXT:  call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([10 x i8]* @.str1, i32 0, i32 0), i32 3)
// CHECK-LL-NEXT:  br label %[[ifend21]]
// CHECK-LL:       [[ifend21]]
// CHECK-LL-NEXT:  br i1 false, label %[[castnull27:.*]], label %[[castnotnull22:.*]]
// CHECK-LL:       [[castnotnull22]]
// CHECK-LL-NEXT:  [[vtable23:%.*]] = load i8** bitcast (%class.test1_D* @test1_d to i8**)
// CHECK-LL-NEXT:  [[vbaseoffsetptr24:%.*]] = getelementptr i8* [[vtable23]], i64 -24
// CHECK-LL-NEXT:  [[v28:%.*]] = bitcast i8* [[vbaseoffsetptr24]] to i64*
// CHECK-LL-NEXT:  [[vbaseoffset25:%.*]] = load i64* [[v28]]
// CHECK-LL-NEXT:  [[addptr26:%.*]] = getelementptr i8* getelementptr inbounds (%class.test1_D* @test1_d, i32 0, i32 0, i32 0), i64 [[vbaseoffset25]]
// CHECK-LL-NEXT:  [[v29:%.*]] = bitcast i8* [[addptr26]] to %class.test1_A*
// CHECK-LL-NEXT:  br label %[[castend28:.*]]
// CHECK-LL:       [[castnull27]]
// CHECK-LL-NEXT:  br label %[[castend28]]
// CHECK-LL:       [[castend28]]
// CHECK-LL-NEXT:  [[v30:%.*]] = phi %class.test1_A* [ [[v29]], %[[castnotnull22]] ], [ null, %[[castnull27]] ]
// CHECK-LL-NEXT:  store %class.test1_A* [[v30]], %class.test1_A** [[ap]]
// CHECK-LL-NEXT:  [[tmp29:%.*]] = load %class.test1_A** [[ap]]
// CHECK-LL-NEXT:  [[cmp30:%.*]] = icmp ne %class.test1_A* [[tmp29]], null
// CHECK-LL-NEXT:  br i1 [[cmp30]], label %[[ifthen31:.*]], label %[[ifelse33:.*]]
// CHECK-LL:       [[ifthen31]]
// CHECK-LL-NEXT:  call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([10 x i8]* @.str, i32 0, i32 0), i32 4)
// CHECK-LL-NEXT:  br label %[[ifend35:.*]]
// CHECK-LL:       [[ifelse33]]
// CHECK-LL-NEXT:  call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([10 x i8]* @.str1, i32 0, i32 0), i32 4)
// CHECK-LL-NEXT:  br label %[[ifend35]]
// CHECK-LL:       [[ifend35]]
// CHECK-LL-NEXT:  br i1 false, label %[[castnull43:.*]], label %[[castnotnull38:.*]]
// CHECK-LL:       [[castnotnull38]]
// CHECK-LL-NEXT:  [[vtable39:%.*]] = load i8** bitcast (%class.test1_F* @test1_f to i8**)
// CHECK-LL-NEXT:  [[vbaseoffsetptr40:%.*]] = getelementptr i8* [[vtable39]], i64 -24
// CHECK-LL-NEXT:  [[v31:%.*]] = bitcast i8* [[vbaseoffsetptr40]] to i64*
// CHECK-LL-NEXT:  [[vbaseoffset41:%.*]] = load i64* [[v31]]
// CHECK-LL-NEXT:  [[addptr42:%.*]] = getelementptr i8* getelementptr inbounds (%class.test1_F* @test1_f, i32 0, i32 0, i32 0), i64 [[vbaseoffset41]]
// CHECK-LL-NEXT:  [[v32:%.*]] = bitcast i8* [[addptr42]] to %class.test1_A*
// CHECK-LL-NEXT:  br label %[[castend44:.*]]
// CHECK-LL:       [[castnull43]]
// CHECK-LL-NEXT:  br label %[[castend44]]
// CHECK-LL:       [[castend44]]
// CHECK-LL-NEXT:  [[v33:%.*]] = phi %class.test1_A* [ [[v32]], %[[castnotnull38]] ], [ null, %[[castnull43]] ]
// CHECK-LL-NEXT:  store %class.test1_A* [[v33]], %class.test1_A** [[ap37]]
// CHECK-LL-NEXT:  [[tmp45:%.*]] = load %class.test1_A** [[ap37]]
// CHECK-LL-NEXT:  [[cmp46:%.*]] = icmp ne %class.test1_A* [[tmp45]], null
// CHECK-LL-NEXT:  br i1 [[cmp46]], label %[[ifthen47:.*]], label %[[ifelse49:.*]]
// CHECK-LL:       [[ifthen47]]
// CHECK-LL-NEXT:  call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([10 x i8]* @.str, i32 0, i32 0), i32 6)
// CHECK-LL-NEXT:  br label %[[ifend51:.*]]
// CHECK-LL:       [[ifelse49]]
// CHECK-LL-NEXT:  call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([10 x i8]* @.str1, i32 0, i32 0), i32 6)
// CHECK-LL-NEXT:  br label %[[ifend51]]
// CHECK-LL:       [[ifend51]]
// CHECK-LL-NEXT:  [[tmp54:%.*]] = load %class.test1_A** [[ap37]]
// CHECK-LL-NEXT:  [[v34:%.*]] = icmp ne %class.test1_A* [[tmp54]], null
// CHECK-LL-NEXT:  br i1 [[v34]], label %[[v35:.*]], label %[[v39:.*]]
// CHECK-LL:       [[v36:%.*]] = bitcast %class.test1_A* [[tmp54]] to i8*
// CHECK-LL-NEXT:  [[v37:%.*]] = call i8* @__dynamic_cast(i8* [[v36]], i8* bitcast ({{.*}} @_ZTI7test1_A to i8*), i8* bitcast ({{.*}} @_ZTI7test1_D to i8*), i64 -1)
// CHECK-LL-NEXT:  [[v38:%.*]] = bitcast i8* [[v37]] to %class.test1_D*
// CHECK-LL-NEXT:  br label %[[v40:.*]]
// CHECK-LL:       br label %[[v40]]
// CHECK-LL:       [[v41:%.*]] = phi %class.test1_D* [ [[v38]], %[[v35]] ], [ null, %[[v39]] ]
// CHECK-LL-NEXT:  store %class.test1_D* [[v41]], %class.test1_D** [[dp53]]
// CHECK-LL-NEXT:  [[tmp55:%.*]] = load %class.test1_D** [[dp53]]
// CHECK-LL-NEXT:  [[cmp56:%.*]] = icmp eq %class.test1_D* [[tmp55]], null
// CHECK-LL-NEXT:  br i1 [[cmp56]], label %[[ifthen57:.*]], label %[[ifelse59:.*]]
// CHECK-LL:       [[ifthen57]]
// CHECK-LL-NEXT:  call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([10 x i8]* @.str, i32 0, i32 0), i32 7)
// CHECK-LL-NEXT:  br label %[[ifend61:.*]]
// CHECK-LL:       [[ifelse59]]
// CHECK-LL-NEXT:  call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([10 x i8]* @.str1, i32 0, i32 0), i32 7)
// CHECK-LL-NEXT:  br label %[[ifend61]]
// CHECK-LL:       [[ifend61]]
// CHECK-LL-NEXT:  [[tmp63:%.*]] = load %class.test1_A** [[ap37]]
// CHECK-LL-NEXT:  [[v42:%.*]] = icmp ne %class.test1_A* [[tmp63]], null
// CHECK-LL-NEXT:  br i1 [[v42]], label %[[v43:.*]], label %[[v47:.*]]
// CHECK-LL:       [[v44:%.*]] = bitcast %class.test1_A* [[tmp63]] to i8*
// CHECK-LL-NEXT:  [[v45:%.*]] = call i8* @__dynamic_cast(i8* [[v44]], i8* bitcast ({{.*}} @_ZTI7test1_A to i8*), i8* bitcast ({{.*}} @_ZTI7test1_E to i8*), i64 -1)
// CHECK-LL-NEXT:  [[v46:%.*]] = bitcast i8* [[v45]] to %class.test1_E*
// CHECK-LL-NEXT:  br label %[[v48:.*]]
// CHECK-LL:       br label %[[v48]]
// CHECK-LL:       [[v49:%.*]] = phi %class.test1_E* [ [[v46]], %[[v43]] ], [ null, %[[v47]] ]
// CHECK-LL-NEXT:  store %class.test1_E* [[v49]], %class.test1_E** [[ep1]]
// CHECK-LL-NEXT:  [[tmp64:%.*]] = load %class.test1_E** [[ep1]]
// CHECK-LL-NEXT:  [[cmp65:%.*]] = icmp ne %class.test1_E* [[tmp64]], null
// CHECK-LL-NEXT:  br i1 [[cmp65]], label %[[ifthen66:.*]], label %[[ifelse68:.*]]
// CHECK-LL:       [[ifthen66]]
// CHECK-LL-NEXT:  call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([10 x i8]* @.str, i32 0, i32 0), i32 8)
// CHECK-LL-NEXT:  br label %[[ifend70:.*]]
// CHECK-LL:       [[ifelse68]]
// CHECK-LL-NEXT:  call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([10 x i8]* @.str1, i32 0, i32 0), i32 8)
// CHECK-LL-NEXT:  br label %[[ifend70]]
// CHECK-LL:       [[ifend70]]
// CHECK-LL-NEXT:  store %class.test1_D* @test1_d, %class.test1_D** [[dp]]
// CHECK-LL-NEXT:  [[tmp71:%.*]] = load %class.test1_D** [[dp]]
// CHECK-LL-NEXT:  [[cmp72:%.*]] = icmp eq %class.test1_D* [[tmp71]], @test1_d
// CHECK-LL-NEXT:  br i1 [[cmp72]], label %[[ifthen73:.*]], label %[[ifelse75:.*]]
// CHECK-LL:       [[ifthen73]]
// CHECK-LL-NEXT:  call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([10 x i8]* @.str, i32 0, i32 0), i32 9)
// CHECK-LL-NEXT:  br label %[[ifend77:.*]]
// CHECK-LL:       [[ifelse75]]
// CHECK-LL-NEXT:  call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([10 x i8]* @.str1, i32 0, i32 0), i32 9)
// CHECK-LL-NEXT:  br label %[[ifend77]]
// CHECK-LL:       [[ifend77]]
// CHECK-LL-NEXT:  store %class.test1_D* @test1_d, %class.test1_D** [[cdp]]
// CHECK-LL-NEXT:  [[tmp79:%.*]] = load %class.test1_D** [[cdp]]
// CHECK-LL-NEXT:  [[cmp80:%.*]] = icmp eq %class.test1_D* [[tmp79]], @test1_d
// CHECK-LL-NEXT:  br i1 [[cmp80]], label %[[ifthen81:.*]], label %[[ifelse83:.*]]
// CHECK-LL:       [[ifthen81]]
// CHECK-LL-NEXT:  call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([10 x i8]* @.str, i32 0, i32 0), i32 10)
// CHECK-LL-NEXT:  br label %[[ifend85:.*]]
// CHECK-LL:       [[ifelse83]]
// CHECK-LL-NEXT:  call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([10 x i8]* @.str1, i32 0, i32 0), i32 10)
// CHECK-LL-NEXT:  br label %[[ifend85]]
// CHECK-LL:       [[ifend85]]
// CHECK-LL-NEXT:  br i1 false, label %[[v50:.*]], label %[[v53:.*]]
// CHECK-LL:       [[v51:%.*]] = call i8* @__dynamic_cast(i8* null, i8* bitcast ({{.*}}* @_ZTI7test1_A to i8*), i8* bitcast ({{.*}} @_ZTI7test1_D to i8*), i64 -1)
// CHECK-LL-NEXT:  [[v52:%.*]] = bitcast i8* [[v51]] to %class.test1_D*
// CHECK-LL-NEXT:  br label %[[v54:.*]]
// CHECK-LL:       br label %[[v54]]
// CHECK-LL:       [[v55:%.*]] = phi %class.test1_D* [ [[v52]], %[[v50]] ], [ null, %[[v53]] ]
// CHECK-LL-NEXT:  store %class.test1_D* [[v55]], %class.test1_D** [[dp]]
// CHECK-LL-NEXT:  [[tmp86:%.*]] = load %class.test1_D** [[dp]]
// CHECK-LL-NEXT:  [[cmp87:%.*]] = icmp eq %class.test1_D* [[tmp86]], null
// CHECK-LL-NEXT:  br i1 [[cmp87]], label %[[ifthen88:.*]], label %[[ifelse90:.*]]
// CHECK-LL:       [[ifthen88]]
// CHECK-LL-NEXT:  call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([10 x i8]* @.str, i32 0, i32 0), i32 11)
// CHECK-LL-NEXT:  br label %[[ifend92:.*]]
// CHECK-LL:       [[ifelse90]]
// CHECK-LL-NEXT:  call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([10 x i8]* @.str1, i32 0, i32 0), i32 11)
// CHECK-LL-NEXT:  br label %[[ifend92]]
// CHECK-LL:       [[ifend92]]
// CHECK-LL-NEXT:  br i1 false, label %[[castnull98:.*]], label %[[castnotnull93:.*]]
// CHECK-LL:       [[castnotnull93]]
// CHECK-LL-NEXT:  [[vtable94:%.*]] = load i8** bitcast (%class.test1_D* @test1_d to i8**)
// CHECK-LL-NEXT:  [[vbaseoffsetptr95:%.*]] = getelementptr i8* [[vtable94]], i64 -24
// CHECK-LL-NEXT:  [[v56:%.*]] = bitcast i8* [[vbaseoffsetptr95]] to i64*
// CHECK-LL-NEXT:  [[vbaseoffset96:%.*]] = load i64* [[v56]]
// CHECK-LL-NEXT:  [[addptr97:%.*]] = getelementptr i8* getelementptr inbounds (%class.test1_D* @test1_d, i32 0, i32 0, i32 0), i64 [[vbaseoffset96]]
// CHECK-LL-NEXT:  [[v57:%.*]] = bitcast i8* [[addptr97]] to %class.test1_A*
// CHECK-LL-NEXT:  br label %[[castend99:.*]]
// CHECK-LL:       [[castnull98]]
// CHECK-LL-NEXT:  br label %[[castend99]]
// CHECK-LL:       [[castend99]]
// CHECK-LL-NEXT:  [[v58:%.*]] = phi %class.test1_A* [ [[v57]], %[[castnotnull93]] ], [ null, %[[castnull98]] ]
// CHECK-LL-NEXT:  store %class.test1_A* [[v58]], %class.test1_A** [[ap]]
// CHECK-LL-NEXT:  [[tmp100:%.*]] = load %class.test1_A** [[ap]]
// CHECK-LL-NEXT:  br i1 false, label %[[castnull106:.*]], label %[[castnotnull101:.*]]
// CHECK-LL:       [[castnotnull101]]
// CHECK-LL-NEXT:  [[vtable102:%.*]] = load i8** bitcast (%class.test1_D* @test1_d to i8**)
// CHECK-LL-NEXT:  [[vbaseoffsetptr103:%.*]] = getelementptr i8* [[vtable102]], i64 -24
// CHECK-LL-NEXT:  [[v59:%.*]] = bitcast i8* [[vbaseoffsetptr103]] to i64*
// CHECK-LL-NEXT:  [[vbaseoffset104:%.*]] = load i64* [[v59]]
// CHECK-LL-NEXT:  [[addptr105:%.*]] = getelementptr i8* getelementptr inbounds (%class.test1_D* @test1_d, i32 0, i32 0, i32 0), i64 [[vbaseoffset104]]
// CHECK-LL-NEXT:  [[v60:%.*]] = bitcast i8* [[addptr105]] to %class.test1_A*
// CHECK-LL-NEXT:  br label %[[castend107:.*]]
// CHECK-LL:       [[castnull106]]
// CHECK-LL-NEXT:  br label %[[castend107]]
// CHECK-LL:       [[castend107]]
// CHECK-LL-NEXT:  [[v61:%.*]] = phi %class.test1_A* [ [[v60]], %[[castnotnull101]] ], [ null, %[[castnull106]] ]
// CHECK-LL-NEXT:  [[cmp108:%.*]] = icmp eq %class.test1_A* [[tmp100]], [[v61]]
// CHECK-LL-NEXT:  br i1 [[cmp108]], label %[[ifthen109:.*]], label %[[ifelse111:.*]]
// CHECK-LL:       [[ifthen109]]
// CHECK-LL-NEXT:  call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([10 x i8]* @.str, i32 0, i32 0), i32 12)
// CHECK-LL-NEXT:  br label %[[ifend113:.*]]
// CHECK-LL:       [[ifelse111]]
// CHECK-LL-NEXT:  call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([10 x i8]* @.str1, i32 0, i32 0), i32 12)
// CHECK-LL-NEXT:  br label %[[ifend113]]
// CHECK-LL:       [[ifend113]]
// CHECK-LL-NEXT: store %class.test1_E* bitcast (%class.test1_F* @test1_f to %class.test1_E*), %class.test1_E** [[ep]]
// CHECK-LL-NEXT:  [[tmp118:%.*]] = load %class.test1_E** [[ep]]
// CHECK-LL-NEXT:  [[cmp122:%.*]] = icmp eq %class.test1_E* [[tmp118]], bitcast (%class.test1_F* @test1_f to %class.test1_E*) 

// CHECK-LL-NEXT:  br i1 [[cmp122]], label %[[ifthen123:.*]], label %[[ifelse125:.*]]
// CHECK-LL:       [[ifthen123]]
// CHECK-LL-NEXT:  call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([10 x i8]* @.str, i32 0, i32 0), i32 13)
// CHECK-LL-NEXT:  br label %[[ifend127:.*]]
// CHECK-LL:       [[ifelse125]]
// CHECK-LL-NEXT:  call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([10 x i8]* @.str1, i32 0, i32 0), i32 13)
// CHECK-LL-NEXT:  br label %[[ifend127]]
// CHECK-LL:       [[ifend127]]
// CHECK-LL-NEXT:  [[tmp129:%.*]] = load %class.test1_A** [[ap]]
// CHECK-LL-NEXT:  [[v64:%.*]] = icmp ne %class.test1_A* [[tmp129]], null
// CHECK-LL-NEXT:  br i1 [[v64]], label %[[v65:.*]], label %[[v70:.*]]
// CHECK-LL:       [[v66:%.*]] = bitcast %class.test1_A* [[tmp129]] to i64**
// CHECK-LL-NEXT:  [[vtable130:%.*]] = load i64** [[v66]]
// CHECK-LL-NEXT:  [[v67:%.*]] = getelementptr inbounds i64* [[vtable130]], i64 -2
// CHECK-LL-NEXT:  [[offsettotop:%.*]] = load i64* [[v67]]
// CHECK-LL-NEXT:  [[v68:%.*]] = bitcast %class.test1_A* [[tmp129]] to i8*
// CHECK-LL-NEXT:  [[v69:%.*]] = getelementptr inbounds i8* [[v68]], i64 [[offsettotop]]
// CHECK-LL-NEXT:  br label %[[v71:.*]]
// CHECK-LL:       br label %[[v71]]
// CHECK-LL:       [[v72:%.*]] = phi i8* [ [[v69]], %[[v65]] ], [ null, %[[v70]] ]
// CHECK-LL-NEXT:  store i8* [[v72]], i8** [[vp]]
// CHECK-LL-NEXT:  [[tmp131:%.*]] = load i8** [[vp]]
// CHECK-LL-NEXT:  [[cmp132:%.*]] = icmp eq i8* [[tmp131]], getelementptr inbounds (%class.test1_D* @test1_d, i32 0, i32 0, i32 0)
// CHECK-LL-NEXT:  br i1 [[cmp132]], label %[[ifthen133:.*]], label %[[ifelse135:.*]]
// CHECK-LL:       [[ifthen133]]
// CHECK-LL-NEXT:  call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([10 x i8]* @.str, i32 0, i32 0), i32 14)
// CHECK-LL-NEXT:  br label %[[ifend137:.*]]
// CHECK-LL:       [[ifelse135]]
// CHECK-LL-NEXT:  call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([10 x i8]* @.str1, i32 0, i32 0), i32 14)
// CHECK-LL-NEXT:  br label %[[ifend137]]
// CHECK-LL:       [[ifend137]]
// CHECK-LL-NEXT:  [[tmp139:%.*]] = load %class.test1_A** [[ap]]
// CHECK-LL-NEXT:  [[v73:%.*]] = icmp ne %class.test1_A* [[tmp139]], null
// CHECK-LL-NEXT:  br i1 [[v73]], label %[[v74:.*]], label %[[v79:.*]]
// CHECK-LL:       [[v75:%.*]] = bitcast %class.test1_A* [[tmp139]] to i64**
// CHECK-LL-NEXT:  [[vtable140:%.*]] = load i64** [[v75]]
// CHECK-LL-NEXT:  [[v76:%.*]] = getelementptr inbounds i64* [[vtable140]], i64 -2
// CHECK-LL-NEXT:  [[offsettotop141:%.*]] = load i64* [[v76]]
// CHECK-LL-NEXT:  [[v77:%.*]] = bitcast %class.test1_A* [[tmp139]] to i8*
// CHECK-LL-NEXT:  [[v78:%.*]] = getelementptr inbounds i8* [[v77]], i64 [[offsettotop141]]
// CHECK-LL-NEXT:  br label %[[v80:.*]]
// CHECK-LL:       br label %[[v80]]
// CHECK-LL:       [[v81:%.*]] = phi i8* [ [[v78]], %[[v74]] ], [ null, %[[v79]] ]
// CHECK-LL-NEXT:  store i8* [[v81]], i8** [[cvp]]
// CHECK-LL-NEXT:  [[tmp142:%.*]] = load i8** [[cvp]]
// CHECK-LL-NEXT:  [[cmp143:%.*]] = icmp eq i8* [[tmp142]], getelementptr inbounds (%class.test1_D* @test1_d, i32 0, i32 0, i32 0)
// CHECK-LL-NEXT:  br i1 [[cmp143]], label %[[ifthen144:.*]], label %[[ifelse146:.*]]
// CHECK-LL:       [[ifthen144]]
// CHECK-LL-NEXT:  call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([10 x i8]* @.str, i32 0, i32 0), i32 15)
// CHECK-LL-NEXT:  br label %[[ifend148:.*]]
// CHECK-LL:       [[ifelse146]]
// CHECK-LL-NEXT:  call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([10 x i8]* @.str1, i32 0, i32 0), i32 15)
// CHECK-LL-NEXT:  br label %[[ifend148]]
// CHECK-LL:       [[ifend148]]
// CHECK-LL-NEXT:  ret void
