// RUN: %clang_cc1 -triple i686-pc-linux-gnu -std=c++98 -emit-llvm -o - %s | FileCheck -check-prefix SANE98 %s
// RUN: %clang_cc1 -triple i686-pc-linux-gnu -std=c++11 -emit-llvm -o - %s | FileCheck -check-prefix SANE11 %s
// RUN: %clang_cc1 -triple i686-pc-win32-msvc -std=c++11 -emit-llvm -o - %s | FileCheck -check-prefix SANE11MS %s
// RUN: %clang_cc1 -triple i686-pc-linux-gnu -emit-llvm -fno-assume-sane-operator-new -o - %s | FileCheck -check-prefix SANENOT %s

class teste {
  int A;
public:
  teste() : A(2) {}
};

void f1() {
  // SANE: declare noalias i8* @_Znwj(
  // SANENOT: declare i8* @_Znwj(
  new teste();
}


// rdar://5739832 - operator new should check for overflow in multiply.
void *f2(long N) {
  return new int[N];

// SANE98:      [[UWO:%.*]] = call {{.*}} @llvm.umul.with.overflow
// SANE98-NEXT: [[OVER:%.*]] = extractvalue {{.*}} [[UWO]], 1
// SANE98-NEXT: [[SUM:%.*]] = extractvalue {{.*}} [[UWO]], 0
// SANE98-NEXT: [[RESULT:%.*]] = select i1 [[OVER]], i32 -1, i32 [[SUM]]
// SANE98-NEXT: call noalias i8* @_Znaj(i32 [[RESULT]])

// SANE11:      [[UWO:%.*]] = call {{.*}} @llvm.umul.with.overflow
// SANE11-NEXT: [[OVER:%.*]] = extractvalue {{.*}} [[UWO]], 1
// SANE11-NEXT: [[SUM:%.*]] = extractvalue {{.*}} [[UWO]], 0
// SANE11-NEXT: br i1 [[OVER]], label %[[BAD:.*]], label %[[GOOD:.*]]
// SANE11: [[BAD]]:
// SANE11-NEXT: call void @__cxa_bad_array_new_length()
// SANE11-NEXT: unreachable
// SANE11: [[GOOD]]:
// SANE11-NEXT: call noalias i8* @_Znaj(i32 [[SUM]])

// FIXME: There should be a call to generate the std::bad_array_new_length
// exception in the Microsoft ABI, however, this is not implemented currently.
// SANE11MS:      [[UWO:%.*]] = call {{.*}} @llvm.umul.with.overflow
// SANE11MS-NEXT: [[OVER:%.*]] = extractvalue {{.*}} [[UWO]], 1
// SANE11MS-NEXT: [[SUM:%.*]] = extractvalue {{.*}} [[UWO]], 0
// SANE11MS-NEXT: [[RESULT:%.*]] = select i1 [[OVER]], i32 -1, i32 [[SUM]]
// SANE11MS-NEXT: call noalias i8* @"\01??_U@YAPAXI@Z"(i32 [[RESULT]])
}

#if __cplusplus > 199711L
void *f3() {
  return new int[0x7FFFFFFF];
// SANE11: br label %[[BAD:.*]]
// SANE11: [[BAD]]:
// SANE11-NEXT: call void @__cxa_bad_array_new_length()
// SANE11-NEXT: unreachable
// SANE11: {{.*}}:
// SANE11-NEXT: call noalias i8* @_Znaj(i32 -1)

// FIXME: There should be a call to generate the std::bad_array_new_length
// exception in the Microsoft ABI, however, this is not implemented currently.
// SANE11MS: call noalias i8* @"\01??_U@YAPAXI@Z"(i32 -1)
}
#endif
