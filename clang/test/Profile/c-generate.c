// Check that the -fprofile-instr-generate= form works.
// RUN: %clang_cc1 -main-file-name c-generate.c %s -o - -emit-llvm -fprofile-instr-generate=c-generate-test.profraw | FileCheck %s

// CHECK: private constant [24 x i8] c"c-generate-test.profraw\00"
// CHECK: call void @__llvm_profile_override_default_filename(i8* getelementptr inbounds ([24 x i8], [24 x i8]* @0, i32 0, i32 0))
// CHECK: declare void @__llvm_profile_override_default_filename(i8*)

int main(void) {
  return 0;
}
