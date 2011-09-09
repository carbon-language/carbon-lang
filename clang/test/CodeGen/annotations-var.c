// RUN: %clang_cc1 -emit-llvm -o %t1 %s
// RUN: FileCheck --check-prefix=LOCAL %s < %t1
// RUN: FileCheck --check-prefix=UNDEF %s < %t1
// RUN: FileCheck --check-prefix=PARAM %s < %t1
// END.

// LOCAL: private unnamed_addr constant [15 x i8] c"localvar_ann_{{.}}\00", section "llvm.metadata"
// LOCAL: private unnamed_addr constant [15 x i8] c"localvar_ann_{{.}}\00", section "llvm.metadata"

// UNDEF: private unnamed_addr constant [15 x i8] c"undefvar_ann_0\00", section "llvm.metadata"

// PARAM: private unnamed_addr constant [12 x i8] c"param_ann_{{.}}\00", section "llvm.metadata"
// PARAM: private unnamed_addr constant [12 x i8] c"param_ann_{{.}}\00", section "llvm.metadata"
// PARAM: private unnamed_addr constant [12 x i8] c"param_ann_{{.}}\00", section "llvm.metadata"
// PARAM: private unnamed_addr constant [12 x i8] c"param_ann_{{.}}\00", section "llvm.metadata"

int foo(int v __attribute__((annotate("param_ann_2"))) __attribute__((annotate("param_ann_3"))));
int foo(int v __attribute__((annotate("param_ann_0"))) __attribute__((annotate("param_ann_1")))) {
    return v + 1;
// PARAM: define {{.*}}@foo
// PARAM:  bitcast i32* %v.addr to i8*
// PARAM-NEXT: call void @llvm.var.annotation(
// PARAM-NEXT: bitcast i32* %v.addr to i8*
// PARAM-NEXT: call void @llvm.var.annotation(
// PARAM-NEXT: bitcast i32* %v.addr to i8*
// PARAM-NEXT: call void @llvm.var.annotation(
// PARAM-NEXT: bitcast i32* %v.addr to i8*
// PARAM-NEXT: call void @llvm.var.annotation(
}

int main(int argc, char **argv) {
    int localvar __attribute__((annotate("localvar_ann_0"))) __attribute__((annotate("localvar_ann_1"))) = 3;
// LOCAL: %localvar1 = bitcast i32* %localvar to i8*
// LOCAL-NEXT: call void @llvm.var.annotation(i8* %localvar1, i8* getelementptr inbounds ([15 x i8]* @{{.*}}), i8* getelementptr inbounds ({{.*}}), i32 32)
// LOCAL-NEXT: %localvar2 = bitcast i32* %localvar to i8*
// LOCAL-NEXT: call void @llvm.var.annotation(i8* %localvar2, i8* getelementptr inbounds ([15 x i8]* @{{.*}}), i8* getelementptr inbounds ({{.*}}), i32 32)
    int undefvar __attribute__((annotate("undefvar_ann_0")));
// UNDEF: %undefvar3 = bitcast i32* %undefvar to i8*
// UNDEF-NEXT: call void @llvm.var.annotation(i8* %undefvar3, i8* getelementptr inbounds ([15 x i8]* @{{.*}}), i8* getelementptr inbounds ({{.*}}), i32 37)
    localvar += argc;
    undefvar = localvar;
    return undefvar + localvar;
}
