// RUN: %clang_cc1 -triple x86_64-apple-darwin10.0.0 -emit-llvm -o - %s -fexceptions -std=c++11 -g | FileCheck %s

auto var = [](int i) { return i+1; };
void *use = &var;

extern "C" auto cvar = []{};

int a() { return []{ return 1; }(); }

int b(int x) { return [x]{return x;}(); }

int c(int x) { return [&x]{return x;}(); }

struct D { D(); D(const D&); int x; };
int d(int x) { D y[10]; return [x,y] { return y[x].x; }(); }

// Randomness for file. -- 6
// CHECK: [[FILE:.*]] = !MDFile(filename: "{{.*}}debug-lambda-expressions.cpp",

// CHECK: ![[INT:[0-9]+]] = !MDBasicType(name: "int"

// A: 10
// CHECK: ![[A_FUNC:.*]] = !MDSubprogram(name: "a"{{.*}}, line: [[A_LINE:[0-9]+]]{{.*}}, isDefinition: true

// B: 14
// CHECK: ![[B_FUNC:.*]] = !MDSubprogram(name: "b"{{.*}}, line: [[B_LINE:[0-9]+]]{{.*}}, isDefinition: true

// C: 17
// CHECK: ![[C_FUNC:.*]] = !MDSubprogram(name: "c"{{.*}}, line: [[C_LINE:[0-9]+]]{{.*}}, isDefinition: true

// D: 18
// CHECK: ![[D_FUNC:.*]] = !MDSubprogram(name: "d"{{.*}}, line: [[D_LINE:[0-9]+]]{{.*}}, isDefinition: true


// Back to A. -- 78
// CHECK: ![[LAM_A:.*]] = !MDCompositeType(tag: DW_TAG_class_type{{.*}}, scope: ![[A_FUNC]]{{.*}}, line: [[A_LINE]],
// CHECK-SAME:                             elements: ![[LAM_A_ARGS:[0-9]+]]
// CHECK: ![[LAM_A_ARGS]] = !{![[CON_LAM_A:[0-9]+]]}
// CHECK: ![[CON_LAM_A]] = !MDSubprogram(name: "operator()"
// CHECK-SAME:                           scope: ![[LAM_A]]
// CHECK-SAME:                           line: [[A_LINE]]
// CHECK-SAME:                           DIFlagPublic

// Back to B. -- 67
// CHECK: ![[LAM_B:.*]] = !MDCompositeType(tag: DW_TAG_class_type{{.*}}, scope: ![[B_FUNC]]{{.*}}, line: [[B_LINE]],
// CHECK-SAME:                             elements: ![[LAM_B_ARGS:[0-9]+]]
// CHECK: ![[LAM_B_ARGS]] = !{![[CAP_B:[0-9]+]], ![[CON_LAM_B:[0-9]+]]}
// CHECK: ![[CAP_B]] = !MDDerivedType(tag: DW_TAG_member, name: "x"
// CHECK-SAME:                        scope: ![[LAM_B]]
// CHECK-SAME:                        line: [[B_LINE]],
// CHECK-SAME:                        baseType: ![[INT]]
// CHECK: ![[CON_LAM_B]] = !MDSubprogram(name: "operator()"
// CHECK-SAME:                           scope: ![[LAM_B]]
// CHECK-SAME:                           line: [[B_LINE]]
// CHECK-SAME:                           DIFlagPublic

// Back to C. -- 55
// CHECK: ![[LAM_C:.*]] = !MDCompositeType(tag: DW_TAG_class_type{{.*}}, scope: ![[C_FUNC]]{{.*}}, line: [[C_LINE]],
// CHECK-SAME:                             elements: ![[LAM_C_ARGS:[0-9]+]]
// CHECK: ![[LAM_C_ARGS]] = !{![[CAP_C:[0-9]+]], ![[CON_LAM_C:[0-9]+]]}
// CHECK: ![[CAP_C]] = !MDDerivedType(tag: DW_TAG_member, name: "x"
// CHECK-SAME:                        scope: ![[LAM_C]]
// CHECK-SAME:                        line: [[C_LINE]],
// CHECK-SAME:                        baseType: ![[TYPE_C_x:[0-9]+]]
// CHECK: ![[TYPE_C_x]] = !MDDerivedType(tag: DW_TAG_reference_type, baseType: ![[INT]]
// CHECK: ![[CON_LAM_C]] = !MDSubprogram(name: "operator()"
// CHECK-SAME:                           scope: ![[LAM_C]]
// CHECK-SAME:                           line: [[C_LINE]]
// CHECK-SAME:                           DIFlagPublic

// Back to D. -- 24
// CHECK: ![[LAM_D:.*]] = !MDCompositeType(tag: DW_TAG_class_type{{.*}}, scope: ![[D_FUNC]]{{.*}}, line: [[D_LINE]],
// CHECK-SAME:                             elements: ![[LAM_D_ARGS:[0-9]+]]
// CHECK: ![[LAM_D_ARGS]] = !{![[CAP_D_X:[0-9]+]], ![[CAP_D_Y:[0-9]+]], ![[CON_LAM_D:[0-9]+]]}
// CHECK: ![[CAP_D_X]] = !MDDerivedType(tag: DW_TAG_member, name: "x"
// CHECK-SAME:                          scope: ![[LAM_D]]
// CHECK-SAME:                          line: [[D_LINE]],
// CHECK: ![[CAP_D_Y]] = !MDDerivedType(tag: DW_TAG_member, name: "y"
// CHECK-SAME:                          scope: ![[LAM_D]]
// CHECK-SAME:                          line: [[D_LINE]],
// CHECK: ![[CON_LAM_D]] = !MDSubprogram(name: "operator()"
// CHECK-SAME:                           scope: ![[LAM_D]]
// CHECK-SAME:                           line: [[D_LINE]]
// CHECK-SAME:                           DIFlagPublic

// CVAR:
// CHECK: !MDGlobalVariable(name: "cvar"
// CHECK-SAME:              line: [[CVAR_LINE:[0-9]+]]
// CHECK-SAME:              type: ![[CVAR_T:[0-9]+]]
// CHECK: ![[CVAR_T]] = !MDCompositeType(tag: DW_TAG_class_type
// CHECK-SAME:                           line: [[CVAR_LINE]],
// CHECK-SAME:                           elements: ![[CVAR_ARGS:[0-9]+]]
// CHECK: ![[CVAR_ARGS]] = !{!{{[0-9]+}}}

// VAR:
// CHECK: !MDGlobalVariable(name: "var"
// CHECK-SAME:              line: [[VAR_LINE:[0-9]+]]
// CHECK-SAME:              type: ![[VAR_T:[0-9]+]]
// CHECK: ![[VAR_T]] = !MDCompositeType(tag: DW_TAG_class_type
// CHECK-SAME:                          line: [[VAR_LINE]],
// CHECK-SAME:                          elements: ![[VAR_ARGS:[0-9]+]]
// CHECK: ![[VAR_ARGS]] = !{!{{[0-9]+}}}
