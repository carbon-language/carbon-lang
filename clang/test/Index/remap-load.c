// RUN: c-index-test -test-load-source all -remap-file="%s;%S/Inputs/remap-load-to.c" %s | FileCheck -check-prefix=CHECK %s
// RUN: CINDEXTEST_USE_EXTERNAL_AST_GENERATION=1 c-index-test -test-load-source all -remap-file="%s;%S/Inputs/remap-load-to.c" %s | FileCheck -check-prefix=CHECK %s

// CHECK: remap-load.c:1:5: FunctionDecl=foo:1:5 (Definition) [Extent=1:5:3:1]
// CHECK: remap-load.c:1:13: ParmDecl=parm1:1:13 (Definition) [Extent=1:9:1:17]
// CHECK: remap-load.c:1:26: ParmDecl=parm2:1:26 (Definition) [Extent=1:20:1:30]
// CHECK: remap-load.c:1:5: UnexposedStmt=foo [Extent=1:33:3:1]
// CHECK: remap-load.c:1:5: UnexposedStmt=foo [Extent=2:3:2:22]
// CHECK: remap-load.c:2:10: UnexposedExpr= [Extent=2:10:2:22]
// CHECK: remap-load.c:2:10: UnexposedExpr= [Extent=2:10:2:22]
// CHECK: remap-load.c:2:10: UnexposedExpr=parm1:1:13 [Extent=2:10:2:14]
// CHECK: remap-load.c:2:10: DeclRefExpr=parm1:1:13 [Extent=2:10:2:14]
// CHECK: remap-load.c:2:18: DeclRefExpr=parm2:1:26 [Extent=2:18:2:22]
