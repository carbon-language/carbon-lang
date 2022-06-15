// Test without serialization:
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -Wno-strict-prototypes -ast-dump -ast-dump-filter Test %s \
// RUN: | FileCheck --strict-whitespace %s
//
// Test with serialization:
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -Wno-strict-prototypes -emit-pch -o %t %s
// RUN: %clang_cc1 -x c -triple x86_64-unknown-unknown -Wno-strict-prototypes -include-pch %t \
// RUN: -ast-dump-all -ast-dump-filter Test /dev/null \
// RUN: | sed -e "s/ <undeserialized declarations>//" -e "s/ imported//" \
// RUN: | FileCheck --strict-whitespace %s
//
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -Wno-strict-prototypes -ast-dump %s \
// RUN: | FileCheck -check-prefix CHECK-TU --strict-whitespace %s
//
// RUN: %clang_cc1 -fmodules -fmodules-local-submodule-visibility -fmodule-name=X \
// RUN: -triple x86_64-unknown-unknown -Wno-strict-prototypes -fmodule-map-file=%S/Inputs/module.modulemap \
// RUN: -ast-dump -ast-dump-filter Test %s -DMODULES \
// RUN: | FileCheck -check-prefix CHECK -check-prefix CHECK-MODULES --strict-whitespace %s

int TestLocation;
// CHECK: VarDecl 0x{{[^ ]*}} <{{.*}}:[[@LINE-1]]:1, col:5> col:5 TestLocation

#ifdef MODULES
#pragma clang module begin X
#endif

struct TestIndent {
  int x;
};
// CHECK:      {{^}}RecordDecl{{.*TestIndent[^()]*$}}
// CHECK-NEXT: {{^}}`-FieldDecl{{.*x[^()]*$}}

struct TestChildren {
  int x;
  struct y {
    int z;
  };
};
// CHECK:      RecordDecl{{.*}}TestChildren
// CHECK-NEXT:   FieldDecl{{.*}}x
// CHECK-NEXT:   RecordDecl{{.*}}y
// CHECK-NEXT:     FieldDecl{{.*}}z

// CHECK-TU: TranslationUnitDecl

void testLabelDecl(void) {
  __label__ TestLabelDecl;
  TestLabelDecl: goto TestLabelDecl;
}
// CHECK:      LabelDecl{{.*}} TestLabelDecl

typedef int TestTypedefDecl;
// CHECK:      TypedefDecl{{.*}} TestTypedefDecl 'int'

__module_private__ typedef int TestTypedefDeclPrivate;
// CHECK-MODULES:      TypedefDecl{{.*}} TestTypedefDeclPrivate 'int' __module_private__

enum TestEnumDecl {
  testEnumDecl
};
// CHECK:      EnumDecl{{.*}} TestEnumDecl
// CHECK-NEXT:   EnumConstantDecl{{.*}} testEnumDecl

struct TestEnumDeclAnon {
  enum {
    testEnumDeclAnon
  } e;
};
// CHECK:      RecordDecl{{.*}} TestEnumDeclAnon
// CHECK-NEXT:   EnumDecl{{.*> .*$}}

enum TestEnumDeclForward;
// CHECK:      EnumDecl{{.*}} TestEnumDeclForward

__module_private__ enum TestEnumDeclPrivate;
// CHECK-MODULE:      EnumDecl{{.*}} TestEnumDeclPrivate __module_private__

struct TestRecordDecl {
  int i;
};
// CHECK:      RecordDecl{{.*}} struct TestRecordDecl
// CHECK-NEXT:   FieldDecl

struct TestRecordDeclEmpty {
};
// CHECK:      RecordDecl{{.*}} struct TestRecordDeclEmpty

struct TestRecordDeclAnon1 {
  struct {
  } testRecordDeclAnon1;
};
// CHECK:      RecordDecl{{.*}} struct TestRecordDeclAnon1
// CHECK-NEXT:   RecordDecl{{.*}} struct

struct TestRecordDeclAnon2 {
  struct {
  };
};
// CHECK:      RecordDecl{{.*}} struct TestRecordDeclAnon2
// CHECK-NEXT:   RecordDecl{{.*}} struct

struct TestRecordDeclForward;
// CHECK:      RecordDecl{{.*}} struct TestRecordDeclForward

__module_private__ struct TestRecordDeclPrivate;
// CHECK-MODULE:      RecordDecl{{.*}} struct TestRecordDeclPrivate __module_private__

enum testEnumConstantDecl {
  TestEnumConstantDecl,
  TestEnumConstantDeclInit = 1
};
// CHECK:      EnumConstantDecl{{.*}} TestEnumConstantDecl 'int'
// CHECK:      EnumConstantDecl{{.*}} TestEnumConstantDeclInit 'int'
// CHECK-NEXT:   ConstantExpr
// CHECK-NEXT:     value: Int 1
// CHECK-NEXT:       IntegerLiteral

struct testIndirectFieldDecl {
  struct {
    int TestIndirectFieldDecl;
  };
};
// CHECK:      IndirectFieldDecl{{.*}} TestIndirectFieldDecl 'int'
// CHECK-NEXT:   Field{{.*}} ''
// CHECK-NEXT:   Field{{.*}} 'TestIndirectFieldDecl'

// FIXME: It would be nice to dump the enum and its enumerators.
int TestFunctionDecl(int x, enum { e } y) {
  return x;
}
// CHECK:      FunctionDecl{{.*}} TestFunctionDecl 'int (int, enum {{.*}})'
// CHECK-NEXT:   ParmVarDecl{{.*}} x
// CHECK-NEXT:   ParmVarDecl{{.*}} y
// CHECK-NEXT:   CompoundStmt

// FIXME: It would be nice to 'Enum' and 'e'.
int TestFunctionDecl2(enum Enum { e } x) { return x; }
// CHECK:      FunctionDecl{{.*}} TestFunctionDecl2 'int (enum {{.*}})'
// CHECK-NEXT:   ParmVarDecl{{.*}} x
// CHECK-NEXT:   CompoundStmt


int TestFunctionDeclProto(int x);
// CHECK:      FunctionDecl{{.*}} TestFunctionDeclProto 'int (int)'
// CHECK-NEXT:   ParmVarDecl{{.*}} x

void TestFunctionDeclNoProto();
// CHECK:      FunctionDecl{{.*}} TestFunctionDeclNoProto 'void ()'

extern int TestFunctionDeclSC(void);
// CHECK:      FunctionDecl{{.*}} TestFunctionDeclSC 'int (void)' extern

inline int TestFunctionDeclInline(void);
// CHECK:      FunctionDecl{{.*}} TestFunctionDeclInline 'int (void)' inline

struct testFieldDecl {
  int TestFieldDecl;
  int TestFieldDeclWidth : 1;
  __module_private__ int TestFieldDeclPrivate;
};
// CHECK:      FieldDecl{{.*}} TestFieldDecl 'int'
// CHECK:      FieldDecl{{.*}} TestFieldDeclWidth 'int'
// CHECK-NEXT:   ConstantExpr
// CHECK-NEXT:     value: Int 1
// CHECK-NEXT:       IntegerLiteral
// CHECK-MODULE:      FieldDecl{{.*}} TestFieldDeclPrivate 'int' __module_private__

int TestVarDecl;
// CHECK:      VarDecl{{.*}} TestVarDecl 'int'

extern int TestVarDeclSC;
// CHECK:      VarDecl{{.*}} TestVarDeclSC 'int' extern

__thread int TestVarDeclThread;
// CHECK:      VarDecl{{.*}} TestVarDeclThread 'int' tls{{$}}

__module_private__ int TestVarDeclPrivate;
// CHECK-MODULE:      VarDecl{{.*}} TestVarDeclPrivate 'int' __module_private__

int TestVarDeclInit = 0;
// CHECK:      VarDecl{{.*}} TestVarDeclInit 'int'
// CHECK-NEXT:   IntegerLiteral

void testParmVarDecl(int TestParmVarDecl);
// CHECK: ParmVarDecl{{.*}} TestParmVarDecl 'int'

#ifdef MODULES
#pragma clang module end
#endif

