// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=51 \
// RUN:   -fsyntax-only -verify %s

// expected-no-diagnostics

// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=51 \
// RUN:   -ast-print %s | FileCheck %s --check-prefix=PRINT

// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=51 \
// RUN:   -ast-dump  %s | FileCheck %s --check-prefix=DUMP

// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=51 \
// RUN:   -emit-pch -o %t %s

// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=51 \
// RUN:   -include-pch %t -ast-dump-all %s | FileCheck %s --check-prefix=DUMP

// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=51 \
// RUN:   -include-pch %t -ast-print %s | FileCheck %s --check-prefix=PRINT

#ifndef HEADER
#define HEADER

//PRINT: template <typename T, int C> void templ_foo(T t) {
//PRINT:   T j, z;
//PRINT:   #pragma omp loop collapse(C) reduction(+: z) lastprivate(j) bind(thread)
//PRINT:   for (T i = 0; i < t; ++i)
//PRINT:       for (j = 0; j < t; ++j)
//PRINT:           z += i + j;
//PRINT: }
//DUMP: FunctionTemplateDecl{{.*}}templ_foo
//DUMP: TemplateTypeParmDecl{{.*}}T
//DUMP: NonTypeTemplateParmDecl{{.*}}C
//DUMP: OMPGenericLoopDirective
//DUMP: OMPCollapseClause
//DUMP: DeclRefExpr{{.*}}'C' 'int'
//DUMP: OMPReductionClause
//DUMP: DeclRefExpr{{.*}}'z' 'T'
//DUMP: OMPLastprivateClause
//DUMP: DeclRefExpr{{.*}}'j' 'T'
//DUMP: OMPBindClause
//DUMP: ForStmt
//DUMP: ForStmt

//PRINT: template<> void templ_foo<int, 2>(int t) {
//PRINT:     int j, z;
//PRINT:     #pragma omp loop collapse(2) reduction(+: z) lastprivate(j) bind(thread)
//PRINT:         for (int i = 0; i < t; ++i)
//PRINT:             for (j = 0; j < t; ++j)
//PRINT:                 z += i + j;
//PRINT: }
//DUMP: FunctionDecl{{.*}}templ_foo 'void (int)'
//DUMP: TemplateArgument type 'int'
//DUMP: TemplateArgument integral 2
//DUMP: ParmVarDecl{{.*}}'int':'int'
//DUMP: OMPGenericLoopDirective
//DUMP: OMPCollapseClause
//DUMP: ConstantExpr{{.*}}'int'
//DUMP: value: Int 2
//DUMP: OMPReductionClause
//DUMP: DeclRefExpr{{.*}}'z' 'int':'int'
//DUMP: OMPLastprivateClause
//DUMP: DeclRefExpr{{.*}}'j' 'int':'int'
//DUMP: OMPBindClause
//DUMP: ForStmt
template <typename T, int C>
void templ_foo(T t) {

  T j,z;
  #pragma omp loop collapse(C) reduction(+:z) lastprivate(j) bind(thread)
  for (T i = 0; i<t; ++i)
    for (j = 0; j<t; ++j)
      z += i+j;
}


//PRINT: void test() {
//DUMP: FunctionDecl {{.*}}test 'void ()'
void test() {
  constexpr int N = 100;
  float MTX[N][N];
  int aaa[1000];

  //PRINT: #pragma omp target teams distribute parallel for map(tofrom: MTX)
  //PRINT: #pragma omp loop
  //DUMP: OMPTargetTeamsDistributeParallelForDirective
  //DUMP: CapturedStmt
  //DUMP: ForStmt
  //DUMP: CompoundStmt
  //DUMP: OMPGenericLoopDirective
  #pragma omp target teams distribute parallel for map(MTX)
  for (auto i = 0; i < N; ++i) {
    #pragma omp loop
    for (auto j = 0; j < N; ++j) {
      MTX[i][j] = 0;
    }
  }

  //PRINT: #pragma omp target teams
  //PRINT: #pragma omp loop
  //DUMP: OMPTargetTeamsDirective
  //DUMP: CapturedStmt
  //DUMP: ForStmt
  //DUMP: OMPGenericLoopDirective
  #pragma omp target teams
  for (int i=0; i<1000; ++i) {
    #pragma omp loop
    for (int j=0; j<100; j++) {
      aaa[i] += i + j;
    }
  }

  int j, z, z1;
  //PRINT: #pragma omp loop collapse(2) private(z) lastprivate(j) order(concurrent) reduction(+: z1) bind(parallel)
  //DUMP: OMPGenericLoopDirective
  //DUMP: OMPCollapseClause
  //DUMP: IntegerLiteral{{.*}}2
  //DUMP: OMPPrivateClause
  //DUMP-NEXT: DeclRefExpr{{.*}}'z'
  //DUMP: OMPLastprivateClause
  //DUMP-NEXT: DeclRefExpr{{.*}}'j'
  //DUMP: OMPOrderClause
  //DUMP: OMPReductionClause
  //DUMP-NEXT: DeclRefExpr{{.*}}'z1'
  //DUMP: OMPBindClause
  //DUMP: ForStmt
  //DUMP: ForStmt
  #pragma omp loop collapse(2) private(z) lastprivate(j) order(concurrent) \
                   reduction(+:z1) bind(parallel)
  for (auto i = 0; i < N; ++i) {
    for (j = 0; j < N; ++j) {
      z = i+j;
      MTX[i][j] = z;
      z1 += z;
    }
  }

  //PRINT: #pragma omp target teams
  //PRINT: #pragma omp loop bind(teams)
  //DUMP: OMPTargetTeamsDirective
  //DUMP: OMPGenericLoopDirective
  //DUMP: OMPBindClause
  //DUMP: ForStmt
  #pragma omp target teams
  #pragma omp loop bind(teams)
  for (auto i = 0; i < N; ++i) { }

  //PRINT: #pragma omp target
  //PRINT: #pragma omp teams
  //PRINT: #pragma omp loop bind(teams)
  //DUMP: OMPTargetDirective
  //DUMP: OMPTeamsDirective
  //DUMP: OMPGenericLoopDirective
  //DUMP: OMPBindClause
  //DUMP: ForStmt
  #pragma omp target
  #pragma omp teams
  #pragma omp loop bind(teams)
  for (auto i = 0; i < N; ++i) { }
}

//PRINT: void nobindingfunc() {
//DUMP: FunctionDecl {{.*}}nobindingfunc 'void ()'
void nobindingfunc()
{
  //PRINT: #pragma omp loop
  //DUMP: OMPGenericLoopDirective
  //DUMP: ForStmt
  #pragma omp loop
  for (int i=0; i<10; ++i) { }
}

void bar()
{
  templ_foo<int,2>(8);
}

#endif // HEADER
