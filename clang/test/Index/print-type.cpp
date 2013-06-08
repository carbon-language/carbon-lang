namespace outer {

template<typename T>
struct Foo {
  T t;
};

namespace inner {

struct Bar {
  Bar(outer::Foo<bool>* foo) { };

  typedef int FooType;
  int *p;
  int *f(int *p, char *x, FooType z) {
    const FooType w = z;
    return p + z;
  }
  typedef double OtherType;
  typedef int ArrayType[5];
};

}
}

template <typename T>
T tbar(int);

template <typename T>
T tbar(int[5]);

// RUN: c-index-test -test-print-type %s | FileCheck %s
// CHECK: Namespace=outer:1:11 (Definition) [type=] [typekind=Invalid] [isPOD=0]
// CHECK: ClassTemplate=Foo:4:8 (Definition) [type=] [typekind=Invalid] [isPOD=0]
// CHECK: TemplateTypeParameter=T:3:19 (Definition) [type=T] [typekind=Unexposed] [canonicaltype=type-parameter-0-0] [canonicaltypekind=Unexposed] [isPOD=0]
// CHECK: FieldDecl=t:5:5 (Definition) [type=T] [typekind=Unexposed] [canonicaltype=type-parameter-0-0] [canonicaltypekind=Unexposed] [isPOD=0]
// CHECK: TypeRef=T:3:19 [type=T] [typekind=Unexposed] [canonicaltype=type-parameter-0-0] [canonicaltypekind=Unexposed] [isPOD=0]
// CHECK: Namespace=inner:8:11 (Definition) [type=] [typekind=Invalid] [isPOD=0]
// CHECK: StructDecl=Bar:10:8 (Definition) [type=outer::inner::Bar] [typekind=Record] [isPOD=0]
// CHECK: CXXConstructor=Bar:11:3 (Definition) [type=void (outer::Foo<bool> *)] [typekind=FunctionProto] [canonicaltype=void (outer::Foo<bool> *)] [canonicaltypekind=FunctionProto] [resulttype=void] [resulttypekind=Void] [args= [outer::Foo<bool> *] [Pointer]] [isPOD=0]
// CHECK: ParmDecl=foo:11:25 (Definition) [type=outer::Foo<bool> *] [typekind=Pointer] [canonicaltype=outer::Foo<bool> *] [canonicaltypekind=Pointer] [isPOD=1]
// CHECK: NamespaceRef=outer:1:11 [type=] [typekind=Invalid] [isPOD=0]
// CHECK: TemplateRef=Foo:4:8 [type=] [typekind=Invalid] [isPOD=0]
// CHECK: CompoundStmt= [type=] [typekind=Invalid] [isPOD=0]
// CHECK: TypedefDecl=FooType:13:15 (Definition) [type=FooType] [typekind=Typedef] [canonicaltype=int] [canonicaltypekind=Int] [isPOD=1]
// CHECK: FieldDecl=p:14:8 (Definition) [type=int *] [typekind=Pointer] [isPOD=1]
// CHECK: CXXMethod=f:15:8 (Definition) [type=int *(int *, char *, FooType)] [typekind=FunctionProto] [canonicaltype=int *(int *, char *, int)] [canonicaltypekind=FunctionProto] [resulttype=int *] [resulttypekind=Pointer] [args= [int *] [Pointer] [char *] [Pointer] [FooType] [Typedef]] [isPOD=0]
// CHECK: ParmDecl=p:15:15 (Definition) [type=int *] [typekind=Pointer] [isPOD=1]
// CHECK: ParmDecl=x:15:24 (Definition) [type=char *] [typekind=Pointer] [isPOD=1]
// CHECK: ParmDecl=z:15:35 (Definition) [type=FooType] [typekind=Typedef] [canonicaltype=int] [canonicaltypekind=Int] [isPOD=1]
// CHECK: TypeRef=FooType:13:15 [type=FooType] [typekind=Typedef] [canonicaltype=int] [canonicaltypekind=Int] [isPOD=1]
// CHECK: CompoundStmt= [type=] [typekind=Invalid] [isPOD=0]
// CHECK: DeclStmt= [type=] [typekind=Invalid] [isPOD=0]
// CHECK: VarDecl=w:16:19 (Definition) [type=const FooType] [typekind=Typedef] const [canonicaltype=const int] [canonicaltypekind=Int] [isPOD=1]
// CHECK: TypeRef=FooType:13:15 [type=FooType] [typekind=Typedef] [canonicaltype=int] [canonicaltypekind=Int] [isPOD=1]
// CHECK: DeclRefExpr=z:15:35 [type=FooType] [typekind=Typedef] [canonicaltype=int] [canonicaltypekind=Int] [isPOD=1]
// CHECK: ReturnStmt= [type=] [typekind=Invalid] [isPOD=0]
// CHECK: BinaryOperator= [type=int *] [typekind=Pointer] [isPOD=1]
// CHECK: DeclRefExpr=p:15:15 [type=int *] [typekind=Pointer] [isPOD=1]
// CHECK: DeclRefExpr=z:15:35 [type=FooType] [typekind=Typedef] [canonicaltype=int] [canonicaltypekind=Int] [isPOD=1]
// CHECK: TypedefDecl=OtherType:19:18 (Definition) [type=OtherType] [typekind=Typedef] [canonicaltype=double] [canonicaltypekind=Double] [isPOD=1]
// CHECK: TypedefDecl=ArrayType:20:15 (Definition) [type=ArrayType] [typekind=Typedef] [canonicaltype=int [5]] [canonicaltypekind=ConstantArray] [isPOD=1]
// CHECK: FunctionTemplate=tbar:27:3 [type=T (int)] [typekind=FunctionProto] [canonicaltype=type-parameter-0-0 (int)] [canonicaltypekind=FunctionProto] [resulttype=T] [resulttypekind=Unexposed] [isPOD=0]
// CHECK: TemplateTypeParameter=T:26:20 (Definition) [type=T] [typekind=Unexposed] [canonicaltype=type-parameter-0-0] [canonicaltypekind=Unexposed] [isPOD=0]
// CHECK: FunctionTemplate=tbar:30:3 [type=T (int [5])] [typekind=FunctionProto] [canonicaltype=type-parameter-0-0 (int *)] [canonicaltypekind=FunctionProto] [resulttype=T] [resulttypekind=Unexposed] [isPOD=0]
// CHECK: ParmDecl=:30:11 (Definition) [type=int [5]] [typekind=ConstantArray] [isPOD=1]
