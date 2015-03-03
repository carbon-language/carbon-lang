// RUN: %clang -S -emit-llvm -target x86_64-unknown_unknown -g %s -o - -std=c++11 | FileCheck %s

// CHECK: !MDCompileUnit(
// CHECK-SAME:           retainedTypes: [[RETAIN:![0-9]*]]
// CHECK: [[EMPTY:![0-9]*]] = !{}
// CHECK: [[RETAIN]] = !{!{{[0-9]]*}}, [[FOO:![0-9]*]],


// CHECK: [[TC:![0-9]*]] = !MDCompositeType(tag: DW_TAG_structure_type, name: "TC<unsigned int, 2, &glb, &foo::e, &foo::f, &foo::g, 1, 2, 3>"
// CHECK-SAME:                              templateParams: [[TCARGS:![0-9]*]]
// CHECK: [[TCARGS]] = !{[[TCARG1:![0-9]*]], [[TCARG2:![0-9]*]], [[TCARG3:![0-9]*]], [[TCARG4:![0-9]*]], [[TCARG5:![0-9]*]], [[TCARG6:![0-9]*]], [[TCARG7:![0-9]*]]}
//
// CHECK: [[TCARG1]] = !MDTemplateTypeParameter(name: "T", type: [[UINT:![0-9]*]])
// CHECK: [[UINT:![0-9]*]] = !MDBasicType(name: "unsigned int"
// CHECK: [[TCARG2]] = !MDTemplateValueParameter(type: [[UINT]], value: i32 2)
// CHECK: [[TCARG3]] = !MDTemplateValueParameter(name: "x", type: [[CINTPTR:![0-9]*]], value: i32* @glb)
// CHECK: [[CINTPTR]] = !MDDerivedType(tag: DW_TAG_pointer_type, {{.*}}baseType: [[CINT:![0-9]+]]
// CHECK: [[CINT]] = !MDDerivedType(tag: DW_TAG_const_type, {{.*}}baseType: [[INT:![0-9]+]]
// CHECK: [[INT]] = !MDBasicType(name: "int"
// CHECK: [[TCARG4]] = !MDTemplateValueParameter(name: "a", type: [[MEMINTPTR:![0-9]*]], value: i64 8)
// CHECK: [[MEMINTPTR]] = !MDDerivedType(tag: DW_TAG_ptr_to_member_type, {{.*}}baseType: [[INT]], {{.*}}extraData: !"_ZTS3foo")
//
// Currently Clang emits the pointer-to-member-function value, but LLVM doesn't
// use it (GCC doesn't emit a value for pointers to member functions either - so
// it's not clear what, if any, format would be acceptable to GDB)
//
// CHECK: [[TCARG5]] = !MDTemplateValueParameter(name: "b", type: [[MEMFUNPTR:![0-9]*]], value: { i64, i64 } { i64 ptrtoint (void (%struct.foo*)* @_ZN3foo1fEv to i64), i64 0 })
// CHECK: [[MEMFUNPTR]] = !MDDerivedType(tag: DW_TAG_ptr_to_member_type, {{.*}}baseType: [[FTYPE:![0-9]*]], {{.*}}extraData: !"_ZTS3foo")
// CHECK: [[FTYPE]] = !MDSubroutineType(types: [[FARGS:![0-9]*]])
// CHECK: [[FARGS]] = !{null, [[FARG1:![0-9]*]]}
// CHECK: [[FARG1]] = !MDDerivedType(tag: DW_TAG_pointer_type,
// CHECK-SAME:                       baseType: !"_ZTS3foo"
// CHECK-NOT:                        line:
// CHECK-SAME:                       size: 64, align: 64
// CHECK-NOT:                        offset: 0
// CHECK-SAME:                       DIFlagArtificial
//
// CHECK: [[TCARG6]] = !MDTemplateValueParameter(name: "f", type: [[FUNPTR:![0-9]*]], value: void ()* @_ZN3foo1gEv)
// CHECK: [[FUNPTR]] = !MDDerivedType(tag: DW_TAG_pointer_type, baseType: [[FUNTYPE:![0-9]*]]
// CHECK: [[FUNTYPE]] = !MDSubroutineType(types: [[FUNARGS:![0-9]*]])
// CHECK: [[FUNARGS]] = !{null}
// CHECK: [[TCARG7]] = !MDTemplateValueParameter(tag: DW_TAG_GNU_template_parameter_pack, name: "Is", value: [[TCARG7_VALS:![0-9]*]])
// CHECK: [[TCARG7_VALS]] = !{[[TCARG7_1:![0-9]*]], [[TCARG7_2:![0-9]*]], [[TCARG7_3:![0-9]*]]}
// CHECK: [[TCARG7_1]] = !MDTemplateValueParameter(type: [[INT]], value: i32 1)
// CHECK: [[TCARG7_2]] = !MDTemplateValueParameter(type: [[INT]], value: i32 2)
// CHECK: [[TCARG7_3]] = !MDTemplateValueParameter(type: [[INT]], value: i32 3)
//
// We could just emit a declaration of 'foo' here, rather than the entire
// definition (same goes for any time we emit a member (function or data)
// pointer type)
// CHECK: [[FOO]] = !MDCompositeType(tag: DW_TAG_structure_type, name: "foo", {{.*}}identifier: "_ZTS3foo")
// CHECK: !MDSubprogram(name: "f", linkageName: "_ZN3foo1fEv", {{.*}}type: [[FTYPE:![0-9]*]]
//

// CHECK: !MDCompositeType(tag: DW_TAG_structure_type, name: "nested",
// CHECK-SAME:             scope: !"_ZTS2TCIjLj2EXadL_Z3glbEEXadL_ZN3foo1eEEEXadL_ZNS0_1fEvEEXadL_ZNS0_1gEvEEJLi1ELi2ELi3EEE"
// CHECK-SAME:             identifier: "[[TCNESTED:.*]]")
// CHECK: !MDCompositeType(tag: DW_TAG_structure_type, name: "TC<int, -3, nullptr, nullptr, nullptr, nullptr>"
// CHECK-SAME:             templateParams: [[TCNARGS:![0-9]*]]
// CHECK-SAME:             identifier: "[[TCNT:.*]]")
// CHECK: [[TCNARGS]] = !{[[TCNARG1:![0-9]*]], [[TCNARG2:![0-9]*]], [[TCNARG3:![0-9]*]], [[TCNARG4:![0-9]*]], [[TCNARG5:![0-9]*]], [[TCNARG6:![0-9]*]], [[TCNARG7:![0-9]*]]}
// CHECK: [[TCNARG1]] = !MDTemplateTypeParameter(name: "T", type: [[INT]])
// CHECK: [[TCNARG2]] = !MDTemplateValueParameter(type: [[INT]], value: i32 -3)
// CHECK: [[TCNARG3]] = !MDTemplateValueParameter(name: "x", type: [[CINTPTR]], value: i8 0)

// The interesting null pointer: -1 for member data pointers (since they are
// just an offset in an object, they can be zero and non-null for the first
// member)

// CHECK: [[TCNARG4]] = !MDTemplateValueParameter(name: "a", type: [[MEMINTPTR]], value: i64 -1)
//
// In some future iteration we could possibly emit the value of a null member
// function pointer as '{ i64, i64 } zeroinitializer' as it may be handled
// naturally from the LLVM CodeGen side once we decide how to handle non-null
// member function pointers. For now, it's simpler just to emit the 'i8 0'.
//
// CHECK: [[TCNARG5]] = !MDTemplateValueParameter(name: "b", type: [[MEMFUNPTR]], value: i8 0)
// CHECK: [[TCNARG6]] = !MDTemplateValueParameter(name: "f", type: [[FUNPTR]], value: i8 0)
// CHECK: [[TCNARG7]] = !MDTemplateValueParameter(tag: DW_TAG_GNU_template_parameter_pack, name: "Is", value: [[EMPTY]])

// FIXME: these parameters should probably be rendered as 'glb' rather than
// '&glb', since they're references, not pointers.
// CHECK: !MDCompositeType(tag: DW_TAG_structure_type, name: "NN<tmpl_impl, &glb, &glb>",
// CHECK-SAME:             templateParams: [[NNARGS:![0-9]*]]
// CHECK-SAME:             identifier: "[[NNT:.*]]")
// CHECK: [[NNARGS]] = !{[[NNARG1:![0-9]*]], [[NNARG2:![0-9]*]], [[NNARG3:![0-9]*]]}
// CHECK: [[NNARG1]] = !MDTemplateValueParameter(tag: DW_TAG_GNU_template_template_param, name: "tmpl", value: !"tmpl_impl")
// CHECK: [[NNARG2]] = !MDTemplateValueParameter(name: "lvr", type: [[INTLVR:![0-9]*]], value: i32* @glb)
// CHECK: [[INTLVR]] = !MDDerivedType(tag: DW_TAG_reference_type, baseType: [[INT]]
// CHECK: [[NNARG3]] = !MDTemplateValueParameter(name: "rvr", type: [[INTRVR:![0-9]*]], value: i32* @glb)
// CHECK: [[INTRVR]] = !MDDerivedType(tag: DW_TAG_rvalue_reference_type, baseType: [[INT]]

// CHECK: !MDCompositeType(tag: DW_TAG_structure_type, name: "PaddingAtEndTemplate<&PaddedObj>"
// CHECK-SAME:             templateParams: [[PTOARGS:![0-9]*]]
// CHECK: [[PTOARGS]] = !{[[PTOARG1:![0-9]*]]}
// CHECK: [[PTOARG1]] = !MDTemplateValueParameter(type: [[CONST_PADDINGATEND_PTR:![0-9]*]], value: %struct.PaddingAtEnd* @PaddedObj)
// CHECK: [[CONST_PADDINGATEND_PTR]] = !MDDerivedType(tag: DW_TAG_pointer_type, baseType: !"_ZTS12PaddingAtEnd", size: 64, align: 64)

// CHECK: !MDGlobalVariable(name: "tci",
// CHECK-SAME:              type: !"[[TCNESTED]]"
// CHECK-SAME:              variable: %"struct.TC<unsigned int, 2, &glb, &foo::e, &foo::f, &foo::g, 1, 2, 3>::nested"* @tci

// CHECK: !MDGlobalVariable(name: "tcn"
// CHECK-SAME:              type: !"[[TCNT]]"
// CHECK-SAME:              variable: %struct.TC* @tcn

// CHECK: !MDGlobalVariable(name: "nn"
// CHECK-SAME:              type: !"[[NNT]]"
// CHECK-SAME:              variable: %struct.NN* @nn
struct foo {
  char pad[8]; // make the member pointer to 'e' a bit more interesting (nonzero)
  int e;
  void f();
  static void g();
};

typedef int foo::*foo_mem;

template<typename T, T, const int *x, foo_mem a, void (foo::*b)(), void (*f)(), int ...Is>
struct TC {
  struct nested {
  };
};

int glb;
void func();

TC<unsigned, 2, &glb, &foo::e, &foo::f, &foo::g, 1, 2, 3>::nested tci;
TC<int, -3, nullptr, nullptr, nullptr, nullptr> tcn;

template<typename>
struct tmpl_impl {
};

template <template <typename> class tmpl, int &lvr, int &&rvr>
struct NN {
};

NN<tmpl_impl, glb, glb> nn;

struct PaddingAtEnd {
  int i;
  char c;
};

PaddingAtEnd PaddedObj = {};

template <PaddingAtEnd *>
struct PaddingAtEndTemplate {
};

PaddingAtEndTemplate<&PaddedObj> PaddedTemplateObj;
