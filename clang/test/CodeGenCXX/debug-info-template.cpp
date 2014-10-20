// RUN: %clang -S -emit-llvm -target x86_64-unknown_unknown -g %s -o - -std=c++11 | FileCheck %s

// CHECK: metadata !"0x11\00{{.*}}"{{, [^,]+, [^,]+}}, metadata [[RETAIN:![0-9]*]], {{.*}} ; [ DW_TAG_compile_unit ]
// CHECK: [[EMPTY:![0-9]*]] = metadata !{}
// CHECK: [[RETAIN]] = metadata !{metadata !{{[0-9]]*}}, metadata [[FOO:![0-9]*]],


// CHECK: [[TC:![0-9]*]] = {{.*}}, metadata [[TCARGS:![0-9]*]], metadata !"{{.*}}"} ; [ DW_TAG_structure_type ] [TC<unsigned int, 2, &glb, &foo::e, &foo::f, &foo::g, 1, 2, 3>]
// CHECK: [[TCARGS]] = metadata !{metadata [[TCARG1:![0-9]*]], metadata [[TCARG2:![0-9]*]], metadata [[TCARG3:![0-9]*]], metadata [[TCARG4:![0-9]*]], metadata [[TCARG5:![0-9]*]], metadata [[TCARG6:![0-9]*]], metadata [[TCARG7:![0-9]*]]}
//
// We seem to be missing file/line/col info on template value parameters -
// metadata supports it but it's not populated. GCC doesn't emit it either,
// perhaps we should just drop it from the metadata.
//
// CHECK: [[TCARG1]] = metadata !{metadata !"0x2f\00T\000\000", null, metadata [[UINT:![0-9]*]], null} ; [ DW_TAG_template_type_parameter ]
// CHECK: [[UINT:![0-9]*]] = {{.*}} ; [ DW_TAG_base_type ] [unsigned int]
// CHECK: [[TCARG2]] = metadata !{metadata !"0x30\00\00{{.*}}", {{[^,]+}}, metadata [[UINT]], i32 2, {{.*}} ; [ DW_TAG_template_value_parameter ]
// CHECK: [[TCARG3]] = metadata !{metadata !"0x30\00x\00{{.*}}", {{[^,]+}}, metadata [[CINTPTR:![0-9]*]], i32* @glb, {{.*}} ; [ DW_TAG_template_value_parameter ]
// CHECK: [[CINTPTR]] = {{.*}}, metadata [[CINT:![0-9]*]]} ; [ DW_TAG_pointer_type ] {{.*}} [from ]
// CHECK: [[CINT]] = {{.*}}, metadata [[INT:![0-9]*]]} ; [ DW_TAG_const_type ] {{.*}} [from int]
// CHECK: [[INT]] = {{.*}} ; [ DW_TAG_base_type ] [int]
// CHECK: [[TCARG4]] = metadata !{metadata !"0x30\00a\00{{.*}}", {{[^,]+}}, metadata [[MEMINTPTR:![0-9]*]], i64 8, {{.*}} ; [ DW_TAG_template_value_parameter ]
// CHECK: [[MEMINTPTR]] = {{.*}}, metadata !"_ZTS3foo"} ; [ DW_TAG_ptr_to_member_type ] {{.*}}[from int]
//
// Currently Clang emits the pointer-to-member-function value, but LLVM doesn't
// use it (GCC doesn't emit a value for pointers to member functions either - so
// it's not clear what, if any, format would be acceptable to GDB)
//
// CHECK: [[TCARG5]] = metadata !{metadata !"0x30\00b\00{{.*}}", {{[^,]+}}, metadata [[MEMFUNPTR:![0-9]*]], { i64, i64 } { i64 ptrtoint (void (%struct.foo*)* @_ZN3foo1fEv to i64), i64 0 }, {{.*}} ; [ DW_TAG_template_value_parameter ]
// CHECK: [[MEMFUNPTR]] = {{.*}}, metadata [[FTYPE:![0-9]*]], metadata !"_ZTS3foo"} ; [ DW_TAG_ptr_to_member_type ]
// CHECK: [[FTYPE]] = {{.*}}, metadata [[FARGS:![0-9]*]], null, null, null} ; [ DW_TAG_subroutine_type ]
// CHECK: [[FARGS]] = metadata !{null, metadata [[FARG1:![0-9]*]]}
// CHECK: [[FARG1]] = {{.*}} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS3foo]
//
// CHECK: [[TCARG6]] = metadata !{metadata !"0x30\00f\00{{.*}}", {{[^,]+}}, metadata [[FUNPTR:![0-9]*]], void ()* @_ZN3foo1gEv, {{.*}} ; [ DW_TAG_template_value_parameter ]
// CHECK: [[FUNPTR]] = {{.*}}, metadata [[FUNTYPE:![0-9]*]]} ; [ DW_TAG_pointer_type ]
// CHECK: [[FUNTYPE]] = {{.*}}, metadata [[FUNARGS:![0-9]*]], null, null, null} ; [ DW_TAG_subroutine_type ]
// CHECK: [[FUNARGS]] = metadata !{null}
// CHECK: [[TCARG7]] = metadata !{metadata !"0x4107\00Is\000\000", null, null, metadata [[TCARG7_VALS:![0-9]*]], null} ; [ DW_TAG_GNU_template_parameter_pack ]
// CHECK: [[TCARG7_VALS]] = metadata !{metadata [[TCARG7_1:![0-9]*]], metadata [[TCARG7_2:![0-9]*]], metadata [[TCARG7_3:![0-9]*]]}
// CHECK: [[TCARG7_1]] = metadata !{metadata !"0x30\00\00{{.*}}", {{[^,]+}}, metadata [[INT]], i32 1, {{.*}} ; [ DW_TAG_template_value_parameter ]
// CHECK: [[TCARG7_2]] = metadata !{metadata !"0x30\00\00{{.*}}", {{[^,]+}}, metadata [[INT]], i32 2, {{.*}} ; [ DW_TAG_template_value_parameter ]
// CHECK: [[TCARG7_3]] = metadata !{metadata !"0x30\00\00{{.*}}", {{[^,]+}}, metadata [[INT]], i32 3, {{.*}} ; [ DW_TAG_template_value_parameter ]
//
// We could just emit a declaration of 'foo' here, rather than the entire
// definition (same goes for any time we emit a member (function or data)
// pointer type)
// CHECK: [[FOO]] = {{.*}}, metadata !"_ZTS3foo"} ; [ DW_TAG_structure_type ] [foo]
// CHECK: metadata !"0x2e\00f\00f\00_ZN3foo1fEv\00{{.*}}", metadata [[FTYPE:![0-9]*]], {{.*}} ; [ DW_TAG_subprogram ]
//

// CHECK: metadata !{metadata !"0x13\00{{.*}}", metadata !{{[0-9]*}}, metadata !"_ZTS2TCIjLj2EXadL_Z3glbEEXadL_ZN3foo1eEEEXadL_ZNS0_1fEvEEXadL_ZNS0_1gEvEEJLi1ELi2ELi3EEE", {{.*}}, metadata !"[[TCNESTED:.*]]"} ; [ DW_TAG_structure_type ] [nested]
// CHECK: metadata [[TCNARGS:![0-9]*]], metadata !"[[TCNT:.*]]"} ; [ DW_TAG_structure_type ] [TC<int, -3, nullptr, nullptr, nullptr, nullptr>]
// CHECK: [[TCNARGS]] = metadata !{metadata [[TCNARG1:![0-9]*]], metadata [[TCNARG2:![0-9]*]], metadata [[TCNARG3:![0-9]*]], metadata [[TCNARG4:![0-9]*]], metadata [[TCNARG5:![0-9]*]], metadata [[TCNARG6:![0-9]*]], metadata [[TCNARG7:![0-9]*]]}
// CHECK: [[TCNARG1]] = metadata !{metadata !"0x2f\00T\000\000", null, metadata [[INT]], null} ; [ DW_TAG_template_type_parameter ]
// CHECK: [[TCNARG2]] = metadata !{metadata !"0x30\00\000\000", null, metadata [[INT]], i32 -3, null} ; [ DW_TAG_template_value_parameter ]
// CHECK: [[TCNARG3]] = metadata !{metadata !"0x30\00x\000\000", null, metadata [[CINTPTR]], i8 0, null} ; [ DW_TAG_template_value_parameter ]

// The interesting null pointer: -1 for member data pointers (since they are
// just an offset in an object, they can be zero and non-null for the first
// member)

// CHECK: [[TCNARG4]] = metadata !{metadata !"0x30\00a\000\000", null, metadata [[MEMINTPTR]], i64 -1, null} ; [ DW_TAG_template_value_parameter ]
//
// In some future iteration we could possibly emit the value of a null member
// function pointer as '{ i64, i64 } zeroinitializer' as it may be handled
// naturally from the LLVM CodeGen side once we decide how to handle non-null
// member function pointers. For now, it's simpler just to emit the 'i8 0'.
//
// CHECK: [[TCNARG5]] = metadata !{metadata !"0x30\00b\000\000", null, metadata [[MEMFUNPTR]], i8 0, null} ; [ DW_TAG_template_value_parameter ]
// CHECK: [[TCNARG6]] = metadata !{metadata !"0x30\00f\000\000", null, metadata [[FUNPTR]], i8 0, null} ; [ DW_TAG_template_value_parameter ]
// CHECK: [[TCNARG7]] = metadata !{metadata !"0x4107\00Is\000\000", null, null, metadata [[EMPTY]], null} ; [ DW_TAG_GNU_template_parameter_pack ]

// FIXME: these parameters should probably be rendered as 'glb' rather than
// '&glb', since they're references, not pointers.
// CHECK: metadata [[NNARGS:![0-9]*]], metadata !"[[NNT:.*]]"} ; [ DW_TAG_structure_type ] [NN<tmpl_impl, &glb, &glb>]
// CHECK: [[NNARGS]] = metadata !{metadata [[NNARG1:![0-9]*]], metadata [[NNARG2:![0-9]*]], metadata [[NNARG3:![0-9]*]]}
// CHECK: [[NNARG1]] = metadata !{metadata !"0x4106\00tmpl\000\000", null, null, metadata !"tmpl_impl", null} ; [ DW_TAG_GNU_template_template_param ]
// CHECK: [[NNARG2]] = metadata !{metadata !"0x30\00lvr\00{{.*}}", {{[^,]+}}, metadata [[INTLVR:![0-9]*]], i32* @glb, {{.*}} ; [ DW_TAG_template_value_parameter ]
// CHECK: [[INTLVR]] = {{.*}}, metadata [[INT]]} ; [ DW_TAG_reference_type ] {{.*}} [from int]
// CHECK: [[NNARG3]] = metadata !{metadata !"0x30\00rvr\00{{.*}}", {{[^,]+}}, metadata [[INTRVR:![0-9]*]], i32* @glb, {{.*}} ; [ DW_TAG_template_value_parameter ]
// CHECK: [[INTRVR]] = {{.*}}, metadata [[INT]]} ; [ DW_TAG_rvalue_reference_type ] {{.*}} [from int]

// CHECK: metadata [[PTOARGS:![0-9]*]], metadata !"{{.*}}"} ; [ DW_TAG_structure_type ] [PaddingAtEndTemplate<&PaddedObj>]
// CHECK: [[PTOARGS]] = metadata !{metadata [[PTOARG1:![0-9]*]]}
// CHECK: [[PTOARG1]] = metadata !{metadata !"0x30\00\000\000", null, metadata [[CONST_PADDINGATEND_PTR:![0-9]*]], %struct.PaddingAtEnd* @PaddedObj, null} ; [ DW_TAG_template_value_parameter ]
// CHECK: [[CONST_PADDINGATEND_PTR]] = {{.*}} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from _ZTS12PaddingAtEnd]

// CHECK: metadata !"[[TCNESTED]]", %"struct.TC<unsigned int, 2, &glb, &foo::e, &foo::f, &foo::g, 1, 2, 3>::nested"* @tci, null} ; [ DW_TAG_variable ] [tci]

// CHECK: metadata !"[[TCNT]]", %struct.TC* @tcn, null} ; [ DW_TAG_variable ] [tcn]

// CHECK: metadata !"[[NNT]]", %struct.NN* @nn, null} ; [ DW_TAG_variable ] [nn]
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
