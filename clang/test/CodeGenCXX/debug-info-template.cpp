// RUN: %clang -S -emit-llvm -target x86_64-unknown_unknown -g %s -o - -std=c++11 | FileCheck %s

// CHECK: !"0x11\00{{.*}}"{{, [^,]+, [^,]+}}, [[RETAIN:![0-9]*]], {{.*}} ; [ DW_TAG_compile_unit ]
// CHECK: [[EMPTY:![0-9]*]] = !{}
// CHECK: [[RETAIN]] = !{!{{[0-9]]*}}, [[FOO:![0-9]*]],


// CHECK: [[TC:![0-9]*]] = {{.*}}, [[TCARGS:![0-9]*]], !"{{.*}}"} ; [ DW_TAG_structure_type ] [TC<unsigned int, 2, &glb, &foo::e, &foo::f, &foo::g, 1, 2, 3>]
// CHECK: [[TCARGS]] = !{[[TCARG1:![0-9]*]], [[TCARG2:![0-9]*]], [[TCARG3:![0-9]*]], [[TCARG4:![0-9]*]], [[TCARG5:![0-9]*]], [[TCARG6:![0-9]*]], [[TCARG7:![0-9]*]]}
//
// We seem to be missing file/line/col info on template value parameters -
// metadata supports it but it's not populated. GCC doesn't emit it either,
// perhaps we should just drop it from the metadata.
//
// CHECK: [[TCARG1]] = !{!"0x2f\00T\000\000", null, [[UINT:![0-9]*]], null} ; [ DW_TAG_template_type_parameter ]
// CHECK: [[UINT:![0-9]*]] = {{.*}} ; [ DW_TAG_base_type ] [unsigned int]
// CHECK: [[TCARG2]] = !{!"0x30\00\00{{.*}}", {{[^,]+}}, [[UINT]], i32 2, {{.*}} ; [ DW_TAG_template_value_parameter ]
// CHECK: [[TCARG3]] = !{!"0x30\00x\00{{.*}}", {{[^,]+}}, [[CINTPTR:![0-9]*]], i32* @glb, {{.*}} ; [ DW_TAG_template_value_parameter ]
// CHECK: [[CINTPTR]] = {{.*}}, [[CINT:![0-9]*]]} ; [ DW_TAG_pointer_type ] {{.*}} [from ]
// CHECK: [[CINT]] = {{.*}}, [[INT:![0-9]*]]} ; [ DW_TAG_const_type ] {{.*}} [from int]
// CHECK: [[INT]] = {{.*}} ; [ DW_TAG_base_type ] [int]
// CHECK: [[TCARG4]] = !{!"0x30\00a\00{{.*}}", {{[^,]+}}, [[MEMINTPTR:![0-9]*]], i64 8, {{.*}} ; [ DW_TAG_template_value_parameter ]
// CHECK: [[MEMINTPTR]] = {{.*}}, !"_ZTS3foo"} ; [ DW_TAG_ptr_to_member_type ] {{.*}}[from int]
//
// Currently Clang emits the pointer-to-member-function value, but LLVM doesn't
// use it (GCC doesn't emit a value for pointers to member functions either - so
// it's not clear what, if any, format would be acceptable to GDB)
//
// CHECK: [[TCARG5]] = !{!"0x30\00b\00{{.*}}", {{[^,]+}}, [[MEMFUNPTR:![0-9]*]], { i64, i64 } { i64 ptrtoint (void (%struct.foo*)* @_ZN3foo1fEv to i64), i64 0 }, {{.*}} ; [ DW_TAG_template_value_parameter ]
// CHECK: [[MEMFUNPTR]] = {{.*}}, [[FTYPE:![0-9]*]], !"_ZTS3foo"} ; [ DW_TAG_ptr_to_member_type ]
// CHECK: [[FTYPE]] = {{.*}}, [[FARGS:![0-9]*]], null, null, null} ; [ DW_TAG_subroutine_type ]
// CHECK: [[FARGS]] = !{null, [[FARG1:![0-9]*]]}
// CHECK: [[FARG1]] = {{.*}} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS3foo]
//
// CHECK: [[TCARG6]] = !{!"0x30\00f\00{{.*}}", {{[^,]+}}, [[FUNPTR:![0-9]*]], void ()* @_ZN3foo1gEv, {{.*}} ; [ DW_TAG_template_value_parameter ]
// CHECK: [[FUNPTR]] = {{.*}}, [[FUNTYPE:![0-9]*]]} ; [ DW_TAG_pointer_type ]
// CHECK: [[FUNTYPE]] = {{.*}}, [[FUNARGS:![0-9]*]], null, null, null} ; [ DW_TAG_subroutine_type ]
// CHECK: [[FUNARGS]] = !{null}
// CHECK: [[TCARG7]] = !{!"0x4107\00Is\000\000", null, null, [[TCARG7_VALS:![0-9]*]], null} ; [ DW_TAG_GNU_template_parameter_pack ]
// CHECK: [[TCARG7_VALS]] = !{[[TCARG7_1:![0-9]*]], [[TCARG7_2:![0-9]*]], [[TCARG7_3:![0-9]*]]}
// CHECK: [[TCARG7_1]] = !{!"0x30\00\00{{.*}}", {{[^,]+}}, [[INT]], i32 1, {{.*}} ; [ DW_TAG_template_value_parameter ]
// CHECK: [[TCARG7_2]] = !{!"0x30\00\00{{.*}}", {{[^,]+}}, [[INT]], i32 2, {{.*}} ; [ DW_TAG_template_value_parameter ]
// CHECK: [[TCARG7_3]] = !{!"0x30\00\00{{.*}}", {{[^,]+}}, [[INT]], i32 3, {{.*}} ; [ DW_TAG_template_value_parameter ]
//
// We could just emit a declaration of 'foo' here, rather than the entire
// definition (same goes for any time we emit a member (function or data)
// pointer type)
// CHECK: [[FOO]] = {{.*}}, !"_ZTS3foo"} ; [ DW_TAG_structure_type ] [foo]
// CHECK: !"0x2e\00f\00f\00_ZN3foo1fEv\00{{.*}}", [[FTYPE:![0-9]*]], {{.*}} ; [ DW_TAG_subprogram ]
//

// CHECK:  !"0x13\00{{.*}}", !{{[0-9]*}}, !"_ZTS2TCIjLj2EXadL_Z3glbEEXadL_ZN3foo1eEEEXadL_ZNS0_1fEvEEXadL_ZNS0_1gEvEEJLi1ELi2ELi3EEE", {{.*}}, !"[[TCNESTED:.*]]"} ; [ DW_TAG_structure_type ] [nested]
// CHECK: [[TCNARGS:![0-9]*]], !"[[TCNT:.*]]"} ; [ DW_TAG_structure_type ] [TC<int, -3, nullptr, nullptr, nullptr, nullptr>]
// CHECK: [[TCNARGS]] = !{[[TCNARG1:![0-9]*]], [[TCNARG2:![0-9]*]], [[TCNARG3:![0-9]*]], [[TCNARG4:![0-9]*]], [[TCNARG5:![0-9]*]], [[TCNARG6:![0-9]*]], [[TCNARG7:![0-9]*]]}
// CHECK: [[TCNARG1]] = !{!"0x2f\00T\000\000", null, [[INT]], null} ; [ DW_TAG_template_type_parameter ]
// CHECK: [[TCNARG2]] = !{!"0x30\00\000\000", null, [[INT]], i32 -3, null} ; [ DW_TAG_template_value_parameter ]
// CHECK: [[TCNARG3]] = !{!"0x30\00x\000\000", null, [[CINTPTR]], i8 0, null} ; [ DW_TAG_template_value_parameter ]

// The interesting null pointer: -1 for member data pointers (since they are
// just an offset in an object, they can be zero and non-null for the first
// member)

// CHECK: [[TCNARG4]] = !{!"0x30\00a\000\000", null, [[MEMINTPTR]], i64 -1, null} ; [ DW_TAG_template_value_parameter ]
//
// In some future iteration we could possibly emit the value of a null member
// function pointer as '{ i64, i64 } zeroinitializer' as it may be handled
// naturally from the LLVM CodeGen side once we decide how to handle non-null
// member function pointers. For now, it's simpler just to emit the 'i8 0'.
//
// CHECK: [[TCNARG5]] = !{!"0x30\00b\000\000", null, [[MEMFUNPTR]], i8 0, null} ; [ DW_TAG_template_value_parameter ]
// CHECK: [[TCNARG6]] = !{!"0x30\00f\000\000", null, [[FUNPTR]], i8 0, null} ; [ DW_TAG_template_value_parameter ]
// CHECK: [[TCNARG7]] = !{!"0x4107\00Is\000\000", null, null, [[EMPTY]], null} ; [ DW_TAG_GNU_template_parameter_pack ]

// FIXME: these parameters should probably be rendered as 'glb' rather than
// '&glb', since they're references, not pointers.
// CHECK: [[NNARGS:![0-9]*]], !"[[NNT:.*]]"} ; [ DW_TAG_structure_type ] [NN<tmpl_impl, &glb, &glb>]
// CHECK: [[NNARGS]] = !{[[NNARG1:![0-9]*]], [[NNARG2:![0-9]*]], [[NNARG3:![0-9]*]]}
// CHECK: [[NNARG1]] = !{!"0x4106\00tmpl\000\000", null, null, !"tmpl_impl", null} ; [ DW_TAG_GNU_template_template_param ]
// CHECK: [[NNARG2]] = !{!"0x30\00lvr\00{{.*}}", {{[^,]+}}, [[INTLVR:![0-9]*]], i32* @glb, {{.*}} ; [ DW_TAG_template_value_parameter ]
// CHECK: [[INTLVR]] = {{.*}}, [[INT]]} ; [ DW_TAG_reference_type ] {{.*}} [from int]
// CHECK: [[NNARG3]] = !{!"0x30\00rvr\00{{.*}}", {{[^,]+}}, [[INTRVR:![0-9]*]], i32* @glb, {{.*}} ; [ DW_TAG_template_value_parameter ]
// CHECK: [[INTRVR]] = {{.*}}, [[INT]]} ; [ DW_TAG_rvalue_reference_type ] {{.*}} [from int]

// CHECK: [[PTOARGS:![0-9]*]], !"{{.*}}"} ; [ DW_TAG_structure_type ] [PaddingAtEndTemplate<&PaddedObj>]
// CHECK: [[PTOARGS]] = !{[[PTOARG1:![0-9]*]]}
// CHECK: [[PTOARG1]] = !{!"0x30\00\000\000", null, [[CONST_PADDINGATEND_PTR:![0-9]*]], %struct.PaddingAtEnd* @PaddedObj, null} ; [ DW_TAG_template_value_parameter ]
// CHECK: [[CONST_PADDINGATEND_PTR]] = {{.*}} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from _ZTS12PaddingAtEnd]

// CHECK: !"[[TCNESTED]]", %"struct.TC<unsigned int, 2, &glb, &foo::e, &foo::f, &foo::g, 1, 2, 3>::nested"* @tci, null} ; [ DW_TAG_variable ] [tci]

// CHECK: !"[[TCNT]]", %struct.TC* @tcn, null} ; [ DW_TAG_variable ] [tcn]

// CHECK: !"[[NNT]]", %struct.NN* @nn, null} ; [ DW_TAG_variable ] [nn]
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
