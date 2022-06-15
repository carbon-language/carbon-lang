# RUN: llvm-mc < %s -filetype obj -triple x86_64 -o - \
# RUN:   | llvm-dwarfdump --name=t1 --show-children - | FileCheck %s

# Assembly generated from the following source:
#
#   struct foo;
#   template <typename... Ts>
#   struct t1 {};
#   namespace ns {
#   struct inner { };
#   }
#   namespace {
#   struct anon_ns_mem { };
#   }
#   template <typename T>
#   struct t2 { };
#   enum e1 {
#     E1
#   };
#   enum class e2 {
#     E2
#   };
#   template<typename T, T V>
#   struct tv { };
#   template<char ...C>
#   struct tc { };
#   t1<
#       // base type
#       int,
#       // nullptr unspecified type
#       decltype(nullptr),
#       // reference type
#       int &,
#       // rvalue reference type
#       int &&,
#       // pointer types
#       int *, const void *const *, const void *const, int *const volatile,
#       int *volatile, void *const,
#       // pointer to member variable
#       int foo::*,
#       // pointer to member functions
#       void (foo::*)(int), void (foo::*const &)() const volatile &&,
#       // arrays
#       int *const (&)[1], int *const[1], const int (&)[1], const int[1],
#       // subroutine types
#       int(), void(int), void(int, int), void(...), void (*)(foo *, int),
#       void (*const)(), void() const, void() volatile &&, void() const volatile &,
#       void(const volatile foo *), void (*(int))(float),
#       // qualified types
#       ns::inner, ns::inner(), ns::inner[1], ns::inner *, ns::inner ns::inner::*,
#       ns::inner (ns::inner::*)(ns::inner), const ns::inner, anon_ns_mem,
#       // templates
#       t2<t2<int>>,
#       // enum literals
#       tv<e1, E1>, tv<e1, (e1)1>, tv<e2, e2::E2>,
#       // char literals
#       tv<unsigned char, 'x'>,
#       tc<'x', '\\', '\'', '\a', '\b', '\f', '\n', '\r', '\t', '\v', (char)127>,
#       // integral literals
#       tv<bool, true>, tv<bool, false>, tv<short, 0>, tv<unsigned short, 0>,
#       tv<int, 0>, tv<long, 0L>, tv<long long, 0LL>, tv<unsigned, 0U>,
#       tv<unsigned long, 0UL>, tv<unsigned long long, 0ULL>
#       // end of template parameter list
#       >
#       v1;
#   // extern function to force the code to be emitted - otherwise v1 (having
#   // internal linkage due to being of a type instantiated with an internal
#   // linkage type) would be optimized away as unused.
#   __attribute__((nodebug)) void*f1() {
#     return &v1;
#   }
#
# With the name of the "t1<...>" type renamed to "t1" (or using
# -gsimple-template-names) to make it easy to query for.
# Note that the llvm-dwarfdump command uses --name=t1 --show-children to only
# dump the template type parameters, making it easy to order the types as
# intended and to avoid visiting subtypes that aren't intended to be tested
# separately.


# base_type
# CHECK:   DW_AT_type{{.*}}"int"

# FIXME: This might change in clang to std::nullptr_t which would be more
# accurate and not be ambiguous with some arbitrary ::nullptr_t a user could
# define. If that change is made, this code/test should be fixed too.
# nullptr_t unspecified type
# CHECK:   DW_AT_type{{.*}}"std::nullptr_t"

# reference_type
# CHECK:   DW_AT_type{{.*}}"int &"

# rvalue_reference_type
# CHECK:   DW_AT_type{{.*}}"int &&"

# pointer_type
# CHECK:   DW_AT_type{{.*}}"int *"
# CHECK:   DW_AT_type{{.*}}"const void *const *")
# CHECK:   DW_AT_type{{.*}}"const void *const")
# CHECK:   DW_AT_type{{.*}}"int *const volatile")
# CHECK:   DW_AT_type{{.*}}"int *volatile")
# CHECK:   DW_AT_type{{.*}}"void *const")

# ptr_to_member_type
# CHECK:   DW_AT_type{{.*}}"int foo::*"

# ptr_to_member_type to a member function
# CHECK:   DW_AT_type{{.*}}"void (foo::*)(int)"
# const reference to a pointer to member function (with const, volatile, rvalue ref qualifiers)
# CHECK:   DW_AT_type{{.*}}"void (foo::*const &)() const volatile &&")

# CHECK:   DW_AT_type{{.*}}"int *const (&)[1]")
# CHECK:   DW_AT_type{{.*}}"int *const[1]")
# CHECK:   DW_AT_type{{.*}}"const int (&)[1]")
# CHECK:   DW_AT_type{{.*}}"const int[1]")

# subroutine types
# CHECK:   DW_AT_type{{.*}}"int ()"
# CHECK:   DW_AT_type{{.*}}"void (int)"
# CHECK:   DW_AT_type{{.*}}"void (int, int)"
# CHECK:   DW_AT_type{{.*}}"void (...)"
# CHECK:   DW_AT_type{{.*}}"void (*)(foo *, int)"
# CHECK:   DW_AT_type{{.*}}"void (*const)()")
# CHECK:   DW_AT_type{{.*}}"void () const")
# CHECK:   DW_AT_type{{.*}}"void () volatile &&")
# CHECK:   DW_AT_type{{.*}}"void () const volatile &")
# CHECK:   DW_AT_type{{.*}}"void (const volatile foo *)")
# CHECK:   DW_AT_type{{.*}}"void (*(int))(float)")

# qualified types
# CHECK:   DW_AT_type{{.*}}"ns::inner"
# CHECK:   DW_AT_type{{.*}}"ns::inner ()"
# CHECK:   DW_AT_type{{.*}}"ns::inner[1]"
# CHECK:   DW_AT_type{{.*}}"ns::inner *"
# CHECK:   DW_AT_type{{.*}}"ns::inner (ns::inner::*)(ns::inner)"
# CHECK:   DW_AT_type{{.*}}"const ns::inner"
# CHECK:   DW_AT_type{{.*}}"(anonymous namespace)::anon_ns_mem"

# template types
# CHECK:   DW_AT_type{{.*}}"t2<t2<int> >"

# enum literals
# CHECK:   DW_AT_type{{.*}}"tv<e1, (e1)0>")
# CHECK:   DW_AT_type{{.*}}"tv<e1, (e1)1>")
# CHECK:   DW_AT_type{{.*}}"tv<e2, (e2)0>")

# char literals
# CHECK:   DW_AT_type{{.*}}"tv<unsigned char, (unsigned char)'x'>")
# CHECK:   DW_AT_type{{.*}}"tc<'x', '\\', '\'', '\a', '\b', '\f', '\n', '\r', '\t', '\v', '\x7f'>")

# bool literals
# CHECK:   DW_AT_type{{.*}}"tv<bool, true>")
# CHECK:   DW_AT_type{{.*}}"tv<bool, false>")

# int literals
# CHECK:   DW_AT_type{{.*}}"tv<short, (short)0>"
# CHECK:   DW_AT_type{{.*}}"tv<unsigned short, (unsigned short)0>"
# CHECK:   DW_AT_type{{.*}}"tv<int, 0>"
# CHECK:   DW_AT_type{{.*}}"tv<long, 0L>"
# CHECK:   DW_AT_type{{.*}}"tv<long long, 0LL>"
# CHECK:   DW_AT_type{{.*}}"tv<unsigned int, 0U>"
# CHECK:   DW_AT_type{{.*}}"tv<unsigned long, 0UL>"
# CHECK:   DW_AT_type{{.*}}"tv<unsigned long long, 0ULL>"

	.text
	.file	"test.cpp"
	.file	1 "/usr/local/google/home/blaikie/dev/scratch" "test.cpp"
	.globl	_Z2f1v                          # -- Begin function _Z2f1v
	.p2align	4, 0x90
	.type	_Z2f1v,@function
_Z2f1v:                                 # @_Z2f1v
.Lfunc_begin0:
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movabsq	$v1, %rax
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end0:
	.size	_Z2f1v, .Lfunc_end0-_Z2f1v
	.cfi_endproc
                                        # -- End function
	.type	v1,@object                      # @v1
	.local	v1
	.comm	v1,1,1
	.section	.debug_abbrev,"",@progbits
	.byte	1                               # Abbreviation Code
	.byte	17                              # DW_TAG_compile_unit
	.byte	1                               # DW_CHILDREN_yes
	.byte	37                              # DW_AT_producer
	.byte	14                              # DW_FORM_strp
	.byte	19                              # DW_AT_language
	.byte	5                               # DW_FORM_data2
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	16                              # DW_AT_stmt_list
	.byte	23                              # DW_FORM_sec_offset
	.byte	27                              # DW_AT_comp_dir
	.byte	14                              # DW_FORM_strp
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	2                               # Abbreviation Code
	.byte	52                              # DW_TAG_variable
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	3                               # Abbreviation Code
	.byte	19                              # DW_TAG_structure_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	54                              # DW_AT_calling_convention
	.byte	11                              # DW_FORM_data1
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	11                              # DW_AT_byte_size
	.byte	11                              # DW_FORM_data1
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	4                               # Abbreviation Code
	.ascii	"\207\202\001"                  # DW_TAG_GNU_template_parameter_pack
	.byte	1                               # DW_CHILDREN_yes
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	5                               # Abbreviation Code
	.byte	47                              # DW_TAG_template_type_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	6                               # Abbreviation Code
	.byte	36                              # DW_TAG_base_type
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	62                              # DW_AT_encoding
	.byte	11                              # DW_FORM_data1
	.byte	11                              # DW_AT_byte_size
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	7                               # Abbreviation Code
	.byte	59                              # DW_TAG_unspecified_type
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	8                               # Abbreviation Code
	.byte	16                              # DW_TAG_reference_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	9                               # Abbreviation Code
	.byte	66                              # DW_TAG_rvalue_reference_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	10                              # Abbreviation Code
	.byte	15                              # DW_TAG_pointer_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	11                              # Abbreviation Code
	.byte	38                              # DW_TAG_const_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	12                              # Abbreviation Code
	.byte	38                              # DW_TAG_const_type
	.byte	0                               # DW_CHILDREN_no
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	13                              # Abbreviation Code
	.byte	53                              # DW_TAG_volatile_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	14                              # Abbreviation Code
	.byte	15                              # DW_TAG_pointer_type
	.byte	0                               # DW_CHILDREN_no
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	15                              # Abbreviation Code
	.byte	31                              # DW_TAG_ptr_to_member_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	29                              # DW_AT_containing_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	16                              # Abbreviation Code
	.byte	19                              # DW_TAG_structure_type
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	60                              # DW_AT_declaration
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	17                              # Abbreviation Code
	.byte	21                              # DW_TAG_subroutine_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	18                              # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	52                              # DW_AT_artificial
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	19                              # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	20                              # Abbreviation Code
	.byte	21                              # DW_TAG_subroutine_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	120                             # DW_AT_rvalue_reference
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	21                              # Abbreviation Code
	.byte	1                               # DW_TAG_array_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	22                              # Abbreviation Code
	.byte	33                              # DW_TAG_subrange_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	55                              # DW_AT_count
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	23                              # Abbreviation Code
	.byte	36                              # DW_TAG_base_type
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	11                              # DW_AT_byte_size
	.byte	11                              # DW_FORM_data1
	.byte	62                              # DW_AT_encoding
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	24                              # Abbreviation Code
	.byte	21                              # DW_TAG_subroutine_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	25                              # Abbreviation Code
	.byte	24                              # DW_TAG_unspecified_parameters
	.byte	0                               # DW_CHILDREN_no
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	26                              # Abbreviation Code
	.byte	21                              # DW_TAG_subroutine_type
	.byte	0                               # DW_CHILDREN_no
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	27                              # Abbreviation Code
	.byte	21                              # DW_TAG_subroutine_type
	.byte	0                               # DW_CHILDREN_no
	.byte	120                             # DW_AT_rvalue_reference
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	28                              # Abbreviation Code
	.byte	21                              # DW_TAG_subroutine_type
	.byte	0                               # DW_CHILDREN_no
	.byte	119                             # DW_AT_reference
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	29                              # Abbreviation Code
	.byte	21                              # DW_TAG_subroutine_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	30                              # Abbreviation Code
	.byte	57                              # DW_TAG_namespace
	.byte	1                               # DW_CHILDREN_yes
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	31                              # Abbreviation Code
	.byte	57                              # DW_TAG_namespace
	.byte	1                               # DW_CHILDREN_yes
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	32                              # Abbreviation Code
	.byte	19                              # DW_TAG_structure_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	60                              # DW_AT_declaration
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	33                              # Abbreviation Code
	.byte	47                              # DW_TAG_template_type_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	34                              # Abbreviation Code
	.byte	48                              # DW_TAG_template_value_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	28                              # DW_AT_const_value
	.byte	15                              # DW_FORM_udata
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	35                              # Abbreviation Code
	.byte	4                               # DW_TAG_enumeration_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	11                              # DW_AT_byte_size
	.byte	11                              # DW_FORM_data1
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	36                              # Abbreviation Code
	.byte	40                              # DW_TAG_enumerator
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	28                              # DW_AT_const_value
	.byte	15                              # DW_FORM_udata
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	37                              # Abbreviation Code
	.byte	48                              # DW_TAG_template_value_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	28                              # DW_AT_const_value
	.byte	13                              # DW_FORM_sdata
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	38                              # Abbreviation Code
	.byte	4                               # DW_TAG_enumeration_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	109                             # DW_AT_enum_class
	.byte	25                              # DW_FORM_flag_present
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	11                              # DW_AT_byte_size
	.byte	11                              # DW_FORM_data1
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	39                              # Abbreviation Code
	.byte	40                              # DW_TAG_enumerator
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	28                              # DW_AT_const_value
	.byte	13                              # DW_FORM_sdata
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	40                              # Abbreviation Code
	.byte	48                              # DW_TAG_template_value_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	28                              # DW_AT_const_value
	.byte	13                              # DW_FORM_sdata
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	0                               # EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.short	4                               # DWARF version number
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	8                               # Address Size (in bytes)
	.byte	1                               # Abbrev [1] 0xb:0x505 DW_TAG_compile_unit
	.long	.Linfo_string0                  # DW_AT_producer
	.short	33                              # DW_AT_language
	.long	.Linfo_string1                  # DW_AT_name
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.long	.Linfo_string2                  # DW_AT_comp_dir
	.byte	2                               # Abbrev [2] 0x1e:0x15 DW_TAG_variable
	.long	.Linfo_string3                  # DW_AT_name
	.long	51                              # DW_AT_type
	.byte	1                               # DW_AT_decl_file
	.byte	60                              # DW_AT_decl_line
	.byte	9                               # DW_AT_location
	.byte	3
	.quad	v1
	.byte	3                               # Abbrev [3] 0x33:0x114 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string33                 # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	4                               # Abbrev [4] 0x3c:0x10a DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string4                  # DW_AT_name
	.byte	5                               # Abbrev [5] 0x41:0x5 DW_TAG_template_type_parameter
	.long	327                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x46:0x5 DW_TAG_template_type_parameter
	.long	334                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x4b:0x5 DW_TAG_template_type_parameter
	.long	339                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x50:0x5 DW_TAG_template_type_parameter
	.long	344                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x55:0x5 DW_TAG_template_type_parameter
	.long	349                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x5a:0x5 DW_TAG_template_type_parameter
	.long	354                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x5f:0x5 DW_TAG_template_type_parameter
	.long	359                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x64:0x5 DW_TAG_template_type_parameter
	.long	370                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x69:0x5 DW_TAG_template_type_parameter
	.long	375                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x6e:0x5 DW_TAG_template_type_parameter
	.long	380                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x73:0x5 DW_TAG_template_type_parameter
	.long	386                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x78:0x5 DW_TAG_template_type_parameter
	.long	400                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x7d:0x5 DW_TAG_template_type_parameter
	.long	426                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x82:0x5 DW_TAG_template_type_parameter
	.long	467                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x87:0x5 DW_TAG_template_type_parameter
	.long	472                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x8c:0x5 DW_TAG_template_type_parameter
	.long	496                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x91:0x5 DW_TAG_template_type_parameter
	.long	501                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x96:0x5 DW_TAG_template_type_parameter
	.long	518                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x9b:0x5 DW_TAG_template_type_parameter
	.long	523                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0xa0:0x5 DW_TAG_template_type_parameter
	.long	530                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0xa5:0x5 DW_TAG_template_type_parameter
	.long	542                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0xaa:0x5 DW_TAG_template_type_parameter
	.long	545                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0xaf:0x5 DW_TAG_template_type_parameter
	.long	567                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0xb4:0x5 DW_TAG_template_type_parameter
	.long	578                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0xb9:0x5 DW_TAG_template_type_parameter
	.long	583                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0xbe:0x5 DW_TAG_template_type_parameter
	.long	589                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0xc3:0x5 DW_TAG_template_type_parameter
	.long	600                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0xc8:0x5 DW_TAG_template_type_parameter
	.long	612                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0xcd:0x5 DW_TAG_template_type_parameter
	.long	647                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0xd2:0x5 DW_TAG_template_type_parameter
	.long	653                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0xd7:0x5 DW_TAG_template_type_parameter
	.long	658                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0xdc:0x5 DW_TAG_template_type_parameter
	.long	670                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0xe1:0x5 DW_TAG_template_type_parameter
	.long	675                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0xe6:0x5 DW_TAG_template_type_parameter
	.long	684                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0xeb:0x5 DW_TAG_template_type_parameter
	.long	714                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0xf0:0x5 DW_TAG_template_type_parameter
	.long	720                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0xf5:0x5 DW_TAG_template_type_parameter
	.long	726                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0xfa:0x5 DW_TAG_template_type_parameter
	.long	756                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0xff:0x5 DW_TAG_template_type_parameter
	.long	807                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x104:0x5 DW_TAG_template_type_parameter
	.long	832                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x109:0x5 DW_TAG_template_type_parameter
	.long	876                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x10e:0x5 DW_TAG_template_type_parameter
	.long	908                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x113:0x5 DW_TAG_template_type_parameter
	.long	996                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x118:0x5 DW_TAG_template_type_parameter
	.long	1028                            # DW_AT_type
	.byte	5                               # Abbrev [5] 0x11d:0x5 DW_TAG_template_type_parameter
	.long	1053                            # DW_AT_type
	.byte	5                               # Abbrev [5] 0x122:0x5 DW_TAG_template_type_parameter
	.long	1085                            # DW_AT_type
	.byte	5                               # Abbrev [5] 0x127:0x5 DW_TAG_template_type_parameter
	.long	1117                            # DW_AT_type
	.byte	5                               # Abbrev [5] 0x12c:0x5 DW_TAG_template_type_parameter
	.long	1142                            # DW_AT_type
	.byte	5                               # Abbrev [5] 0x131:0x5 DW_TAG_template_type_parameter
	.long	1174                            # DW_AT_type
	.byte	5                               # Abbrev [5] 0x136:0x5 DW_TAG_template_type_parameter
	.long	1206                            # DW_AT_type
	.byte	5                               # Abbrev [5] 0x13b:0x5 DW_TAG_template_type_parameter
	.long	1231                            # DW_AT_type
	.byte	5                               # Abbrev [5] 0x140:0x5 DW_TAG_template_type_parameter
	.long	1263                            # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	6                               # Abbrev [6] 0x147:0x7 DW_TAG_base_type
	.long	.Linfo_string5                  # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	7                               # Abbrev [7] 0x14e:0x5 DW_TAG_unspecified_type
	.long	.Linfo_string6                  # DW_AT_name
	.byte	8                               # Abbrev [8] 0x153:0x5 DW_TAG_reference_type
	.long	327                             # DW_AT_type
	.byte	9                               # Abbrev [9] 0x158:0x5 DW_TAG_rvalue_reference_type
	.long	327                             # DW_AT_type
	.byte	10                              # Abbrev [10] 0x15d:0x5 DW_TAG_pointer_type
	.long	327                             # DW_AT_type
	.byte	10                              # Abbrev [10] 0x162:0x5 DW_TAG_pointer_type
	.long	359                             # DW_AT_type
	.byte	11                              # Abbrev [11] 0x167:0x5 DW_TAG_const_type
	.long	364                             # DW_AT_type
	.byte	10                              # Abbrev [10] 0x16c:0x5 DW_TAG_pointer_type
	.long	369                             # DW_AT_type
	.byte	12                              # Abbrev [12] 0x171:0x1 DW_TAG_const_type
	.byte	11                              # Abbrev [11] 0x172:0x5 DW_TAG_const_type
	.long	375                             # DW_AT_type
	.byte	13                              # Abbrev [13] 0x177:0x5 DW_TAG_volatile_type
	.long	349                             # DW_AT_type
	.byte	11                              # Abbrev [11] 0x17c:0x5 DW_TAG_const_type
	.long	385                             # DW_AT_type
	.byte	14                              # Abbrev [14] 0x181:0x1 DW_TAG_pointer_type
	.byte	15                              # Abbrev [15] 0x182:0x9 DW_TAG_ptr_to_member_type
	.long	327                             # DW_AT_type
	.long	395                             # DW_AT_containing_type
	.byte	16                              # Abbrev [16] 0x18b:0x5 DW_TAG_structure_type
	.long	.Linfo_string7                  # DW_AT_name
                                        # DW_AT_declaration
	.byte	15                              # Abbrev [15] 0x190:0x9 DW_TAG_ptr_to_member_type
	.long	409                             # DW_AT_type
	.long	395                             # DW_AT_containing_type
	.byte	17                              # Abbrev [17] 0x199:0xc DW_TAG_subroutine_type
	.byte	18                              # Abbrev [18] 0x19a:0x5 DW_TAG_formal_parameter
	.long	421                             # DW_AT_type
                                        # DW_AT_artificial
	.byte	19                              # Abbrev [19] 0x19f:0x5 DW_TAG_formal_parameter
	.long	327                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x1a5:0x5 DW_TAG_pointer_type
	.long	395                             # DW_AT_type
	.byte	8                               # Abbrev [8] 0x1aa:0x5 DW_TAG_reference_type
	.long	431                             # DW_AT_type
	.byte	11                              # Abbrev [11] 0x1af:0x5 DW_TAG_const_type
	.long	436                             # DW_AT_type
	.byte	15                              # Abbrev [15] 0x1b4:0x9 DW_TAG_ptr_to_member_type
	.long	445                             # DW_AT_type
	.long	395                             # DW_AT_containing_type
	.byte	20                              # Abbrev [20] 0x1bd:0x7 DW_TAG_subroutine_type
                                        # DW_AT_rvalue_reference
	.byte	18                              # Abbrev [18] 0x1be:0x5 DW_TAG_formal_parameter
	.long	452                             # DW_AT_type
                                        # DW_AT_artificial
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x1c4:0x5 DW_TAG_pointer_type
	.long	457                             # DW_AT_type
	.byte	11                              # Abbrev [11] 0x1c9:0x5 DW_TAG_const_type
	.long	462                             # DW_AT_type
	.byte	13                              # Abbrev [13] 0x1ce:0x5 DW_TAG_volatile_type
	.long	395                             # DW_AT_type
	.byte	8                               # Abbrev [8] 0x1d3:0x5 DW_TAG_reference_type
	.long	472                             # DW_AT_type
	.byte	11                              # Abbrev [11] 0x1d8:0x5 DW_TAG_const_type
	.long	477                             # DW_AT_type
	.byte	21                              # Abbrev [21] 0x1dd:0xc DW_TAG_array_type
	.long	349                             # DW_AT_type
	.byte	22                              # Abbrev [22] 0x1e2:0x6 DW_TAG_subrange_type
	.long	489                             # DW_AT_type
	.byte	1                               # DW_AT_count
	.byte	0                               # End Of Children Mark
	.byte	23                              # Abbrev [23] 0x1e9:0x7 DW_TAG_base_type
	.long	.Linfo_string8                  # DW_AT_name
	.byte	8                               # DW_AT_byte_size
	.byte	7                               # DW_AT_encoding
	.byte	8                               # Abbrev [8] 0x1f0:0x5 DW_TAG_reference_type
	.long	501                             # DW_AT_type
	.byte	11                              # Abbrev [11] 0x1f5:0x5 DW_TAG_const_type
	.long	506                             # DW_AT_type
	.byte	21                              # Abbrev [21] 0x1fa:0xc DW_TAG_array_type
	.long	327                             # DW_AT_type
	.byte	22                              # Abbrev [22] 0x1ff:0x6 DW_TAG_subrange_type
	.long	489                             # DW_AT_type
	.byte	1                               # DW_AT_count
	.byte	0                               # End Of Children Mark
	.byte	24                              # Abbrev [24] 0x206:0x5 DW_TAG_subroutine_type
	.long	327                             # DW_AT_type
	.byte	17                              # Abbrev [17] 0x20b:0x7 DW_TAG_subroutine_type
	.byte	19                              # Abbrev [19] 0x20c:0x5 DW_TAG_formal_parameter
	.long	327                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	17                              # Abbrev [17] 0x212:0xc DW_TAG_subroutine_type
	.byte	19                              # Abbrev [19] 0x213:0x5 DW_TAG_formal_parameter
	.long	327                             # DW_AT_type
	.byte	19                              # Abbrev [19] 0x218:0x5 DW_TAG_formal_parameter
	.long	327                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	17                              # Abbrev [17] 0x21e:0x3 DW_TAG_subroutine_type
	.byte	25                              # Abbrev [25] 0x21f:0x1 DW_TAG_unspecified_parameters
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x221:0x5 DW_TAG_pointer_type
	.long	550                             # DW_AT_type
	.byte	17                              # Abbrev [17] 0x226:0xc DW_TAG_subroutine_type
	.byte	19                              # Abbrev [19] 0x227:0x5 DW_TAG_formal_parameter
	.long	562                             # DW_AT_type
	.byte	19                              # Abbrev [19] 0x22c:0x5 DW_TAG_formal_parameter
	.long	327                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x232:0x5 DW_TAG_pointer_type
	.long	395                             # DW_AT_type
	.byte	11                              # Abbrev [11] 0x237:0x5 DW_TAG_const_type
	.long	572                             # DW_AT_type
	.byte	10                              # Abbrev [10] 0x23c:0x5 DW_TAG_pointer_type
	.long	577                             # DW_AT_type
	.byte	26                              # Abbrev [26] 0x241:0x1 DW_TAG_subroutine_type
	.byte	11                              # Abbrev [11] 0x242:0x5 DW_TAG_const_type
	.long	577                             # DW_AT_type
	.byte	13                              # Abbrev [13] 0x247:0x5 DW_TAG_volatile_type
	.long	588                             # DW_AT_type
	.byte	27                              # Abbrev [27] 0x24c:0x1 DW_TAG_subroutine_type
                                        # DW_AT_rvalue_reference
	.byte	11                              # Abbrev [11] 0x24d:0x5 DW_TAG_const_type
	.long	594                             # DW_AT_type
	.byte	13                              # Abbrev [13] 0x252:0x5 DW_TAG_volatile_type
	.long	599                             # DW_AT_type
	.byte	28                              # Abbrev [28] 0x257:0x1 DW_TAG_subroutine_type
                                        # DW_AT_reference
	.byte	17                              # Abbrev [17] 0x258:0x7 DW_TAG_subroutine_type
	.byte	19                              # Abbrev [19] 0x259:0x5 DW_TAG_formal_parameter
	.long	607                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x25f:0x5 DW_TAG_pointer_type
	.long	457                             # DW_AT_type
	.byte	29                              # Abbrev [29] 0x264:0xb DW_TAG_subroutine_type
	.long	623                             # DW_AT_type
	.byte	19                              # Abbrev [19] 0x269:0x5 DW_TAG_formal_parameter
	.long	327                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x26f:0x5 DW_TAG_pointer_type
	.long	628                             # DW_AT_type
	.byte	17                              # Abbrev [17] 0x274:0x7 DW_TAG_subroutine_type
	.byte	19                              # Abbrev [19] 0x275:0x5 DW_TAG_formal_parameter
	.long	635                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	6                               # Abbrev [6] 0x27b:0x7 DW_TAG_base_type
	.long	.Linfo_string9                  # DW_AT_name
	.byte	4                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	30                              # Abbrev [30] 0x282:0xb DW_TAG_namespace
	.long	.Linfo_string10                 # DW_AT_name
	.byte	16                              # Abbrev [16] 0x287:0x5 DW_TAG_structure_type
	.long	.Linfo_string11                 # DW_AT_name
                                        # DW_AT_declaration
	.byte	0                               # End Of Children Mark
	.byte	24                              # Abbrev [24] 0x28d:0x5 DW_TAG_subroutine_type
	.long	647                             # DW_AT_type
	.byte	21                              # Abbrev [21] 0x292:0xc DW_TAG_array_type
	.long	647                             # DW_AT_type
	.byte	22                              # Abbrev [22] 0x297:0x6 DW_TAG_subrange_type
	.long	489                             # DW_AT_type
	.byte	1                               # DW_AT_count
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x29e:0x5 DW_TAG_pointer_type
	.long	647                             # DW_AT_type
	.byte	15                              # Abbrev [15] 0x2a3:0x9 DW_TAG_ptr_to_member_type
	.long	647                             # DW_AT_type
	.long	647                             # DW_AT_containing_type
	.byte	15                              # Abbrev [15] 0x2ac:0x9 DW_TAG_ptr_to_member_type
	.long	693                             # DW_AT_type
	.long	647                             # DW_AT_containing_type
	.byte	29                              # Abbrev [29] 0x2b5:0x10 DW_TAG_subroutine_type
	.long	647                             # DW_AT_type
	.byte	18                              # Abbrev [18] 0x2ba:0x5 DW_TAG_formal_parameter
	.long	709                             # DW_AT_type
                                        # DW_AT_artificial
	.byte	19                              # Abbrev [19] 0x2bf:0x5 DW_TAG_formal_parameter
	.long	647                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x2c5:0x5 DW_TAG_pointer_type
	.long	647                             # DW_AT_type
	.byte	11                              # Abbrev [11] 0x2ca:0x5 DW_TAG_const_type
	.long	647                             # DW_AT_type
	.byte	31                              # Abbrev [31] 0x2cf:0x7 DW_TAG_namespace
	.byte	16                              # Abbrev [16] 0x2d0:0x5 DW_TAG_structure_type
	.long	.Linfo_string12                 # DW_AT_name
                                        # DW_AT_declaration
	.byte	0                               # End Of Children Mark
	.byte	32                              # Abbrev [32] 0x2d6:0xf DW_TAG_structure_type
	.long	.Linfo_string14                 # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x2db:0x9 DW_TAG_template_type_parameter
	.long	741                             # DW_AT_type
	.long	.Linfo_string13                 # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	32                              # Abbrev [32] 0x2e5:0xf DW_TAG_structure_type
	.long	.Linfo_string14                 # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x2ea:0x9 DW_TAG_template_type_parameter
	.long	327                             # DW_AT_type
	.long	.Linfo_string13                 # DW_AT_name
	.byte	0                               # End Of Children Mark
	.byte	32                              # Abbrev [32] 0x2f4:0x19 DW_TAG_structure_type
	.long	.Linfo_string19                 # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x2f9:0x9 DW_TAG_template_type_parameter
	.long	781                             # DW_AT_type
	.long	.Linfo_string13                 # DW_AT_name
	.byte	34                              # Abbrev [34] 0x302:0xa DW_TAG_template_value_parameter
	.long	781                             # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	0                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	35                              # Abbrev [35] 0x30d:0x13 DW_TAG_enumeration_type
	.long	800                             # DW_AT_type
	.long	.Linfo_string17                 # DW_AT_name
	.byte	4                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	36                              # Abbrev [36] 0x319:0x6 DW_TAG_enumerator
	.long	.Linfo_string16                 # DW_AT_name
	.byte	0                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	6                               # Abbrev [6] 0x320:0x7 DW_TAG_base_type
	.long	.Linfo_string15                 # DW_AT_name
	.byte	7                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	32                              # Abbrev [32] 0x327:0x19 DW_TAG_structure_type
	.long	.Linfo_string19                 # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x32c:0x9 DW_TAG_template_type_parameter
	.long	781                             # DW_AT_type
	.long	.Linfo_string13                 # DW_AT_name
	.byte	34                              # Abbrev [34] 0x335:0xa DW_TAG_template_value_parameter
	.long	781                             # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	1                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	32                              # Abbrev [32] 0x340:0x19 DW_TAG_structure_type
	.long	.Linfo_string19                 # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x345:0x9 DW_TAG_template_type_parameter
	.long	857                             # DW_AT_type
	.long	.Linfo_string13                 # DW_AT_name
	.byte	37                              # Abbrev [37] 0x34e:0xa DW_TAG_template_value_parameter
	.long	857                             # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	0                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	38                              # Abbrev [38] 0x359:0x13 DW_TAG_enumeration_type
	.long	327                             # DW_AT_type
                                        # DW_AT_enum_class
	.long	.Linfo_string21                 # DW_AT_name
	.byte	4                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	15                              # DW_AT_decl_line
	.byte	39                              # Abbrev [39] 0x365:0x6 DW_TAG_enumerator
	.long	.Linfo_string20                 # DW_AT_name
	.byte	0                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	32                              # Abbrev [32] 0x36c:0x19 DW_TAG_structure_type
	.long	.Linfo_string19                 # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x371:0x9 DW_TAG_template_type_parameter
	.long	901                             # DW_AT_type
	.long	.Linfo_string13                 # DW_AT_name
	.byte	34                              # Abbrev [34] 0x37a:0xa DW_TAG_template_value_parameter
	.long	901                             # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	120                             # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	6                               # Abbrev [6] 0x385:0x7 DW_TAG_base_type
	.long	.Linfo_string22                 # DW_AT_name
	.byte	8                               # DW_AT_encoding
	.byte	1                               # DW_AT_byte_size
	.byte	32                              # Abbrev [32] 0x38c:0x51 DW_TAG_structure_type
	.long	.Linfo_string25                 # DW_AT_name
                                        # DW_AT_declaration
	.byte	4                               # Abbrev [4] 0x391:0x4b DW_TAG_GNU_template_parameter_pack
	.long	.Linfo_string23                 # DW_AT_name
	.byte	40                              # Abbrev [40] 0x396:0x7 DW_TAG_template_value_parameter
	.long	989                             # DW_AT_type
	.asciz	"\370"                          # DW_AT_const_value
	.byte	40                              # Abbrev [40] 0x39d:0x7 DW_TAG_template_value_parameter
	.long	989                             # DW_AT_type
	.asciz	"\334"                          # DW_AT_const_value
	.byte	40                              # Abbrev [40] 0x3a4:0x6 DW_TAG_template_value_parameter
	.long	989                             # DW_AT_type
	.byte	39                              # DW_AT_const_value
	.byte	40                              # Abbrev [40] 0x3aa:0x6 DW_TAG_template_value_parameter
	.long	989                             # DW_AT_type
	.byte	7                               # DW_AT_const_value
	.byte	40                              # Abbrev [40] 0x3b0:0x6 DW_TAG_template_value_parameter
	.long	989                             # DW_AT_type
	.byte	8                               # DW_AT_const_value
	.byte	40                              # Abbrev [40] 0x3b6:0x6 DW_TAG_template_value_parameter
	.long	989                             # DW_AT_type
	.byte	12                              # DW_AT_const_value
	.byte	40                              # Abbrev [40] 0x3bc:0x6 DW_TAG_template_value_parameter
	.long	989                             # DW_AT_type
	.byte	10                              # DW_AT_const_value
	.byte	40                              # Abbrev [40] 0x3c2:0x6 DW_TAG_template_value_parameter
	.long	989                             # DW_AT_type
	.byte	13                              # DW_AT_const_value
	.byte	40                              # Abbrev [40] 0x3c8:0x6 DW_TAG_template_value_parameter
	.long	989                             # DW_AT_type
	.byte	9                               # DW_AT_const_value
	.byte	40                              # Abbrev [40] 0x3ce:0x6 DW_TAG_template_value_parameter
	.long	989                             # DW_AT_type
	.byte	11                              # DW_AT_const_value
	.byte	40                              # Abbrev [40] 0x3d4:0x7 DW_TAG_template_value_parameter
	.long	989                             # DW_AT_type
	.asciz	"\377"                          # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	6                               # Abbrev [6] 0x3dd:0x7 DW_TAG_base_type
	.long	.Linfo_string24                 # DW_AT_name
	.byte	6                               # DW_AT_encoding
	.byte	1                               # DW_AT_byte_size
	.byte	32                              # Abbrev [32] 0x3e4:0x19 DW_TAG_structure_type
	.long	.Linfo_string19                 # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x3e9:0x9 DW_TAG_template_type_parameter
	.long	1021                            # DW_AT_type
	.long	.Linfo_string13                 # DW_AT_name
	.byte	34                              # Abbrev [34] 0x3f2:0xa DW_TAG_template_value_parameter
	.long	1021                            # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	1                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	6                               # Abbrev [6] 0x3fd:0x7 DW_TAG_base_type
	.long	.Linfo_string26                 # DW_AT_name
	.byte	2                               # DW_AT_encoding
	.byte	1                               # DW_AT_byte_size
	.byte	32                              # Abbrev [32] 0x404:0x19 DW_TAG_structure_type
	.long	.Linfo_string19                 # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x409:0x9 DW_TAG_template_type_parameter
	.long	1021                            # DW_AT_type
	.long	.Linfo_string13                 # DW_AT_name
	.byte	34                              # Abbrev [34] 0x412:0xa DW_TAG_template_value_parameter
	.long	1021                            # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	0                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	32                              # Abbrev [32] 0x41d:0x19 DW_TAG_structure_type
	.long	.Linfo_string19                 # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x422:0x9 DW_TAG_template_type_parameter
	.long	1078                            # DW_AT_type
	.long	.Linfo_string13                 # DW_AT_name
	.byte	37                              # Abbrev [37] 0x42b:0xa DW_TAG_template_value_parameter
	.long	1078                            # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	0                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	6                               # Abbrev [6] 0x436:0x7 DW_TAG_base_type
	.long	.Linfo_string27                 # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	2                               # DW_AT_byte_size
	.byte	32                              # Abbrev [32] 0x43d:0x19 DW_TAG_structure_type
	.long	.Linfo_string19                 # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x442:0x9 DW_TAG_template_type_parameter
	.long	1110                            # DW_AT_type
	.long	.Linfo_string13                 # DW_AT_name
	.byte	34                              # Abbrev [34] 0x44b:0xa DW_TAG_template_value_parameter
	.long	1110                            # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	0                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	6                               # Abbrev [6] 0x456:0x7 DW_TAG_base_type
	.long	.Linfo_string28                 # DW_AT_name
	.byte	7                               # DW_AT_encoding
	.byte	2                               # DW_AT_byte_size
	.byte	32                              # Abbrev [32] 0x45d:0x19 DW_TAG_structure_type
	.long	.Linfo_string19                 # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x462:0x9 DW_TAG_template_type_parameter
	.long	327                             # DW_AT_type
	.long	.Linfo_string13                 # DW_AT_name
	.byte	37                              # Abbrev [37] 0x46b:0xa DW_TAG_template_value_parameter
	.long	327                             # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	0                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	32                              # Abbrev [32] 0x476:0x19 DW_TAG_structure_type
	.long	.Linfo_string19                 # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x47b:0x9 DW_TAG_template_type_parameter
	.long	1167                            # DW_AT_type
	.long	.Linfo_string13                 # DW_AT_name
	.byte	37                              # Abbrev [37] 0x484:0xa DW_TAG_template_value_parameter
	.long	1167                            # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	0                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	6                               # Abbrev [6] 0x48f:0x7 DW_TAG_base_type
	.long	.Linfo_string29                 # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	8                               # DW_AT_byte_size
	.byte	32                              # Abbrev [32] 0x496:0x19 DW_TAG_structure_type
	.long	.Linfo_string19                 # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x49b:0x9 DW_TAG_template_type_parameter
	.long	1199                            # DW_AT_type
	.long	.Linfo_string13                 # DW_AT_name
	.byte	37                              # Abbrev [37] 0x4a4:0xa DW_TAG_template_value_parameter
	.long	1199                            # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	0                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	6                               # Abbrev [6] 0x4af:0x7 DW_TAG_base_type
	.long	.Linfo_string30                 # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	8                               # DW_AT_byte_size
	.byte	32                              # Abbrev [32] 0x4b6:0x19 DW_TAG_structure_type
	.long	.Linfo_string19                 # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x4bb:0x9 DW_TAG_template_type_parameter
	.long	800                             # DW_AT_type
	.long	.Linfo_string13                 # DW_AT_name
	.byte	34                              # Abbrev [34] 0x4c4:0xa DW_TAG_template_value_parameter
	.long	800                             # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	0                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	32                              # Abbrev [32] 0x4cf:0x19 DW_TAG_structure_type
	.long	.Linfo_string19                 # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x4d4:0x9 DW_TAG_template_type_parameter
	.long	1256                            # DW_AT_type
	.long	.Linfo_string13                 # DW_AT_name
	.byte	34                              # Abbrev [34] 0x4dd:0xa DW_TAG_template_value_parameter
	.long	1256                            # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	0                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	6                               # Abbrev [6] 0x4e8:0x7 DW_TAG_base_type
	.long	.Linfo_string31                 # DW_AT_name
	.byte	7                               # DW_AT_encoding
	.byte	8                               # DW_AT_byte_size
	.byte	32                              # Abbrev [32] 0x4ef:0x19 DW_TAG_structure_type
	.long	.Linfo_string19                 # DW_AT_name
                                        # DW_AT_declaration
	.byte	33                              # Abbrev [33] 0x4f4:0x9 DW_TAG_template_type_parameter
	.long	1288                            # DW_AT_type
	.long	.Linfo_string13                 # DW_AT_name
	.byte	34                              # Abbrev [34] 0x4fd:0xa DW_TAG_template_value_parameter
	.long	1288                            # DW_AT_type
	.long	.Linfo_string18                 # DW_AT_name
	.byte	0                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	6                               # Abbrev [6] 0x508:0x7 DW_TAG_base_type
	.long	.Linfo_string32                 # DW_AT_name
	.byte	7                               # DW_AT_encoding
	.byte	8                               # DW_AT_byte_size
	.byte	0                               # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 14.0.0 (git@github.com:llvm/llvm-project.git e1e74f6cd6ce41ce8303a5a91f29736808fccc36)" # string offset=0
.Linfo_string1:
	.asciz	"test.cpp"                      # string offset=101
.Linfo_string2:
	.asciz	"/usr/local/google/home/blaikie/dev/scratch" # string offset=110
.Linfo_string3:
	.asciz	"v1"                            # string offset=153
.Linfo_string4:
	.asciz	"Ts"                            # string offset=156
.Linfo_string5:
	.asciz	"int"                           # string offset=159
.Linfo_string6:
	.asciz	"decltype(nullptr)"             # string offset=163
.Linfo_string7:
	.asciz	"foo"                           # string offset=181
.Linfo_string8:
	.asciz	"__ARRAY_SIZE_TYPE__"           # string offset=185
.Linfo_string9:
	.asciz	"float"                         # string offset=205
.Linfo_string10:
	.asciz	"ns"                            # string offset=211
.Linfo_string11:
	.asciz	"inner"                         # string offset=214
.Linfo_string12:
	.asciz	"anon_ns_mem"                   # string offset=220
.Linfo_string13:
	.asciz	"T"                             # string offset=232
.Linfo_string14:
	.asciz	"t2"                            # string offset=234
.Linfo_string15:
	.asciz	"unsigned int"                  # string offset=237
.Linfo_string16:
	.asciz	"E1"                            # string offset=250
.Linfo_string17:
	.asciz	"e1"                            # string offset=253
.Linfo_string18:
	.asciz	"V"                             # string offset=256
.Linfo_string19:
	.asciz	"tv"                            # string offset=258
.Linfo_string20:
	.asciz	"E2"                            # string offset=261
.Linfo_string21:
	.asciz	"e2"                            # string offset=264
.Linfo_string22:
	.asciz	"unsigned char"                 # string offset=267
.Linfo_string23:
	.asciz	"C"                             # string offset=281
.Linfo_string24:
	.asciz	"char"                          # string offset=283
.Linfo_string25:
	.asciz	"tc"                            # string offset=288
.Linfo_string26:
	.asciz	"bool"                          # string offset=291
.Linfo_string27:
	.asciz	"short"                         # string offset=296
.Linfo_string28:
	.asciz	"unsigned short"                # string offset=302
.Linfo_string29:
	.asciz	"long"                          # string offset=317
.Linfo_string30:
	.asciz	"long long"                     # string offset=322
.Linfo_string31:
	.asciz	"unsigned long"                 # string offset=332
.Linfo_string32:
	.asciz	"unsigned long long"            # string offset=346
.Linfo_string33:
	.asciz	"t1"                            # string offset=365
	.ident	"clang version 14.0.0 (git@github.com:llvm/llvm-project.git e1e74f6cd6ce41ce8303a5a91f29736808fccc36)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym v1
	.section	.debug_line,"",@progbits
.Lline_table_start0:
