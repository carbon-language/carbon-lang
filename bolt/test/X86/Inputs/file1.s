	.text
	.file	"file1.cpp"
	.section	.text._Z7doStuffii,"ax",@progbits
	.globl	_Z7doStuffii                    # -- Begin function _Z7doStuffii
	.p2align	4, 0x90
	.type	_Z7doStuffii,@function
_Z7doStuffii:                           # @_Z7doStuffii
.Lfunc_begin0:
	.file	1 "/home/ayermolo/local/fbsource/fbcode/scripts/ayermolo/debugExperiment" "file1.cpp"
	.loc	1 9 0                           # file1.cpp:9:0
	.cfi_startproc
# %bb.0:                                # %entry
	#DEBUG_VALUE: doStuff:var1 <- $edi
	#DEBUG_VALUE: doStuff:var1 <- $edi
	#DEBUG_VALUE: doStuff:var2 <- $esi
	#DEBUG_VALUE: doStuff:var2 <- $esi
	#DEBUG_VALUE: doStuff:retVal <- $edi
	#DEBUG_VALUE: doStuff:retVal <- $edi
	#DEBUG_VALUE: doStuff:f <- [DW_OP_LLVM_fragment 0 32] $edi
	#DEBUG_VALUE: doStuff:f <- [DW_OP_LLVM_fragment 0 32] $edi
	#DEBUG_VALUE: doStuff:f <- [DW_OP_LLVM_fragment 32 32] 3
	.loc	1 12 7 prologue_end             # file1.cpp:12:7
	movl	globalVar(%rip), %ecx
	movl	%ecx, %eax
	negl	%eax
.Ltmp0:
	.loc	1 12 12 is_stmt 0               # file1.cpp:12:12
	cmpl	%esi, %edi
.Ltmp1:
	.loc	1 12 7                          # file1.cpp:12:7
	cmovel	%ecx, %eax
	addl	%edi, %eax
.Ltmp2:
	#DEBUG_VALUE: doStuff:f <- [DW_OP_LLVM_fragment 0 32] $eax
	.loc	1 18 3 is_stmt 1                # file1.cpp:18:3
	retq
.Ltmp3:
.Lfunc_end0:
	.size	_Z7doStuffii, .Lfunc_end0-_Z7doStuffii
	.cfi_endproc
                                        # -- End function
	.section	.debug_types,"G",@progbits,7448148824980338162,comdat
	.long	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.short	4                               # DWARF version number
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	8                               # Address Size (in bytes)
	.quad	7448148824980338162             # Type Signature
	.long	30                              # Type DIE Offset
	.byte	1                               # Abbrev [1] 0x17:0x33 DW_TAG_type_unit
	.short	33                              # DW_AT_language
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.byte	2                               # Abbrev [2] 0x1e:0x24 DW_TAG_class_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string12                 # DW_AT_name
	.byte	8                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	3                               # Abbrev [3] 0x27:0xd DW_TAG_member
	.long	.Linfo_string10                 # DW_AT_name
	.long	66                              # DW_AT_type
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	0                               # DW_AT_data_member_location
	.byte	1                               # DW_AT_accessibility
                                        # DW_ACCESS_public
	.byte	3                               # Abbrev [3] 0x34:0xd DW_TAG_member
	.long	.Linfo_string11                 # DW_AT_name
	.long	66                              # DW_AT_type
	.byte	1                               # DW_AT_decl_file
	.byte	6                               # DW_AT_decl_line
	.byte	4                               # DW_AT_data_member_location
	.byte	1                               # DW_AT_accessibility
                                        # DW_ACCESS_public
	.byte	0                               # End Of Children Mark
	.byte	4                               # Abbrev [4] 0x42:0x7 DW_TAG_base_type
	.long	.Linfo_string5                  # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	0                               # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_loc,"",@progbits
.Ldebug_loc0:
	.quad	.Lfunc_begin0-.Lfunc_begin0
	.quad	.Ltmp2-.Lfunc_begin0
	.short	7                               # Loc expr size
	.byte	85                              # super-register DW_OP_reg5
	.byte	147                             # DW_OP_piece
	.byte	4                               # 4
	.byte	51                              # DW_OP_lit3
	.byte	159                             # DW_OP_stack_value
	.byte	147                             # DW_OP_piece
	.byte	4                               # 4
	.quad	.Ltmp2-.Lfunc_begin0
	.quad	.Lfunc_end0-.Lfunc_begin0
	.short	7                               # Loc expr size
	.byte	80                              # super-register DW_OP_reg0
	.byte	147                             # DW_OP_piece
	.byte	4                               # 4
	.byte	51                              # DW_OP_lit3
	.byte	159                             # DW_OP_stack_value
	.byte	147                             # DW_OP_piece
	.byte	4                               # 4
	.quad	0
	.quad	0
	.section	.debug_abbrev,"",@progbits
	.byte	1                               # Abbreviation Code
	.byte	65                              # DW_TAG_type_unit
	.byte	1                               # DW_CHILDREN_yes
	.byte	19                              # DW_AT_language
	.byte	5                               # DW_FORM_data2
	.byte	16                              # DW_AT_stmt_list
	.byte	23                              # DW_FORM_sec_offset
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	2                               # Abbreviation Code
	.byte	2                               # DW_TAG_class_type
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
	.byte	3                               # Abbreviation Code
	.byte	13                              # DW_TAG_member
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	56                              # DW_AT_data_member_location
	.byte	11                              # DW_FORM_data1
	.byte	50                              # DW_AT_accessibility
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	4                               # Abbreviation Code
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
	.byte	5                               # Abbreviation Code
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
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	6                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
	.ascii	"\227B"                         # DW_AT_GNU_all_call_sites
	.byte	25                              # DW_FORM_flag_present
	.byte	110                             # DW_AT_linkage_name
	.byte	14                              # DW_FORM_strp
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	7                               # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	8                               # Abbreviation Code
	.byte	52                              # DW_TAG_variable
	.byte	0                               # DW_CHILDREN_no
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	9                               # Abbreviation Code
	.byte	52                              # DW_TAG_variable
	.byte	0                               # DW_CHILDREN_no
	.byte	2                               # DW_AT_location
	.byte	23                              # DW_FORM_sec_offset
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	10                              # Abbreviation Code
	.byte	38                              # DW_TAG_const_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	11                              # Abbreviation Code
	.byte	2                               # DW_TAG_class_type
	.byte	0                               # DW_CHILDREN_no
	.byte	60                              # DW_AT_declaration
	.byte	25                              # DW_FORM_flag_present
	.byte	105                             # DW_AT_signature
	.byte	32                              # DW_FORM_ref_sig8
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	0                               # EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end1-.Ldebug_info_start1 # Length of Unit
.Ldebug_info_start1:
	.short	4                               # DWARF version number
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	8                               # Address Size (in bytes)
	.byte	5                               # Abbrev [5] 0xb:0x89 DW_TAG_compile_unit
	.long	.Linfo_string0                  # DW_AT_producer
	.short	33                              # DW_AT_language
	.long	.Linfo_string1                  # DW_AT_name
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.long	.Linfo_string2                  # DW_AT_comp_dir
	.quad	.Lfunc_begin0                   # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	6                               # Abbrev [6] 0x2a:0x54 DW_TAG_subprogram
	.quad	.Lfunc_begin0                   # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	87
                                        # DW_AT_GNU_all_call_sites
	.long	.Linfo_string3                  # DW_AT_linkage_name
	.long	.Linfo_string4                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	9                               # DW_AT_decl_line
	.long	126                             # DW_AT_type
                                        # DW_AT_external
	.byte	7                               # Abbrev [7] 0x47:0xd DW_TAG_formal_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.long	.Linfo_string6                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	9                               # DW_AT_decl_line
	.long	126                             # DW_AT_type
	.byte	7                               # Abbrev [7] 0x54:0xd DW_TAG_formal_parameter
	.byte	1                               # DW_AT_location
	.byte	84
	.long	.Linfo_string7                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	9                               # DW_AT_decl_line
	.long	133                             # DW_AT_type
	.byte	8                               # Abbrev [8] 0x61:0xd DW_TAG_variable
	.byte	1                               # DW_AT_location
	.byte	85
	.long	.Linfo_string8                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	10                              # DW_AT_decl_line
	.long	126                             # DW_AT_type
	.byte	9                               # Abbrev [9] 0x6e:0xf DW_TAG_variable
	.long	.Ldebug_loc0                    # DW_AT_location
	.long	.Linfo_string9                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	11                              # DW_AT_decl_line
	.long	138                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	4                               # Abbrev [4] 0x7e:0x7 DW_TAG_base_type
	.long	.Linfo_string5                  # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	10                              # Abbrev [10] 0x85:0x5 DW_TAG_const_type
	.long	126                             # DW_AT_type
	.byte	11                              # Abbrev [11] 0x8a:0x9 DW_TAG_class_type
                                        # DW_AT_declaration
	.quad	7448148824980338162             # DW_AT_signature
	.byte	0                               # End Of Children Mark
.Ldebug_info_end1:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 12.0.0 (ssh://git.vip.facebook.com/data/gitrepos/osmeta/external/llvm-project 865a43501716d59b5b257119d6cf021fe4239642)" # string offset=0
.Linfo_string1:
	.asciz	"file1.cpp"                     # string offset=134
.Linfo_string2:
	.asciz	"/home/ayermolo/local/fbsource/fbcode/scripts/ayermolo/debugExperiment" # string offset=144
.Linfo_string3:
	.asciz	"_Z7doStuffii"                  # string offset=214
.Linfo_string4:
	.asciz	"doStuff"                       # string offset=227
.Linfo_string5:
	.asciz	"int"                           # string offset=235
.Linfo_string6:
	.asciz	"var1"                          # string offset=239
.Linfo_string7:
	.asciz	"var2"                          # string offset=244
.Linfo_string8:
	.asciz	"retVal"                        # string offset=249
.Linfo_string9:
	.asciz	"f"                             # string offset=256
.Linfo_string10:
	.asciz	"x1"                            # string offset=258
.Linfo_string11:
	.asciz	"y1"                            # string offset=261
.Linfo_string12:
	.asciz	"Foo"                           # string offset=264
	.ident	"clang version 12.0.0 (ssh://git.vip.facebook.com/data/gitrepos/osmeta/external/llvm-project 865a43501716d59b5b257119d6cf021fe4239642)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.section	.debug_line,"",@progbits
.Lline_table_start0:
