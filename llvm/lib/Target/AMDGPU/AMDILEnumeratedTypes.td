//===-- AMDILEnumeratedTypes.td - IL Type definitions --*- tablegen -*-----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//==-----------------------------------------------------------------------===//
// ILEnumreatedTypes.td - The IL Enumerated Types
//===--------------------------------------------------------------------===//

// Section 5.1  IL Shader
class ILShader<bits<8> val> {
    bits<8> Value = val;
}
// Table 5-1
def IL_SHADER_PIXEL : ILShader<0>;
def IL_SHADER_COMPUTE : ILShader<1>;

// Section 5.2 IL RegType
class ILRegType<bits<6> val> {
    bits<6> Value = val;
}
// Table 5-2
def IL_REGTYPE_TEMP      : ILRegType<0>;
def IL_REGTYPE_WINCOORD  : ILRegType<1>;
def IL_REGTYPE_CONST_BUF : ILRegType<2>;
def IL_REGTYPE_LITERAL   : ILRegType<3>;
def IL_REGTYPE_ITEMP     : ILRegType<4>;
def IL_REGTYPE_GLOBAL    : ILRegType<5>;

// Section 5.3 IL Component Select
class ILComponentSelect<bits<3> val, string text> {
     bits<3> Value = val;
     string Text = text;
}
// Table 5-3
def IL_COMPSEL_X : ILComponentSelect<0, "x">;
def IL_COMPSEL_Y : ILComponentSelect<1, "y">;
def IL_COMPSEL_Z : ILComponentSelect<2, "z">;
def IL_COMPSEL_W : ILComponentSelect<3, "w">;
def IL_COMPSEL_0 : ILComponentSelect<4, "0">;
def IL_COMPSEL_1 : ILComponentSelect<5, "1">;

// Section 5.4 IL Mod Dst Comp
class ILModDstComp<bits<2> val, string text> {
    bits<2> Value = val;
    string Text = text;
}
// Table 5-4
def IL_MODCOMP_NOWRITE : ILModDstComp<0, "_">;
def IL_MODCOMP_WRITE_X : ILModDstComp<1, "x">;
def IL_MODCOMP_WRITE_y : ILModDstComp<1, "y">;
def IL_MODCOMP_WRITE_z : ILModDstComp<1, "z">;
def IL_MODCOMP_WRITE_w : ILModDstComp<1, "w">;
def IL_MODCOMP_0       : ILModDstComp<2, "0">;
def IL_MODCOMP_1       : ILModDstComp<3, "1">;

// Section 5.5 IL Import Usage
class ILImportUsage<bits<1> val, string usage> {
    bits<1> Value = val;
    string Text = usage;
}
// Table 5-5
def IL_IMPORTUSAGE_WINCOORD : ILImportUsage<0, "_usage(wincoord)">;

// Section 5.6 Il Shift Scale
class ILShiftScale<bits<4> val, string scale> {
    bits<4> Value = val;
    string Text = scale;
}

// Table 5-6
def IL_SHIFT_NONE   : ILShiftScale<0, "">;
def IL_SHIFT_X2     : ILShiftScale<1, "_x2">;
def IL_SHIFT_X4     : ILShiftScale<2, "_x4">;
def IL_SHIFT_X8     : ILShiftScale<3, "_x8">;
def IL_SHIFT_D2     : ILShiftScale<4, "_d2">;
def IL_SHIFT_D4     : ILShiftScale<5, "_d4">;
def IL_SHIFT_D8     : ILShiftScale<6, "_d8">;

// Section 5.7 IL Divide Component
class ILDivComp<bits<3> val, string divcomp> {
    bits<3> Value = val;
    string Text = divcomp;
}

// Table 5-7
def IL_DIVCOMP_NONE : ILDivComp<0, "_divcomp(none)">;
def IL_DIVCOMP_Y    : ILDivComp<1, "_divcomp(y)">;
def IL_DIVCOMP_Z    : ILDivComp<2, "_divcomp(z)">;
def IL_DIVCOMP_W    : ILDivComp<3, "_divcomp(w)">;
//def IL_DIVCOMP_UNKNOWN : ILDivComp<4, "_divcomp(unknown)">;

// Section 5.8 IL Relational Op
class ILRelOp<bits<3> val, string op> {
    bits<3> Value = val;
    string Text = op;
}

// Table 5-8
def IL_RELOP_EQ : ILRelOp<0, "_relop(eq)">;
def IL_RELOP_NE : ILRelOp<1, "_relop(ne)">;
def IL_RELOP_GT : ILRelOp<2, "_relop(gt)">;
def IL_RELOP_GE : ILRelOp<3, "_relop(ge)">;
def IL_RELOP_LT : ILRelOp<4, "_relop(lt)">;
def IL_RELOP_LE : ILRelOp<5, "_relop(le)">;

// Section 5.9 IL Zero Op
class ILZeroOp<bits<3> val, string behavior> {
    bits<3> Value = val;
    string Text = behavior;
}

// Table 5-9
def IL_ZEROOP_FLTMAX    : ILZeroOp<0, "_zeroop(fltmax)">;
def IL_ZEROOP_0         : ILZeroOp<1, "_zeroop(zero)">;
def IL_ZEROOP_INFINITY  : ILZeroOp<2, "_zeroop(infinity)">;
def IL_ZEROOP_INF_ELSE_MAX : ILZeroOp<3, "_zeroop(inf_else_max)">;

// Section 5.10 IL Cmp Value
class ILCmpValue<bits<3> val, string num> {
    bits<3> Value = val;
    string Text = num;
}

// Table 5-10
def IL_CMPVAL_0_0     : ILCmpValue<0, "0.0">;
def IL_CMPVAL_0_5     : ILCmpValue<1, "0.5">;
def IL_CMPVAL_1_0     : ILCmpValue<2, "1.0">;
def IL_CMPVAL_NEG_0_5 : ILCmpValue<3, "-0.5">;
def IL_CMPVAL_NEG_1_0 : ILCmpValue<4, "-1.0">;

// Section 5.11 IL Addressing
class ILAddressing<bits<3> val> {
    bits<3> Value = val;
}

// Table 5-11
def IL_ADDR_ABSOLUTE     : ILAddressing<0>;
def IL_ADDR_RELATIVE     : ILAddressing<1>;
def IL_ADDR_REG_RELATIVE : ILAddressing<2>;

// Section 5.11 IL Element Format
class ILElementFormat<bits<5> val> {
    bits<5> Value = val;
}

// Table 5-11
def IL_ELEMENTFORMAT_UNKNOWN : ILElementFormat<0>;
def IL_ELEMENTFORMAT_SNORM   : ILElementFormat<1>;
def IL_ELEMENTFORMAT_UNORM   : ILElementFormat<2>;
def IL_ELEMENTFORMAT_SINT    : ILElementFormat<3>;
def IL_ELEMENTFORMAT_UINT    : ILElementFormat<4>;
def IL_ELEMENTFORMAT_FLOAT   : ILElementFormat<5>;
def IL_ELEMENTFORMAT_SRGB    : ILElementFormat<6>;
def IL_ELEMENTFORMAT_MIXED   : ILElementFormat<7>;
def IL_ELEMENTFORMAT_Last    : ILElementFormat<8>;

// Section 5.12 IL Op Code
class ILOpCode<bits<16> val = -1, string cmd> {
    bits<16> Value = val;
    string Text = cmd;
}

// Table 5-12
def IL_DCL_CONST_BUFFER         : ILOpCode<0, "dcl_cb">;
def IL_DCL_INDEXED_TEMP_ARRAY   : ILOpCode<1, "dcl_index_temp_array">;
def IL_DCL_INPUT                : ILOpCode<2, "dcl_input">;
def IL_DCL_LITERAL              : ILOpCode<3, "dcl_literal">;
def IL_DCL_OUTPUT               : ILOpCode<4, "dcl_output">;
def IL_DCL_RESOURCE             : ILOpCode<5, "dcl_resource">;
def IL_OP_ABS                   : ILOpCode<6, "abs">;
def IL_OP_ADD                   : ILOpCode<7, "add">;
def IL_OP_AND                   : ILOpCode<8, "iand">;
def IL_OP_BREAK                 : ILOpCode<9, "break">;
def IL_OP_BREAK_LOGICALNZ       : ILOpCode<10, "break_logicalnz">;
def IL_OP_BREAK_LOGICALZ        : ILOpCode<11, "break_logicalz">;
def IL_OP_BREAKC                : ILOpCode<12, "breakc">;
def IL_OP_CALL                  : ILOpCode<13, "call">;
def IL_OP_CALL_LOGICALNZ        : ILOpCode<14, "call_logicalnz">;
def IL_OP_CALL_LOGICALZ         : ILOpCode<15, "call_logicalz">;
def IL_OP_CASE                  : ILOpCode<16, "case">;
def IL_OP_CLG                   : ILOpCode<17, "clg">;
def IL_OP_CMOV                  : ILOpCode<18, "cmov">;
def IL_OP_CMOV_LOGICAL          : ILOpCode<19, "cmov_logical">;
def IL_OP_CMP                   : ILOpCode<20, "cmp">;
def IL_OP_CONTINUE              : ILOpCode<21, "continue">;
def IL_OP_CONTINUE_LOGICALNZ    : ILOpCode<22, "continue_logicalnz">;
def IL_OP_CONTINUE_LOGICALZ     : ILOpCode<23, "continue_logicalz">;
def IL_OP_CONTINUEC             : ILOpCode<24, "continuec">;
def IL_OP_COS                   : ILOpCode<25, "cos">;
def IL_OP_COS_VEC               : ILOpCode<26, "cos_vec">;
def IL_OP_D_2_F                 : ILOpCode<27, "d2f">;
def IL_OP_D_ADD                 : ILOpCode<28, "dadd">;
def IL_OP_D_EQ                  : ILOpCode<29, "deq">;
def IL_OP_D_FRC                 : ILOpCode<30, "dfrac">;
def IL_OP_D_FREXP               : ILOpCode<31, "dfrexp">;
def IL_OP_D_GE                  : ILOpCode<32, "dge">;
def IL_OP_D_LDEXP               : ILOpCode<33, "dldexp">;
def IL_OP_D_LT                  : ILOpCode<34, "dlt">;
def IL_OP_D_MAD                 : ILOpCode<35, "dmad">;
def IL_OP_D_MUL                 : ILOpCode<36, "dmul">;
def IL_OP_D_NE                  : ILOpCode<37, "dne">;
def IL_OP_DEFAULT               : ILOpCode<38, "default">;
def IL_OP_DISCARD_LOGICALNZ     : ILOpCode<39, "discard_logicalnz">;
def IL_OP_DISCARD_LOGICALZ      : ILOpCode<40, "discard_logicalz">;
def IL_OP_DIV                   : ILOpCode<41, "div_zeroop(infinity)">;
def IL_OP_DP2                   : ILOpCode<42, "dp2">;
def IL_OP_DP3                   : ILOpCode<43, "dp3">;
def IL_OP_DP4                   : ILOpCode<44, "dp4">;
def IL_OP_ELSE                  : ILOpCode<45, "else">;
def IL_OP_END                   : ILOpCode<46, "end">;
def IL_OP_ENDFUNC               : ILOpCode<47, "endfunc">;
def IL_OP_ENDIF                 : ILOpCode<48, "endif">;
def IL_OP_ENDLOOP               : ILOpCode<49, "endloop">;
def IL_OP_ENDMAIN               : ILOpCode<50, "endmain">;
def IL_OP_ENDSWITCH             : ILOpCode<51, "endswitch">;
def IL_OP_EQ                    : ILOpCode<52, "eq">;
def IL_OP_EXP                   : ILOpCode<53, "exp">;
def IL_OP_EXP_VEC               : ILOpCode<54, "exp_vec">;
def IL_OP_F_2_D                 : ILOpCode<55, "f2d">;
def IL_OP_FLR                   : ILOpCode<56, "flr">;
def IL_OP_FRC                   : ILOpCode<57, "frc">;
def IL_OP_FTOI                  : ILOpCode<58, "ftoi">;
def IL_OP_FTOU                  : ILOpCode<59, "ftou">;
def IL_OP_FUNC                  : ILOpCode<60, "func">;
def IL_OP_GE                    : ILOpCode<61, "ge">;
def IL_OP_I_ADD                 : ILOpCode<62, "iadd">;
def IL_OP_I_EQ                  : ILOpCode<63, "ieq">;
def IL_OP_I_GE                  : ILOpCode<64, "ige">;
def IL_OP_I_LT                  : ILOpCode<65, "ilt">;
def IL_OP_I_MAD                 : ILOpCode<66, "imad">;
def IL_OP_I_MAX                 : ILOpCode<67, "imax">;
def IL_OP_I_MIN                 : ILOpCode<68, "imin">;
def IL_OP_I_MUL                 : ILOpCode<69, "imul">;
def IL_OP_I_MUL_HIGH            : ILOpCode<70, "imul_high">;
def IL_OP_I_NE                  : ILOpCode<71, "ine">;
def IL_OP_I_NEGATE              : ILOpCode<72, "inegate">;
def IL_OP_I_NOT                 : ILOpCode<73, "inot">;
def IL_OP_I_OR                  : ILOpCode<74, "ior">;
def IL_OP_I_SHL                 : ILOpCode<75, "ishl">;
def IL_OP_I_SHR                 : ILOpCode<76, "ishr">;
def IL_OP_I_XOR                 : ILOpCode<77, "ixor">;
def IL_OP_IF_LOGICALNZ          : ILOpCode<78, "if_logicalnz">;
def IL_OP_IF_LOGICALZ           : ILOpCode<79, "if_logicalz">;
def IL_OP_IFC                   : ILOpCode<80, "ifc">;
def IL_OP_ITOF                  : ILOpCode<81, "itof">;
def IL_OP_LN                    : ILOpCode<82, "ln">;
def IL_OP_LOG                   : ILOpCode<83, "log">;
def IL_OP_LOG_VEC               : ILOpCode<84, "log_vec">;
def IL_OP_LOOP                  : ILOpCode<85, "loop">;
def IL_OP_LT                    : ILOpCode<86, "lt">;
def IL_OP_MAD                   : ILOpCode<87, "mad_ieee">;
def IL_OP_MAX                   : ILOpCode<88, "max_ieee">;
def IL_OP_MIN                   : ILOpCode<89, "min_ieee">;
def IL_OP_MOD                   : ILOpCode<90, "mod_ieee">;
def IL_OP_MOV                   : ILOpCode<91, "mov">;
def IL_OP_MUL_IEEE              : ILOpCode<92, "mul_ieee">;
def IL_OP_NE                    : ILOpCode<93, "ne">;
def IL_OP_NRM                   : ILOpCode<94, "nrm_nrm4_zeroop(zero)">;
def IL_OP_POW                   : ILOpCode<95, "pow">;
def IL_OP_RCP                   : ILOpCode<96, "rcp">;
def IL_OP_RET                   : ILOpCode<97, "ret">;
def IL_OP_RET_DYN               : ILOpCode<98, "ret_dyn">;
def IL_OP_RET_LOGICALNZ         : ILOpCode<99, "ret_logicalnz">;
def IL_OP_RET_LOGICALZ          : ILOpCode<100, "ret_logicalz">;
def IL_OP_RND                   : ILOpCode<101, "rnd">;
def IL_OP_ROUND_NEAR            : ILOpCode<102, "round_nearest">;
def IL_OP_ROUND_NEG_INF         : ILOpCode<103, "round_neginf">;
def IL_OP_ROUND_POS_INF         : ILOpCode<104, "round_plusinf">;
def IL_OP_ROUND_ZERO            : ILOpCode<105, "round_z">;
def IL_OP_RSQ                   : ILOpCode<106, "rsq">;
def IL_OP_RSQ_VEC               : ILOpCode<107, "rsq_vec">;
def IL_OP_SAMPLE                : ILOpCode<108, "sample">;
def IL_OP_SAMPLE_L              : ILOpCode<109, "sample_l">;
def IL_OP_SET                   : ILOpCode<110, "set">;
def IL_OP_SGN                   : ILOpCode<111, "sgn">;
def IL_OP_SIN                   : ILOpCode<112, "sin">;
def IL_OP_SIN_VEC               : ILOpCode<113, "sin_vec">;
def IL_OP_SUB                   : ILOpCode<114, "sub">;
def IL_OP_SWITCH                : ILOpCode<115, "switch">;
def IL_OP_TRC                   : ILOpCode<116, "trc">;
def IL_OP_U_DIV                 : ILOpCode<117, "udiv">;
def IL_OP_U_GE                  : ILOpCode<118, "uge">;
def IL_OP_U_LT                  : ILOpCode<119, "ult">;
def IL_OP_U_MAD                 : ILOpCode<120, "umad">;
def IL_OP_U_MAX                 : ILOpCode<121, "umax">;
def IL_OP_U_MIN                 : ILOpCode<122, "umin">;
def IL_OP_U_MOD                 : ILOpCode<123, "umod">;
def IL_OP_U_MUL                 : ILOpCode<124, "umul">;
def IL_OP_U_MUL_HIGH            : ILOpCode<125, "umul_high">;
def IL_OP_U_SHR                 : ILOpCode<126, "ushr">;
def IL_OP_UTOF                  : ILOpCode<127, "utof">;
def IL_OP_WHILE                 : ILOpCode<128, "whileloop">;
// SC IL instructions that are not in CAL IL
def IL_OP_ACOS                  : ILOpCode<129, "acos">;
def IL_OP_ASIN                  : ILOpCode<130, "asin">;
def IL_OP_EXN                   : ILOpCode<131, "exn">;
def IL_OP_UBIT_REVERSE          : ILOpCode<132, "ubit_reverse">;
def IL_OP_UBIT_EXTRACT          : ILOpCode<133, "ubit_extract">;
def IL_OP_IBIT_EXTRACT          : ILOpCode<134, "ibit_extract">;
def IL_OP_SQRT                  : ILOpCode<135, "sqrt">;
def IL_OP_SQRT_VEC              : ILOpCode<136, "sqrt_vec">;
def IL_OP_ATAN                  : ILOpCode<137, "atan">;
def IL_OP_TAN                   : ILOpCode<137, "tan">;
def IL_OP_D_DIV                 : ILOpCode<138, "ddiv">;
def IL_OP_F_NEG                 : ILOpCode<139, "mov">;
def IL_OP_GT                    : ILOpCode<140, "gt">;
def IL_OP_LE                    : ILOpCode<141, "lt">;
def IL_OP_DIST                  : ILOpCode<142, "dist">;
def IL_OP_LEN                   : ILOpCode<143, "len">;
def IL_OP_MACRO                 : ILOpCode<144, "mcall">;
def IL_OP_INTR                  : ILOpCode<145, "call">;
def IL_OP_I_FFB_HI              : ILOpCode<146, "ffb_hi">;
def IL_OP_I_FFB_LO              : ILOpCode<147, "ffb_lo">;
def IL_OP_BARRIER               : ILOpCode<148, "fence_threads_memory_lds">;
def IL_OP_BARRIER_LOCAL         : ILOpCode<149, "fence_threads_lds">;
def IL_OP_BARRIER_GLOBAL        : ILOpCode<150, "fence_threads_memory">;
def IL_OP_FENCE                 : ILOpCode<151, "fence_lds_memory">;
def IL_OP_FENCE_READ_ONLY       : ILOpCode<152, "fence_lds_mem_read_only">;
def IL_OP_FENCE_WRITE_ONLY      : ILOpCode<153, "fence_lds_mem_write_only">;
def IL_PSEUDO_INST              : ILOpCode<154, ";Pseudo Op">;
def IL_OP_UNPACK_0              : ILOpCode<155, "unpack0">;
def IL_OP_UNPACK_1              : ILOpCode<156, "unpack1">;
def IL_OP_UNPACK_2              : ILOpCode<157, "unpack2">;
def IL_OP_UNPACK_3              : ILOpCode<158, "unpack3">;
def IL_OP_PI_REDUCE             : ILOpCode<159, "pireduce">;
def IL_OP_IBIT_COUNT            : ILOpCode<160, "icbits">;
def IL_OP_I_FFB_SGN             : ILOpCode<161, "ffb_shi">;
def IL_OP_F2U4                  : ILOpCode<162, "f_2_u4">;
def IL_OP_BIT_ALIGN             : ILOpCode<163, "bitalign">;
def IL_OP_BYTE_ALIGN            : ILOpCode<164, "bytealign">;
def IL_OP_U4_LERP               : ILOpCode<165, "u4lerp">;
def IL_OP_SAD                   : ILOpCode<166, "sad">;
def IL_OP_SAD_HI                : ILOpCode<167, "sadhi">;
def IL_OP_SAD4                  : ILOpCode<168, "sad4">;
def IL_OP_UBIT_INSERT           : ILOpCode<169, "ubit_insert">;
def IL_OP_I_CARRY               : ILOpCode<170, "icarry">;
def IL_OP_I_BORROW              : ILOpCode<171, "iborrow">;
def IL_OP_U_MAD24               : ILOpCode<172, "umad24">;
def IL_OP_U_MUL24               : ILOpCode<173, "umul24">;
def IL_OP_I_MAD24               : ILOpCode<174, "imad24">;
def IL_OP_I_MUL24               : ILOpCode<175, "imul24">;
def IL_OP_CLAMP                 : ILOpCode<176, "clamp">;
def IL_OP_LERP                  : ILOpCode<177, "lrp">;
def IL_OP_FMA                   : ILOpCode<178, "fma">;
def IL_OP_D_MIN                 : ILOpCode<179, "dmin">;
def IL_OP_D_MAX                 : ILOpCode<180, "dmax">;
def IL_OP_D_SQRT                : ILOpCode<181, "dsqrt">;
def IL_OP_DP2_ADD               : ILOpCode<182, "dp2add">;
def IL_OP_F16_TO_F32            : ILOpCode<183, "f162f">;
def IL_OP_F32_TO_F16            : ILOpCode<184, "f2f16">;
def IL_REG_LOCAL_ID_FLAT        : ILOpCode<185, "vTidInGrpFlat">;
def IL_REG_LOCAL_ID             : ILOpCode<186, "vTidInGrp">;
def IL_REG_GLOBAL_ID_FLAT       : ILOpCode<187, "vAbsTidFlag">;
def IL_REG_GLOBAL_ID            : ILOpCode<188, "vAbsTid">;
def IL_REG_GROUP_ID_FLAT        : ILOpCode<189, "vThreadGrpIDFlat">;
def IL_REG_GROUP_ID             : ILOpCode<190, "vThreadGrpID">;
def IL_OP_D_RCP                 : ILOpCode<191, "drcp_zeroop(infinity)">;
def IL_OP_D_RSQ                 : ILOpCode<192, "drsq_zeroop(infinity)">;
def IL_OP_D_MOV                 : ILOpCode<193, "dmov">;
def IL_OP_D_MOVC                : ILOpCode<194, "dmovc">;
def IL_OP_NOP                   : ILOpCode<195, "nop">;
def IL_OP_UAV_ADD               : ILOpCode<196, "uav_add">;
def IL_OP_UAV_AND               : ILOpCode<197, "uav_and">;
def IL_OP_UAV_MAX               : ILOpCode<198, "uav_max">;
def IL_OP_UAV_MIN               : ILOpCode<199, "uav_min">;
def IL_OP_UAV_OR                : ILOpCode<200, "uav_or">;
def IL_OP_UAV_RSUB              : ILOpCode<201, "uav_rsub">;
def IL_OP_UAV_SUB               : ILOpCode<202, "uav_sub">;
def IL_OP_UAV_UMAX              : ILOpCode<203, "uav_umax">;
def IL_OP_UAV_UMIN              : ILOpCode<204, "uav_umin">;
def IL_OP_UAV_XOR               : ILOpCode<205, "uav_xor">;
def IL_OP_UAV_INC               : ILOpCode<206, "uav_uinc">;
def IL_OP_UAV_DEC               : ILOpCode<207, "uav_udec">;
def IL_OP_UAV_CMP               : ILOpCode<208, "uav_cmp">;
def IL_OP_UAV_READ_ADD          : ILOpCode<209, "uav_read_add">;
def IL_OP_UAV_READ_AND          : ILOpCode<210, "uav_read_and">;
def IL_OP_UAV_READ_MAX          : ILOpCode<211, "uav_read_max">;
def IL_OP_UAV_READ_MIN          : ILOpCode<212, "uav_read_min">;
def IL_OP_UAV_READ_OR           : ILOpCode<213, "uav_read_or">;
def IL_OP_UAV_READ_RSUB         : ILOpCode<214, "uav_read_rsub">;
def IL_OP_UAV_READ_SUB          : ILOpCode<215, "uav_read_sub">;
def IL_OP_UAV_READ_UMAX         : ILOpCode<216, "uav_read_umax">;
def IL_OP_UAV_READ_UMIN         : ILOpCode<217, "uav_read_umin">;
def IL_OP_UAV_READ_XOR          : ILOpCode<218, "uav_read_xor">;
def IL_OP_UAV_READ_INC          : ILOpCode<219, "uav_read_uinc">;
def IL_OP_UAV_READ_DEC          : ILOpCode<220, "uav_read_udec">;
def IL_OP_UAV_READ_XCHG         : ILOpCode<221, "uav_read_xchg">;
def IL_OP_UAV_READ_CMPXCHG      : ILOpCode<222, "uav_read_cmp_xchg">;
def IL_OP_LDS_ADD               : ILOpCode<223, "lds_add">;
def IL_OP_LDS_AND               : ILOpCode<224, "lds_and">;
def IL_OP_LDS_MAX               : ILOpCode<225, "lds_max">;
def IL_OP_LDS_MIN               : ILOpCode<226, "lds_min">;
def IL_OP_LDS_OR                : ILOpCode<227, "lds_or">;
def IL_OP_LDS_RSUB              : ILOpCode<228, "lds_rsub">;
def IL_OP_LDS_SUB               : ILOpCode<229, "lds_sub">;
def IL_OP_LDS_UMAX              : ILOpCode<230, "lds_umax">;
def IL_OP_LDS_UMIN              : ILOpCode<231, "lds_umin">;
def IL_OP_LDS_XOR               : ILOpCode<232, "lds_xor">;
def IL_OP_LDS_INC               : ILOpCode<233, "lds_inc">;
def IL_OP_LDS_DEC               : ILOpCode<234, "lds_dec">;
def IL_OP_LDS_CMP               : ILOpCode<235, "lds_cmp">;
def IL_OP_LDS_READ_ADD          : ILOpCode<236, "lds_read_add">;
def IL_OP_LDS_READ_AND          : ILOpCode<237, "lds_read_and">;
def IL_OP_LDS_READ_MAX          : ILOpCode<238, "lds_read_max">;
def IL_OP_LDS_READ_MIN          : ILOpCode<239, "lds_read_min">;
def IL_OP_LDS_READ_OR           : ILOpCode<240, "lds_read_or">;
def IL_OP_LDS_READ_RSUB         : ILOpCode<241, "lds_read_rsub">;
def IL_OP_LDS_READ_SUB          : ILOpCode<242, "lds_read_sub">;
def IL_OP_LDS_READ_UMAX         : ILOpCode<243, "lds_read_umax">;
def IL_OP_LDS_READ_UMIN         : ILOpCode<244, "lds_read_umin">;
def IL_OP_LDS_READ_XOR          : ILOpCode<245, "lds_read_xor">;
def IL_OP_LDS_READ_INC          : ILOpCode<246, "lds_read_inc">;
def IL_OP_LDS_READ_DEC          : ILOpCode<247, "lds_read_dec">;
def IL_OP_LDS_READ_XCHG         : ILOpCode<248, "lds_read_xchg">;
def IL_OP_LDS_READ_CMPXCHG      : ILOpCode<249, "lds_read_cmp_xchg">;
def IL_OP_GDS_ADD               : ILOpCode<250, "gds_add">;
def IL_OP_GDS_AND               : ILOpCode<251, "gds_and">;
def IL_OP_GDS_MAX               : ILOpCode<252, "gds_max">;
def IL_OP_GDS_MIN               : ILOpCode<253, "gds_min">;
def IL_OP_GDS_OR                : ILOpCode<254, "gds_or">;
def IL_OP_GDS_RSUB              : ILOpCode<255, "gds_rsub">;
def IL_OP_GDS_SUB               : ILOpCode<256, "gds_sub">;
def IL_OP_GDS_UMAX              : ILOpCode<257, "gds_umax">;
def IL_OP_GDS_UMIN              : ILOpCode<258, "gds_umin">;
def IL_OP_GDS_MSKOR             : ILOpCode<259, "gds_mskor">;
def IL_OP_GDS_XOR               : ILOpCode<260, "gds_xor">;
def IL_OP_GDS_INC               : ILOpCode<261, "gds_inc">;
def IL_OP_GDS_DEC               : ILOpCode<262, "gds_dec">;
def IL_OP_GDS_CMP               : ILOpCode<263, "gds_cmp">;
def IL_OP_GDS_READ_ADD          : ILOpCode<264, "gds_read_add">;
def IL_OP_GDS_READ_AND          : ILOpCode<265, "gds_read_and">;
def IL_OP_GDS_READ_MAX          : ILOpCode<266, "gds_read_max">;
def IL_OP_GDS_READ_MIN          : ILOpCode<267, "gds_read_min">;
def IL_OP_GDS_READ_OR           : ILOpCode<268, "gds_read_or">;
def IL_OP_GDS_READ_RSUB         : ILOpCode<269, "gds_read_rsub">;
def IL_OP_GDS_READ_SUB          : ILOpCode<270, "gds_read_sub">;
def IL_OP_GDS_READ_UMAX         : ILOpCode<271, "gds_read_umax">;
def IL_OP_GDS_READ_UMIN         : ILOpCode<272, "gds_read_umin">;
def IL_OP_GDS_READ_MSKOR        : ILOpCode<273, "gds_read_mskor">;
def IL_OP_GDS_READ_XOR          : ILOpCode<274, "gds_read_xor">;
def IL_OP_GDS_READ_INC          : ILOpCode<275, "gds_read_inc">;
def IL_OP_GDS_READ_DEC          : ILOpCode<276, "gds_read_dec">;
def IL_OP_GDS_READ_XCHG         : ILOpCode<277, "gds_read_xchg">;
def IL_OP_GDS_READ_CMPXCHG      : ILOpCode<278, "gds_read_cmp_xchg">;
def IL_OP_APPEND_BUF_ALLOC      : ILOpCode<279, "append_buf_alloc">;
def IL_OP_APPEND_BUF_CONSUME    : ILOpCode<280, "append_buf_consume">;
def IL_OP_I64_ADD               : ILOpCode<281, "i64add">;
def IL_OP_I64_MAX               : ILOpCode<282, "i64max">;
def IL_OP_U64_MAX               : ILOpCode<283, "u64max">;
def IL_OP_I64_MIN               : ILOpCode<284, "i64min">;
def IL_OP_U64_MIN               : ILOpCode<285, "u64min">;
def IL_OP_I64_NEGATE            : ILOpCode<286, "i64negate">;
def IL_OP_I64_SHL               : ILOpCode<287, "i64shl">;
def IL_OP_I64_SHR               : ILOpCode<288, "i64shr">;
def IL_OP_U64_SHR               : ILOpCode<289, "u64shr">;
def IL_OP_I64_EQ                : ILOpCode<290, "i64eq">;
def IL_OP_I64_GE                : ILOpCode<291, "i64ge">;
def IL_OP_U64_GE                : ILOpCode<292, "u64ge">;
def IL_OP_I64_LT                : ILOpCode<293, "i64lt">;
def IL_OP_U64_LT                : ILOpCode<294, "u64lt">;
def IL_OP_I64_NE                : ILOpCode<295, "i64ne">;
def IL_OP_U_MULHI24             : ILOpCode<296, "umul24_high">;
def IL_OP_I_MULHI24             : ILOpCode<297, "imul24_high">;
def IL_OP_GDS_LOAD              : ILOpCode<298, "gds_load">;
def IL_OP_GDS_STORE             : ILOpCode<299, "gds_store">;
def IL_OP_LDS_LOAD              : ILOpCode<300, "lds_load">;
def IL_OP_LDS_LOAD_VEC          : ILOpCode<301, "lds_load_vec">;
def IL_OP_LDS_LOAD_BYTE         : ILOpCode<302, "lds_load_byte">;
def IL_OP_LDS_LOAD_UBYTE        : ILOpCode<303, "lds_load_ubyte">;
def IL_OP_LDS_LOAD_SHORT        : ILOpCode<304, "lds_load_short">;
def IL_OP_LDS_LOAD_USHORT       : ILOpCode<305, "lds_load_ushort">;
def IL_OP_LDS_STORE             : ILOpCode<306, "lds_store">;
def IL_OP_LDS_STORE_VEC         : ILOpCode<307, "lds_store_vec">;
def IL_OP_LDS_STORE_BYTE        : ILOpCode<308, "lds_store_byte">;
def IL_OP_LDS_STORE_SHORT       : ILOpCode<309, "lds_store_short">;
def IL_OP_RAW_UAV_LOAD          : ILOpCode<310, "uav_raw_load">;
def IL_OP_RAW_UAV_STORE         : ILOpCode<311, "uav_raw_store">;
def IL_OP_ARENA_UAV_LOAD        : ILOpCode<312, "uav_arena_load">;
def IL_OP_ARENA_UAV_STORE       : ILOpCode<313, "uav_arena_store">;
def IL_OP_LDS_MSKOR             : ILOpCode<314, "lds_mskor">;
def IL_OP_LDS_READ_MSKOR        : ILOpCode<315, "lds_read_mskor">;
def IL_OP_UAV_BYTE_LOAD         : ILOpCode<316, "uav_byte_load">;
def IL_OP_UAV_UBYTE_LOAD        : ILOpCode<317, "uav_ubyte_load">;
def IL_OP_UAV_SHORT_LOAD        : ILOpCode<318, "uav_short_load">;
def IL_OP_UAV_USHORT_LOAD       : ILOpCode<319, "uav_ushort_load">;
def IL_OP_UAV_BYTE_STORE        : ILOpCode<320, "uav_byte_store">;
def IL_OP_UAV_SHORT_STORE       : ILOpCode<320, "uav_short_store">;
def IL_OP_UAV_STORE             : ILOpCode<321, "uav_store">;
def IL_OP_UAV_LOAD              : ILOpCode<322, "uav_load">;
def IL_OP_MUL                   : ILOpCode<323, "mul">;
def IL_OP_DIV_INF               : ILOpCode<324, "div_zeroop(infinity)">;
def IL_OP_DIV_FLTMAX            : ILOpCode<325, "div_zeroop(fltmax)">;
def IL_OP_DIV_ZERO              : ILOpCode<326, "div_zeroop(zero)">;
def IL_OP_DIV_INFELSEMAX        : ILOpCode<327, "div_zeroop(inf_else_max)">;
def IL_OP_FTOI_FLR              : ILOpCode<328, "ftoi_flr">;
def IL_OP_FTOI_RPI              : ILOpCode<329, "ftoi_rpi">;
def IL_OP_F32_TO_F16_NEAR       : ILOpCode<330, "f2f16_near">;
def IL_OP_F32_TO_F16_NEG_INF    : ILOpCode<331, "f2f16_neg_inf">;
def IL_OP_F32_TO_F16_PLUS_INF   : ILOpCode<332, "f2f16_plus_inf">;
def IL_OP_I64_MUL               : ILOpCode<333, "i64mul">;
def IL_OP_U64_MUL               : ILOpCode<334, "u64mul">;
def IL_OP_CU_ID                 : ILOpCode<355, "cu_id">;
def IL_OP_WAVE_ID               : ILOpCode<356, "wave_id">;
def IL_OP_I64_SUB               : ILOpCode<357, "i64sub">;
def IL_OP_I64_DIV               : ILOpCode<358, "i64div">;
def IL_OP_U64_DIV               : ILOpCode<359, "u64div">;
def IL_OP_I64_MOD               : ILOpCode<360, "i64mod">;
def IL_OP_U64_MOD               : ILOpCode<361, "u64mod">;
def IL_DCL_GWS_THREAD_COUNT     : ILOpCode<362, "dcl_gws_thread_count">;
def IL_DCL_SEMAPHORE            : ILOpCode<363, "dcl_semaphore">;
def IL_OP_SEMAPHORE_INIT        : ILOpCode<364, "init_semaphore">;
def IL_OP_SEMAPHORE_WAIT        : ILOpCode<365, "semaphore_wait">;
def IL_OP_SEMAPHORE_SIGNAL      : ILOpCode<366, "semaphore_signal">;
def IL_OP_BARRIER_REGION        : ILOpCode<377, "fence_threads_gds">;
def IL_OP_BFI                   : ILOpCode<394, "bfi">;
def IL_OP_BFM                   : ILOpCode<395, "bfm">;
def IL_DBG_STRING               : ILOpCode<396, "dbg_string">;
def IL_DBG_LINE                 : ILOpCode<397, "dbg_line">;
def IL_DBG_TEMPLOC              : ILOpCode<398, "dbg_temploc">;
