//===-- TargetLoweringBase.cpp - Implement the TargetLoweringBase class ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements the TargetLoweringBase class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetLowering.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Triple.h"
#include "llvm/CodeGen/Analysis.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/CodeGen/StackMaps.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Mangler.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include <cctype>
using namespace llvm;

/// InitLibcallNames - Set default libcall names.
///
static void InitLibcallNames(const char **Names, const TargetMachine &TM) {
  Names[RTLIB::SHL_I16] = "__ashlhi3";
  Names[RTLIB::SHL_I32] = "__ashlsi3";
  Names[RTLIB::SHL_I64] = "__ashldi3";
  Names[RTLIB::SHL_I128] = "__ashlti3";
  Names[RTLIB::SRL_I16] = "__lshrhi3";
  Names[RTLIB::SRL_I32] = "__lshrsi3";
  Names[RTLIB::SRL_I64] = "__lshrdi3";
  Names[RTLIB::SRL_I128] = "__lshrti3";
  Names[RTLIB::SRA_I16] = "__ashrhi3";
  Names[RTLIB::SRA_I32] = "__ashrsi3";
  Names[RTLIB::SRA_I64] = "__ashrdi3";
  Names[RTLIB::SRA_I128] = "__ashrti3";
  Names[RTLIB::MUL_I8] = "__mulqi3";
  Names[RTLIB::MUL_I16] = "__mulhi3";
  Names[RTLIB::MUL_I32] = "__mulsi3";
  Names[RTLIB::MUL_I64] = "__muldi3";
  Names[RTLIB::MUL_I128] = "__multi3";
  Names[RTLIB::MULO_I32] = "__mulosi4";
  Names[RTLIB::MULO_I64] = "__mulodi4";
  Names[RTLIB::MULO_I128] = "__muloti4";
  Names[RTLIB::SDIV_I8] = "__divqi3";
  Names[RTLIB::SDIV_I16] = "__divhi3";
  Names[RTLIB::SDIV_I32] = "__divsi3";
  Names[RTLIB::SDIV_I64] = "__divdi3";
  Names[RTLIB::SDIV_I128] = "__divti3";
  Names[RTLIB::UDIV_I8] = "__udivqi3";
  Names[RTLIB::UDIV_I16] = "__udivhi3";
  Names[RTLIB::UDIV_I32] = "__udivsi3";
  Names[RTLIB::UDIV_I64] = "__udivdi3";
  Names[RTLIB::UDIV_I128] = "__udivti3";
  Names[RTLIB::SREM_I8] = "__modqi3";
  Names[RTLIB::SREM_I16] = "__modhi3";
  Names[RTLIB::SREM_I32] = "__modsi3";
  Names[RTLIB::SREM_I64] = "__moddi3";
  Names[RTLIB::SREM_I128] = "__modti3";
  Names[RTLIB::UREM_I8] = "__umodqi3";
  Names[RTLIB::UREM_I16] = "__umodhi3";
  Names[RTLIB::UREM_I32] = "__umodsi3";
  Names[RTLIB::UREM_I64] = "__umoddi3";
  Names[RTLIB::UREM_I128] = "__umodti3";

  // These are generally not available.
  Names[RTLIB::SDIVREM_I8] = 0;
  Names[RTLIB::SDIVREM_I16] = 0;
  Names[RTLIB::SDIVREM_I32] = 0;
  Names[RTLIB::SDIVREM_I64] = 0;
  Names[RTLIB::SDIVREM_I128] = 0;
  Names[RTLIB::UDIVREM_I8] = 0;
  Names[RTLIB::UDIVREM_I16] = 0;
  Names[RTLIB::UDIVREM_I32] = 0;
  Names[RTLIB::UDIVREM_I64] = 0;
  Names[RTLIB::UDIVREM_I128] = 0;

  Names[RTLIB::NEG_I32] = "__negsi2";
  Names[RTLIB::NEG_I64] = "__negdi2";
  Names[RTLIB::ADD_F32] = "__addsf3";
  Names[RTLIB::ADD_F64] = "__adddf3";
  Names[RTLIB::ADD_F80] = "__addxf3";
  Names[RTLIB::ADD_F128] = "__addtf3";
  Names[RTLIB::ADD_PPCF128] = "__gcc_qadd";
  Names[RTLIB::SUB_F32] = "__subsf3";
  Names[RTLIB::SUB_F64] = "__subdf3";
  Names[RTLIB::SUB_F80] = "__subxf3";
  Names[RTLIB::SUB_F128] = "__subtf3";
  Names[RTLIB::SUB_PPCF128] = "__gcc_qsub";
  Names[RTLIB::MUL_F32] = "__mulsf3";
  Names[RTLIB::MUL_F64] = "__muldf3";
  Names[RTLIB::MUL_F80] = "__mulxf3";
  Names[RTLIB::MUL_F128] = "__multf3";
  Names[RTLIB::MUL_PPCF128] = "__gcc_qmul";
  Names[RTLIB::DIV_F32] = "__divsf3";
  Names[RTLIB::DIV_F64] = "__divdf3";
  Names[RTLIB::DIV_F80] = "__divxf3";
  Names[RTLIB::DIV_F128] = "__divtf3";
  Names[RTLIB::DIV_PPCF128] = "__gcc_qdiv";
  Names[RTLIB::REM_F32] = "fmodf";
  Names[RTLIB::REM_F64] = "fmod";
  Names[RTLIB::REM_F80] = "fmodl";
  Names[RTLIB::REM_F128] = "fmodl";
  Names[RTLIB::REM_PPCF128] = "fmodl";
  Names[RTLIB::FMA_F32] = "fmaf";
  Names[RTLIB::FMA_F64] = "fma";
  Names[RTLIB::FMA_F80] = "fmal";
  Names[RTLIB::FMA_F128] = "fmal";
  Names[RTLIB::FMA_PPCF128] = "fmal";
  Names[RTLIB::POWI_F32] = "__powisf2";
  Names[RTLIB::POWI_F64] = "__powidf2";
  Names[RTLIB::POWI_F80] = "__powixf2";
  Names[RTLIB::POWI_F128] = "__powitf2";
  Names[RTLIB::POWI_PPCF128] = "__powitf2";
  Names[RTLIB::SQRT_F32] = "sqrtf";
  Names[RTLIB::SQRT_F64] = "sqrt";
  Names[RTLIB::SQRT_F80] = "sqrtl";
  Names[RTLIB::SQRT_F128] = "sqrtl";
  Names[RTLIB::SQRT_PPCF128] = "sqrtl";
  Names[RTLIB::LOG_F32] = "logf";
  Names[RTLIB::LOG_F64] = "log";
  Names[RTLIB::LOG_F80] = "logl";
  Names[RTLIB::LOG_F128] = "logl";
  Names[RTLIB::LOG_PPCF128] = "logl";
  Names[RTLIB::LOG2_F32] = "log2f";
  Names[RTLIB::LOG2_F64] = "log2";
  Names[RTLIB::LOG2_F80] = "log2l";
  Names[RTLIB::LOG2_F128] = "log2l";
  Names[RTLIB::LOG2_PPCF128] = "log2l";
  Names[RTLIB::LOG10_F32] = "log10f";
  Names[RTLIB::LOG10_F64] = "log10";
  Names[RTLIB::LOG10_F80] = "log10l";
  Names[RTLIB::LOG10_F128] = "log10l";
  Names[RTLIB::LOG10_PPCF128] = "log10l";
  Names[RTLIB::EXP_F32] = "expf";
  Names[RTLIB::EXP_F64] = "exp";
  Names[RTLIB::EXP_F80] = "expl";
  Names[RTLIB::EXP_F128] = "expl";
  Names[RTLIB::EXP_PPCF128] = "expl";
  Names[RTLIB::EXP2_F32] = "exp2f";
  Names[RTLIB::EXP2_F64] = "exp2";
  Names[RTLIB::EXP2_F80] = "exp2l";
  Names[RTLIB::EXP2_F128] = "exp2l";
  Names[RTLIB::EXP2_PPCF128] = "exp2l";
  Names[RTLIB::SIN_F32] = "sinf";
  Names[RTLIB::SIN_F64] = "sin";
  Names[RTLIB::SIN_F80] = "sinl";
  Names[RTLIB::SIN_F128] = "sinl";
  Names[RTLIB::SIN_PPCF128] = "sinl";
  Names[RTLIB::COS_F32] = "cosf";
  Names[RTLIB::COS_F64] = "cos";
  Names[RTLIB::COS_F80] = "cosl";
  Names[RTLIB::COS_F128] = "cosl";
  Names[RTLIB::COS_PPCF128] = "cosl";
  Names[RTLIB::POW_F32] = "powf";
  Names[RTLIB::POW_F64] = "pow";
  Names[RTLIB::POW_F80] = "powl";
  Names[RTLIB::POW_F128] = "powl";
  Names[RTLIB::POW_PPCF128] = "powl";
  Names[RTLIB::CEIL_F32] = "ceilf";
  Names[RTLIB::CEIL_F64] = "ceil";
  Names[RTLIB::CEIL_F80] = "ceill";
  Names[RTLIB::CEIL_F128] = "ceill";
  Names[RTLIB::CEIL_PPCF128] = "ceill";
  Names[RTLIB::TRUNC_F32] = "truncf";
  Names[RTLIB::TRUNC_F64] = "trunc";
  Names[RTLIB::TRUNC_F80] = "truncl";
  Names[RTLIB::TRUNC_F128] = "truncl";
  Names[RTLIB::TRUNC_PPCF128] = "truncl";
  Names[RTLIB::RINT_F32] = "rintf";
  Names[RTLIB::RINT_F64] = "rint";
  Names[RTLIB::RINT_F80] = "rintl";
  Names[RTLIB::RINT_F128] = "rintl";
  Names[RTLIB::RINT_PPCF128] = "rintl";
  Names[RTLIB::NEARBYINT_F32] = "nearbyintf";
  Names[RTLIB::NEARBYINT_F64] = "nearbyint";
  Names[RTLIB::NEARBYINT_F80] = "nearbyintl";
  Names[RTLIB::NEARBYINT_F128] = "nearbyintl";
  Names[RTLIB::NEARBYINT_PPCF128] = "nearbyintl";
  Names[RTLIB::ROUND_F32] = "roundf";
  Names[RTLIB::ROUND_F64] = "round";
  Names[RTLIB::ROUND_F80] = "roundl";
  Names[RTLIB::ROUND_F128] = "roundl";
  Names[RTLIB::ROUND_PPCF128] = "roundl";
  Names[RTLIB::FLOOR_F32] = "floorf";
  Names[RTLIB::FLOOR_F64] = "floor";
  Names[RTLIB::FLOOR_F80] = "floorl";
  Names[RTLIB::FLOOR_F128] = "floorl";
  Names[RTLIB::FLOOR_PPCF128] = "floorl";
  Names[RTLIB::COPYSIGN_F32] = "copysignf";
  Names[RTLIB::COPYSIGN_F64] = "copysign";
  Names[RTLIB::COPYSIGN_F80] = "copysignl";
  Names[RTLIB::COPYSIGN_F128] = "copysignl";
  Names[RTLIB::COPYSIGN_PPCF128] = "copysignl";
  Names[RTLIB::FPEXT_F64_F128] = "__extenddftf2";
  Names[RTLIB::FPEXT_F32_F128] = "__extendsftf2";
  Names[RTLIB::FPEXT_F32_F64] = "__extendsfdf2";
  Names[RTLIB::FPEXT_F16_F32] = "__gnu_h2f_ieee";
  Names[RTLIB::FPROUND_F32_F16] = "__gnu_f2h_ieee";
  Names[RTLIB::FPROUND_F64_F32] = "__truncdfsf2";
  Names[RTLIB::FPROUND_F80_F32] = "__truncxfsf2";
  Names[RTLIB::FPROUND_F128_F32] = "__trunctfsf2";
  Names[RTLIB::FPROUND_PPCF128_F32] = "__trunctfsf2";
  Names[RTLIB::FPROUND_F80_F64] = "__truncxfdf2";
  Names[RTLIB::FPROUND_F128_F64] = "__trunctfdf2";
  Names[RTLIB::FPROUND_PPCF128_F64] = "__trunctfdf2";
  Names[RTLIB::FPTOSINT_F32_I8] = "__fixsfqi";
  Names[RTLIB::FPTOSINT_F32_I16] = "__fixsfhi";
  Names[RTLIB::FPTOSINT_F32_I32] = "__fixsfsi";
  Names[RTLIB::FPTOSINT_F32_I64] = "__fixsfdi";
  Names[RTLIB::FPTOSINT_F32_I128] = "__fixsfti";
  Names[RTLIB::FPTOSINT_F64_I8] = "__fixdfqi";
  Names[RTLIB::FPTOSINT_F64_I16] = "__fixdfhi";
  Names[RTLIB::FPTOSINT_F64_I32] = "__fixdfsi";
  Names[RTLIB::FPTOSINT_F64_I64] = "__fixdfdi";
  Names[RTLIB::FPTOSINT_F64_I128] = "__fixdfti";
  Names[RTLIB::FPTOSINT_F80_I32] = "__fixxfsi";
  Names[RTLIB::FPTOSINT_F80_I64] = "__fixxfdi";
  Names[RTLIB::FPTOSINT_F80_I128] = "__fixxfti";
  Names[RTLIB::FPTOSINT_F128_I32] = "__fixtfsi";
  Names[RTLIB::FPTOSINT_F128_I64] = "__fixtfdi";
  Names[RTLIB::FPTOSINT_F128_I128] = "__fixtfti";
  Names[RTLIB::FPTOSINT_PPCF128_I32] = "__fixtfsi";
  Names[RTLIB::FPTOSINT_PPCF128_I64] = "__fixtfdi";
  Names[RTLIB::FPTOSINT_PPCF128_I128] = "__fixtfti";
  Names[RTLIB::FPTOUINT_F32_I8] = "__fixunssfqi";
  Names[RTLIB::FPTOUINT_F32_I16] = "__fixunssfhi";
  Names[RTLIB::FPTOUINT_F32_I32] = "__fixunssfsi";
  Names[RTLIB::FPTOUINT_F32_I64] = "__fixunssfdi";
  Names[RTLIB::FPTOUINT_F32_I128] = "__fixunssfti";
  Names[RTLIB::FPTOUINT_F64_I8] = "__fixunsdfqi";
  Names[RTLIB::FPTOUINT_F64_I16] = "__fixunsdfhi";
  Names[RTLIB::FPTOUINT_F64_I32] = "__fixunsdfsi";
  Names[RTLIB::FPTOUINT_F64_I64] = "__fixunsdfdi";
  Names[RTLIB::FPTOUINT_F64_I128] = "__fixunsdfti";
  Names[RTLIB::FPTOUINT_F80_I32] = "__fixunsxfsi";
  Names[RTLIB::FPTOUINT_F80_I64] = "__fixunsxfdi";
  Names[RTLIB::FPTOUINT_F80_I128] = "__fixunsxfti";
  Names[RTLIB::FPTOUINT_F128_I32] = "__fixunstfsi";
  Names[RTLIB::FPTOUINT_F128_I64] = "__fixunstfdi";
  Names[RTLIB::FPTOUINT_F128_I128] = "__fixunstfti";
  Names[RTLIB::FPTOUINT_PPCF128_I32] = "__fixunstfsi";
  Names[RTLIB::FPTOUINT_PPCF128_I64] = "__fixunstfdi";
  Names[RTLIB::FPTOUINT_PPCF128_I128] = "__fixunstfti";
  Names[RTLIB::SINTTOFP_I32_F32] = "__floatsisf";
  Names[RTLIB::SINTTOFP_I32_F64] = "__floatsidf";
  Names[RTLIB::SINTTOFP_I32_F80] = "__floatsixf";
  Names[RTLIB::SINTTOFP_I32_F128] = "__floatsitf";
  Names[RTLIB::SINTTOFP_I32_PPCF128] = "__floatsitf";
  Names[RTLIB::SINTTOFP_I64_F32] = "__floatdisf";
  Names[RTLIB::SINTTOFP_I64_F64] = "__floatdidf";
  Names[RTLIB::SINTTOFP_I64_F80] = "__floatdixf";
  Names[RTLIB::SINTTOFP_I64_F128] = "__floatditf";
  Names[RTLIB::SINTTOFP_I64_PPCF128] = "__floatditf";
  Names[RTLIB::SINTTOFP_I128_F32] = "__floattisf";
  Names[RTLIB::SINTTOFP_I128_F64] = "__floattidf";
  Names[RTLIB::SINTTOFP_I128_F80] = "__floattixf";
  Names[RTLIB::SINTTOFP_I128_F128] = "__floattitf";
  Names[RTLIB::SINTTOFP_I128_PPCF128] = "__floattitf";
  Names[RTLIB::UINTTOFP_I32_F32] = "__floatunsisf";
  Names[RTLIB::UINTTOFP_I32_F64] = "__floatunsidf";
  Names[RTLIB::UINTTOFP_I32_F80] = "__floatunsixf";
  Names[RTLIB::UINTTOFP_I32_F128] = "__floatunsitf";
  Names[RTLIB::UINTTOFP_I32_PPCF128] = "__floatunsitf";
  Names[RTLIB::UINTTOFP_I64_F32] = "__floatundisf";
  Names[RTLIB::UINTTOFP_I64_F64] = "__floatundidf";
  Names[RTLIB::UINTTOFP_I64_F80] = "__floatundixf";
  Names[RTLIB::UINTTOFP_I64_F128] = "__floatunditf";
  Names[RTLIB::UINTTOFP_I64_PPCF128] = "__floatunditf";
  Names[RTLIB::UINTTOFP_I128_F32] = "__floatuntisf";
  Names[RTLIB::UINTTOFP_I128_F64] = "__floatuntidf";
  Names[RTLIB::UINTTOFP_I128_F80] = "__floatuntixf";
  Names[RTLIB::UINTTOFP_I128_F128] = "__floatuntitf";
  Names[RTLIB::UINTTOFP_I128_PPCF128] = "__floatuntitf";
  Names[RTLIB::OEQ_F32] = "__eqsf2";
  Names[RTLIB::OEQ_F64] = "__eqdf2";
  Names[RTLIB::OEQ_F128] = "__eqtf2";
  Names[RTLIB::UNE_F32] = "__nesf2";
  Names[RTLIB::UNE_F64] = "__nedf2";
  Names[RTLIB::UNE_F128] = "__netf2";
  Names[RTLIB::OGE_F32] = "__gesf2";
  Names[RTLIB::OGE_F64] = "__gedf2";
  Names[RTLIB::OGE_F128] = "__getf2";
  Names[RTLIB::OLT_F32] = "__ltsf2";
  Names[RTLIB::OLT_F64] = "__ltdf2";
  Names[RTLIB::OLT_F128] = "__lttf2";
  Names[RTLIB::OLE_F32] = "__lesf2";
  Names[RTLIB::OLE_F64] = "__ledf2";
  Names[RTLIB::OLE_F128] = "__letf2";
  Names[RTLIB::OGT_F32] = "__gtsf2";
  Names[RTLIB::OGT_F64] = "__gtdf2";
  Names[RTLIB::OGT_F128] = "__gttf2";
  Names[RTLIB::UO_F32] = "__unordsf2";
  Names[RTLIB::UO_F64] = "__unorddf2";
  Names[RTLIB::UO_F128] = "__unordtf2";
  Names[RTLIB::O_F32] = "__unordsf2";
  Names[RTLIB::O_F64] = "__unorddf2";
  Names[RTLIB::O_F128] = "__unordtf2";
  Names[RTLIB::MEMCPY] = "memcpy";
  Names[RTLIB::MEMMOVE] = "memmove";
  Names[RTLIB::MEMSET] = "memset";
  Names[RTLIB::UNWIND_RESUME] = "_Unwind_Resume";
  Names[RTLIB::SYNC_VAL_COMPARE_AND_SWAP_1] = "__sync_val_compare_and_swap_1";
  Names[RTLIB::SYNC_VAL_COMPARE_AND_SWAP_2] = "__sync_val_compare_and_swap_2";
  Names[RTLIB::SYNC_VAL_COMPARE_AND_SWAP_4] = "__sync_val_compare_and_swap_4";
  Names[RTLIB::SYNC_VAL_COMPARE_AND_SWAP_8] = "__sync_val_compare_and_swap_8";
  Names[RTLIB::SYNC_VAL_COMPARE_AND_SWAP_16] = "__sync_val_compare_and_swap_16";
  Names[RTLIB::SYNC_LOCK_TEST_AND_SET_1] = "__sync_lock_test_and_set_1";
  Names[RTLIB::SYNC_LOCK_TEST_AND_SET_2] = "__sync_lock_test_and_set_2";
  Names[RTLIB::SYNC_LOCK_TEST_AND_SET_4] = "__sync_lock_test_and_set_4";
  Names[RTLIB::SYNC_LOCK_TEST_AND_SET_8] = "__sync_lock_test_and_set_8";
  Names[RTLIB::SYNC_LOCK_TEST_AND_SET_16] = "__sync_lock_test_and_set_16";
  Names[RTLIB::SYNC_FETCH_AND_ADD_1] = "__sync_fetch_and_add_1";
  Names[RTLIB::SYNC_FETCH_AND_ADD_2] = "__sync_fetch_and_add_2";
  Names[RTLIB::SYNC_FETCH_AND_ADD_4] = "__sync_fetch_and_add_4";
  Names[RTLIB::SYNC_FETCH_AND_ADD_8] = "__sync_fetch_and_add_8";
  Names[RTLIB::SYNC_FETCH_AND_ADD_16] = "__sync_fetch_and_add_16";
  Names[RTLIB::SYNC_FETCH_AND_SUB_1] = "__sync_fetch_and_sub_1";
  Names[RTLIB::SYNC_FETCH_AND_SUB_2] = "__sync_fetch_and_sub_2";
  Names[RTLIB::SYNC_FETCH_AND_SUB_4] = "__sync_fetch_and_sub_4";
  Names[RTLIB::SYNC_FETCH_AND_SUB_8] = "__sync_fetch_and_sub_8";
  Names[RTLIB::SYNC_FETCH_AND_SUB_16] = "__sync_fetch_and_sub_16";
  Names[RTLIB::SYNC_FETCH_AND_AND_1] = "__sync_fetch_and_and_1";
  Names[RTLIB::SYNC_FETCH_AND_AND_2] = "__sync_fetch_and_and_2";
  Names[RTLIB::SYNC_FETCH_AND_AND_4] = "__sync_fetch_and_and_4";
  Names[RTLIB::SYNC_FETCH_AND_AND_8] = "__sync_fetch_and_and_8";
  Names[RTLIB::SYNC_FETCH_AND_AND_16] = "__sync_fetch_and_and_16";
  Names[RTLIB::SYNC_FETCH_AND_OR_1] = "__sync_fetch_and_or_1";
  Names[RTLIB::SYNC_FETCH_AND_OR_2] = "__sync_fetch_and_or_2";
  Names[RTLIB::SYNC_FETCH_AND_OR_4] = "__sync_fetch_and_or_4";
  Names[RTLIB::SYNC_FETCH_AND_OR_8] = "__sync_fetch_and_or_8";
  Names[RTLIB::SYNC_FETCH_AND_OR_16] = "__sync_fetch_and_or_16";
  Names[RTLIB::SYNC_FETCH_AND_XOR_1] = "__sync_fetch_and_xor_1";
  Names[RTLIB::SYNC_FETCH_AND_XOR_2] = "__sync_fetch_and_xor_2";
  Names[RTLIB::SYNC_FETCH_AND_XOR_4] = "__sync_fetch_and_xor_4";
  Names[RTLIB::SYNC_FETCH_AND_XOR_8] = "__sync_fetch_and_xor_8";
  Names[RTLIB::SYNC_FETCH_AND_XOR_16] = "__sync_fetch_and_xor_16";
  Names[RTLIB::SYNC_FETCH_AND_NAND_1] = "__sync_fetch_and_nand_1";
  Names[RTLIB::SYNC_FETCH_AND_NAND_2] = "__sync_fetch_and_nand_2";
  Names[RTLIB::SYNC_FETCH_AND_NAND_4] = "__sync_fetch_and_nand_4";
  Names[RTLIB::SYNC_FETCH_AND_NAND_8] = "__sync_fetch_and_nand_8";
  Names[RTLIB::SYNC_FETCH_AND_NAND_16] = "__sync_fetch_and_nand_16";
  Names[RTLIB::SYNC_FETCH_AND_MAX_1] = "__sync_fetch_and_max_1";
  Names[RTLIB::SYNC_FETCH_AND_MAX_2] = "__sync_fetch_and_max_2";
  Names[RTLIB::SYNC_FETCH_AND_MAX_4] = "__sync_fetch_and_max_4";
  Names[RTLIB::SYNC_FETCH_AND_MAX_8] = "__sync_fetch_and_max_8";
  Names[RTLIB::SYNC_FETCH_AND_MAX_16] = "__sync_fetch_and_max_16";
  Names[RTLIB::SYNC_FETCH_AND_UMAX_1] = "__sync_fetch_and_umax_1";
  Names[RTLIB::SYNC_FETCH_AND_UMAX_2] = "__sync_fetch_and_umax_2";
  Names[RTLIB::SYNC_FETCH_AND_UMAX_4] = "__sync_fetch_and_umax_4";
  Names[RTLIB::SYNC_FETCH_AND_UMAX_8] = "__sync_fetch_and_umax_8";
  Names[RTLIB::SYNC_FETCH_AND_UMAX_16] = "__sync_fetch_and_umax_16";
  Names[RTLIB::SYNC_FETCH_AND_MIN_1] = "__sync_fetch_and_min_1";
  Names[RTLIB::SYNC_FETCH_AND_MIN_2] = "__sync_fetch_and_min_2";
  Names[RTLIB::SYNC_FETCH_AND_MIN_4] = "__sync_fetch_and_min_4";
  Names[RTLIB::SYNC_FETCH_AND_MIN_8] = "__sync_fetch_and_min_8";
  Names[RTLIB::SYNC_FETCH_AND_MIN_16] = "__sync_fetch_and_min_16";
  Names[RTLIB::SYNC_FETCH_AND_UMIN_1] = "__sync_fetch_and_umin_1";
  Names[RTLIB::SYNC_FETCH_AND_UMIN_2] = "__sync_fetch_and_umin_2";
  Names[RTLIB::SYNC_FETCH_AND_UMIN_4] = "__sync_fetch_and_umin_4";
  Names[RTLIB::SYNC_FETCH_AND_UMIN_8] = "__sync_fetch_and_umin_8";
  Names[RTLIB::SYNC_FETCH_AND_UMIN_16] = "__sync_fetch_and_umin_16";
  
  if (Triple(TM.getTargetTriple()).getEnvironment() == Triple::GNU) {
    Names[RTLIB::SINCOS_F32] = "sincosf";
    Names[RTLIB::SINCOS_F64] = "sincos";
    Names[RTLIB::SINCOS_F80] = "sincosl";
    Names[RTLIB::SINCOS_F128] = "sincosl";
    Names[RTLIB::SINCOS_PPCF128] = "sincosl";
  } else {
    // These are generally not available.
    Names[RTLIB::SINCOS_F32] = 0;
    Names[RTLIB::SINCOS_F64] = 0;
    Names[RTLIB::SINCOS_F80] = 0;
    Names[RTLIB::SINCOS_F128] = 0;
    Names[RTLIB::SINCOS_PPCF128] = 0;
  }

  if (Triple(TM.getTargetTriple()).getOS() != Triple::OpenBSD) {
    Names[RTLIB::STACKPROTECTOR_CHECK_FAIL] = "__stack_chk_fail";
  } else {
    // These are generally not available.
    Names[RTLIB::STACKPROTECTOR_CHECK_FAIL] = 0;
  }
}

/// InitLibcallCallingConvs - Set default libcall CallingConvs.
///
static void InitLibcallCallingConvs(CallingConv::ID *CCs) {
  for (int i = 0; i < RTLIB::UNKNOWN_LIBCALL; ++i) {
    CCs[i] = CallingConv::C;
  }
}

/// getFPEXT - Return the FPEXT_*_* value for the given types, or
/// UNKNOWN_LIBCALL if there is none.
RTLIB::Libcall RTLIB::getFPEXT(EVT OpVT, EVT RetVT) {
  if (OpVT == MVT::f32) {
    if (RetVT == MVT::f64)
      return FPEXT_F32_F64;
    if (RetVT == MVT::f128)
      return FPEXT_F32_F128;
  } else if (OpVT == MVT::f64) {
    if (RetVT == MVT::f128)
      return FPEXT_F64_F128;
  }

  return UNKNOWN_LIBCALL;
}

/// getFPROUND - Return the FPROUND_*_* value for the given types, or
/// UNKNOWN_LIBCALL if there is none.
RTLIB::Libcall RTLIB::getFPROUND(EVT OpVT, EVT RetVT) {
  if (RetVT == MVT::f32) {
    if (OpVT == MVT::f64)
      return FPROUND_F64_F32;
    if (OpVT == MVT::f80)
      return FPROUND_F80_F32;
    if (OpVT == MVT::f128)
      return FPROUND_F128_F32;
    if (OpVT == MVT::ppcf128)
      return FPROUND_PPCF128_F32;
  } else if (RetVT == MVT::f64) {
    if (OpVT == MVT::f80)
      return FPROUND_F80_F64;
    if (OpVT == MVT::f128)
      return FPROUND_F128_F64;
    if (OpVT == MVT::ppcf128)
      return FPROUND_PPCF128_F64;
  }

  return UNKNOWN_LIBCALL;
}

/// getFPTOSINT - Return the FPTOSINT_*_* value for the given types, or
/// UNKNOWN_LIBCALL if there is none.
RTLIB::Libcall RTLIB::getFPTOSINT(EVT OpVT, EVT RetVT) {
  if (OpVT == MVT::f32) {
    if (RetVT == MVT::i8)
      return FPTOSINT_F32_I8;
    if (RetVT == MVT::i16)
      return FPTOSINT_F32_I16;
    if (RetVT == MVT::i32)
      return FPTOSINT_F32_I32;
    if (RetVT == MVT::i64)
      return FPTOSINT_F32_I64;
    if (RetVT == MVT::i128)
      return FPTOSINT_F32_I128;
  } else if (OpVT == MVT::f64) {
    if (RetVT == MVT::i8)
      return FPTOSINT_F64_I8;
    if (RetVT == MVT::i16)
      return FPTOSINT_F64_I16;
    if (RetVT == MVT::i32)
      return FPTOSINT_F64_I32;
    if (RetVT == MVT::i64)
      return FPTOSINT_F64_I64;
    if (RetVT == MVT::i128)
      return FPTOSINT_F64_I128;
  } else if (OpVT == MVT::f80) {
    if (RetVT == MVT::i32)
      return FPTOSINT_F80_I32;
    if (RetVT == MVT::i64)
      return FPTOSINT_F80_I64;
    if (RetVT == MVT::i128)
      return FPTOSINT_F80_I128;
  } else if (OpVT == MVT::f128) {
    if (RetVT == MVT::i32)
      return FPTOSINT_F128_I32;
    if (RetVT == MVT::i64)
      return FPTOSINT_F128_I64;
    if (RetVT == MVT::i128)
      return FPTOSINT_F128_I128;
  } else if (OpVT == MVT::ppcf128) {
    if (RetVT == MVT::i32)
      return FPTOSINT_PPCF128_I32;
    if (RetVT == MVT::i64)
      return FPTOSINT_PPCF128_I64;
    if (RetVT == MVT::i128)
      return FPTOSINT_PPCF128_I128;
  }
  return UNKNOWN_LIBCALL;
}

/// getFPTOUINT - Return the FPTOUINT_*_* value for the given types, or
/// UNKNOWN_LIBCALL if there is none.
RTLIB::Libcall RTLIB::getFPTOUINT(EVT OpVT, EVT RetVT) {
  if (OpVT == MVT::f32) {
    if (RetVT == MVT::i8)
      return FPTOUINT_F32_I8;
    if (RetVT == MVT::i16)
      return FPTOUINT_F32_I16;
    if (RetVT == MVT::i32)
      return FPTOUINT_F32_I32;
    if (RetVT == MVT::i64)
      return FPTOUINT_F32_I64;
    if (RetVT == MVT::i128)
      return FPTOUINT_F32_I128;
  } else if (OpVT == MVT::f64) {
    if (RetVT == MVT::i8)
      return FPTOUINT_F64_I8;
    if (RetVT == MVT::i16)
      return FPTOUINT_F64_I16;
    if (RetVT == MVT::i32)
      return FPTOUINT_F64_I32;
    if (RetVT == MVT::i64)
      return FPTOUINT_F64_I64;
    if (RetVT == MVT::i128)
      return FPTOUINT_F64_I128;
  } else if (OpVT == MVT::f80) {
    if (RetVT == MVT::i32)
      return FPTOUINT_F80_I32;
    if (RetVT == MVT::i64)
      return FPTOUINT_F80_I64;
    if (RetVT == MVT::i128)
      return FPTOUINT_F80_I128;
  } else if (OpVT == MVT::f128) {
    if (RetVT == MVT::i32)
      return FPTOUINT_F128_I32;
    if (RetVT == MVT::i64)
      return FPTOUINT_F128_I64;
    if (RetVT == MVT::i128)
      return FPTOUINT_F128_I128;
  } else if (OpVT == MVT::ppcf128) {
    if (RetVT == MVT::i32)
      return FPTOUINT_PPCF128_I32;
    if (RetVT == MVT::i64)
      return FPTOUINT_PPCF128_I64;
    if (RetVT == MVT::i128)
      return FPTOUINT_PPCF128_I128;
  }
  return UNKNOWN_LIBCALL;
}

/// getSINTTOFP - Return the SINTTOFP_*_* value for the given types, or
/// UNKNOWN_LIBCALL if there is none.
RTLIB::Libcall RTLIB::getSINTTOFP(EVT OpVT, EVT RetVT) {
  if (OpVT == MVT::i32) {
    if (RetVT == MVT::f32)
      return SINTTOFP_I32_F32;
    if (RetVT == MVT::f64)
      return SINTTOFP_I32_F64;
    if (RetVT == MVT::f80)
      return SINTTOFP_I32_F80;
    if (RetVT == MVT::f128)
      return SINTTOFP_I32_F128;
    if (RetVT == MVT::ppcf128)
      return SINTTOFP_I32_PPCF128;
  } else if (OpVT == MVT::i64) {
    if (RetVT == MVT::f32)
      return SINTTOFP_I64_F32;
    if (RetVT == MVT::f64)
      return SINTTOFP_I64_F64;
    if (RetVT == MVT::f80)
      return SINTTOFP_I64_F80;
    if (RetVT == MVT::f128)
      return SINTTOFP_I64_F128;
    if (RetVT == MVT::ppcf128)
      return SINTTOFP_I64_PPCF128;
  } else if (OpVT == MVT::i128) {
    if (RetVT == MVT::f32)
      return SINTTOFP_I128_F32;
    if (RetVT == MVT::f64)
      return SINTTOFP_I128_F64;
    if (RetVT == MVT::f80)
      return SINTTOFP_I128_F80;
    if (RetVT == MVT::f128)
      return SINTTOFP_I128_F128;
    if (RetVT == MVT::ppcf128)
      return SINTTOFP_I128_PPCF128;
  }
  return UNKNOWN_LIBCALL;
}

/// getUINTTOFP - Return the UINTTOFP_*_* value for the given types, or
/// UNKNOWN_LIBCALL if there is none.
RTLIB::Libcall RTLIB::getUINTTOFP(EVT OpVT, EVT RetVT) {
  if (OpVT == MVT::i32) {
    if (RetVT == MVT::f32)
      return UINTTOFP_I32_F32;
    if (RetVT == MVT::f64)
      return UINTTOFP_I32_F64;
    if (RetVT == MVT::f80)
      return UINTTOFP_I32_F80;
    if (RetVT == MVT::f128)
      return UINTTOFP_I32_F128;
    if (RetVT == MVT::ppcf128)
      return UINTTOFP_I32_PPCF128;
  } else if (OpVT == MVT::i64) {
    if (RetVT == MVT::f32)
      return UINTTOFP_I64_F32;
    if (RetVT == MVT::f64)
      return UINTTOFP_I64_F64;
    if (RetVT == MVT::f80)
      return UINTTOFP_I64_F80;
    if (RetVT == MVT::f128)
      return UINTTOFP_I64_F128;
    if (RetVT == MVT::ppcf128)
      return UINTTOFP_I64_PPCF128;
  } else if (OpVT == MVT::i128) {
    if (RetVT == MVT::f32)
      return UINTTOFP_I128_F32;
    if (RetVT == MVT::f64)
      return UINTTOFP_I128_F64;
    if (RetVT == MVT::f80)
      return UINTTOFP_I128_F80;
    if (RetVT == MVT::f128)
      return UINTTOFP_I128_F128;
    if (RetVT == MVT::ppcf128)
      return UINTTOFP_I128_PPCF128;
  }
  return UNKNOWN_LIBCALL;
}

/// InitCmpLibcallCCs - Set default comparison libcall CC.
///
static void InitCmpLibcallCCs(ISD::CondCode *CCs) {
  memset(CCs, ISD::SETCC_INVALID, sizeof(ISD::CondCode)*RTLIB::UNKNOWN_LIBCALL);
  CCs[RTLIB::OEQ_F32] = ISD::SETEQ;
  CCs[RTLIB::OEQ_F64] = ISD::SETEQ;
  CCs[RTLIB::OEQ_F128] = ISD::SETEQ;
  CCs[RTLIB::UNE_F32] = ISD::SETNE;
  CCs[RTLIB::UNE_F64] = ISD::SETNE;
  CCs[RTLIB::UNE_F128] = ISD::SETNE;
  CCs[RTLIB::OGE_F32] = ISD::SETGE;
  CCs[RTLIB::OGE_F64] = ISD::SETGE;
  CCs[RTLIB::OGE_F128] = ISD::SETGE;
  CCs[RTLIB::OLT_F32] = ISD::SETLT;
  CCs[RTLIB::OLT_F64] = ISD::SETLT;
  CCs[RTLIB::OLT_F128] = ISD::SETLT;
  CCs[RTLIB::OLE_F32] = ISD::SETLE;
  CCs[RTLIB::OLE_F64] = ISD::SETLE;
  CCs[RTLIB::OLE_F128] = ISD::SETLE;
  CCs[RTLIB::OGT_F32] = ISD::SETGT;
  CCs[RTLIB::OGT_F64] = ISD::SETGT;
  CCs[RTLIB::OGT_F128] = ISD::SETGT;
  CCs[RTLIB::UO_F32] = ISD::SETNE;
  CCs[RTLIB::UO_F64] = ISD::SETNE;
  CCs[RTLIB::UO_F128] = ISD::SETNE;
  CCs[RTLIB::O_F32] = ISD::SETEQ;
  CCs[RTLIB::O_F64] = ISD::SETEQ;
  CCs[RTLIB::O_F128] = ISD::SETEQ;
}

/// NOTE: The constructor takes ownership of TLOF.
TargetLoweringBase::TargetLoweringBase(const TargetMachine &tm,
                                       const TargetLoweringObjectFile *tlof)
  : TM(tm), DL(TM.getDataLayout()), TLOF(*tlof) {
  initActions();

  // Perform these initializations only once.
  IsLittleEndian = DL->isLittleEndian();
  MaxStoresPerMemset = MaxStoresPerMemcpy = MaxStoresPerMemmove = 8;
  MaxStoresPerMemsetOptSize = MaxStoresPerMemcpyOptSize
    = MaxStoresPerMemmoveOptSize = 4;
  UseUnderscoreSetJmp = false;
  UseUnderscoreLongJmp = false;
  SelectIsExpensive = false;
  HasMultipleConditionRegisters = false;
  IntDivIsCheap = false;
  Pow2DivIsCheap = false;
  JumpIsExpensive = false;
  PredictableSelectIsExpensive = false;
  StackPointerRegisterToSaveRestore = 0;
  ExceptionPointerRegister = 0;
  ExceptionSelectorRegister = 0;
  BooleanContents = UndefinedBooleanContent;
  BooleanVectorContents = UndefinedBooleanContent;
  SchedPreferenceInfo = Sched::ILP;
  JumpBufSize = 0;
  JumpBufAlignment = 0;
  MinFunctionAlignment = 0;
  PrefFunctionAlignment = 0;
  PrefLoopAlignment = 0;
  MinStackArgumentAlignment = 1;
  InsertFencesForAtomic = false;
  SupportJumpTables = true;
  MinimumJumpTableEntries = 4;

  InitLibcallNames(LibcallRoutineNames, TM);
  InitCmpLibcallCCs(CmpLibcallCCs);
  InitLibcallCallingConvs(LibcallCallingConvs);
}

TargetLoweringBase::~TargetLoweringBase() {
  delete &TLOF;
}

void TargetLoweringBase::initActions() {
  // All operations default to being supported.
  memset(OpActions, 0, sizeof(OpActions));
  memset(LoadExtActions, 0, sizeof(LoadExtActions));
  memset(TruncStoreActions, 0, sizeof(TruncStoreActions));
  memset(IndexedModeActions, 0, sizeof(IndexedModeActions));
  memset(CondCodeActions, 0, sizeof(CondCodeActions));
  memset(RegClassForVT, 0,MVT::LAST_VALUETYPE*sizeof(TargetRegisterClass*));
  memset(TargetDAGCombineArray, 0, array_lengthof(TargetDAGCombineArray));

  // Set default actions for various operations.
  for (unsigned VT = 0; VT != (unsigned)MVT::LAST_VALUETYPE; ++VT) {
    // Default all indexed load / store to expand.
    for (unsigned IM = (unsigned)ISD::PRE_INC;
         IM != (unsigned)ISD::LAST_INDEXED_MODE; ++IM) {
      setIndexedLoadAction(IM, (MVT::SimpleValueType)VT, Expand);
      setIndexedStoreAction(IM, (MVT::SimpleValueType)VT, Expand);
    }

    // These operations default to expand.
    setOperationAction(ISD::FGETSIGN, (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::CONCAT_VECTORS, (MVT::SimpleValueType)VT, Expand);

    // These library functions default to expand.
    setOperationAction(ISD::FROUND, (MVT::SimpleValueType)VT, Expand);

    // These operations default to expand for vector types.
    if (VT >= MVT::FIRST_VECTOR_VALUETYPE &&
        VT <= MVT::LAST_VECTOR_VALUETYPE)
      setOperationAction(ISD::FCOPYSIGN, (MVT::SimpleValueType)VT, Expand);
  }

  // Most targets ignore the @llvm.prefetch intrinsic.
  setOperationAction(ISD::PREFETCH, MVT::Other, Expand);

  // ConstantFP nodes default to expand.  Targets can either change this to
  // Legal, in which case all fp constants are legal, or use isFPImmLegal()
  // to optimize expansions for certain constants.
  setOperationAction(ISD::ConstantFP, MVT::f16, Expand);
  setOperationAction(ISD::ConstantFP, MVT::f32, Expand);
  setOperationAction(ISD::ConstantFP, MVT::f64, Expand);
  setOperationAction(ISD::ConstantFP, MVT::f80, Expand);
  setOperationAction(ISD::ConstantFP, MVT::f128, Expand);

  // These library functions default to expand.
  setOperationAction(ISD::FLOG ,  MVT::f16, Expand);
  setOperationAction(ISD::FLOG2,  MVT::f16, Expand);
  setOperationAction(ISD::FLOG10, MVT::f16, Expand);
  setOperationAction(ISD::FEXP ,  MVT::f16, Expand);
  setOperationAction(ISD::FEXP2,  MVT::f16, Expand);
  setOperationAction(ISD::FFLOOR, MVT::f16, Expand);
  setOperationAction(ISD::FNEARBYINT, MVT::f16, Expand);
  setOperationAction(ISD::FCEIL,  MVT::f16, Expand);
  setOperationAction(ISD::FRINT,  MVT::f16, Expand);
  setOperationAction(ISD::FTRUNC, MVT::f16, Expand);
  setOperationAction(ISD::FLOG ,  MVT::f32, Expand);
  setOperationAction(ISD::FLOG2,  MVT::f32, Expand);
  setOperationAction(ISD::FLOG10, MVT::f32, Expand);
  setOperationAction(ISD::FEXP ,  MVT::f32, Expand);
  setOperationAction(ISD::FEXP2,  MVT::f32, Expand);
  setOperationAction(ISD::FFLOOR, MVT::f32, Expand);
  setOperationAction(ISD::FNEARBYINT, MVT::f32, Expand);
  setOperationAction(ISD::FCEIL,  MVT::f32, Expand);
  setOperationAction(ISD::FRINT,  MVT::f32, Expand);
  setOperationAction(ISD::FTRUNC, MVT::f32, Expand);
  setOperationAction(ISD::FLOG ,  MVT::f64, Expand);
  setOperationAction(ISD::FLOG2,  MVT::f64, Expand);
  setOperationAction(ISD::FLOG10, MVT::f64, Expand);
  setOperationAction(ISD::FEXP ,  MVT::f64, Expand);
  setOperationAction(ISD::FEXP2,  MVT::f64, Expand);
  setOperationAction(ISD::FFLOOR, MVT::f64, Expand);
  setOperationAction(ISD::FNEARBYINT, MVT::f64, Expand);
  setOperationAction(ISD::FCEIL,  MVT::f64, Expand);
  setOperationAction(ISD::FRINT,  MVT::f64, Expand);
  setOperationAction(ISD::FTRUNC, MVT::f64, Expand);
  setOperationAction(ISD::FLOG ,  MVT::f128, Expand);
  setOperationAction(ISD::FLOG2,  MVT::f128, Expand);
  setOperationAction(ISD::FLOG10, MVT::f128, Expand);
  setOperationAction(ISD::FEXP ,  MVT::f128, Expand);
  setOperationAction(ISD::FEXP2,  MVT::f128, Expand);
  setOperationAction(ISD::FFLOOR, MVT::f128, Expand);
  setOperationAction(ISD::FNEARBYINT, MVT::f128, Expand);
  setOperationAction(ISD::FCEIL,  MVT::f128, Expand);
  setOperationAction(ISD::FRINT,  MVT::f128, Expand);
  setOperationAction(ISD::FTRUNC, MVT::f128, Expand);

  // Default ISD::TRAP to expand (which turns it into abort).
  setOperationAction(ISD::TRAP, MVT::Other, Expand);

  // On most systems, DEBUGTRAP and TRAP have no difference. The "Expand"
  // here is to inform DAG Legalizer to replace DEBUGTRAP with TRAP.
  //
  setOperationAction(ISD::DEBUGTRAP, MVT::Other, Expand);
}

MVT TargetLoweringBase::getPointerTy(uint32_t AS) const {
  return MVT::getIntegerVT(getPointerSizeInBits(AS));
}

unsigned TargetLoweringBase::getPointerSizeInBits(uint32_t AS) const {
  return DL->getPointerSizeInBits(AS);
}

unsigned TargetLoweringBase::getPointerTypeSizeInBits(Type *Ty) const {
  assert(Ty->isPointerTy());
  return getPointerSizeInBits(Ty->getPointerAddressSpace());
}

MVT TargetLoweringBase::getScalarShiftAmountTy(EVT LHSTy) const {
  return MVT::getIntegerVT(8*DL->getPointerSize(0));
}

EVT TargetLoweringBase::getShiftAmountTy(EVT LHSTy) const {
  assert(LHSTy.isInteger() && "Shift amount is not an integer type!");
  if (LHSTy.isVector())
    return LHSTy;
  return getScalarShiftAmountTy(LHSTy);
}

/// canOpTrap - Returns true if the operation can trap for the value type.
/// VT must be a legal type.
bool TargetLoweringBase::canOpTrap(unsigned Op, EVT VT) const {
  assert(isTypeLegal(VT));
  switch (Op) {
  default:
    return false;
  case ISD::FDIV:
  case ISD::FREM:
  case ISD::SDIV:
  case ISD::UDIV:
  case ISD::SREM:
  case ISD::UREM:
    return true;
  }
}


static unsigned getVectorTypeBreakdownMVT(MVT VT, MVT &IntermediateVT,
                                          unsigned &NumIntermediates,
                                          MVT &RegisterVT,
                                          TargetLoweringBase *TLI) {
  // Figure out the right, legal destination reg to copy into.
  unsigned NumElts = VT.getVectorNumElements();
  MVT EltTy = VT.getVectorElementType();

  unsigned NumVectorRegs = 1;

  // FIXME: We don't support non-power-of-2-sized vectors for now.  Ideally we
  // could break down into LHS/RHS like LegalizeDAG does.
  if (!isPowerOf2_32(NumElts)) {
    NumVectorRegs = NumElts;
    NumElts = 1;
  }

  // Divide the input until we get to a supported size.  This will always
  // end with a scalar if the target doesn't support vectors.
  while (NumElts > 1 && !TLI->isTypeLegal(MVT::getVectorVT(EltTy, NumElts))) {
    NumElts >>= 1;
    NumVectorRegs <<= 1;
  }

  NumIntermediates = NumVectorRegs;

  MVT NewVT = MVT::getVectorVT(EltTy, NumElts);
  if (!TLI->isTypeLegal(NewVT))
    NewVT = EltTy;
  IntermediateVT = NewVT;

  unsigned NewVTSize = NewVT.getSizeInBits();

  // Convert sizes such as i33 to i64.
  if (!isPowerOf2_32(NewVTSize))
    NewVTSize = NextPowerOf2(NewVTSize);

  MVT DestVT = TLI->getRegisterType(NewVT);
  RegisterVT = DestVT;
  if (EVT(DestVT).bitsLT(NewVT))    // Value is expanded, e.g. i64 -> i16.
    return NumVectorRegs*(NewVTSize/DestVT.getSizeInBits());

  // Otherwise, promotion or legal types use the same number of registers as
  // the vector decimated to the appropriate level.
  return NumVectorRegs;
}

/// isLegalRC - Return true if the value types that can be represented by the
/// specified register class are all legal.
bool TargetLoweringBase::isLegalRC(const TargetRegisterClass *RC) const {
  for (TargetRegisterClass::vt_iterator I = RC->vt_begin(), E = RC->vt_end();
       I != E; ++I) {
    if (isTypeLegal(*I))
      return true;
  }
  return false;
}

/// Replace/modify any TargetFrameIndex operands with a targte-dependent
/// sequence of memory operands that is recognized by PrologEpilogInserter.
MachineBasicBlock*
TargetLoweringBase::emitPatchPoint(MachineInstr *MI,
                                   MachineBasicBlock *MBB) const {
  const TargetMachine &TM = getTargetMachine();
  MachineFunction &MF = *MI->getParent()->getParent();

  // MI changes inside this loop as we grow operands.
  for(unsigned OperIdx = 0; OperIdx != MI->getNumOperands(); ++OperIdx) {
    MachineOperand &MO = MI->getOperand(OperIdx);
    if (!MO.isFI())
      continue;

    // foldMemoryOperand builds a new MI after replacing a single FI operand
    // with the canonical set of five x86 addressing-mode operands.
    int FI = MO.getIndex();
    MachineInstrBuilder MIB = BuildMI(MF, MI->getDebugLoc(), MI->getDesc());

    // Copy operands before the frame-index.
    for (unsigned i = 0; i < OperIdx; ++i)
      MIB.addOperand(MI->getOperand(i));
    // Add frame index operands: direct-mem-ref tag, #FI, offset.
    MIB.addImm(StackMaps::DirectMemRefOp);
    MIB.addOperand(MI->getOperand(OperIdx));
    MIB.addImm(0);
    // Copy the operands after the frame index.
    for (unsigned i = OperIdx + 1; i != MI->getNumOperands(); ++i)
      MIB.addOperand(MI->getOperand(i));

    // Inherit previous memory operands.
    MIB->setMemRefs(MI->memoperands_begin(), MI->memoperands_end());
    assert(MIB->mayLoad() && "Folded a stackmap use to a non-load!");

    // Add a new memory operand for this FI.
    const MachineFrameInfo &MFI = *MF.getFrameInfo();
    assert(MFI.getObjectOffset(FI) != -1);
    MachineMemOperand *MMO =
      MF.getMachineMemOperand(MachinePointerInfo::getFixedStack(FI),
                              MachineMemOperand::MOLoad,
                              TM.getDataLayout()->getPointerSize(),
                              MFI.getObjectAlignment(FI));
    MIB->addMemOperand(MF, MMO);

    // Replace the instruction and update the operand index.
    MBB->insert(MachineBasicBlock::iterator(MI), MIB);
    OperIdx += (MIB->getNumOperands() - MI->getNumOperands()) - 1;
    MI->eraseFromParent();
    MI = MIB;
  }
  return MBB;
}

/// findRepresentativeClass - Return the largest legal super-reg register class
/// of the register class for the specified type and its associated "cost".
std::pair<const TargetRegisterClass*, uint8_t>
TargetLoweringBase::findRepresentativeClass(MVT VT) const {
  const TargetRegisterInfo *TRI = getTargetMachine().getRegisterInfo();
  const TargetRegisterClass *RC = RegClassForVT[VT.SimpleTy];
  if (!RC)
    return std::make_pair(RC, 0);

  // Compute the set of all super-register classes.
  BitVector SuperRegRC(TRI->getNumRegClasses());
  for (SuperRegClassIterator RCI(RC, TRI); RCI.isValid(); ++RCI)
    SuperRegRC.setBitsInMask(RCI.getMask());

  // Find the first legal register class with the largest spill size.
  const TargetRegisterClass *BestRC = RC;
  for (int i = SuperRegRC.find_first(); i >= 0; i = SuperRegRC.find_next(i)) {
    const TargetRegisterClass *SuperRC = TRI->getRegClass(i);
    // We want the largest possible spill size.
    if (SuperRC->getSize() <= BestRC->getSize())
      continue;
    if (!isLegalRC(SuperRC))
      continue;
    BestRC = SuperRC;
  }
  return std::make_pair(BestRC, 1);
}

/// computeRegisterProperties - Once all of the register classes are added,
/// this allows us to compute derived properties we expose.
void TargetLoweringBase::computeRegisterProperties() {
  assert(MVT::LAST_VALUETYPE <= MVT::MAX_ALLOWED_VALUETYPE &&
         "Too many value types for ValueTypeActions to hold!");

  // Everything defaults to needing one register.
  for (unsigned i = 0; i != MVT::LAST_VALUETYPE; ++i) {
    NumRegistersForVT[i] = 1;
    RegisterTypeForVT[i] = TransformToType[i] = (MVT::SimpleValueType)i;
  }
  // ...except isVoid, which doesn't need any registers.
  NumRegistersForVT[MVT::isVoid] = 0;

  // Find the largest integer register class.
  unsigned LargestIntReg = MVT::LAST_INTEGER_VALUETYPE;
  for (; RegClassForVT[LargestIntReg] == 0; --LargestIntReg)
    assert(LargestIntReg != MVT::i1 && "No integer registers defined!");

  // Every integer value type larger than this largest register takes twice as
  // many registers to represent as the previous ValueType.
  for (unsigned ExpandedReg = LargestIntReg + 1;
       ExpandedReg <= MVT::LAST_INTEGER_VALUETYPE; ++ExpandedReg) {
    NumRegistersForVT[ExpandedReg] = 2*NumRegistersForVT[ExpandedReg-1];
    RegisterTypeForVT[ExpandedReg] = (MVT::SimpleValueType)LargestIntReg;
    TransformToType[ExpandedReg] = (MVT::SimpleValueType)(ExpandedReg - 1);
    ValueTypeActions.setTypeAction((MVT::SimpleValueType)ExpandedReg,
                                   TypeExpandInteger);
  }

  // Inspect all of the ValueType's smaller than the largest integer
  // register to see which ones need promotion.
  unsigned LegalIntReg = LargestIntReg;
  for (unsigned IntReg = LargestIntReg - 1;
       IntReg >= (unsigned)MVT::i1; --IntReg) {
    MVT IVT = (MVT::SimpleValueType)IntReg;
    if (isTypeLegal(IVT)) {
      LegalIntReg = IntReg;
    } else {
      RegisterTypeForVT[IntReg] = TransformToType[IntReg] =
        (const MVT::SimpleValueType)LegalIntReg;
      ValueTypeActions.setTypeAction(IVT, TypePromoteInteger);
    }
  }

  // ppcf128 type is really two f64's.
  if (!isTypeLegal(MVT::ppcf128)) {
    NumRegistersForVT[MVT::ppcf128] = 2*NumRegistersForVT[MVT::f64];
    RegisterTypeForVT[MVT::ppcf128] = MVT::f64;
    TransformToType[MVT::ppcf128] = MVT::f64;
    ValueTypeActions.setTypeAction(MVT::ppcf128, TypeExpandFloat);
  }

  // Decide how to handle f128. If the target does not have native f128 support,
  // expand it to i128 and we will be generating soft float library calls.
  if (!isTypeLegal(MVT::f128)) {
    NumRegistersForVT[MVT::f128] = NumRegistersForVT[MVT::i128];
    RegisterTypeForVT[MVT::f128] = RegisterTypeForVT[MVT::i128];
    TransformToType[MVT::f128] = MVT::i128;
    ValueTypeActions.setTypeAction(MVT::f128, TypeSoftenFloat);
  }

  // Decide how to handle f64. If the target does not have native f64 support,
  // expand it to i64 and we will be generating soft float library calls.
  if (!isTypeLegal(MVT::f64)) {
    NumRegistersForVT[MVT::f64] = NumRegistersForVT[MVT::i64];
    RegisterTypeForVT[MVT::f64] = RegisterTypeForVT[MVT::i64];
    TransformToType[MVT::f64] = MVT::i64;
    ValueTypeActions.setTypeAction(MVT::f64, TypeSoftenFloat);
  }

  // Decide how to handle f32. If the target does not have native support for
  // f32, promote it to f64 if it is legal. Otherwise, expand it to i32.
  if (!isTypeLegal(MVT::f32)) {
    if (isTypeLegal(MVT::f64)) {
      NumRegistersForVT[MVT::f32] = NumRegistersForVT[MVT::f64];
      RegisterTypeForVT[MVT::f32] = RegisterTypeForVT[MVT::f64];
      TransformToType[MVT::f32] = MVT::f64;
      ValueTypeActions.setTypeAction(MVT::f32, TypePromoteInteger);
    } else {
      NumRegistersForVT[MVT::f32] = NumRegistersForVT[MVT::i32];
      RegisterTypeForVT[MVT::f32] = RegisterTypeForVT[MVT::i32];
      TransformToType[MVT::f32] = MVT::i32;
      ValueTypeActions.setTypeAction(MVT::f32, TypeSoftenFloat);
    }
  }

  // Loop over all of the vector value types to see which need transformations.
  for (unsigned i = MVT::FIRST_VECTOR_VALUETYPE;
       i <= (unsigned)MVT::LAST_VECTOR_VALUETYPE; ++i) {
    MVT VT = (MVT::SimpleValueType)i;
    if (isTypeLegal(VT)) continue;

    // Determine if there is a legal wider type.  If so, we should promote to
    // that wider vector type.
    MVT EltVT = VT.getVectorElementType();
    unsigned NElts = VT.getVectorNumElements();
    if (NElts != 1 && !shouldSplitVectorElementType(EltVT)) {
      bool IsLegalWiderType = false;
      // First try to promote the elements of integer vectors. If no legal
      // promotion was found, fallback to the widen-vector method.
      for (unsigned nVT = i+1; nVT <= MVT::LAST_VECTOR_VALUETYPE; ++nVT) {
        MVT SVT = (MVT::SimpleValueType)nVT;
        // Promote vectors of integers to vectors with the same number
        // of elements, with a wider element type.
        if (SVT.getVectorElementType().getSizeInBits() > EltVT.getSizeInBits()
            && SVT.getVectorNumElements() == NElts &&
            isTypeLegal(SVT) && SVT.getScalarType().isInteger()) {
          TransformToType[i] = SVT;
          RegisterTypeForVT[i] = SVT;
          NumRegistersForVT[i] = 1;
          ValueTypeActions.setTypeAction(VT, TypePromoteInteger);
          IsLegalWiderType = true;
          break;
        }
      }

      if (IsLegalWiderType) continue;

      // Try to widen the vector.
      for (unsigned nVT = i+1; nVT <= MVT::LAST_VECTOR_VALUETYPE; ++nVT) {
        MVT SVT = (MVT::SimpleValueType)nVT;
        if (SVT.getVectorElementType() == EltVT &&
            SVT.getVectorNumElements() > NElts &&
            isTypeLegal(SVT)) {
          TransformToType[i] = SVT;
          RegisterTypeForVT[i] = SVT;
          NumRegistersForVT[i] = 1;
          ValueTypeActions.setTypeAction(VT, TypeWidenVector);
          IsLegalWiderType = true;
          break;
        }
      }
      if (IsLegalWiderType) continue;
    }

    MVT IntermediateVT;
    MVT RegisterVT;
    unsigned NumIntermediates;
    NumRegistersForVT[i] =
      getVectorTypeBreakdownMVT(VT, IntermediateVT, NumIntermediates,
                                RegisterVT, this);
    RegisterTypeForVT[i] = RegisterVT;

    MVT NVT = VT.getPow2VectorType();
    if (NVT == VT) {
      // Type is already a power of 2.  The default action is to split.
      TransformToType[i] = MVT::Other;
      unsigned NumElts = VT.getVectorNumElements();
      ValueTypeActions.setTypeAction(VT,
            NumElts > 1 ? TypeSplitVector : TypeScalarizeVector);
    } else {
      TransformToType[i] = NVT;
      ValueTypeActions.setTypeAction(VT, TypeWidenVector);
    }
  }

  // Determine the 'representative' register class for each value type.
  // An representative register class is the largest (meaning one which is
  // not a sub-register class / subreg register class) legal register class for
  // a group of value types. For example, on i386, i8, i16, and i32
  // representative would be GR32; while on x86_64 it's GR64.
  for (unsigned i = 0; i != MVT::LAST_VALUETYPE; ++i) {
    const TargetRegisterClass* RRC;
    uint8_t Cost;
    std::tie(RRC, Cost) = findRepresentativeClass((MVT::SimpleValueType)i);
    RepRegClassForVT[i] = RRC;
    RepRegClassCostForVT[i] = Cost;
  }
}

EVT TargetLoweringBase::getSetCCResultType(LLVMContext &, EVT VT) const {
  assert(!VT.isVector() && "No default SetCC type for vectors!");
  return getPointerTy(0).SimpleTy;
}

MVT::SimpleValueType TargetLoweringBase::getCmpLibcallReturnType() const {
  return MVT::i32; // return the default value
}

/// getVectorTypeBreakdown - Vector types are broken down into some number of
/// legal first class types.  For example, MVT::v8f32 maps to 2 MVT::v4f32
/// with Altivec or SSE1, or 8 promoted MVT::f64 values with the X86 FP stack.
/// Similarly, MVT::v2i64 turns into 4 MVT::i32 values with both PPC and X86.
///
/// This method returns the number of registers needed, and the VT for each
/// register.  It also returns the VT and quantity of the intermediate values
/// before they are promoted/expanded.
///
unsigned TargetLoweringBase::getVectorTypeBreakdown(LLVMContext &Context, EVT VT,
                                                EVT &IntermediateVT,
                                                unsigned &NumIntermediates,
                                                MVT &RegisterVT) const {
  unsigned NumElts = VT.getVectorNumElements();

  // If there is a wider vector type with the same element type as this one,
  // or a promoted vector type that has the same number of elements which
  // are wider, then we should convert to that legal vector type.
  // This handles things like <2 x float> -> <4 x float> and
  // <4 x i1> -> <4 x i32>.
  LegalizeTypeAction TA = getTypeAction(Context, VT);
  if (NumElts != 1 && (TA == TypeWidenVector || TA == TypePromoteInteger)) {
    EVT RegisterEVT = getTypeToTransformTo(Context, VT);
    if (isTypeLegal(RegisterEVT)) {
      IntermediateVT = RegisterEVT;
      RegisterVT = RegisterEVT.getSimpleVT();
      NumIntermediates = 1;
      return 1;
    }
  }

  // Figure out the right, legal destination reg to copy into.
  EVT EltTy = VT.getVectorElementType();

  unsigned NumVectorRegs = 1;

  // FIXME: We don't support non-power-of-2-sized vectors for now.  Ideally we
  // could break down into LHS/RHS like LegalizeDAG does.
  if (!isPowerOf2_32(NumElts)) {
    NumVectorRegs = NumElts;
    NumElts = 1;
  }

  // Divide the input until we get to a supported size.  This will always
  // end with a scalar if the target doesn't support vectors.
  while (NumElts > 1 && !isTypeLegal(
                                   EVT::getVectorVT(Context, EltTy, NumElts))) {
    NumElts >>= 1;
    NumVectorRegs <<= 1;
  }

  NumIntermediates = NumVectorRegs;

  EVT NewVT = EVT::getVectorVT(Context, EltTy, NumElts);
  if (!isTypeLegal(NewVT))
    NewVT = EltTy;
  IntermediateVT = NewVT;

  MVT DestVT = getRegisterType(Context, NewVT);
  RegisterVT = DestVT;
  unsigned NewVTSize = NewVT.getSizeInBits();

  // Convert sizes such as i33 to i64.
  if (!isPowerOf2_32(NewVTSize))
    NewVTSize = NextPowerOf2(NewVTSize);

  if (EVT(DestVT).bitsLT(NewVT))   // Value is expanded, e.g. i64 -> i16.
    return NumVectorRegs*(NewVTSize/DestVT.getSizeInBits());

  // Otherwise, promotion or legal types use the same number of registers as
  // the vector decimated to the appropriate level.
  return NumVectorRegs;
}

/// Get the EVTs and ArgFlags collections that represent the legalized return
/// type of the given function.  This does not require a DAG or a return value,
/// and is suitable for use before any DAGs for the function are constructed.
/// TODO: Move this out of TargetLowering.cpp.
void llvm::GetReturnInfo(Type* ReturnType, AttributeSet attr,
                         SmallVectorImpl<ISD::OutputArg> &Outs,
                         const TargetLowering &TLI) {
  SmallVector<EVT, 4> ValueVTs;
  ComputeValueVTs(TLI, ReturnType, ValueVTs);
  unsigned NumValues = ValueVTs.size();
  if (NumValues == 0) return;

  for (unsigned j = 0, f = NumValues; j != f; ++j) {
    EVT VT = ValueVTs[j];
    ISD::NodeType ExtendKind = ISD::ANY_EXTEND;

    if (attr.hasAttribute(AttributeSet::ReturnIndex, Attribute::SExt))
      ExtendKind = ISD::SIGN_EXTEND;
    else if (attr.hasAttribute(AttributeSet::ReturnIndex, Attribute::ZExt))
      ExtendKind = ISD::ZERO_EXTEND;

    // FIXME: C calling convention requires the return type to be promoted to
    // at least 32-bit. But this is not necessary for non-C calling
    // conventions. The frontend should mark functions whose return values
    // require promoting with signext or zeroext attributes.
    if (ExtendKind != ISD::ANY_EXTEND && VT.isInteger()) {
      MVT MinVT = TLI.getRegisterType(ReturnType->getContext(), MVT::i32);
      if (VT.bitsLT(MinVT))
        VT = MinVT;
    }

    unsigned NumParts = TLI.getNumRegisters(ReturnType->getContext(), VT);
    MVT PartVT = TLI.getRegisterType(ReturnType->getContext(), VT);

    // 'inreg' on function refers to return value
    ISD::ArgFlagsTy Flags = ISD::ArgFlagsTy();
    if (attr.hasAttribute(AttributeSet::ReturnIndex, Attribute::InReg))
      Flags.setInReg();

    // Propagate extension type if any
    if (attr.hasAttribute(AttributeSet::ReturnIndex, Attribute::SExt))
      Flags.setSExt();
    else if (attr.hasAttribute(AttributeSet::ReturnIndex, Attribute::ZExt))
      Flags.setZExt();

    for (unsigned i = 0; i < NumParts; ++i)
      Outs.push_back(ISD::OutputArg(Flags, PartVT, VT, /*isFixed=*/true, 0, 0));
  }
}

/// getByValTypeAlignment - Return the desired alignment for ByVal aggregate
/// function arguments in the caller parameter area.  This is the actual
/// alignment, not its logarithm.
unsigned TargetLoweringBase::getByValTypeAlignment(Type *Ty) const {
  return DL->getABITypeAlignment(Ty);
}

//===----------------------------------------------------------------------===//
//  TargetTransformInfo Helpers
//===----------------------------------------------------------------------===//

int TargetLoweringBase::InstructionOpcodeToISD(unsigned Opcode) const {
  enum InstructionOpcodes {
#define HANDLE_INST(NUM, OPCODE, CLASS) OPCODE = NUM,
#define LAST_OTHER_INST(NUM) InstructionOpcodesCount = NUM
#include "llvm/IR/Instruction.def"
  };
  switch (static_cast<InstructionOpcodes>(Opcode)) {
  case Ret:            return 0;
  case Br:             return 0;
  case Switch:         return 0;
  case IndirectBr:     return 0;
  case Invoke:         return 0;
  case Resume:         return 0;
  case Unreachable:    return 0;
  case Add:            return ISD::ADD;
  case FAdd:           return ISD::FADD;
  case Sub:            return ISD::SUB;
  case FSub:           return ISD::FSUB;
  case Mul:            return ISD::MUL;
  case FMul:           return ISD::FMUL;
  case UDiv:           return ISD::UDIV;
  case SDiv:           return ISD::UDIV;
  case FDiv:           return ISD::FDIV;
  case URem:           return ISD::UREM;
  case SRem:           return ISD::SREM;
  case FRem:           return ISD::FREM;
  case Shl:            return ISD::SHL;
  case LShr:           return ISD::SRL;
  case AShr:           return ISD::SRA;
  case And:            return ISD::AND;
  case Or:             return ISD::OR;
  case Xor:            return ISD::XOR;
  case Alloca:         return 0;
  case Load:           return ISD::LOAD;
  case Store:          return ISD::STORE;
  case GetElementPtr:  return 0;
  case Fence:          return 0;
  case AtomicCmpXchg:  return 0;
  case AtomicRMW:      return 0;
  case Trunc:          return ISD::TRUNCATE;
  case ZExt:           return ISD::ZERO_EXTEND;
  case SExt:           return ISD::SIGN_EXTEND;
  case FPToUI:         return ISD::FP_TO_UINT;
  case FPToSI:         return ISD::FP_TO_SINT;
  case UIToFP:         return ISD::UINT_TO_FP;
  case SIToFP:         return ISD::SINT_TO_FP;
  case FPTrunc:        return ISD::FP_ROUND;
  case FPExt:          return ISD::FP_EXTEND;
  case PtrToInt:       return ISD::BITCAST;
  case IntToPtr:       return ISD::BITCAST;
  case BitCast:        return ISD::BITCAST;
  case AddrSpaceCast:  return ISD::ADDRSPACECAST;
  case ICmp:           return ISD::SETCC;
  case FCmp:           return ISD::SETCC;
  case PHI:            return 0;
  case Call:           return 0;
  case Select:         return ISD::SELECT;
  case UserOp1:        return 0;
  case UserOp2:        return 0;
  case VAArg:          return 0;
  case ExtractElement: return ISD::EXTRACT_VECTOR_ELT;
  case InsertElement:  return ISD::INSERT_VECTOR_ELT;
  case ShuffleVector:  return ISD::VECTOR_SHUFFLE;
  case ExtractValue:   return ISD::MERGE_VALUES;
  case InsertValue:    return ISD::MERGE_VALUES;
  case LandingPad:     return 0;
  }

  llvm_unreachable("Unknown instruction type encountered!");
}

std::pair<unsigned, MVT>
TargetLoweringBase::getTypeLegalizationCost(Type *Ty) const {
  LLVMContext &C = Ty->getContext();
  EVT MTy = getValueType(Ty);

  unsigned Cost = 1;
  // We keep legalizing the type until we find a legal kind. We assume that
  // the only operation that costs anything is the split. After splitting
  // we need to handle two types.
  while (true) {
    LegalizeKind LK = getTypeConversion(C, MTy);

    if (LK.first == TypeLegal)
      return std::make_pair(Cost, MTy.getSimpleVT());

    if (LK.first == TypeSplitVector || LK.first == TypeExpandInteger)
      Cost *= 2;

    // Keep legalizing the type.
    MTy = LK.second;
  }
}

//===----------------------------------------------------------------------===//
//  Loop Strength Reduction hooks
//===----------------------------------------------------------------------===//

/// isLegalAddressingMode - Return true if the addressing mode represented
/// by AM is legal for this target, for a load/store of the specified type.
bool TargetLoweringBase::isLegalAddressingMode(const AddrMode &AM,
                                           Type *Ty) const {
  // The default implementation of this implements a conservative RISCy, r+r and
  // r+i addr mode.

  // Allows a sign-extended 16-bit immediate field.
  if (AM.BaseOffs <= -(1LL << 16) || AM.BaseOffs >= (1LL << 16)-1)
    return false;

  // No global is ever allowed as a base.
  if (AM.BaseGV)
    return false;

  // Only support r+r,
  switch (AM.Scale) {
  case 0:  // "r+i" or just "i", depending on HasBaseReg.
    break;
  case 1:
    if (AM.HasBaseReg && AM.BaseOffs)  // "r+r+i" is not allowed.
      return false;
    // Otherwise we have r+r or r+i.
    break;
  case 2:
    if (AM.HasBaseReg || AM.BaseOffs)  // 2*r+r  or  2*r+i is not allowed.
      return false;
    // Allow 2*r as r+r.
    break;
  default: // Don't allow n * r
    return false;
  }

  return true;
}
