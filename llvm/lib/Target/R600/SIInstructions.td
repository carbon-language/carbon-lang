//===-- SIInstructions.td - SI Instruction Defintions ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This file was originally auto-generated from a GPU register header file and
// all the instruction definitions were originally commented out.  Instructions
// that are not yet supported remain commented out.
//===----------------------------------------------------------------------===//

class InterpSlots {
int P0 = 2;
int P10 = 0;
int P20 = 1;
}
def INTERP : InterpSlots;

def InterpSlot : Operand<i32> {
  let PrintMethod = "printInterpSlot";
}

def SendMsgImm : Operand<i32> {
  let PrintMethod = "printSendMsg";
}

def isSI : Predicate<"Subtarget.getGeneration() "
                      ">= AMDGPUSubtarget::SOUTHERN_ISLANDS">;

def isCI : Predicate<"Subtarget.getGeneration() "
                      ">= AMDGPUSubtarget::SEA_ISLANDS">;

def isCFDepth0 : Predicate<"isCFDepth0()">;

def WAIT_FLAG : InstFlag<"printWaitFlag">;

let SubtargetPredicate = isSI in {
let OtherPredicates  = [isCFDepth0] in {

//===----------------------------------------------------------------------===//
// SMRD Instructions
//===----------------------------------------------------------------------===//

let mayLoad = 1 in {

// We are using the SGPR_32 and not the SReg_32 register class for 32-bit
// SMRD instructions, because the SGPR_32 register class does not include M0
// and writing to M0 from an SMRD instruction will hang the GPU.
defm S_LOAD_DWORD : SMRD_Helper <0x00, "S_LOAD_DWORD", SReg_64, SGPR_32>;
defm S_LOAD_DWORDX2 : SMRD_Helper <0x01, "S_LOAD_DWORDX2", SReg_64, SReg_64>;
defm S_LOAD_DWORDX4 : SMRD_Helper <0x02, "S_LOAD_DWORDX4", SReg_64, SReg_128>;
defm S_LOAD_DWORDX8 : SMRD_Helper <0x03, "S_LOAD_DWORDX8", SReg_64, SReg_256>;
defm S_LOAD_DWORDX16 : SMRD_Helper <0x04, "S_LOAD_DWORDX16", SReg_64, SReg_512>;

defm S_BUFFER_LOAD_DWORD : SMRD_Helper <
  0x08, "S_BUFFER_LOAD_DWORD", SReg_128, SGPR_32
>;

defm S_BUFFER_LOAD_DWORDX2 : SMRD_Helper <
  0x09, "S_BUFFER_LOAD_DWORDX2", SReg_128, SReg_64
>;

defm S_BUFFER_LOAD_DWORDX4 : SMRD_Helper <
  0x0a, "S_BUFFER_LOAD_DWORDX4", SReg_128, SReg_128
>;

defm S_BUFFER_LOAD_DWORDX8 : SMRD_Helper <
  0x0b, "S_BUFFER_LOAD_DWORDX8", SReg_128, SReg_256
>;

defm S_BUFFER_LOAD_DWORDX16 : SMRD_Helper <
  0x0c, "S_BUFFER_LOAD_DWORDX16", SReg_128, SReg_512
>;

} // mayLoad = 1

//def S_MEMTIME : SMRD_ <0x0000001e, "S_MEMTIME", []>;
//def S_DCACHE_INV : SMRD_ <0x0000001f, "S_DCACHE_INV", []>;

//===----------------------------------------------------------------------===//
// SOP1 Instructions
//===----------------------------------------------------------------------===//

let isMoveImm = 1 in {
def S_MOV_B32 : SOP1_32 <0x00000003, "S_MOV_B32", []>;
def S_MOV_B64 : SOP1_64 <0x00000004, "S_MOV_B64", []>;
def S_CMOV_B32 : SOP1_32 <0x00000005, "S_CMOV_B32", []>;
def S_CMOV_B64 : SOP1_64 <0x00000006, "S_CMOV_B64", []>;
} // End isMoveImm = 1

def S_NOT_B32 : SOP1_32 <0x00000007, "S_NOT_B32",
  [(set i32:$dst, (not i32:$src0))]
>;

def S_NOT_B64 : SOP1_64 <0x00000008, "S_NOT_B64",
  [(set i64:$dst, (not i64:$src0))]
>;
def S_WQM_B32 : SOP1_32 <0x00000009, "S_WQM_B32", []>;
def S_WQM_B64 : SOP1_64 <0x0000000a, "S_WQM_B64", []>;
def S_BREV_B32 : SOP1_32 <0x0000000b, "S_BREV_B32",
  [(set i32:$dst, (AMDGPUbrev i32:$src0))]
>;
def S_BREV_B64 : SOP1_64 <0x0000000c, "S_BREV_B64", []>;

////def S_BCNT0_I32_B32 : SOP1_BCNT0 <0x0000000d, "S_BCNT0_I32_B32", []>;
////def S_BCNT0_I32_B64 : SOP1_BCNT0 <0x0000000e, "S_BCNT0_I32_B64", []>;
def S_BCNT1_I32_B32 : SOP1_32 <0x0000000f, "S_BCNT1_I32_B32",
  [(set i32:$dst, (ctpop i32:$src0))]
>;
def S_BCNT1_I32_B64 : SOP1_32_64 <0x00000010, "S_BCNT1_I32_B64", []>;

////def S_FF0_I32_B32 : SOP1_32 <0x00000011, "S_FF0_I32_B32", []>;
////def S_FF0_I32_B64 : SOP1_FF0 <0x00000012, "S_FF0_I32_B64", []>;
def S_FF1_I32_B32 : SOP1_32 <0x00000013, "S_FF1_I32_B32",
  [(set i32:$dst, (cttz_zero_undef i32:$src0))]
>;
////def S_FF1_I32_B64 : SOP1_FF1 <0x00000014, "S_FF1_I32_B64", []>;

def S_FLBIT_I32_B32 : SOP1_32 <0x00000015, "S_FLBIT_I32_B32",
  [(set i32:$dst, (ctlz_zero_undef i32:$src0))]
>;

//def S_FLBIT_I32_B64 : SOP1_32 <0x00000016, "S_FLBIT_I32_B64", []>;
def S_FLBIT_I32 : SOP1_32 <0x00000017, "S_FLBIT_I32", []>;
//def S_FLBIT_I32_I64 : SOP1_32 <0x00000018, "S_FLBIT_I32_I64", []>;
def S_SEXT_I32_I8 : SOP1_32 <0x00000019, "S_SEXT_I32_I8",
  [(set i32:$dst, (sext_inreg i32:$src0, i8))]
>;
def S_SEXT_I32_I16 : SOP1_32 <0x0000001a, "S_SEXT_I32_I16",
  [(set i32:$dst, (sext_inreg i32:$src0, i16))]
>;

////def S_BITSET0_B32 : SOP1_BITSET0 <0x0000001b, "S_BITSET0_B32", []>;
////def S_BITSET0_B64 : SOP1_BITSET0 <0x0000001c, "S_BITSET0_B64", []>;
////def S_BITSET1_B32 : SOP1_BITSET1 <0x0000001d, "S_BITSET1_B32", []>;
////def S_BITSET1_B64 : SOP1_BITSET1 <0x0000001e, "S_BITSET1_B64", []>;
def S_GETPC_B64 : SOP1 <
  0x0000001f, (outs SReg_64:$dst), (ins), "S_GETPC_B64 $dst", []
> {
  let SSRC0 = 0;
}
def S_SETPC_B64 : SOP1_64 <0x00000020, "S_SETPC_B64", []>;
def S_SWAPPC_B64 : SOP1_64 <0x00000021, "S_SWAPPC_B64", []>;
def S_RFE_B64 : SOP1_64 <0x00000022, "S_RFE_B64", []>;

let hasSideEffects = 1, Uses = [EXEC], Defs = [EXEC] in {

def S_AND_SAVEEXEC_B64 : SOP1_64 <0x00000024, "S_AND_SAVEEXEC_B64", []>;
def S_OR_SAVEEXEC_B64 : SOP1_64 <0x00000025, "S_OR_SAVEEXEC_B64", []>;
def S_XOR_SAVEEXEC_B64 : SOP1_64 <0x00000026, "S_XOR_SAVEEXEC_B64", []>;
def S_ANDN2_SAVEEXEC_B64 : SOP1_64 <0x00000027, "S_ANDN2_SAVEEXEC_B64", []>;
def S_ORN2_SAVEEXEC_B64 : SOP1_64 <0x00000028, "S_ORN2_SAVEEXEC_B64", []>;
def S_NAND_SAVEEXEC_B64 : SOP1_64 <0x00000029, "S_NAND_SAVEEXEC_B64", []>;
def S_NOR_SAVEEXEC_B64 : SOP1_64 <0x0000002a, "S_NOR_SAVEEXEC_B64", []>;
def S_XNOR_SAVEEXEC_B64 : SOP1_64 <0x0000002b, "S_XNOR_SAVEEXEC_B64", []>;

} // End hasSideEffects = 1

def S_QUADMASK_B32 : SOP1_32 <0x0000002c, "S_QUADMASK_B32", []>;
def S_QUADMASK_B64 : SOP1_64 <0x0000002d, "S_QUADMASK_B64", []>;
def S_MOVRELS_B32 : SOP1_32 <0x0000002e, "S_MOVRELS_B32", []>;
def S_MOVRELS_B64 : SOP1_64 <0x0000002f, "S_MOVRELS_B64", []>;
def S_MOVRELD_B32 : SOP1_32 <0x00000030, "S_MOVRELD_B32", []>;
def S_MOVRELD_B64 : SOP1_64 <0x00000031, "S_MOVRELD_B64", []>;
//def S_CBRANCH_JOIN : SOP1_ <0x00000032, "S_CBRANCH_JOIN", []>;
def S_MOV_REGRD_B32 : SOP1_32 <0x00000033, "S_MOV_REGRD_B32", []>;
def S_ABS_I32 : SOP1_32 <0x00000034, "S_ABS_I32", []>;
def S_MOV_FED_B32 : SOP1_32 <0x00000035, "S_MOV_FED_B32", []>;

//===----------------------------------------------------------------------===//
// SOP2 Instructions
//===----------------------------------------------------------------------===//

let Defs = [SCC] in { // Carry out goes to SCC
let isCommutable = 1 in {
def S_ADD_U32 : SOP2_32 <0x00000000, "S_ADD_U32", []>;
def S_ADD_I32 : SOP2_32 <0x00000002, "S_ADD_I32",
  [(set i32:$dst, (add SSrc_32:$src0, SSrc_32:$src1))]
>;
} // End isCommutable = 1

def S_SUB_U32 : SOP2_32 <0x00000001, "S_SUB_U32", []>;
def S_SUB_I32 : SOP2_32 <0x00000003, "S_SUB_I32",
  [(set i32:$dst, (sub SSrc_32:$src0, SSrc_32:$src1))]
>;

let Uses = [SCC] in { // Carry in comes from SCC
let isCommutable = 1 in {
def S_ADDC_U32 : SOP2_32 <0x00000004, "S_ADDC_U32",
  [(set i32:$dst, (adde (i32 SSrc_32:$src0), (i32 SSrc_32:$src1)))]>;
} // End isCommutable = 1

def S_SUBB_U32 : SOP2_32 <0x00000005, "S_SUBB_U32",
  [(set i32:$dst, (sube (i32 SSrc_32:$src0), (i32 SSrc_32:$src1)))]>;
} // End Uses = [SCC]
} // End Defs = [SCC]

def S_MIN_I32 : SOP2_32 <0x00000006, "S_MIN_I32",
  [(set i32:$dst, (AMDGPUsmin i32:$src0, i32:$src1))]
>;
def S_MIN_U32 : SOP2_32 <0x00000007, "S_MIN_U32",
  [(set i32:$dst, (AMDGPUumin i32:$src0, i32:$src1))]
>;
def S_MAX_I32 : SOP2_32 <0x00000008, "S_MAX_I32",
  [(set i32:$dst, (AMDGPUsmax i32:$src0, i32:$src1))]
>;
def S_MAX_U32 : SOP2_32 <0x00000009, "S_MAX_U32",
  [(set i32:$dst, (AMDGPUumax i32:$src0, i32:$src1))]
>;

def S_CSELECT_B32 : SOP2 <
  0x0000000a, (outs SReg_32:$dst),
  (ins SReg_32:$src0, SReg_32:$src1, SCCReg:$scc), "S_CSELECT_B32",
  []
>;

def S_CSELECT_B64 : SOP2_64 <0x0000000b, "S_CSELECT_B64", []>;

def S_AND_B32 : SOP2_32 <0x0000000e, "S_AND_B32",
  [(set i32:$dst, (and i32:$src0, i32:$src1))]
>;

def S_AND_B64 : SOP2_64 <0x0000000f, "S_AND_B64",
  [(set i64:$dst, (and i64:$src0, i64:$src1))]
>;

def S_OR_B32 : SOP2_32 <0x00000010, "S_OR_B32",
  [(set i32:$dst, (or i32:$src0, i32:$src1))]
>;

def S_OR_B64 : SOP2_64 <0x00000011, "S_OR_B64",
  [(set i64:$dst, (or i64:$src0, i64:$src1))]
>;

def S_XOR_B32 : SOP2_32 <0x00000012, "S_XOR_B32",
  [(set i32:$dst, (xor i32:$src0, i32:$src1))]
>;

def S_XOR_B64 : SOP2_64 <0x00000013, "S_XOR_B64",
  [(set i64:$dst, (xor i64:$src0, i64:$src1))]
>;
def S_ANDN2_B32 : SOP2_32 <0x00000014, "S_ANDN2_B32", []>;
def S_ANDN2_B64 : SOP2_64 <0x00000015, "S_ANDN2_B64", []>;
def S_ORN2_B32 : SOP2_32 <0x00000016, "S_ORN2_B32", []>;
def S_ORN2_B64 : SOP2_64 <0x00000017, "S_ORN2_B64", []>;
def S_NAND_B32 : SOP2_32 <0x00000018, "S_NAND_B32", []>;
def S_NAND_B64 : SOP2_64 <0x00000019, "S_NAND_B64", []>;
def S_NOR_B32 : SOP2_32 <0x0000001a, "S_NOR_B32", []>;
def S_NOR_B64 : SOP2_64 <0x0000001b, "S_NOR_B64", []>;
def S_XNOR_B32 : SOP2_32 <0x0000001c, "S_XNOR_B32", []>;
def S_XNOR_B64 : SOP2_64 <0x0000001d, "S_XNOR_B64", []>;

// Use added complexity so these patterns are preferred to the VALU patterns.
let AddedComplexity = 1 in {

def S_LSHL_B32 : SOP2_32 <0x0000001e, "S_LSHL_B32",
  [(set i32:$dst, (shl i32:$src0, i32:$src1))]
>;
def S_LSHL_B64 : SOP2_SHIFT_64 <0x0000001f, "S_LSHL_B64",
  [(set i64:$dst, (shl i64:$src0, i32:$src1))]
>;
def S_LSHR_B32 : SOP2_32 <0x00000020, "S_LSHR_B32",
  [(set i32:$dst, (srl i32:$src0, i32:$src1))]
>;
def S_LSHR_B64 : SOP2_SHIFT_64 <0x00000021, "S_LSHR_B64",
  [(set i64:$dst, (srl i64:$src0, i32:$src1))]
>;
def S_ASHR_I32 : SOP2_32 <0x00000022, "S_ASHR_I32",
  [(set i32:$dst, (sra i32:$src0, i32:$src1))]
>;
def S_ASHR_I64 : SOP2_SHIFT_64 <0x00000023, "S_ASHR_I64",
  [(set i64:$dst, (sra i64:$src0, i32:$src1))]
>;

} // End AddedComplexity = 1

def S_BFM_B32 : SOP2_32 <0x00000024, "S_BFM_B32", []>;
def S_BFM_B64 : SOP2_64 <0x00000025, "S_BFM_B64", []>;
def S_MUL_I32 : SOP2_32 <0x00000026, "S_MUL_I32", []>;
def S_BFE_U32 : SOP2_32 <0x00000027, "S_BFE_U32", []>;
def S_BFE_I32 : SOP2_32 <0x00000028, "S_BFE_I32", []>;
def S_BFE_U64 : SOP2_64 <0x00000029, "S_BFE_U64", []>;
def S_BFE_I64 : SOP2_64 <0x0000002a, "S_BFE_I64", []>;
//def S_CBRANCH_G_FORK : SOP2_ <0x0000002b, "S_CBRANCH_G_FORK", []>;
def S_ABSDIFF_I32 : SOP2_32 <0x0000002c, "S_ABSDIFF_I32", []>;

//===----------------------------------------------------------------------===//
// SOPC Instructions
//===----------------------------------------------------------------------===//

def S_CMP_EQ_I32 : SOPC_32 <0x00000000, "S_CMP_EQ_I32">;
def S_CMP_LG_I32 : SOPC_32 <0x00000001, "S_CMP_LG_I32">;
def S_CMP_GT_I32 : SOPC_32 <0x00000002, "S_CMP_GT_I32">;
def S_CMP_GE_I32 : SOPC_32 <0x00000003, "S_CMP_GE_I32">;
def S_CMP_LT_I32 : SOPC_32 <0x00000004, "S_CMP_LT_I32">;
def S_CMP_LE_I32 : SOPC_32 <0x00000005, "S_CMP_LE_I32">;
def S_CMP_EQ_U32 : SOPC_32 <0x00000006, "S_CMP_EQ_U32">;
def S_CMP_LG_U32 : SOPC_32 <0x00000007, "S_CMP_LG_U32">;
def S_CMP_GT_U32 : SOPC_32 <0x00000008, "S_CMP_GT_U32">;
def S_CMP_GE_U32 : SOPC_32 <0x00000009, "S_CMP_GE_U32">;
def S_CMP_LT_U32 : SOPC_32 <0x0000000a, "S_CMP_LT_U32">;
def S_CMP_LE_U32 : SOPC_32 <0x0000000b, "S_CMP_LE_U32">;
////def S_BITCMP0_B32 : SOPC_BITCMP0 <0x0000000c, "S_BITCMP0_B32", []>;
////def S_BITCMP1_B32 : SOPC_BITCMP1 <0x0000000d, "S_BITCMP1_B32", []>;
////def S_BITCMP0_B64 : SOPC_BITCMP0 <0x0000000e, "S_BITCMP0_B64", []>;
////def S_BITCMP1_B64 : SOPC_BITCMP1 <0x0000000f, "S_BITCMP1_B64", []>;
//def S_SETVSKIP : SOPC_ <0x00000010, "S_SETVSKIP", []>;

//===----------------------------------------------------------------------===//
// SOPK Instructions
//===----------------------------------------------------------------------===//

def S_MOVK_I32 : SOPK_32 <0x00000000, "S_MOVK_I32", []>;
def S_CMOVK_I32 : SOPK_32 <0x00000002, "S_CMOVK_I32", []>;

/*
This instruction is disabled for now until we can figure out how to teach
the instruction selector to correctly use the  S_CMP* vs V_CMP*
instructions.

When this instruction is enabled the code generator sometimes produces this
invalid sequence:

SCC = S_CMPK_EQ_I32 SGPR0, imm
VCC = COPY SCC
VGPR0 = V_CNDMASK VCC, VGPR0, VGPR1

def S_CMPK_EQ_I32 : SOPK <
  0x00000003, (outs SCCReg:$dst), (ins SReg_32:$src0, i32imm:$src1),
  "S_CMPK_EQ_I32",
  [(set i1:$dst, (setcc i32:$src0, imm:$src1, SETEQ))]
>;
*/

let isCompare = 1, Defs = [SCC] in {
def S_CMPK_LG_I32 : SOPK_32 <0x00000004, "S_CMPK_LG_I32", []>;
def S_CMPK_GT_I32 : SOPK_32 <0x00000005, "S_CMPK_GT_I32", []>;
def S_CMPK_GE_I32 : SOPK_32 <0x00000006, "S_CMPK_GE_I32", []>;
def S_CMPK_LT_I32 : SOPK_32 <0x00000007, "S_CMPK_LT_I32", []>;
def S_CMPK_LE_I32 : SOPK_32 <0x00000008, "S_CMPK_LE_I32", []>;
def S_CMPK_EQ_U32 : SOPK_32 <0x00000009, "S_CMPK_EQ_U32", []>;
def S_CMPK_LG_U32 : SOPK_32 <0x0000000a, "S_CMPK_LG_U32", []>;
def S_CMPK_GT_U32 : SOPK_32 <0x0000000b, "S_CMPK_GT_U32", []>;
def S_CMPK_GE_U32 : SOPK_32 <0x0000000c, "S_CMPK_GE_U32", []>;
def S_CMPK_LT_U32 : SOPK_32 <0x0000000d, "S_CMPK_LT_U32", []>;
def S_CMPK_LE_U32 : SOPK_32 <0x0000000e, "S_CMPK_LE_U32", []>;
} // End isCompare = 1, Defs = [SCC]

let Defs = [SCC], isCommutable = 1 in {
  def S_ADDK_I32 : SOPK_32 <0x0000000f, "S_ADDK_I32", []>;
  def S_MULK_I32 : SOPK_32 <0x00000010, "S_MULK_I32", []>;
}

//def S_CBRANCH_I_FORK : SOPK_ <0x00000011, "S_CBRANCH_I_FORK", []>;
def S_GETREG_B32 : SOPK_32 <0x00000012, "S_GETREG_B32", []>;
def S_SETREG_B32 : SOPK_32 <0x00000013, "S_SETREG_B32", []>;
def S_GETREG_REGRD_B32 : SOPK_32 <0x00000014, "S_GETREG_REGRD_B32", []>;
//def S_SETREG_IMM32_B32 : SOPK_32 <0x00000015, "S_SETREG_IMM32_B32", []>;
//def EXP : EXP_ <0x00000000, "EXP", []>;

} // End let OtherPredicates = [isCFDepth0]

//===----------------------------------------------------------------------===//
// SOPP Instructions
//===----------------------------------------------------------------------===//

def S_NOP : SOPP <0x00000000, (ins i16imm:$simm16), "S_NOP $simm16", []>;

let isTerminator = 1 in {

def S_ENDPGM : SOPP <0x00000001, (ins), "S_ENDPGM",
  [(IL_retflag)]> {
  let simm16 = 0;
  let isBarrier = 1;
  let hasCtrlDep = 1;
}

let isBranch = 1 in {
def S_BRANCH : SOPP <
  0x00000002, (ins sopp_brtarget:$simm16), "S_BRANCH $simm16",
  [(br bb:$simm16)]> {
  let isBarrier = 1;
}

let DisableEncoding = "$scc" in {
def S_CBRANCH_SCC0 : SOPP <
  0x00000004, (ins sopp_brtarget:$simm16, SCCReg:$scc),
  "S_CBRANCH_SCC0 $simm16", []
>;
def S_CBRANCH_SCC1 : SOPP <
  0x00000005, (ins sopp_brtarget:$simm16, SCCReg:$scc),
  "S_CBRANCH_SCC1 $simm16",
  []
>;
} // End DisableEncoding = "$scc"

def S_CBRANCH_VCCZ : SOPP <
  0x00000006, (ins sopp_brtarget:$simm16, VCCReg:$vcc),
  "S_CBRANCH_VCCZ $simm16",
  []
>;
def S_CBRANCH_VCCNZ : SOPP <
  0x00000007, (ins sopp_brtarget:$simm16, VCCReg:$vcc),
  "S_CBRANCH_VCCNZ $simm16",
  []
>;

let DisableEncoding = "$exec" in {
def S_CBRANCH_EXECZ : SOPP <
  0x00000008, (ins sopp_brtarget:$simm16, EXECReg:$exec),
  "S_CBRANCH_EXECZ $simm16",
  []
>;
def S_CBRANCH_EXECNZ : SOPP <
  0x00000009, (ins sopp_brtarget:$simm16, EXECReg:$exec),
  "S_CBRANCH_EXECNZ $simm16",
  []
>;
} // End DisableEncoding = "$exec"


} // End isBranch = 1
} // End isTerminator = 1

let hasSideEffects = 1 in {
def S_BARRIER : SOPP <0x0000000a, (ins), "S_BARRIER",
  [(int_AMDGPU_barrier_local)]
> {
  let simm16 = 0;
  let isBarrier = 1;
  let hasCtrlDep = 1;
  let mayLoad = 1;
  let mayStore = 1;
}

def S_WAITCNT : SOPP <0x0000000c, (ins WAIT_FLAG:$simm16), "S_WAITCNT $simm16",
  []
>;
//def S_SETHALT : SOPP_ <0x0000000d, "S_SETHALT", []>;
//def S_SLEEP : SOPP_ <0x0000000e, "S_SLEEP", []>;
//def S_SETPRIO : SOPP_ <0x0000000f, "S_SETPRIO", []>;

let Uses = [EXEC] in {
  def S_SENDMSG : SOPP <0x00000010, (ins SendMsgImm:$simm16, M0Reg:$m0), "S_SENDMSG $simm16",
      [(int_SI_sendmsg imm:$simm16, M0Reg:$m0)]
  > {
    let DisableEncoding = "$m0";
  }
} // End Uses = [EXEC]

//def S_SENDMSGHALT : SOPP_ <0x00000011, "S_SENDMSGHALT", []>;
//def S_TRAP : SOPP_ <0x00000012, "S_TRAP", []>;
//def S_ICACHE_INV : SOPP_ <0x00000013, "S_ICACHE_INV", []>;
//def S_INCPERFLEVEL : SOPP_ <0x00000014, "S_INCPERFLEVEL", []>;
//def S_DECPERFLEVEL : SOPP_ <0x00000015, "S_DECPERFLEVEL", []>;
//def S_TTRACEDATA : SOPP_ <0x00000016, "S_TTRACEDATA", []>;
} // End hasSideEffects

//===----------------------------------------------------------------------===//
// VOPC Instructions
//===----------------------------------------------------------------------===//

let isCompare = 1 in {

defm V_CMP_F_F32 : VOPC_F32 <0x00000000, "V_CMP_F_F32">;
defm V_CMP_LT_F32 : VOPC_F32 <0x00000001, "V_CMP_LT_F32", COND_OLT>;
defm V_CMP_EQ_F32 : VOPC_F32 <0x00000002, "V_CMP_EQ_F32", COND_OEQ>;
defm V_CMP_LE_F32 : VOPC_F32 <0x00000003, "V_CMP_LE_F32", COND_OLE>;
defm V_CMP_GT_F32 : VOPC_F32 <0x00000004, "V_CMP_GT_F32", COND_OGT>;
defm V_CMP_LG_F32 : VOPC_F32 <0x00000005, "V_CMP_LG_F32">;
defm V_CMP_GE_F32 : VOPC_F32 <0x00000006, "V_CMP_GE_F32", COND_OGE>;
defm V_CMP_O_F32 : VOPC_F32 <0x00000007, "V_CMP_O_F32", COND_O>;
defm V_CMP_U_F32 : VOPC_F32 <0x00000008, "V_CMP_U_F32", COND_UO>;
defm V_CMP_NGE_F32 : VOPC_F32 <0x00000009, "V_CMP_NGE_F32">;
defm V_CMP_NLG_F32 : VOPC_F32 <0x0000000a, "V_CMP_NLG_F32">;
defm V_CMP_NGT_F32 : VOPC_F32 <0x0000000b, "V_CMP_NGT_F32">;
defm V_CMP_NLE_F32 : VOPC_F32 <0x0000000c, "V_CMP_NLE_F32">;
defm V_CMP_NEQ_F32 : VOPC_F32 <0x0000000d, "V_CMP_NEQ_F32", COND_UNE>;
defm V_CMP_NLT_F32 : VOPC_F32 <0x0000000e, "V_CMP_NLT_F32">;
defm V_CMP_TRU_F32 : VOPC_F32 <0x0000000f, "V_CMP_TRU_F32">;

let hasSideEffects = 1 in {

defm V_CMPX_F_F32 : VOPCX_F32 <0x00000010, "V_CMPX_F_F32">;
defm V_CMPX_LT_F32 : VOPCX_F32 <0x00000011, "V_CMPX_LT_F32">;
defm V_CMPX_EQ_F32 : VOPCX_F32 <0x00000012, "V_CMPX_EQ_F32">;
defm V_CMPX_LE_F32 : VOPCX_F32 <0x00000013, "V_CMPX_LE_F32">;
defm V_CMPX_GT_F32 : VOPCX_F32 <0x00000014, "V_CMPX_GT_F32">;
defm V_CMPX_LG_F32 : VOPCX_F32 <0x00000015, "V_CMPX_LG_F32">;
defm V_CMPX_GE_F32 : VOPCX_F32 <0x00000016, "V_CMPX_GE_F32">;
defm V_CMPX_O_F32 : VOPCX_F32 <0x00000017, "V_CMPX_O_F32">;
defm V_CMPX_U_F32 : VOPCX_F32 <0x00000018, "V_CMPX_U_F32">;
defm V_CMPX_NGE_F32 : VOPCX_F32 <0x00000019, "V_CMPX_NGE_F32">;
defm V_CMPX_NLG_F32 : VOPCX_F32 <0x0000001a, "V_CMPX_NLG_F32">;
defm V_CMPX_NGT_F32 : VOPCX_F32 <0x0000001b, "V_CMPX_NGT_F32">;
defm V_CMPX_NLE_F32 : VOPCX_F32 <0x0000001c, "V_CMPX_NLE_F32">;
defm V_CMPX_NEQ_F32 : VOPCX_F32 <0x0000001d, "V_CMPX_NEQ_F32">;
defm V_CMPX_NLT_F32 : VOPCX_F32 <0x0000001e, "V_CMPX_NLT_F32">;
defm V_CMPX_TRU_F32 : VOPCX_F32 <0x0000001f, "V_CMPX_TRU_F32">;

} // End hasSideEffects = 1

defm V_CMP_F_F64 : VOPC_F64 <0x00000020, "V_CMP_F_F64">;
defm V_CMP_LT_F64 : VOPC_F64 <0x00000021, "V_CMP_LT_F64", COND_OLT>;
defm V_CMP_EQ_F64 : VOPC_F64 <0x00000022, "V_CMP_EQ_F64", COND_OEQ>;
defm V_CMP_LE_F64 : VOPC_F64 <0x00000023, "V_CMP_LE_F64", COND_OLE>;
defm V_CMP_GT_F64 : VOPC_F64 <0x00000024, "V_CMP_GT_F64", COND_OGT>;
defm V_CMP_LG_F64 : VOPC_F64 <0x00000025, "V_CMP_LG_F64">;
defm V_CMP_GE_F64 : VOPC_F64 <0x00000026, "V_CMP_GE_F64", COND_OGE>;
defm V_CMP_O_F64 : VOPC_F64 <0x00000027, "V_CMP_O_F64", COND_O>;
defm V_CMP_U_F64 : VOPC_F64 <0x00000028, "V_CMP_U_F64", COND_UO>;
defm V_CMP_NGE_F64 : VOPC_F64 <0x00000029, "V_CMP_NGE_F64">;
defm V_CMP_NLG_F64 : VOPC_F64 <0x0000002a, "V_CMP_NLG_F64">;
defm V_CMP_NGT_F64 : VOPC_F64 <0x0000002b, "V_CMP_NGT_F64">;
defm V_CMP_NLE_F64 : VOPC_F64 <0x0000002c, "V_CMP_NLE_F64">;
defm V_CMP_NEQ_F64 : VOPC_F64 <0x0000002d, "V_CMP_NEQ_F64", COND_UNE>;
defm V_CMP_NLT_F64 : VOPC_F64 <0x0000002e, "V_CMP_NLT_F64">;
defm V_CMP_TRU_F64 : VOPC_F64 <0x0000002f, "V_CMP_TRU_F64">;

let hasSideEffects = 1 in {

defm V_CMPX_F_F64 : VOPCX_F64 <0x00000030, "V_CMPX_F_F64">;
defm V_CMPX_LT_F64 : VOPCX_F64 <0x00000031, "V_CMPX_LT_F64">;
defm V_CMPX_EQ_F64 : VOPCX_F64 <0x00000032, "V_CMPX_EQ_F64">;
defm V_CMPX_LE_F64 : VOPCX_F64 <0x00000033, "V_CMPX_LE_F64">;
defm V_CMPX_GT_F64 : VOPCX_F64 <0x00000034, "V_CMPX_GT_F64">;
defm V_CMPX_LG_F64 : VOPCX_F64 <0x00000035, "V_CMPX_LG_F64">;
defm V_CMPX_GE_F64 : VOPCX_F64 <0x00000036, "V_CMPX_GE_F64">;
defm V_CMPX_O_F64 : VOPCX_F64 <0x00000037, "V_CMPX_O_F64">;
defm V_CMPX_U_F64 : VOPCX_F64 <0x00000038, "V_CMPX_U_F64">;
defm V_CMPX_NGE_F64 : VOPCX_F64 <0x00000039, "V_CMPX_NGE_F64">;
defm V_CMPX_NLG_F64 : VOPCX_F64 <0x0000003a, "V_CMPX_NLG_F64">;
defm V_CMPX_NGT_F64 : VOPCX_F64 <0x0000003b, "V_CMPX_NGT_F64">;
defm V_CMPX_NLE_F64 : VOPCX_F64 <0x0000003c, "V_CMPX_NLE_F64">;
defm V_CMPX_NEQ_F64 : VOPCX_F64 <0x0000003d, "V_CMPX_NEQ_F64">;
defm V_CMPX_NLT_F64 : VOPCX_F64 <0x0000003e, "V_CMPX_NLT_F64">;
defm V_CMPX_TRU_F64 : VOPCX_F64 <0x0000003f, "V_CMPX_TRU_F64">;

} // End hasSideEffects = 1

defm V_CMPS_F_F32 : VOPC_F32 <0x00000040, "V_CMPS_F_F32">;
defm V_CMPS_LT_F32 : VOPC_F32 <0x00000041, "V_CMPS_LT_F32">;
defm V_CMPS_EQ_F32 : VOPC_F32 <0x00000042, "V_CMPS_EQ_F32">;
defm V_CMPS_LE_F32 : VOPC_F32 <0x00000043, "V_CMPS_LE_F32">;
defm V_CMPS_GT_F32 : VOPC_F32 <0x00000044, "V_CMPS_GT_F32">;
defm V_CMPS_LG_F32 : VOPC_F32 <0x00000045, "V_CMPS_LG_F32">;
defm V_CMPS_GE_F32 : VOPC_F32 <0x00000046, "V_CMPS_GE_F32">;
defm V_CMPS_O_F32 : VOPC_F32 <0x00000047, "V_CMPS_O_F32">;
defm V_CMPS_U_F32 : VOPC_F32 <0x00000048, "V_CMPS_U_F32">;
defm V_CMPS_NGE_F32 : VOPC_F32 <0x00000049, "V_CMPS_NGE_F32">;
defm V_CMPS_NLG_F32 : VOPC_F32 <0x0000004a, "V_CMPS_NLG_F32">;
defm V_CMPS_NGT_F32 : VOPC_F32 <0x0000004b, "V_CMPS_NGT_F32">;
defm V_CMPS_NLE_F32 : VOPC_F32 <0x0000004c, "V_CMPS_NLE_F32">;
defm V_CMPS_NEQ_F32 : VOPC_F32 <0x0000004d, "V_CMPS_NEQ_F32">;
defm V_CMPS_NLT_F32 : VOPC_F32 <0x0000004e, "V_CMPS_NLT_F32">;
defm V_CMPS_TRU_F32 : VOPC_F32 <0x0000004f, "V_CMPS_TRU_F32">;

let hasSideEffects = 1 in {

defm V_CMPSX_F_F32 : VOPCX_F32 <0x00000050, "V_CMPSX_F_F32">;
defm V_CMPSX_LT_F32 : VOPCX_F32 <0x00000051, "V_CMPSX_LT_F32">;
defm V_CMPSX_EQ_F32 : VOPCX_F32 <0x00000052, "V_CMPSX_EQ_F32">;
defm V_CMPSX_LE_F32 : VOPCX_F32 <0x00000053, "V_CMPSX_LE_F32">;
defm V_CMPSX_GT_F32 : VOPCX_F32 <0x00000054, "V_CMPSX_GT_F32">;
defm V_CMPSX_LG_F32 : VOPCX_F32 <0x00000055, "V_CMPSX_LG_F32">;
defm V_CMPSX_GE_F32 : VOPCX_F32 <0x00000056, "V_CMPSX_GE_F32">;
defm V_CMPSX_O_F32 : VOPCX_F32 <0x00000057, "V_CMPSX_O_F32">;
defm V_CMPSX_U_F32 : VOPCX_F32 <0x00000058, "V_CMPSX_U_F32">;
defm V_CMPSX_NGE_F32 : VOPCX_F32 <0x00000059, "V_CMPSX_NGE_F32">;
defm V_CMPSX_NLG_F32 : VOPCX_F32 <0x0000005a, "V_CMPSX_NLG_F32">;
defm V_CMPSX_NGT_F32 : VOPCX_F32 <0x0000005b, "V_CMPSX_NGT_F32">;
defm V_CMPSX_NLE_F32 : VOPCX_F32 <0x0000005c, "V_CMPSX_NLE_F32">;
defm V_CMPSX_NEQ_F32 : VOPCX_F32 <0x0000005d, "V_CMPSX_NEQ_F32">;
defm V_CMPSX_NLT_F32 : VOPCX_F32 <0x0000005e, "V_CMPSX_NLT_F32">;
defm V_CMPSX_TRU_F32 : VOPCX_F32 <0x0000005f, "V_CMPSX_TRU_F32">;

} // End hasSideEffects = 1

defm V_CMPS_F_F64 : VOPC_F64 <0x00000060, "V_CMPS_F_F64">;
defm V_CMPS_LT_F64 : VOPC_F64 <0x00000061, "V_CMPS_LT_F64">;
defm V_CMPS_EQ_F64 : VOPC_F64 <0x00000062, "V_CMPS_EQ_F64">;
defm V_CMPS_LE_F64 : VOPC_F64 <0x00000063, "V_CMPS_LE_F64">;
defm V_CMPS_GT_F64 : VOPC_F64 <0x00000064, "V_CMPS_GT_F64">;
defm V_CMPS_LG_F64 : VOPC_F64 <0x00000065, "V_CMPS_LG_F64">;
defm V_CMPS_GE_F64 : VOPC_F64 <0x00000066, "V_CMPS_GE_F64">;
defm V_CMPS_O_F64 : VOPC_F64 <0x00000067, "V_CMPS_O_F64">;
defm V_CMPS_U_F64 : VOPC_F64 <0x00000068, "V_CMPS_U_F64">;
defm V_CMPS_NGE_F64 : VOPC_F64 <0x00000069, "V_CMPS_NGE_F64">;
defm V_CMPS_NLG_F64 : VOPC_F64 <0x0000006a, "V_CMPS_NLG_F64">;
defm V_CMPS_NGT_F64 : VOPC_F64 <0x0000006b, "V_CMPS_NGT_F64">;
defm V_CMPS_NLE_F64 : VOPC_F64 <0x0000006c, "V_CMPS_NLE_F64">;
defm V_CMPS_NEQ_F64 : VOPC_F64 <0x0000006d, "V_CMPS_NEQ_F64">;
defm V_CMPS_NLT_F64 : VOPC_F64 <0x0000006e, "V_CMPS_NLT_F64">;
defm V_CMPS_TRU_F64 : VOPC_F64 <0x0000006f, "V_CMPS_TRU_F64">;

let hasSideEffects = 1, Defs = [EXEC] in {

defm V_CMPSX_F_F64 : VOPC_F64 <0x00000070, "V_CMPSX_F_F64">;
defm V_CMPSX_LT_F64 : VOPC_F64 <0x00000071, "V_CMPSX_LT_F64">;
defm V_CMPSX_EQ_F64 : VOPC_F64 <0x00000072, "V_CMPSX_EQ_F64">;
defm V_CMPSX_LE_F64 : VOPC_F64 <0x00000073, "V_CMPSX_LE_F64">;
defm V_CMPSX_GT_F64 : VOPC_F64 <0x00000074, "V_CMPSX_GT_F64">;
defm V_CMPSX_LG_F64 : VOPC_F64 <0x00000075, "V_CMPSX_LG_F64">;
defm V_CMPSX_GE_F64 : VOPC_F64 <0x00000076, "V_CMPSX_GE_F64">;
defm V_CMPSX_O_F64 : VOPC_F64 <0x00000077, "V_CMPSX_O_F64">;
defm V_CMPSX_U_F64 : VOPC_F64 <0x00000078, "V_CMPSX_U_F64">;
defm V_CMPSX_NGE_F64 : VOPC_F64 <0x00000079, "V_CMPSX_NGE_F64">;
defm V_CMPSX_NLG_F64 : VOPC_F64 <0x0000007a, "V_CMPSX_NLG_F64">;
defm V_CMPSX_NGT_F64 : VOPC_F64 <0x0000007b, "V_CMPSX_NGT_F64">;
defm V_CMPSX_NLE_F64 : VOPC_F64 <0x0000007c, "V_CMPSX_NLE_F64">;
defm V_CMPSX_NEQ_F64 : VOPC_F64 <0x0000007d, "V_CMPSX_NEQ_F64">;
defm V_CMPSX_NLT_F64 : VOPC_F64 <0x0000007e, "V_CMPSX_NLT_F64">;
defm V_CMPSX_TRU_F64 : VOPC_F64 <0x0000007f, "V_CMPSX_TRU_F64">;

} // End hasSideEffects = 1, Defs = [EXEC]

defm V_CMP_F_I32 : VOPC_I32 <0x00000080, "V_CMP_F_I32">;
defm V_CMP_LT_I32 : VOPC_I32 <0x00000081, "V_CMP_LT_I32", COND_SLT>;
defm V_CMP_EQ_I32 : VOPC_I32 <0x00000082, "V_CMP_EQ_I32", COND_EQ>;
defm V_CMP_LE_I32 : VOPC_I32 <0x00000083, "V_CMP_LE_I32", COND_SLE>;
defm V_CMP_GT_I32 : VOPC_I32 <0x00000084, "V_CMP_GT_I32", COND_SGT>;
defm V_CMP_NE_I32 : VOPC_I32 <0x00000085, "V_CMP_NE_I32", COND_NE>;
defm V_CMP_GE_I32 : VOPC_I32 <0x00000086, "V_CMP_GE_I32", COND_SGE>;
defm V_CMP_T_I32 : VOPC_I32 <0x00000087, "V_CMP_T_I32">;

let hasSideEffects = 1 in {

defm V_CMPX_F_I32 : VOPCX_I32 <0x00000090, "V_CMPX_F_I32">;
defm V_CMPX_LT_I32 : VOPCX_I32 <0x00000091, "V_CMPX_LT_I32">;
defm V_CMPX_EQ_I32 : VOPCX_I32 <0x00000092, "V_CMPX_EQ_I32">;
defm V_CMPX_LE_I32 : VOPCX_I32 <0x00000093, "V_CMPX_LE_I32">;
defm V_CMPX_GT_I32 : VOPCX_I32 <0x00000094, "V_CMPX_GT_I32">;
defm V_CMPX_NE_I32 : VOPCX_I32 <0x00000095, "V_CMPX_NE_I32">;
defm V_CMPX_GE_I32 : VOPCX_I32 <0x00000096, "V_CMPX_GE_I32">;
defm V_CMPX_T_I32 : VOPCX_I32 <0x00000097, "V_CMPX_T_I32">;

} // End hasSideEffects = 1

defm V_CMP_F_I64 : VOPC_I64 <0x000000a0, "V_CMP_F_I64">;
defm V_CMP_LT_I64 : VOPC_I64 <0x000000a1, "V_CMP_LT_I64", COND_SLT>;
defm V_CMP_EQ_I64 : VOPC_I64 <0x000000a2, "V_CMP_EQ_I64", COND_EQ>;
defm V_CMP_LE_I64 : VOPC_I64 <0x000000a3, "V_CMP_LE_I64", COND_SLE>;
defm V_CMP_GT_I64 : VOPC_I64 <0x000000a4, "V_CMP_GT_I64", COND_SGT>;
defm V_CMP_NE_I64 : VOPC_I64 <0x000000a5, "V_CMP_NE_I64", COND_NE>;
defm V_CMP_GE_I64 : VOPC_I64 <0x000000a6, "V_CMP_GE_I64", COND_SGE>;
defm V_CMP_T_I64 : VOPC_I64 <0x000000a7, "V_CMP_T_I64">;

let hasSideEffects = 1 in {

defm V_CMPX_F_I64 : VOPCX_I64 <0x000000b0, "V_CMPX_F_I64">;
defm V_CMPX_LT_I64 : VOPCX_I64 <0x000000b1, "V_CMPX_LT_I64">;
defm V_CMPX_EQ_I64 : VOPCX_I64 <0x000000b2, "V_CMPX_EQ_I64">;
defm V_CMPX_LE_I64 : VOPCX_I64 <0x000000b3, "V_CMPX_LE_I64">;
defm V_CMPX_GT_I64 : VOPCX_I64 <0x000000b4, "V_CMPX_GT_I64">;
defm V_CMPX_NE_I64 : VOPCX_I64 <0x000000b5, "V_CMPX_NE_I64">;
defm V_CMPX_GE_I64 : VOPCX_I64 <0x000000b6, "V_CMPX_GE_I64">;
defm V_CMPX_T_I64 : VOPCX_I64 <0x000000b7, "V_CMPX_T_I64">;

} // End hasSideEffects = 1

defm V_CMP_F_U32 : VOPC_I32 <0x000000c0, "V_CMP_F_U32">;
defm V_CMP_LT_U32 : VOPC_I32 <0x000000c1, "V_CMP_LT_U32", COND_ULT>;
defm V_CMP_EQ_U32 : VOPC_I32 <0x000000c2, "V_CMP_EQ_U32", COND_EQ>;
defm V_CMP_LE_U32 : VOPC_I32 <0x000000c3, "V_CMP_LE_U32", COND_ULE>;
defm V_CMP_GT_U32 : VOPC_I32 <0x000000c4, "V_CMP_GT_U32", COND_UGT>;
defm V_CMP_NE_U32 : VOPC_I32 <0x000000c5, "V_CMP_NE_U32", COND_NE>;
defm V_CMP_GE_U32 : VOPC_I32 <0x000000c6, "V_CMP_GE_U32", COND_UGE>;
defm V_CMP_T_U32 : VOPC_I32 <0x000000c7, "V_CMP_T_U32">;

let hasSideEffects = 1 in {

defm V_CMPX_F_U32 : VOPCX_I32 <0x000000d0, "V_CMPX_F_U32">;
defm V_CMPX_LT_U32 : VOPCX_I32 <0x000000d1, "V_CMPX_LT_U32">;
defm V_CMPX_EQ_U32 : VOPCX_I32 <0x000000d2, "V_CMPX_EQ_U32">;
defm V_CMPX_LE_U32 : VOPCX_I32 <0x000000d3, "V_CMPX_LE_U32">;
defm V_CMPX_GT_U32 : VOPCX_I32 <0x000000d4, "V_CMPX_GT_U32">;
defm V_CMPX_NE_U32 : VOPCX_I32 <0x000000d5, "V_CMPX_NE_U32">;
defm V_CMPX_GE_U32 : VOPCX_I32 <0x000000d6, "V_CMPX_GE_U32">;
defm V_CMPX_T_U32 : VOPCX_I32 <0x000000d7, "V_CMPX_T_U32">;

} // End hasSideEffects = 1

defm V_CMP_F_U64 : VOPC_I64 <0x000000e0, "V_CMP_F_U64">;
defm V_CMP_LT_U64 : VOPC_I64 <0x000000e1, "V_CMP_LT_U64", COND_ULT>;
defm V_CMP_EQ_U64 : VOPC_I64 <0x000000e2, "V_CMP_EQ_U64", COND_EQ>;
defm V_CMP_LE_U64 : VOPC_I64 <0x000000e3, "V_CMP_LE_U64", COND_ULE>;
defm V_CMP_GT_U64 : VOPC_I64 <0x000000e4, "V_CMP_GT_U64", COND_UGT>;
defm V_CMP_NE_U64 : VOPC_I64 <0x000000e5, "V_CMP_NE_U64", COND_NE>;
defm V_CMP_GE_U64 : VOPC_I64 <0x000000e6, "V_CMP_GE_U64", COND_UGE>;
defm V_CMP_T_U64 : VOPC_I64 <0x000000e7, "V_CMP_T_U64">;

let hasSideEffects = 1 in {

defm V_CMPX_F_U64 : VOPCX_I64 <0x000000f0, "V_CMPX_F_U64">;
defm V_CMPX_LT_U64 : VOPCX_I64 <0x000000f1, "V_CMPX_LT_U64">;
defm V_CMPX_EQ_U64 : VOPCX_I64 <0x000000f2, "V_CMPX_EQ_U64">;
defm V_CMPX_LE_U64 : VOPCX_I64 <0x000000f3, "V_CMPX_LE_U64">;
defm V_CMPX_GT_U64 : VOPCX_I64 <0x000000f4, "V_CMPX_GT_U64">;
defm V_CMPX_NE_U64 : VOPCX_I64 <0x000000f5, "V_CMPX_NE_U64">;
defm V_CMPX_GE_U64 : VOPCX_I64 <0x000000f6, "V_CMPX_GE_U64">;
defm V_CMPX_T_U64 : VOPCX_I64 <0x000000f7, "V_CMPX_T_U64">;

} // End hasSideEffects = 1

defm V_CMP_CLASS_F32 : VOPC_F32 <0x00000088, "V_CMP_CLASS_F32">;

let hasSideEffects = 1 in {
defm V_CMPX_CLASS_F32 : VOPCX_F32 <0x00000098, "V_CMPX_CLASS_F32">;
} // End hasSideEffects = 1

defm V_CMP_CLASS_F64 : VOPC_F64 <0x000000a8, "V_CMP_CLASS_F64">;

let hasSideEffects = 1 in {
defm V_CMPX_CLASS_F64 : VOPCX_F64 <0x000000b8, "V_CMPX_CLASS_F64">;
} // End hasSideEffects = 1

} // End isCompare = 1

//===----------------------------------------------------------------------===//
// DS Instructions
//===----------------------------------------------------------------------===//


def DS_ADD_U32 : DS_1A1D_NORET <0x0, "DS_ADD_U32", VReg_32>;
def DS_SUB_U32 : DS_1A1D_NORET <0x1, "DS_SUB_U32", VReg_32>;
def DS_RSUB_U32 : DS_1A1D_NORET <0x2, "DS_RSUB_U32", VReg_32>;
def DS_INC_U32 : DS_1A1D_NORET <0x3, "DS_INC_U32", VReg_32>;
def DS_DEC_U32 : DS_1A1D_NORET <0x4, "DS_DEC_U32", VReg_32>;
def DS_MIN_I32 : DS_1A1D_NORET <0x5, "DS_MIN_I32", VReg_32>;
def DS_MAX_I32 : DS_1A1D_NORET <0x6, "DS_MAX_I32", VReg_32>;
def DS_MIN_U32 : DS_1A1D_NORET <0x7, "DS_MIN_U32", VReg_32>;
def DS_MAX_U32 : DS_1A1D_NORET <0x8, "DS_MAX_U32", VReg_32>;
def DS_AND_B32 : DS_1A1D_NORET <0x9, "DS_AND_B32", VReg_32>;
def DS_OR_B32 : DS_1A1D_NORET <0xa, "DS_OR_B32", VReg_32>;
def DS_XOR_B32 : DS_1A1D_NORET <0xb, "DS_XOR_B32", VReg_32>;
def DS_MSKOR_B32 : DS_1A1D_NORET <0xc, "DS_MSKOR_B32", VReg_32>;
def DS_CMPST_B32 : DS_1A2D_NORET <0x10, "DS_CMPST_B32", VReg_32>;
def DS_CMPST_F32 : DS_1A2D_NORET <0x11, "DS_CMPST_F32", VReg_32>;
def DS_MIN_F32 : DS_1A1D_NORET <0x12, "DS_MIN_F32", VReg_32>;
def DS_MAX_F32 : DS_1A1D_NORET <0x13, "DS_MAX_F32", VReg_32>;

def DS_ADD_RTN_U32 : DS_1A1D_RET <0x20, "DS_ADD_RTN_U32", VReg_32>;
def DS_SUB_RTN_U32 : DS_1A1D_RET <0x21, "DS_SUB_RTN_U32", VReg_32>;
def DS_RSUB_RTN_U32 : DS_1A1D_RET <0x22, "DS_RSUB_RTN_U32", VReg_32>;
def DS_INC_RTN_U32 : DS_1A1D_RET <0x23, "DS_INC_RTN_U32", VReg_32>;
def DS_DEC_RTN_U32 : DS_1A1D_RET <0x24, "DS_DEC_RTN_U32", VReg_32>;
def DS_MIN_RTN_I32 : DS_1A1D_RET <0x25, "DS_MIN_RTN_I32", VReg_32>;
def DS_MAX_RTN_I32 : DS_1A1D_RET <0x26, "DS_MAX_RTN_I32", VReg_32>;
def DS_MIN_RTN_U32 : DS_1A1D_RET <0x27, "DS_MIN_RTN_U32", VReg_32>;
def DS_MAX_RTN_U32 : DS_1A1D_RET <0x28, "DS_MAX_RTN_U32", VReg_32>;
def DS_AND_RTN_B32 : DS_1A1D_RET <0x29, "DS_AND_RTN_B32", VReg_32>;
def DS_OR_RTN_B32 : DS_1A1D_RET <0x2a, "DS_OR_RTN_B32", VReg_32>;
def DS_XOR_RTN_B32 : DS_1A1D_RET <0x2b, "DS_XOR_RTN_B32", VReg_32>;
def DS_MSKOR_RTN_B32 : DS_1A1D_RET <0x2c, "DS_MSKOR_RTN_B32", VReg_32>;
def DS_WRXCHG_RTN_B32 : DS_1A1D_RET <0x2d, "DS_WRXCHG_RTN_B32", VReg_32>;
//def DS_WRXCHG2_RTN_B32 : DS_2A0D_RET <0x2e, "DS_WRXCHG2_RTN_B32", VReg_32>;
//def DS_WRXCHG2ST64_RTN_B32 : DS_2A0D_RET <0x2f, "DS_WRXCHG2_RTN_B32", VReg_32>;
def DS_CMPST_RTN_B32 : DS_1A2D_RET <0x30, "DS_CMPST_RTN_B32", VReg_32>;
def DS_CMPST_RTN_F32 : DS_1A2D_RET <0x31, "DS_CMPST_RTN_F32", VReg_32>;
def DS_MIN_RTN_F32 : DS_1A1D_RET <0x32, "DS_MIN_RTN_F32", VReg_32>;
def DS_MAX_RTN_F32 : DS_1A1D_RET <0x33, "DS_MAX_RTN_F32", VReg_32>;

let SubtargetPredicate = isCI in {
def DS_WRAP_RTN_F32 : DS_1A1D_RET <0x34, "DS_WRAP_RTN_F32", VReg_32>;
} // End isCI


def DS_ADD_U64 : DS_1A1D_NORET <0x40, "DS_ADD_U64", VReg_32>;
def DS_SUB_U64 : DS_1A1D_NORET <0x41, "DS_SUB_U64", VReg_32>;
def DS_RSUB_U64 : DS_1A1D_NORET <0x42, "DS_RSUB_U64", VReg_32>;
def DS_INC_U64 : DS_1A1D_NORET <0x43, "DS_INC_U64", VReg_32>;
def DS_DEC_U64 : DS_1A1D_NORET <0x44, "DS_DEC_U64", VReg_32>;
def DS_MIN_I64 : DS_1A1D_NORET <0x45, "DS_MIN_I64", VReg_64>;
def DS_MAX_I64 : DS_1A1D_NORET <0x46, "DS_MAX_I64", VReg_64>;
def DS_MIN_U64 : DS_1A1D_NORET <0x47, "DS_MIN_U64", VReg_64>;
def DS_MAX_U64 : DS_1A1D_NORET <0x48, "DS_MAX_U64", VReg_64>;
def DS_AND_B64 : DS_1A1D_NORET <0x49, "DS_AND_B64", VReg_64>;
def DS_OR_B64 : DS_1A1D_NORET <0x4a, "DS_OR_B64", VReg_64>;
def DS_XOR_B64 : DS_1A1D_NORET <0x4b, "DS_XOR_B64", VReg_64>;
def DS_MSKOR_B64 : DS_1A1D_NORET <0x4c, "DS_MSKOR_B64", VReg_64>;
def DS_CMPST_B64 : DS_1A2D_NORET <0x50, "DS_CMPST_B64", VReg_64>;
def DS_CMPST_F64 : DS_1A2D_NORET <0x51, "DS_CMPST_F64", VReg_64>;
def DS_MIN_F64 : DS_1A1D_NORET <0x52, "DS_MIN_F64", VReg_64>;
def DS_MAX_F64 : DS_1A1D_NORET <0x53, "DS_MAX_F64", VReg_64>;

def DS_ADD_RTN_U64 : DS_1A1D_RET <0x60, "DS_ADD_RTN_U64", VReg_64>;
def DS_SUB_RTN_U64 : DS_1A1D_RET <0x61, "DS_SUB_RTN_U64", VReg_64>;
def DS_RSUB_RTN_U64 : DS_1A1D_RET <0x62, "DS_RSUB_RTN_U64", VReg_64>;
def DS_INC_RTN_U64 : DS_1A1D_RET <0x63, "DS_INC_RTN_U64", VReg_64>;
def DS_DEC_RTN_U64 : DS_1A1D_RET <0x64, "DS_DEC_RTN_U64", VReg_64>;
def DS_MIN_RTN_I64 : DS_1A1D_RET <0x65, "DS_MIN_RTN_I64", VReg_64>;
def DS_MAX_RTN_I64 : DS_1A1D_RET <0x66, "DS_MAX_RTN_I64", VReg_64>;
def DS_MIN_RTN_U64 : DS_1A1D_RET <0x67, "DS_MIN_RTN_U64", VReg_64>;
def DS_MAX_RTN_U64 : DS_1A1D_RET <0x68, "DS_MAX_RTN_U64", VReg_64>;
def DS_AND_RTN_B64 : DS_1A1D_RET <0x69, "DS_AND_RTN_B64", VReg_64>;
def DS_OR_RTN_B64 : DS_1A1D_RET <0x6a, "DS_OR_RTN_B64", VReg_64>;
def DS_XOR_RTN_B64 : DS_1A1D_RET <0x6b, "DS_XOR_RTN_B64", VReg_64>;
def DS_MSKOR_RTN_B64 : DS_1A1D_RET <0x6c, "DS_MSKOR_RTN_B64", VReg_64>;
def DS_WRXCHG_RTN_B64 : DS_1A1D_RET <0x6d, "DS_WRXCHG_RTN_B64", VReg_64>;
//def DS_WRXCHG2_RTN_B64 : DS_2A0D_RET <0x6e, "DS_WRXCHG2_RTN_B64", VReg_64>;
//def DS_WRXCHG2ST64_RTN_B64 : DS_2A0D_RET <0x6f, "DS_WRXCHG2_RTN_B64", VReg_64>;
def DS_CMPST_RTN_B64 : DS_1A2D_RET <0x70, "DS_CMPST_RTN_B64", VReg_64>;
def DS_CMPST_RTN_F64 : DS_1A2D_RET <0x71, "DS_CMPST_RTN_F64", VReg_64>;
def DS_MIN_RTN_F64 : DS_1A1D_RET <0x72, "DS_MIN_F64", VReg_64>;
def DS_MAX_RTN_F64 : DS_1A1D_RET <0x73, "DS_MAX_F64", VReg_64>;

//let SubtargetPredicate = isCI in {
// DS_CONDXCHG32_RTN_B64
// DS_CONDXCHG32_RTN_B128
//} // End isCI

// TODO: _SRC2_* forms

def DS_WRITE_B32 : DS_Store_Helper <0x0000000d, "DS_WRITE_B32", VReg_32>;
def DS_WRITE_B8 : DS_Store_Helper <0x00000001e, "DS_WRITE_B8", VReg_32>;
def DS_WRITE_B16 : DS_Store_Helper <0x00000001f, "DS_WRITE_B16", VReg_32>;
def DS_WRITE_B64 : DS_Store_Helper <0x00000004d, "DS_WRITE_B64", VReg_64>;

def DS_READ_B32 : DS_Load_Helper <0x00000036, "DS_READ_B32", VReg_32>;
def DS_READ_I8 : DS_Load_Helper <0x00000039, "DS_READ_I8", VReg_32>;
def DS_READ_U8 : DS_Load_Helper <0x0000003a, "DS_READ_U8", VReg_32>;
def DS_READ_I16 : DS_Load_Helper <0x0000003b, "DS_READ_I16", VReg_32>;
def DS_READ_U16 : DS_Load_Helper <0x0000003c, "DS_READ_U16", VReg_32>;
def DS_READ_B64 : DS_Load_Helper <0x00000076, "DS_READ_B64", VReg_64>;

// 2 forms.
def DS_WRITE2_B32 : DS_Store2_Helper <0x0000000E, "DS_WRITE2_B32", VReg_32>;
def DS_WRITE2ST64_B32 : DS_Store2_Helper <0x0000000F, "DS_WRITE2ST64_B32", VReg_32>;
def DS_WRITE2_B64 : DS_Store2_Helper <0x0000004E, "DS_WRITE2_B64", VReg_64>;
def DS_WRITE2ST64_B64 : DS_Store2_Helper <0x0000004F, "DS_WRITE2ST64_B64", VReg_64>;

def DS_READ2_B32 : DS_Load2_Helper <0x00000037, "DS_READ2_B32", VReg_64>;
def DS_READ2ST64_B32 : DS_Load2_Helper <0x00000038, "DS_READ2ST64_B32", VReg_64>;
def DS_READ2_B64 : DS_Load2_Helper <0x00000075, "DS_READ2_B64", VReg_128>;
def DS_READ2ST64_B64 : DS_Load2_Helper <0x00000076, "DS_READ2ST64_B64", VReg_128>;

//===----------------------------------------------------------------------===//
// MUBUF Instructions
//===----------------------------------------------------------------------===//

//def BUFFER_LOAD_FORMAT_X : MUBUF_ <0x00000000, "BUFFER_LOAD_FORMAT_X", []>;
//def BUFFER_LOAD_FORMAT_XY : MUBUF_ <0x00000001, "BUFFER_LOAD_FORMAT_XY", []>;
//def BUFFER_LOAD_FORMAT_XYZ : MUBUF_ <0x00000002, "BUFFER_LOAD_FORMAT_XYZ", []>;
defm BUFFER_LOAD_FORMAT_XYZW : MUBUF_Load_Helper <0x00000003, "BUFFER_LOAD_FORMAT_XYZW", VReg_128>;
//def BUFFER_STORE_FORMAT_X : MUBUF_ <0x00000004, "BUFFER_STORE_FORMAT_X", []>;
//def BUFFER_STORE_FORMAT_XY : MUBUF_ <0x00000005, "BUFFER_STORE_FORMAT_XY", []>;
//def BUFFER_STORE_FORMAT_XYZ : MUBUF_ <0x00000006, "BUFFER_STORE_FORMAT_XYZ", []>;
//def BUFFER_STORE_FORMAT_XYZW : MUBUF_ <0x00000007, "BUFFER_STORE_FORMAT_XYZW", []>;
defm BUFFER_LOAD_UBYTE : MUBUF_Load_Helper <
  0x00000008, "BUFFER_LOAD_UBYTE", VReg_32, i32, az_extloadi8_global
>;
defm BUFFER_LOAD_SBYTE : MUBUF_Load_Helper <
  0x00000009, "BUFFER_LOAD_SBYTE", VReg_32, i32, sextloadi8_global
>;
defm BUFFER_LOAD_USHORT : MUBUF_Load_Helper <
  0x0000000a, "BUFFER_LOAD_USHORT", VReg_32, i32, az_extloadi16_global
>;
defm BUFFER_LOAD_SSHORT : MUBUF_Load_Helper <
  0x0000000b, "BUFFER_LOAD_SSHORT", VReg_32, i32, sextloadi16_global
>;
defm BUFFER_LOAD_DWORD : MUBUF_Load_Helper <
  0x0000000c, "BUFFER_LOAD_DWORD", VReg_32, i32, global_load
>;
defm BUFFER_LOAD_DWORDX2 : MUBUF_Load_Helper <
  0x0000000d, "BUFFER_LOAD_DWORDX2", VReg_64, v2i32, global_load
>;
defm BUFFER_LOAD_DWORDX4 : MUBUF_Load_Helper <
  0x0000000e, "BUFFER_LOAD_DWORDX4", VReg_128, v4i32, global_load
>;

defm BUFFER_STORE_BYTE : MUBUF_Store_Helper <
  0x00000018, "BUFFER_STORE_BYTE", VReg_32, i32, truncstorei8_global
>;

defm BUFFER_STORE_SHORT : MUBUF_Store_Helper <
  0x0000001a, "BUFFER_STORE_SHORT", VReg_32, i32, truncstorei16_global
>;

defm BUFFER_STORE_DWORD : MUBUF_Store_Helper <
  0x0000001c, "BUFFER_STORE_DWORD", VReg_32, i32, global_store
>;

defm BUFFER_STORE_DWORDX2 : MUBUF_Store_Helper <
  0x0000001d, "BUFFER_STORE_DWORDX2", VReg_64, v2i32, global_store
>;

defm BUFFER_STORE_DWORDX4 : MUBUF_Store_Helper <
  0x0000001e, "BUFFER_STORE_DWORDX4", VReg_128, v4i32, global_store
>;
//def BUFFER_ATOMIC_SWAP : MUBUF_ <0x00000030, "BUFFER_ATOMIC_SWAP", []>;
//def BUFFER_ATOMIC_CMPSWAP : MUBUF_ <0x00000031, "BUFFER_ATOMIC_CMPSWAP", []>;
//def BUFFER_ATOMIC_ADD : MUBUF_ <0x00000032, "BUFFER_ATOMIC_ADD", []>;
//def BUFFER_ATOMIC_SUB : MUBUF_ <0x00000033, "BUFFER_ATOMIC_SUB", []>;
//def BUFFER_ATOMIC_RSUB : MUBUF_ <0x00000034, "BUFFER_ATOMIC_RSUB", []>;
//def BUFFER_ATOMIC_SMIN : MUBUF_ <0x00000035, "BUFFER_ATOMIC_SMIN", []>;
//def BUFFER_ATOMIC_UMIN : MUBUF_ <0x00000036, "BUFFER_ATOMIC_UMIN", []>;
//def BUFFER_ATOMIC_SMAX : MUBUF_ <0x00000037, "BUFFER_ATOMIC_SMAX", []>;
//def BUFFER_ATOMIC_UMAX : MUBUF_ <0x00000038, "BUFFER_ATOMIC_UMAX", []>;
//def BUFFER_ATOMIC_AND : MUBUF_ <0x00000039, "BUFFER_ATOMIC_AND", []>;
//def BUFFER_ATOMIC_OR : MUBUF_ <0x0000003a, "BUFFER_ATOMIC_OR", []>;
//def BUFFER_ATOMIC_XOR : MUBUF_ <0x0000003b, "BUFFER_ATOMIC_XOR", []>;
//def BUFFER_ATOMIC_INC : MUBUF_ <0x0000003c, "BUFFER_ATOMIC_INC", []>;
//def BUFFER_ATOMIC_DEC : MUBUF_ <0x0000003d, "BUFFER_ATOMIC_DEC", []>;
//def BUFFER_ATOMIC_FCMPSWAP : MUBUF_ <0x0000003e, "BUFFER_ATOMIC_FCMPSWAP", []>;
//def BUFFER_ATOMIC_FMIN : MUBUF_ <0x0000003f, "BUFFER_ATOMIC_FMIN", []>;
//def BUFFER_ATOMIC_FMAX : MUBUF_ <0x00000040, "BUFFER_ATOMIC_FMAX", []>;
//def BUFFER_ATOMIC_SWAP_X2 : MUBUF_X2 <0x00000050, "BUFFER_ATOMIC_SWAP_X2", []>;
//def BUFFER_ATOMIC_CMPSWAP_X2 : MUBUF_X2 <0x00000051, "BUFFER_ATOMIC_CMPSWAP_X2", []>;
//def BUFFER_ATOMIC_ADD_X2 : MUBUF_X2 <0x00000052, "BUFFER_ATOMIC_ADD_X2", []>;
//def BUFFER_ATOMIC_SUB_X2 : MUBUF_X2 <0x00000053, "BUFFER_ATOMIC_SUB_X2", []>;
//def BUFFER_ATOMIC_RSUB_X2 : MUBUF_X2 <0x00000054, "BUFFER_ATOMIC_RSUB_X2", []>;
//def BUFFER_ATOMIC_SMIN_X2 : MUBUF_X2 <0x00000055, "BUFFER_ATOMIC_SMIN_X2", []>;
//def BUFFER_ATOMIC_UMIN_X2 : MUBUF_X2 <0x00000056, "BUFFER_ATOMIC_UMIN_X2", []>;
//def BUFFER_ATOMIC_SMAX_X2 : MUBUF_X2 <0x00000057, "BUFFER_ATOMIC_SMAX_X2", []>;
//def BUFFER_ATOMIC_UMAX_X2 : MUBUF_X2 <0x00000058, "BUFFER_ATOMIC_UMAX_X2", []>;
//def BUFFER_ATOMIC_AND_X2 : MUBUF_X2 <0x00000059, "BUFFER_ATOMIC_AND_X2", []>;
//def BUFFER_ATOMIC_OR_X2 : MUBUF_X2 <0x0000005a, "BUFFER_ATOMIC_OR_X2", []>;
//def BUFFER_ATOMIC_XOR_X2 : MUBUF_X2 <0x0000005b, "BUFFER_ATOMIC_XOR_X2", []>;
//def BUFFER_ATOMIC_INC_X2 : MUBUF_X2 <0x0000005c, "BUFFER_ATOMIC_INC_X2", []>;
//def BUFFER_ATOMIC_DEC_X2 : MUBUF_X2 <0x0000005d, "BUFFER_ATOMIC_DEC_X2", []>;
//def BUFFER_ATOMIC_FCMPSWAP_X2 : MUBUF_X2 <0x0000005e, "BUFFER_ATOMIC_FCMPSWAP_X2", []>;
//def BUFFER_ATOMIC_FMIN_X2 : MUBUF_X2 <0x0000005f, "BUFFER_ATOMIC_FMIN_X2", []>;
//def BUFFER_ATOMIC_FMAX_X2 : MUBUF_X2 <0x00000060, "BUFFER_ATOMIC_FMAX_X2", []>;
//def BUFFER_WBINVL1_SC : MUBUF_WBINVL1 <0x00000070, "BUFFER_WBINVL1_SC", []>;
//def BUFFER_WBINVL1 : MUBUF_WBINVL1 <0x00000071, "BUFFER_WBINVL1", []>;

//===----------------------------------------------------------------------===//
// MTBUF Instructions
//===----------------------------------------------------------------------===//

//def TBUFFER_LOAD_FORMAT_X : MTBUF_ <0x00000000, "TBUFFER_LOAD_FORMAT_X", []>;
//def TBUFFER_LOAD_FORMAT_XY : MTBUF_ <0x00000001, "TBUFFER_LOAD_FORMAT_XY", []>;
//def TBUFFER_LOAD_FORMAT_XYZ : MTBUF_ <0x00000002, "TBUFFER_LOAD_FORMAT_XYZ", []>;
def TBUFFER_LOAD_FORMAT_XYZW : MTBUF_Load_Helper <0x00000003, "TBUFFER_LOAD_FORMAT_XYZW", VReg_128>;
def TBUFFER_STORE_FORMAT_X : MTBUF_Store_Helper <0x00000004, "TBUFFER_STORE_FORMAT_X", VReg_32>;
def TBUFFER_STORE_FORMAT_XY : MTBUF_Store_Helper <0x00000005, "TBUFFER_STORE_FORMAT_XY", VReg_64>;
def TBUFFER_STORE_FORMAT_XYZ : MTBUF_Store_Helper <0x00000006, "TBUFFER_STORE_FORMAT_XYZ", VReg_128>;
def TBUFFER_STORE_FORMAT_XYZW : MTBUF_Store_Helper <0x00000007, "TBUFFER_STORE_FORMAT_XYZW", VReg_128>;

//===----------------------------------------------------------------------===//
// MIMG Instructions
//===----------------------------------------------------------------------===//

defm IMAGE_LOAD : MIMG_NoSampler <0x00000000, "IMAGE_LOAD">;
defm IMAGE_LOAD_MIP : MIMG_NoSampler <0x00000001, "IMAGE_LOAD_MIP">;
//def IMAGE_LOAD_PCK : MIMG_NoPattern_ <"IMAGE_LOAD_PCK", 0x00000002>;
//def IMAGE_LOAD_PCK_SGN : MIMG_NoPattern_ <"IMAGE_LOAD_PCK_SGN", 0x00000003>;
//def IMAGE_LOAD_MIP_PCK : MIMG_NoPattern_ <"IMAGE_LOAD_MIP_PCK", 0x00000004>;
//def IMAGE_LOAD_MIP_PCK_SGN : MIMG_NoPattern_ <"IMAGE_LOAD_MIP_PCK_SGN", 0x00000005>;
//def IMAGE_STORE : MIMG_NoPattern_ <"IMAGE_STORE", 0x00000008>;
//def IMAGE_STORE_MIP : MIMG_NoPattern_ <"IMAGE_STORE_MIP", 0x00000009>;
//def IMAGE_STORE_PCK : MIMG_NoPattern_ <"IMAGE_STORE_PCK", 0x0000000a>;
//def IMAGE_STORE_MIP_PCK : MIMG_NoPattern_ <"IMAGE_STORE_MIP_PCK", 0x0000000b>;
defm IMAGE_GET_RESINFO : MIMG_NoSampler <0x0000000e, "IMAGE_GET_RESINFO">;
//def IMAGE_ATOMIC_SWAP : MIMG_NoPattern_ <"IMAGE_ATOMIC_SWAP", 0x0000000f>;
//def IMAGE_ATOMIC_CMPSWAP : MIMG_NoPattern_ <"IMAGE_ATOMIC_CMPSWAP", 0x00000010>;
//def IMAGE_ATOMIC_ADD : MIMG_NoPattern_ <"IMAGE_ATOMIC_ADD", 0x00000011>;
//def IMAGE_ATOMIC_SUB : MIMG_NoPattern_ <"IMAGE_ATOMIC_SUB", 0x00000012>;
//def IMAGE_ATOMIC_RSUB : MIMG_NoPattern_ <"IMAGE_ATOMIC_RSUB", 0x00000013>;
//def IMAGE_ATOMIC_SMIN : MIMG_NoPattern_ <"IMAGE_ATOMIC_SMIN", 0x00000014>;
//def IMAGE_ATOMIC_UMIN : MIMG_NoPattern_ <"IMAGE_ATOMIC_UMIN", 0x00000015>;
//def IMAGE_ATOMIC_SMAX : MIMG_NoPattern_ <"IMAGE_ATOMIC_SMAX", 0x00000016>;
//def IMAGE_ATOMIC_UMAX : MIMG_NoPattern_ <"IMAGE_ATOMIC_UMAX", 0x00000017>;
//def IMAGE_ATOMIC_AND : MIMG_NoPattern_ <"IMAGE_ATOMIC_AND", 0x00000018>;
//def IMAGE_ATOMIC_OR : MIMG_NoPattern_ <"IMAGE_ATOMIC_OR", 0x00000019>;
//def IMAGE_ATOMIC_XOR : MIMG_NoPattern_ <"IMAGE_ATOMIC_XOR", 0x0000001a>;
//def IMAGE_ATOMIC_INC : MIMG_NoPattern_ <"IMAGE_ATOMIC_INC", 0x0000001b>;
//def IMAGE_ATOMIC_DEC : MIMG_NoPattern_ <"IMAGE_ATOMIC_DEC", 0x0000001c>;
//def IMAGE_ATOMIC_FCMPSWAP : MIMG_NoPattern_ <"IMAGE_ATOMIC_FCMPSWAP", 0x0000001d>;
//def IMAGE_ATOMIC_FMIN : MIMG_NoPattern_ <"IMAGE_ATOMIC_FMIN", 0x0000001e>;
//def IMAGE_ATOMIC_FMAX : MIMG_NoPattern_ <"IMAGE_ATOMIC_FMAX", 0x0000001f>;
defm IMAGE_SAMPLE           : MIMG_Sampler <0x00000020, "IMAGE_SAMPLE">;
defm IMAGE_SAMPLE_CL        : MIMG_Sampler <0x00000021, "IMAGE_SAMPLE_CL">;
defm IMAGE_SAMPLE_D         : MIMG_Sampler <0x00000022, "IMAGE_SAMPLE_D">;
defm IMAGE_SAMPLE_D_CL      : MIMG_Sampler <0x00000023, "IMAGE_SAMPLE_D_CL">;
defm IMAGE_SAMPLE_L         : MIMG_Sampler <0x00000024, "IMAGE_SAMPLE_L">;
defm IMAGE_SAMPLE_B         : MIMG_Sampler <0x00000025, "IMAGE_SAMPLE_B">;
defm IMAGE_SAMPLE_B_CL      : MIMG_Sampler <0x00000026, "IMAGE_SAMPLE_B_CL">;
defm IMAGE_SAMPLE_LZ        : MIMG_Sampler <0x00000027, "IMAGE_SAMPLE_LZ">;
defm IMAGE_SAMPLE_C         : MIMG_Sampler <0x00000028, "IMAGE_SAMPLE_C">;
defm IMAGE_SAMPLE_C_CL      : MIMG_Sampler <0x00000029, "IMAGE_SAMPLE_C_CL">;
defm IMAGE_SAMPLE_C_D       : MIMG_Sampler <0x0000002a, "IMAGE_SAMPLE_C_D">;
defm IMAGE_SAMPLE_C_D_CL    : MIMG_Sampler <0x0000002b, "IMAGE_SAMPLE_C_D_CL">;
defm IMAGE_SAMPLE_C_L       : MIMG_Sampler <0x0000002c, "IMAGE_SAMPLE_C_L">;
defm IMAGE_SAMPLE_C_B       : MIMG_Sampler <0x0000002d, "IMAGE_SAMPLE_C_B">;
defm IMAGE_SAMPLE_C_B_CL    : MIMG_Sampler <0x0000002e, "IMAGE_SAMPLE_C_B_CL">;
defm IMAGE_SAMPLE_C_LZ      : MIMG_Sampler <0x0000002f, "IMAGE_SAMPLE_C_LZ">;
defm IMAGE_SAMPLE_O         : MIMG_Sampler <0x00000030, "IMAGE_SAMPLE_O">;
defm IMAGE_SAMPLE_CL_O      : MIMG_Sampler <0x00000031, "IMAGE_SAMPLE_CL_O">;
defm IMAGE_SAMPLE_D_O       : MIMG_Sampler <0x00000032, "IMAGE_SAMPLE_D_O">;
defm IMAGE_SAMPLE_D_CL_O    : MIMG_Sampler <0x00000033, "IMAGE_SAMPLE_D_CL_O">;
defm IMAGE_SAMPLE_L_O       : MIMG_Sampler <0x00000034, "IMAGE_SAMPLE_L_O">;
defm IMAGE_SAMPLE_B_O       : MIMG_Sampler <0x00000035, "IMAGE_SAMPLE_B_O">;
defm IMAGE_SAMPLE_B_CL_O    : MIMG_Sampler <0x00000036, "IMAGE_SAMPLE_B_CL_O">;
defm IMAGE_SAMPLE_LZ_O      : MIMG_Sampler <0x00000037, "IMAGE_SAMPLE_LZ_O">;
defm IMAGE_SAMPLE_C_O       : MIMG_Sampler <0x00000038, "IMAGE_SAMPLE_C_O">;
defm IMAGE_SAMPLE_C_CL_O    : MIMG_Sampler <0x00000039, "IMAGE_SAMPLE_C_CL_O">;
defm IMAGE_SAMPLE_C_D_O     : MIMG_Sampler <0x0000003a, "IMAGE_SAMPLE_C_D_O">;
defm IMAGE_SAMPLE_C_D_CL_O  : MIMG_Sampler <0x0000003b, "IMAGE_SAMPLE_C_D_CL_O">;
defm IMAGE_SAMPLE_C_L_O     : MIMG_Sampler <0x0000003c, "IMAGE_SAMPLE_C_L_O">;
defm IMAGE_SAMPLE_C_B_O     : MIMG_Sampler <0x0000003d, "IMAGE_SAMPLE_C_B_O">;
defm IMAGE_SAMPLE_C_B_CL_O  : MIMG_Sampler <0x0000003e, "IMAGE_SAMPLE_C_B_CL_O">;
defm IMAGE_SAMPLE_C_LZ_O    : MIMG_Sampler <0x0000003f, "IMAGE_SAMPLE_C_LZ_O">;
defm IMAGE_GATHER4          : MIMG_Gather <0x00000040, "IMAGE_GATHER4">;
defm IMAGE_GATHER4_CL       : MIMG_Gather <0x00000041, "IMAGE_GATHER4_CL">;
defm IMAGE_GATHER4_L        : MIMG_Gather <0x00000044, "IMAGE_GATHER4_L">;
defm IMAGE_GATHER4_B        : MIMG_Gather <0x00000045, "IMAGE_GATHER4_B">;
defm IMAGE_GATHER4_B_CL     : MIMG_Gather <0x00000046, "IMAGE_GATHER4_B_CL">;
defm IMAGE_GATHER4_LZ       : MIMG_Gather <0x00000047, "IMAGE_GATHER4_LZ">;
defm IMAGE_GATHER4_C        : MIMG_Gather <0x00000048, "IMAGE_GATHER4_C">;
defm IMAGE_GATHER4_C_CL     : MIMG_Gather <0x00000049, "IMAGE_GATHER4_C_CL">;
defm IMAGE_GATHER4_C_L      : MIMG_Gather <0x0000004c, "IMAGE_GATHER4_C_L">;
defm IMAGE_GATHER4_C_B      : MIMG_Gather <0x0000004d, "IMAGE_GATHER4_C_B">;
defm IMAGE_GATHER4_C_B_CL   : MIMG_Gather <0x0000004e, "IMAGE_GATHER4_C_B_CL">;
defm IMAGE_GATHER4_C_LZ     : MIMG_Gather <0x0000004f, "IMAGE_GATHER4_C_LZ">;
defm IMAGE_GATHER4_O        : MIMG_Gather <0x00000050, "IMAGE_GATHER4_O">;
defm IMAGE_GATHER4_CL_O     : MIMG_Gather <0x00000051, "IMAGE_GATHER4_CL_O">;
defm IMAGE_GATHER4_L_O      : MIMG_Gather <0x00000054, "IMAGE_GATHER4_L_O">;
defm IMAGE_GATHER4_B_O      : MIMG_Gather <0x00000055, "IMAGE_GATHER4_B_O">;
defm IMAGE_GATHER4_B_CL_O   : MIMG_Gather <0x00000056, "IMAGE_GATHER4_B_CL_O">;
defm IMAGE_GATHER4_LZ_O     : MIMG_Gather <0x00000057, "IMAGE_GATHER4_LZ_O">;
defm IMAGE_GATHER4_C_O      : MIMG_Gather <0x00000058, "IMAGE_GATHER4_C_O">;
defm IMAGE_GATHER4_C_CL_O   : MIMG_Gather <0x00000059, "IMAGE_GATHER4_C_CL_O">;
defm IMAGE_GATHER4_C_L_O    : MIMG_Gather <0x0000005c, "IMAGE_GATHER4_C_L_O">;
defm IMAGE_GATHER4_C_B_O    : MIMG_Gather <0x0000005d, "IMAGE_GATHER4_C_B_O">;
defm IMAGE_GATHER4_C_B_CL_O : MIMG_Gather <0x0000005e, "IMAGE_GATHER4_C_B_CL_O">;
defm IMAGE_GATHER4_C_LZ_O   : MIMG_Gather <0x0000005f, "IMAGE_GATHER4_C_LZ_O">;
defm IMAGE_GET_LOD          : MIMG_Sampler <0x00000060, "IMAGE_GET_LOD">;
defm IMAGE_SAMPLE_CD        : MIMG_Sampler <0x00000068, "IMAGE_SAMPLE_CD">;
defm IMAGE_SAMPLE_CD_CL     : MIMG_Sampler <0x00000069, "IMAGE_SAMPLE_CD_CL">;
defm IMAGE_SAMPLE_C_CD      : MIMG_Sampler <0x0000006a, "IMAGE_SAMPLE_C_CD">;
defm IMAGE_SAMPLE_C_CD_CL   : MIMG_Sampler <0x0000006b, "IMAGE_SAMPLE_C_CD_CL">;
defm IMAGE_SAMPLE_CD_O      : MIMG_Sampler <0x0000006c, "IMAGE_SAMPLE_CD_O">;
defm IMAGE_SAMPLE_CD_CL_O   : MIMG_Sampler <0x0000006d, "IMAGE_SAMPLE_CD_CL_O">;
defm IMAGE_SAMPLE_C_CD_O    : MIMG_Sampler <0x0000006e, "IMAGE_SAMPLE_C_CD_O">;
defm IMAGE_SAMPLE_C_CD_CL_O : MIMG_Sampler <0x0000006f, "IMAGE_SAMPLE_C_CD_CL_O">;
//def IMAGE_RSRC256 : MIMG_NoPattern_RSRC256 <"IMAGE_RSRC256", 0x0000007e>;
//def IMAGE_SAMPLER : MIMG_NoPattern_ <"IMAGE_SAMPLER", 0x0000007f>;

//===----------------------------------------------------------------------===//
// VOP1 Instructions
//===----------------------------------------------------------------------===//

//def V_NOP : VOP1_ <0x00000000, "V_NOP", []>;

let isMoveImm = 1 in {
defm V_MOV_B32 : VOP1Inst <0x00000001, "V_MOV_B32", VOP_I32_I32>;
} // End isMoveImm = 1

let Uses = [EXEC] in {

def V_READFIRSTLANE_B32 : VOP1 <
  0x00000002,
  (outs SReg_32:$vdst),
  (ins VReg_32:$src0),
  "V_READFIRSTLANE_B32 $vdst, $src0",
  []
>;

}

defm V_CVT_I32_F64 : VOP1Inst <0x00000003, "V_CVT_I32_F64",
  VOP_I32_F64, fp_to_sint
>;
defm V_CVT_F64_I32 : VOP1Inst <0x00000004, "V_CVT_F64_I32",
  VOP_F64_I32, sint_to_fp
>;
defm V_CVT_F32_I32 : VOP1Inst <0x00000005, "V_CVT_F32_I32",
  VOP_F32_I32, sint_to_fp
>;
defm V_CVT_F32_U32 : VOP1Inst <0x00000006, "V_CVT_F32_U32",
  VOP_F32_I32, uint_to_fp
>;
defm V_CVT_U32_F32 : VOP1Inst <0x00000007, "V_CVT_U32_F32",
  VOP_I32_F32, fp_to_uint
>;
defm V_CVT_I32_F32 : VOP1Inst <0x00000008, "V_CVT_I32_F32",
  VOP_I32_F32, fp_to_sint
>;
defm V_MOV_FED_B32 : VOP1Inst <0x00000009, "V_MOV_FED_B32", VOP_I32_I32>;
defm V_CVT_F16_F32 : VOP1Inst <0x0000000a, "V_CVT_F16_F32",
  VOP_I32_F32, fp_to_f16
>;
defm V_CVT_F32_F16 : VOP1Inst <0x0000000b, "V_CVT_F32_F16",
  VOP_F32_I32, f16_to_fp
>;
//defm V_CVT_RPI_I32_F32 : VOP1_32 <0x0000000c, "V_CVT_RPI_I32_F32", []>;
//defm V_CVT_FLR_I32_F32 : VOP1_32 <0x0000000d, "V_CVT_FLR_I32_F32", []>;
//defm V_CVT_OFF_F32_I4 : VOP1_32 <0x0000000e, "V_CVT_OFF_F32_I4", []>;
defm V_CVT_F32_F64 : VOP1Inst <0x0000000f, "V_CVT_F32_F64",
  VOP_F32_F64, fround
>;
defm V_CVT_F64_F32 : VOP1Inst <0x00000010, "V_CVT_F64_F32",
  VOP_F64_F32, fextend
>;
defm V_CVT_F32_UBYTE0 : VOP1Inst <0x00000011, "V_CVT_F32_UBYTE0",
  VOP_F32_I32, AMDGPUcvt_f32_ubyte0
>;
defm V_CVT_F32_UBYTE1 : VOP1Inst <0x00000012, "V_CVT_F32_UBYTE1",
  VOP_F32_I32, AMDGPUcvt_f32_ubyte1
>;
defm V_CVT_F32_UBYTE2 : VOP1Inst <0x00000013, "V_CVT_F32_UBYTE2",
  VOP_F32_I32, AMDGPUcvt_f32_ubyte2
>;
defm V_CVT_F32_UBYTE3 : VOP1Inst <0x00000014, "V_CVT_F32_UBYTE3",
  VOP_F32_I32, AMDGPUcvt_f32_ubyte3
>;
defm V_CVT_U32_F64 : VOP1Inst <0x00000015, "V_CVT_U32_F64",
  VOP_I32_F64, fp_to_uint
>;
defm V_CVT_F64_U32 : VOP1Inst <0x00000016, "V_CVT_F64_U32",
  VOP_F64_I32, uint_to_fp
>;

defm V_FRACT_F32 : VOP1Inst <0x00000020, "V_FRACT_F32",
  VOP_F32_F32, AMDGPUfract
>;
defm V_TRUNC_F32 : VOP1Inst <0x00000021, "V_TRUNC_F32",
  VOP_F32_F32, ftrunc
>;
defm V_CEIL_F32 : VOP1Inst <0x00000022, "V_CEIL_F32",
  VOP_F32_F32, fceil
>;
defm V_RNDNE_F32 : VOP1Inst <0x00000023, "V_RNDNE_F32",
  VOP_F32_F32, frint
>;
defm V_FLOOR_F32 : VOP1Inst <0x00000024, "V_FLOOR_F32",
  VOP_F32_F32, ffloor
>;
defm V_EXP_F32 : VOP1Inst <0x00000025, "V_EXP_F32",
  VOP_F32_F32, fexp2
>;
defm V_LOG_CLAMP_F32 : VOP1Inst <0x00000026, "V_LOG_CLAMP_F32", VOP_F32_F32>;
defm V_LOG_F32 : VOP1Inst <0x00000027, "V_LOG_F32",
  VOP_F32_F32, flog2
>;

defm V_RCP_CLAMP_F32 : VOP1Inst <0x00000028, "V_RCP_CLAMP_F32", VOP_F32_F32>;
defm V_RCP_LEGACY_F32 : VOP1Inst <0x00000029, "V_RCP_LEGACY_F32", VOP_F32_F32>;
defm V_RCP_F32 : VOP1Inst <0x0000002a, "V_RCP_F32",
  VOP_F32_F32, AMDGPUrcp
>;
defm V_RCP_IFLAG_F32 : VOP1Inst <0x0000002b, "V_RCP_IFLAG_F32", VOP_F32_F32>;
defm V_RSQ_CLAMP_F32 : VOP1Inst <0x0000002c, "V_RSQ_CLAMP_F32",
  VOP_F32_F32, AMDGPUrsq_clamped
>;
defm V_RSQ_LEGACY_F32 : VOP1Inst <
  0x0000002d, "V_RSQ_LEGACY_F32",
  VOP_F32_F32, AMDGPUrsq_legacy
>;
defm V_RSQ_F32 : VOP1Inst <0x0000002e, "V_RSQ_F32",
  VOP_F32_F32, AMDGPUrsq
>;
defm V_RCP_F64 : VOP1Inst <0x0000002f, "V_RCP_F64",
  VOP_F64_F64, AMDGPUrcp
>;
defm V_RCP_CLAMP_F64 : VOP1Inst <0x00000030, "V_RCP_CLAMP_F64", VOP_F64_F64>;
defm V_RSQ_F64 : VOP1Inst <0x00000031, "V_RSQ_F64",
  VOP_F64_F64, AMDGPUrsq
>;
defm V_RSQ_CLAMP_F64 : VOP1Inst <0x00000032, "V_RSQ_CLAMP_F64",
  VOP_F64_F64, AMDGPUrsq_clamped
>;
defm V_SQRT_F32 : VOP1Inst <0x00000033, "V_SQRT_F32",
  VOP_F32_F32, fsqrt
>;
defm V_SQRT_F64 : VOP1Inst <0x00000034, "V_SQRT_F64",
  VOP_F64_F64, fsqrt
>;
defm V_SIN_F32 : VOP1Inst <0x00000035, "V_SIN_F32",
  VOP_F32_F32, AMDGPUsin
>;
defm V_COS_F32 : VOP1Inst <0x00000036, "V_COS_F32",
  VOP_F32_F32, AMDGPUcos
>;
defm V_NOT_B32 : VOP1Inst <0x00000037, "V_NOT_B32", VOP_I32_I32>;
defm V_BFREV_B32 : VOP1Inst <0x00000038, "V_BFREV_B32", VOP_I32_I32>;
defm V_FFBH_U32 : VOP1Inst <0x00000039, "V_FFBH_U32", VOP_I32_I32>;
defm V_FFBL_B32 : VOP1Inst <0x0000003a, "V_FFBL_B32", VOP_I32_I32>;
defm V_FFBH_I32 : VOP1Inst <0x0000003b, "V_FFBH_I32", VOP_I32_I32>;
//defm V_FREXP_EXP_I32_F64 : VOPInst <0x0000003c, "V_FREXP_EXP_I32_F64", VOP_I32_F32>;
defm V_FREXP_MANT_F64 : VOP1Inst <0x0000003d, "V_FREXP_MANT_F64", VOP_F64_F64>;
defm V_FRACT_F64 : VOP1Inst <0x0000003e, "V_FRACT_F64", VOP_F64_F64>;
//defm V_FREXP_EXP_I32_F32 : VOPInst <0x0000003f, "V_FREXP_EXP_I32_F32", VOP_I32_F32>;
defm V_FREXP_MANT_F32 : VOP1Inst <0x00000040, "V_FREXP_MANT_F32", VOP_F32_F32>;
//def V_CLREXCP : VOP1_ <0x00000041, "V_CLREXCP", []>;
defm V_MOVRELD_B32 : VOP1Inst <0x00000042, "V_MOVRELD_B32", VOP_I32_I32>;
defm V_MOVRELS_B32 : VOP1Inst <0x00000043, "V_MOVRELS_B32", VOP_I32_I32>;
defm V_MOVRELSD_B32 : VOP1Inst <0x00000044, "V_MOVRELSD_B32", VOP_I32_I32>;


//===----------------------------------------------------------------------===//
// VINTRP Instructions
//===----------------------------------------------------------------------===//

def V_INTERP_P1_F32 : VINTRP <
  0x00000000,
  (outs VReg_32:$dst),
  (ins VReg_32:$i, i32imm:$attr_chan, i32imm:$attr, M0Reg:$m0),
  "V_INTERP_P1_F32 $dst, $i, $attr_chan, $attr, [$m0]",
  []> {
  let DisableEncoding = "$m0";
}

def V_INTERP_P2_F32 : VINTRP <
  0x00000001,
  (outs VReg_32:$dst),
  (ins VReg_32:$src0, VReg_32:$j, i32imm:$attr_chan, i32imm:$attr, M0Reg:$m0),
  "V_INTERP_P2_F32 $dst, [$src0], $j, $attr_chan, $attr, [$m0]",
  []> {

  let Constraints = "$src0 = $dst";
  let DisableEncoding = "$src0,$m0";

}

def V_INTERP_MOV_F32 : VINTRP <
  0x00000002,
  (outs VReg_32:$dst),
  (ins InterpSlot:$src0, i32imm:$attr_chan, i32imm:$attr, M0Reg:$m0),
  "V_INTERP_MOV_F32 $dst, $src0, $attr_chan, $attr, [$m0]",
  []> {
  let DisableEncoding = "$m0";
}

//===----------------------------------------------------------------------===//
// VOP2 Instructions
//===----------------------------------------------------------------------===//

def V_CNDMASK_B32_e32 : VOP2 <0x00000000, (outs VReg_32:$dst),
  (ins VSrc_32:$src0, VReg_32:$src1, VCCReg:$vcc),
  "V_CNDMASK_B32_e32 $dst, $src0, $src1, [$vcc]",
  []
>{
  let DisableEncoding = "$vcc";
}

def V_CNDMASK_B32_e64 : VOP3 <0x00000100, (outs VReg_32:$dst),
  (ins VSrc_32:$src0, VSrc_32:$src1, SSrc_64:$src2,
   InstFlag:$abs, InstFlag:$clamp, InstFlag:$omod, InstFlag:$neg),
  "V_CNDMASK_B32_e64 $dst, $src0, $src1, $src2, $abs, $clamp, $omod, $neg",
  [(set i32:$dst, (select i1:$src2, i32:$src1, i32:$src0))]
> {
  let src0_modifiers = 0;
  let src1_modifiers = 0;
  let src2_modifiers = 0;
}

def V_READLANE_B32 : VOP2 <
  0x00000001,
  (outs SReg_32:$vdst),
  (ins VReg_32:$src0, SSrc_32:$vsrc1),
  "V_READLANE_B32 $vdst, $src0, $vsrc1",
  []
>;

def V_WRITELANE_B32 : VOP2 <
  0x00000002,
  (outs VReg_32:$vdst),
  (ins SReg_32:$src0, SSrc_32:$vsrc1),
  "V_WRITELANE_B32 $vdst, $src0, $vsrc1",
  []
>;

let isCommutable = 1 in {
defm V_ADD_F32 : VOP2Inst <0x00000003, "V_ADD_F32",
  VOP_F32_F32_F32, fadd
>;

defm V_SUB_F32 : VOP2Inst <0x00000004, "V_SUB_F32", VOP_F32_F32_F32, fsub>;
defm V_SUBREV_F32 : VOP2Inst <0x00000005, "V_SUBREV_F32",
  VOP_F32_F32_F32, null_frag, "V_SUB_F32"
>;
} // End isCommutable = 1

defm V_MAC_LEGACY_F32 : VOP2Inst <0x00000006, "V_MAC_LEGACY_F32",
  VOP_F32_F32_F32
>;

let isCommutable = 1 in {

defm V_MUL_LEGACY_F32 : VOP2Inst <
  0x00000007, "V_MUL_LEGACY_F32",
  VOP_F32_F32_F32, int_AMDGPU_mul
>;

defm V_MUL_F32 : VOP2Inst <0x00000008, "V_MUL_F32",
  VOP_F32_F32_F32, fmul
>;


defm V_MUL_I32_I24 : VOP2Inst <0x00000009, "V_MUL_I32_I24",
  VOP_I32_I32_I32, AMDGPUmul_i24
>;
//defm V_MUL_HI_I32_I24 : VOP2_32 <0x0000000a, "V_MUL_HI_I32_I24", []>;
defm V_MUL_U32_U24 : VOP2Inst <0x0000000b, "V_MUL_U32_U24",
  VOP_I32_I32_I32, AMDGPUmul_u24
>;
//defm V_MUL_HI_U32_U24 : VOP2_32 <0x0000000c, "V_MUL_HI_U32_U24", []>;


defm V_MIN_LEGACY_F32 : VOP2Inst <0x0000000d, "V_MIN_LEGACY_F32",
  VOP_F32_F32_F32, AMDGPUfmin
>;

defm V_MAX_LEGACY_F32 : VOP2Inst <0x0000000e, "V_MAX_LEGACY_F32",
  VOP_F32_F32_F32, AMDGPUfmax
>;

defm V_MIN_F32 : VOP2Inst <0x0000000f, "V_MIN_F32", VOP_F32_F32_F32>;
defm V_MAX_F32 : VOP2Inst <0x00000010, "V_MAX_F32", VOP_F32_F32_F32>;
defm V_MIN_I32 : VOP2Inst <0x00000011, "V_MIN_I32", VOP_I32_I32_I32, AMDGPUsmin>;
defm V_MAX_I32 : VOP2Inst <0x00000012, "V_MAX_I32", VOP_I32_I32_I32, AMDGPUsmax>;
defm V_MIN_U32 : VOP2Inst <0x00000013, "V_MIN_U32", VOP_I32_I32_I32, AMDGPUumin>;
defm V_MAX_U32 : VOP2Inst <0x00000014, "V_MAX_U32", VOP_I32_I32_I32, AMDGPUumax>;

defm V_LSHR_B32 : VOP2Inst <0x00000015, "V_LSHR_B32", VOP_I32_I32_I32, srl>;

defm V_LSHRREV_B32 : VOP2Inst <
  0x00000016, "V_LSHRREV_B32", VOP_I32_I32_I32, null_frag, "V_LSHR_B32"
>;

defm V_ASHR_I32 : VOP2Inst <0x00000017, "V_ASHR_I32",
  VOP_I32_I32_I32, sra
>;
defm V_ASHRREV_I32 : VOP2Inst <
  0x00000018, "V_ASHRREV_I32", VOP_I32_I32_I32, null_frag, "V_ASHR_I32"
>;

let hasPostISelHook = 1 in {

defm V_LSHL_B32 : VOP2Inst <0x00000019, "V_LSHL_B32", VOP_I32_I32_I32, shl>;

}
defm V_LSHLREV_B32 : VOP2Inst <
  0x0000001a, "V_LSHLREV_B32", VOP_I32_I32_I32, null_frag, "V_LSHL_B32"
>;

defm V_AND_B32 : VOP2Inst <0x0000001b, "V_AND_B32",
  VOP_I32_I32_I32, and>;
defm V_OR_B32 : VOP2Inst <0x0000001c, "V_OR_B32",
  VOP_I32_I32_I32, or
>;
defm V_XOR_B32 : VOP2Inst <0x0000001d, "V_XOR_B32",
  VOP_I32_I32_I32, xor
>;

} // End isCommutable = 1

defm V_BFM_B32 : VOP2Inst <0x0000001e, "V_BFM_B32",
  VOP_I32_I32_I32, AMDGPUbfm>;
defm V_MAC_F32 : VOP2Inst <0x0000001f, "V_MAC_F32", VOP_F32_F32_F32>;
defm V_MADMK_F32 : VOP2Inst <0x00000020, "V_MADMK_F32", VOP_F32_F32_F32>;
defm V_MADAK_F32 : VOP2Inst <0x00000021, "V_MADAK_F32", VOP_F32_F32_F32>;
defm V_BCNT_U32_B32 : VOP2Inst <0x00000022, "V_BCNT_U32_B32", VOP_I32_I32_I32>;
defm V_MBCNT_LO_U32_B32 : VOP2Inst <0x00000023, "V_MBCNT_LO_U32_B32",
  VOP_I32_I32_I32
>;
defm V_MBCNT_HI_U32_B32 : VOP2Inst <0x00000024, "V_MBCNT_HI_U32_B32",
  VOP_I32_I32_I32
>;

let isCommutable = 1, Defs = [VCC] in { // Carry-out goes to VCC
// No patterns so that the scalar instructions are always selected.
// The scalar versions will be replaced with vector when needed later.
defm V_ADD_I32 : VOP2bInst <0x00000025, "V_ADD_I32",
  VOP_I32_I32_I32, add
>;
defm V_SUB_I32 : VOP2bInst <0x00000026, "V_SUB_I32",
  VOP_I32_I32_I32, sub
>;
defm V_SUBREV_I32 : VOP2bInst <0x00000027, "V_SUBREV_I32",
  VOP_I32_I32_I32, null_frag, "V_SUB_I32"
>;

let Uses = [VCC] in { // Carry-in comes from VCC
defm V_ADDC_U32 : VOP2bInst <0x00000028, "V_ADDC_U32",
  VOP_I32_I32_I32_VCC, adde
>;
defm V_SUBB_U32 : VOP2bInst <0x00000029, "V_SUBB_U32",
  VOP_I32_I32_I32_VCC, sube
>;
defm V_SUBBREV_U32 : VOP2bInst <0x0000002a, "V_SUBBREV_U32",
  VOP_I32_I32_I32_VCC, null_frag, "V_SUBB_U32"
>;

} // End Uses = [VCC]
} // End isCommutable = 1, Defs = [VCC]

defm V_LDEXP_F32 : VOP2Inst <0x0000002b, "V_LDEXP_F32",
  VOP_F32_F32_F32
>;
////def V_CVT_PKACCUM_U8_F32 : VOP2_U8 <0x0000002c, "V_CVT_PKACCUM_U8_F32", []>;
////def V_CVT_PKNORM_I16_F32 : VOP2_I16 <0x0000002d, "V_CVT_PKNORM_I16_F32", []>;
////def V_CVT_PKNORM_U16_F32 : VOP2_U16 <0x0000002e, "V_CVT_PKNORM_U16_F32", []>;
defm V_CVT_PKRTZ_F16_F32 : VOP2Inst <0x0000002f, "V_CVT_PKRTZ_F16_F32",
 VOP_I32_F32_F32, int_SI_packf16
>;
////def V_CVT_PK_U16_U32 : VOP2_U16 <0x00000030, "V_CVT_PK_U16_U32", []>;
////def V_CVT_PK_I16_I32 : VOP2_I16 <0x00000031, "V_CVT_PK_I16_I32", []>;

//===----------------------------------------------------------------------===//
// VOP3 Instructions
//===----------------------------------------------------------------------===//

defm V_MAD_LEGACY_F32 : VOP3Inst <0x00000140, "V_MAD_LEGACY_F32",
  VOP_F32_F32_F32_F32
>;
defm V_MAD_F32 : VOP3Inst <0x00000141, "V_MAD_F32",
  VOP_F32_F32_F32_F32, fmad
>;
defm V_MAD_I32_I24 : VOP3Inst <0x00000142, "V_MAD_I32_I24",
  VOP_I32_I32_I32_I32, AMDGPUmad_i24
>;
defm V_MAD_U32_U24 : VOP3Inst <0x00000143, "V_MAD_U32_U24",
  VOP_I32_I32_I32_I32, AMDGPUmad_u24
>;

defm V_CUBEID_F32 : VOP3Inst <0x00000144, "V_CUBEID_F32",
  VOP_F32_F32_F32_F32
>;
defm V_CUBESC_F32 : VOP3Inst <0x00000145, "V_CUBESC_F32",
  VOP_F32_F32_F32_F32
>;
defm V_CUBETC_F32 : VOP3Inst <0x00000146, "V_CUBETC_F32",
  VOP_F32_F32_F32_F32
>;
defm V_CUBEMA_F32 : VOP3Inst <0x00000147, "V_CUBEMA_F32",
  VOP_F32_F32_F32_F32
>;

let neverHasSideEffects = 1, mayLoad = 0, mayStore = 0 in {
defm V_BFE_U32 : VOP3Inst <0x00000148, "V_BFE_U32",
  VOP_I32_I32_I32_I32, AMDGPUbfe_u32
>;
defm V_BFE_I32 : VOP3Inst <0x00000149, "V_BFE_I32",
  VOP_I32_I32_I32_I32, AMDGPUbfe_i32
>;
}

defm V_BFI_B32 : VOP3Inst <0x0000014a, "V_BFI_B32",
  VOP_I32_I32_I32_I32, AMDGPUbfi
>;
defm V_FMA_F32 : VOP3Inst <0x0000014b, "V_FMA_F32",
  VOP_F32_F32_F32_F32, fma
>;
defm V_FMA_F64 : VOP3Inst <0x0000014c, "V_FMA_F64",
  VOP_F64_F64_F64_F64, fma
>;
//def V_LERP_U8 : VOP3_U8 <0x0000014d, "V_LERP_U8", []>;
defm V_ALIGNBIT_B32 : VOP3Inst <0x0000014e, "V_ALIGNBIT_B32",
  VOP_I32_I32_I32_I32
>;
defm V_ALIGNBYTE_B32 : VOP3Inst <0x0000014f, "V_ALIGNBYTE_B32",
  VOP_I32_I32_I32_I32
>;
defm V_MULLIT_F32 : VOP3Inst <0x00000150, "V_MULLIT_F32",
  VOP_F32_F32_F32_F32>;
////def V_MIN3_F32 : VOP3_MIN3 <0x00000151, "V_MIN3_F32", []>;
////def V_MIN3_I32 : VOP3_MIN3 <0x00000152, "V_MIN3_I32", []>;
////def V_MIN3_U32 : VOP3_MIN3 <0x00000153, "V_MIN3_U32", []>;
////def V_MAX3_F32 : VOP3_MAX3 <0x00000154, "V_MAX3_F32", []>;
////def V_MAX3_I32 : VOP3_MAX3 <0x00000155, "V_MAX3_I32", []>;
////def V_MAX3_U32 : VOP3_MAX3 <0x00000156, "V_MAX3_U32", []>;
////def V_MED3_F32 : VOP3_MED3 <0x00000157, "V_MED3_F32", []>;
////def V_MED3_I32 : VOP3_MED3 <0x00000158, "V_MED3_I32", []>;
////def V_MED3_U32 : VOP3_MED3 <0x00000159, "V_MED3_U32", []>;
//def V_SAD_U8 : VOP3_U8 <0x0000015a, "V_SAD_U8", []>;
//def V_SAD_HI_U8 : VOP3_U8 <0x0000015b, "V_SAD_HI_U8", []>;
//def V_SAD_U16 : VOP3_U16 <0x0000015c, "V_SAD_U16", []>;
defm V_SAD_U32 : VOP3Inst <0x0000015d, "V_SAD_U32",
  VOP_I32_I32_I32_I32
>;
////def V_CVT_PK_U8_F32 : VOP3_U8 <0x0000015e, "V_CVT_PK_U8_F32", []>;
defm V_DIV_FIXUP_F32 : VOP3Inst <
  0x0000015f, "V_DIV_FIXUP_F32", VOP_F32_F32_F32_F32, AMDGPUdiv_fixup
>;
defm V_DIV_FIXUP_F64 : VOP3Inst <
  0x00000160, "V_DIV_FIXUP_F64", VOP_F64_F64_F64_F64, AMDGPUdiv_fixup
>;

defm V_LSHL_B64 : VOP3Inst <0x00000161, "V_LSHL_B64",
  VOP_I64_I64_I32, shl
>;
defm V_LSHR_B64 : VOP3Inst <0x00000162, "V_LSHR_B64",
  VOP_I64_I64_I32, srl
>;
defm V_ASHR_I64 : VOP3Inst <0x00000163, "V_ASHR_I64",
  VOP_I64_I64_I32, sra
>;

let isCommutable = 1 in {

defm V_ADD_F64 : VOP3Inst <0x00000164, "V_ADD_F64",
  VOP_F64_F64_F64, fadd
>;
defm V_MUL_F64 : VOP3Inst <0x00000165, "V_MUL_F64",
  VOP_F64_F64_F64, fmul
>;
defm V_MIN_F64 : VOP3Inst <0x00000166, "V_MIN_F64",
  VOP_F64_F64_F64
>;
defm V_MAX_F64 : VOP3Inst <0x00000167, "V_MAX_F64",
  VOP_F64_F64_F64
>;

} // isCommutable = 1

defm V_LDEXP_F64 : VOP3Inst <0x00000168, "V_LDEXP_F64",
  VOP_F32_F32_I32
>;

let isCommutable = 1 in {

defm V_MUL_LO_U32 : VOP3Inst <0x00000169, "V_MUL_LO_U32",
  VOP_I32_I32_I32
>;
defm V_MUL_HI_U32 : VOP3Inst <0x0000016a, "V_MUL_HI_U32",
  VOP_I32_I32_I32
>;
defm V_MUL_LO_I32 : VOP3Inst <0x0000016b, "V_MUL_LO_I32",
  VOP_I32_I32_I32
>;
defm V_MUL_HI_I32 : VOP3Inst <0x0000016c, "V_MUL_HI_I32",
  VOP_I32_I32_I32
>;

} // isCommutable = 1

defm V_DIV_SCALE_F32 : VOP3b_32 <0x0000016d, "V_DIV_SCALE_F32", []>;

// Double precision division pre-scale.
defm V_DIV_SCALE_F64 : VOP3b_64 <0x0000016e, "V_DIV_SCALE_F64", []>;

defm V_DIV_FMAS_F32 : VOP3Inst <0x0000016f, "V_DIV_FMAS_F32",
  VOP_F32_F32_F32_F32, AMDGPUdiv_fmas
>;
defm V_DIV_FMAS_F64 : VOP3Inst <0x00000170, "V_DIV_FMAS_F64",
  VOP_F64_F64_F64_F64, AMDGPUdiv_fmas
>;
//def V_MSAD_U8 : VOP3_U8 <0x00000171, "V_MSAD_U8", []>;
//def V_QSAD_U8 : VOP3_U8 <0x00000172, "V_QSAD_U8", []>;
//def V_MQSAD_U8 : VOP3_U8 <0x00000173, "V_MQSAD_U8", []>;
defm V_TRIG_PREOP_F64 : VOP3Inst <
  0x00000174, "V_TRIG_PREOP_F64", VOP_F64_F64_I32, AMDGPUtrig_preop
>;

//===----------------------------------------------------------------------===//
// Pseudo Instructions
//===----------------------------------------------------------------------===//

let isCodeGenOnly = 1, isPseudo = 1 in {

def V_MOV_I1 : InstSI <
  (outs VReg_1:$dst),
  (ins i1imm:$src),
  "", [(set i1:$dst, (imm:$src))]
>;

def V_AND_I1 : InstSI <
   (outs VReg_1:$dst), (ins VReg_1:$src0, VReg_1:$src1), "",
   [(set i1:$dst, (and i1:$src0, i1:$src1))]
>;

def V_OR_I1 : InstSI <
   (outs VReg_1:$dst), (ins VReg_1:$src0, VReg_1:$src1), "",
   [(set i1:$dst, (or i1:$src0, i1:$src1))]
>;

def V_XOR_I1 : InstSI <
  (outs VReg_1:$dst), (ins VReg_1:$src0, VReg_1:$src1), "",
  [(set i1:$dst, (xor i1:$src0, i1:$src1))]
>;

// SI pseudo instructions. These are used by the CFG structurizer pass
// and should be lowered to ISA instructions prior to codegen.

let mayLoad = 1, mayStore = 1, hasSideEffects = 1,
    Uses = [EXEC], Defs = [EXEC] in {

let isBranch = 1, isTerminator = 1 in {

def SI_IF: InstSI <
  (outs SReg_64:$dst),
  (ins SReg_64:$vcc, brtarget:$target),
  "",
  [(set i64:$dst, (int_SI_if i1:$vcc, bb:$target))]
>;

def SI_ELSE : InstSI <
  (outs SReg_64:$dst),
  (ins SReg_64:$src, brtarget:$target),
  "",
  [(set i64:$dst, (int_SI_else i64:$src, bb:$target))]
> {
  let Constraints = "$src = $dst";
}

def SI_LOOP : InstSI <
  (outs),
  (ins SReg_64:$saved, brtarget:$target),
  "SI_LOOP $saved, $target",
  [(int_SI_loop i64:$saved, bb:$target)]
>;

} // end isBranch = 1, isTerminator = 1

def SI_BREAK : InstSI <
  (outs SReg_64:$dst),
  (ins SReg_64:$src),
  "SI_ELSE $dst, $src",
  [(set i64:$dst, (int_SI_break i64:$src))]
>;

def SI_IF_BREAK : InstSI <
  (outs SReg_64:$dst),
  (ins SReg_64:$vcc, SReg_64:$src),
  "SI_IF_BREAK $dst, $vcc, $src",
  [(set i64:$dst, (int_SI_if_break i1:$vcc, i64:$src))]
>;

def SI_ELSE_BREAK : InstSI <
  (outs SReg_64:$dst),
  (ins SReg_64:$src0, SReg_64:$src1),
  "SI_ELSE_BREAK $dst, $src0, $src1",
  [(set i64:$dst, (int_SI_else_break i64:$src0, i64:$src1))]
>;

def SI_END_CF : InstSI <
  (outs),
  (ins SReg_64:$saved),
  "SI_END_CF $saved",
  [(int_SI_end_cf i64:$saved)]
>;

def SI_KILL : InstSI <
  (outs),
  (ins VSrc_32:$src),
  "SI_KILL $src",
  [(int_AMDGPU_kill f32:$src)]
>;

} // end mayLoad = 1, mayStore = 1, hasSideEffects = 1
  // Uses = [EXEC], Defs = [EXEC]

let Uses = [EXEC], Defs = [EXEC,VCC,M0] in {

//defm SI_ : RegisterLoadStore <VReg_32, FRAMEri, ADDRIndirect>;

let UseNamedOperandTable = 1 in {

def SI_RegisterLoad : InstSI <
  (outs VReg_32:$dst, SReg_64:$temp),
  (ins FRAMEri32:$addr, i32imm:$chan),
  "", []
> {
  let isRegisterLoad = 1;
  let mayLoad = 1;
}

class SIRegStore<dag outs> : InstSI <
  outs,
  (ins VReg_32:$val, FRAMEri32:$addr, i32imm:$chan),
  "", []
> {
  let isRegisterStore = 1;
  let mayStore = 1;
}

let usesCustomInserter = 1 in {
def SI_RegisterStorePseudo : SIRegStore<(outs)>;
} // End usesCustomInserter = 1
def SI_RegisterStore : SIRegStore<(outs SReg_64:$temp)>;


} // End UseNamedOperandTable = 1

def SI_INDIRECT_SRC : InstSI <
  (outs VReg_32:$dst, SReg_64:$temp),
  (ins unknown:$src, VSrc_32:$idx, i32imm:$off),
  "SI_INDIRECT_SRC $dst, $temp, $src, $idx, $off",
  []
>;

class SI_INDIRECT_DST<RegisterClass rc> : InstSI <
  (outs rc:$dst, SReg_64:$temp),
  (ins unknown:$src, VSrc_32:$idx, i32imm:$off, VReg_32:$val),
  "SI_INDIRECT_DST $dst, $temp, $src, $idx, $off, $val",
  []
> {
  let Constraints = "$src = $dst";
}

def SI_INDIRECT_DST_V1 : SI_INDIRECT_DST<VReg_32>;
def SI_INDIRECT_DST_V2 : SI_INDIRECT_DST<VReg_64>;
def SI_INDIRECT_DST_V4 : SI_INDIRECT_DST<VReg_128>;
def SI_INDIRECT_DST_V8 : SI_INDIRECT_DST<VReg_256>;
def SI_INDIRECT_DST_V16 : SI_INDIRECT_DST<VReg_512>;

} // Uses = [EXEC,VCC,M0], Defs = [EXEC,VCC,M0]

let usesCustomInserter = 1 in {

// This pseudo instruction takes a pointer as input and outputs a resource
// constant that can be used with the ADDR64 MUBUF instructions.
def SI_ADDR64_RSRC : InstSI <
  (outs SReg_128:$srsrc),
  (ins SSrc_64:$ptr),
  "", []
>;

def SI_BUFFER_RSRC : InstSI <
  (outs SReg_128:$srsrc),
  (ins SReg_32:$ptr_lo, SReg_32:$ptr_hi, SSrc_32:$data_lo, SSrc_32:$data_hi),
  "", []
>;

def V_SUB_F64 : InstSI <
  (outs VReg_64:$dst),
  (ins VReg_64:$src0, VReg_64:$src1),
  "V_SUB_F64 $dst, $src0, $src1",
  [(set f64:$dst, (fsub f64:$src0, f64:$src1))]
>;

} // end usesCustomInserter

multiclass SI_SPILL_SGPR <RegisterClass sgpr_class> {

  def _SAVE : InstSI <
    (outs VReg_32:$dst),
    (ins sgpr_class:$src, i32imm:$frame_idx),
    "", []
  >;

  def _RESTORE : InstSI <
    (outs sgpr_class:$dst),
    (ins VReg_32:$src, i32imm:$frame_idx),
    "", []
  >;

}

defm SI_SPILL_S32  : SI_SPILL_SGPR <SReg_32>;
defm SI_SPILL_S64  : SI_SPILL_SGPR <SReg_64>;
defm SI_SPILL_S128 : SI_SPILL_SGPR <SReg_128>;
defm SI_SPILL_S256 : SI_SPILL_SGPR <SReg_256>;
defm SI_SPILL_S512 : SI_SPILL_SGPR <SReg_512>;

let Defs = [SCC] in {

def SI_CONSTDATA_PTR : InstSI <
  (outs SReg_64:$dst),
  (ins),
  "", [(set SReg_64:$dst, (i64 SIconstdata_ptr))]
>;

} // End Defs = [SCC]

} // end IsCodeGenOnly, isPseudo

} // end SubtargetPredicate = SI

let Predicates = [isSI] in {

def : Pat<
  (int_AMDGPU_cndlt f32:$src0, f32:$src1, f32:$src2),
  (V_CNDMASK_B32_e64 $src2, $src1,
                     (V_CMP_GT_F32_e64 SRCMODS.NONE, 0, SRCMODS.NONE, $src0,
                                       DSTCLAMP.NONE, DSTOMOD.NONE))
>;

def : Pat <
  (int_AMDGPU_kilp),
  (SI_KILL 0xbf800000)
>;

/* int_SI_vs_load_input */
def : Pat<
  (SIload_input v4i32:$tlst, imm:$attr_offset, i32:$buf_idx_vgpr),
  (BUFFER_LOAD_FORMAT_XYZW_IDXEN $tlst, $buf_idx_vgpr, imm:$attr_offset, 0, 0, 0, 0)
>;

/* int_SI_export */
def : Pat <
  (int_SI_export imm:$en, imm:$vm, imm:$done, imm:$tgt, imm:$compr,
                 f32:$src0, f32:$src1, f32:$src2, f32:$src3),
  (EXP imm:$en, imm:$tgt, imm:$compr, imm:$done, imm:$vm,
       $src0, $src1, $src2, $src3)
>;

//===----------------------------------------------------------------------===//
// SMRD Patterns
//===----------------------------------------------------------------------===//

multiclass SMRD_Pattern <SMRD Instr_IMM, SMRD Instr_SGPR, ValueType vt> {

  // 1. Offset as 8bit DWORD immediate
  def : Pat <
    (constant_load (add i64:$sbase, (i64 IMM8bitDWORD:$offset))),
    (vt (Instr_IMM $sbase, (as_dword_i32imm $offset)))
  >;

  // 2. Offset loaded in an 32bit SGPR
  def : Pat <
    (constant_load (add i64:$sbase, (i64 IMM32bit:$offset))),
    (vt (Instr_SGPR $sbase, (S_MOV_B32 (i32 (as_i32imm $offset)))))
  >;

  // 3. No offset at all
  def : Pat <
    (constant_load i64:$sbase),
    (vt (Instr_IMM $sbase, 0))
  >;
}

defm : SMRD_Pattern <S_LOAD_DWORD_IMM, S_LOAD_DWORD_SGPR, f32>;
defm : SMRD_Pattern <S_LOAD_DWORD_IMM, S_LOAD_DWORD_SGPR, i32>;
defm : SMRD_Pattern <S_LOAD_DWORDX2_IMM, S_LOAD_DWORDX2_SGPR, v2i32>;
defm : SMRD_Pattern <S_LOAD_DWORDX4_IMM, S_LOAD_DWORDX4_SGPR, v4i32>;
defm : SMRD_Pattern <S_LOAD_DWORDX8_IMM, S_LOAD_DWORDX8_SGPR, v32i8>;
defm : SMRD_Pattern <S_LOAD_DWORDX8_IMM, S_LOAD_DWORDX8_SGPR, v8i32>;
defm : SMRD_Pattern <S_LOAD_DWORDX16_IMM, S_LOAD_DWORDX16_SGPR, v16i32>;

// 1. Offset as 8bit DWORD immediate
def : Pat <
  (SIload_constant v4i32:$sbase, IMM8bitDWORD:$offset),
  (S_BUFFER_LOAD_DWORD_IMM $sbase, (as_dword_i32imm $offset))
>;

// 2. Offset loaded in an 32bit SGPR
def : Pat <
  (SIload_constant v4i32:$sbase, imm:$offset),
  (S_BUFFER_LOAD_DWORD_SGPR $sbase, (S_MOV_B32 imm:$offset))
>;

} // Predicates = [isSI] in {

//===----------------------------------------------------------------------===//
// SOP1 Patterns
//===----------------------------------------------------------------------===//

let Predicates = [isSI, isCFDepth0] in {

def : Pat <
  (i64 (ctpop i64:$src)),
  (INSERT_SUBREG (INSERT_SUBREG (i64 (IMPLICIT_DEF)),
    (S_BCNT1_I32_B64 $src), sub0),
    (S_MOV_B32 0), sub1)
>;

//===----------------------------------------------------------------------===//
// SOP2 Patterns
//===----------------------------------------------------------------------===//

// V_ADD_I32_e32/S_ADD_I32 produces carry in VCC/SCC. For the vector
// case, the sgpr-copies pass will fix this to use the vector version.
def : Pat <
  (i32 (addc i32:$src0, i32:$src1)),
  (S_ADD_I32 $src0, $src1)
>;

} // Predicates = [isSI, isCFDepth0]

let  Predicates = [isSI] in {

//===----------------------------------------------------------------------===//
// SOPP Patterns
//===----------------------------------------------------------------------===//

def : Pat <
  (int_AMDGPU_barrier_global),
  (S_BARRIER)
>;

//===----------------------------------------------------------------------===//
// VOP1 Patterns
//===----------------------------------------------------------------------===//

let Predicates = [UnsafeFPMath] in {
def : RcpPat<V_RCP_F64_e32, f64>;
defm : RsqPat<V_RSQ_F64_e32, f64>;
defm : RsqPat<V_RSQ_F32_e32, f32>;
}

//===----------------------------------------------------------------------===//
// VOP2 Patterns
//===----------------------------------------------------------------------===//

class BinOp64Pat <SDNode node, Instruction inst> : Pat <
  (node i64:$src0, i64:$src1),
  (INSERT_SUBREG (INSERT_SUBREG (i64 (IMPLICIT_DEF)),
    (inst  (EXTRACT_SUBREG i64:$src0, sub0),
                  (EXTRACT_SUBREG i64:$src1, sub0)), sub0),
    (inst (EXTRACT_SUBREG i64:$src0, sub1),
                  (EXTRACT_SUBREG i64:$src1, sub1)), sub1)
>;

def : BinOp64Pat <or, V_OR_B32_e32>;
def : BinOp64Pat <xor, V_XOR_B32_e32>;

class SextInReg <ValueType vt, int ShiftAmt> : Pat <
  (sext_inreg i32:$src0, vt),
  (V_ASHRREV_I32_e32 ShiftAmt, (V_LSHLREV_B32_e32 ShiftAmt, $src0))
>;

def : SextInReg <i8, 24>;
def : SextInReg <i16, 16>;

def : Pat <
  (i32 (add (i32 (ctpop i32:$popcnt)), i32:$val)),
  (V_BCNT_U32_B32_e32 $popcnt, $val)
>;

def : Pat <
   (i32 (ctpop i32:$popcnt)),
   (V_BCNT_U32_B32_e64 $popcnt, 0)
>;

def : Pat <
  (i64 (ctpop i64:$src)),
  (INSERT_SUBREG
    (INSERT_SUBREG (i64 (IMPLICIT_DEF)),
      (V_BCNT_U32_B32_e32 (EXTRACT_SUBREG $src, sub1),
        (V_BCNT_U32_B32_e64 (EXTRACT_SUBREG $src, sub0), 0)),
      sub0),
    (V_MOV_B32_e32 0), sub1)
>;

def : Pat <
  (addc i32:$src0, i32:$src1),
  (V_ADD_I32_e32 $src0, $src1)
>;

/********** ======================= **********/
/********** Image sampling patterns **********/
/********** ======================= **********/

// Image + sampler
class SampleRawPattern<SDPatternOperator name, MIMG opcode, ValueType vt> : Pat <
  (name vt:$addr, v8i32:$rsrc, v4i32:$sampler, i32:$dmask, i32:$unorm,
        i32:$r128, i32:$da, i32:$glc, i32:$slc, i32:$tfe, i32:$lwe),
  (opcode (as_i32imm $dmask), (as_i1imm $unorm), (as_i1imm $glc), (as_i1imm $da),
          (as_i1imm $r128), (as_i1imm $tfe), (as_i1imm $lwe), (as_i1imm $slc),
          $addr, $rsrc, $sampler)
>;

multiclass SampleRawPatterns<SDPatternOperator name, string opcode> {
  def : SampleRawPattern<name, !cast<MIMG>(opcode # _V4_V1), i32>;
  def : SampleRawPattern<name, !cast<MIMG>(opcode # _V4_V2), v2i32>;
  def : SampleRawPattern<name, !cast<MIMG>(opcode # _V4_V4), v4i32>;
  def : SampleRawPattern<name, !cast<MIMG>(opcode # _V4_V8), v8i32>;
  def : SampleRawPattern<name, !cast<MIMG>(opcode # _V4_V16), v16i32>;
}

// Image only
class ImagePattern<SDPatternOperator name, MIMG opcode, ValueType vt> : Pat <
  (name vt:$addr, v8i32:$rsrc, i32:$dmask, i32:$unorm,
        i32:$r128, i32:$da, i32:$glc, i32:$slc, i32:$tfe, i32:$lwe),
  (opcode (as_i32imm $dmask), (as_i1imm $unorm), (as_i1imm $glc), (as_i1imm $da),
          (as_i1imm $r128), (as_i1imm $tfe), (as_i1imm $lwe), (as_i1imm $slc),
          $addr, $rsrc)
>;

multiclass ImagePatterns<SDPatternOperator name, string opcode> {
  def : ImagePattern<name, !cast<MIMG>(opcode # _V4_V1), i32>;
  def : ImagePattern<name, !cast<MIMG>(opcode # _V4_V2), v2i32>;
  def : ImagePattern<name, !cast<MIMG>(opcode # _V4_V4), v4i32>;
}

// Basic sample
defm : SampleRawPatterns<int_SI_image_sample,           "IMAGE_SAMPLE">;
defm : SampleRawPatterns<int_SI_image_sample_cl,        "IMAGE_SAMPLE_CL">;
defm : SampleRawPatterns<int_SI_image_sample_d,         "IMAGE_SAMPLE_D">;
defm : SampleRawPatterns<int_SI_image_sample_d_cl,      "IMAGE_SAMPLE_D_CL">;
defm : SampleRawPatterns<int_SI_image_sample_l,         "IMAGE_SAMPLE_L">;
defm : SampleRawPatterns<int_SI_image_sample_b,         "IMAGE_SAMPLE_B">;
defm : SampleRawPatterns<int_SI_image_sample_b_cl,      "IMAGE_SAMPLE_B_CL">;
defm : SampleRawPatterns<int_SI_image_sample_lz,        "IMAGE_SAMPLE_LZ">;
defm : SampleRawPatterns<int_SI_image_sample_cd,        "IMAGE_SAMPLE_CD">;
defm : SampleRawPatterns<int_SI_image_sample_cd_cl,     "IMAGE_SAMPLE_CD_CL">;

// Sample with comparison
defm : SampleRawPatterns<int_SI_image_sample_c,         "IMAGE_SAMPLE_C">;
defm : SampleRawPatterns<int_SI_image_sample_c_cl,      "IMAGE_SAMPLE_C_CL">;
defm : SampleRawPatterns<int_SI_image_sample_c_d,       "IMAGE_SAMPLE_C_D">;
defm : SampleRawPatterns<int_SI_image_sample_c_d_cl,    "IMAGE_SAMPLE_C_D_CL">;
defm : SampleRawPatterns<int_SI_image_sample_c_l,       "IMAGE_SAMPLE_C_L">;
defm : SampleRawPatterns<int_SI_image_sample_c_b,       "IMAGE_SAMPLE_C_B">;
defm : SampleRawPatterns<int_SI_image_sample_c_b_cl,    "IMAGE_SAMPLE_C_B_CL">;
defm : SampleRawPatterns<int_SI_image_sample_c_lz,      "IMAGE_SAMPLE_C_LZ">;
defm : SampleRawPatterns<int_SI_image_sample_c_cd,      "IMAGE_SAMPLE_C_CD">;
defm : SampleRawPatterns<int_SI_image_sample_c_cd_cl,   "IMAGE_SAMPLE_C_CD_CL">;

// Sample with offsets
defm : SampleRawPatterns<int_SI_image_sample_o,         "IMAGE_SAMPLE_O">;
defm : SampleRawPatterns<int_SI_image_sample_cl_o,      "IMAGE_SAMPLE_CL_O">;
defm : SampleRawPatterns<int_SI_image_sample_d_o,       "IMAGE_SAMPLE_D_O">;
defm : SampleRawPatterns<int_SI_image_sample_d_cl_o,    "IMAGE_SAMPLE_D_CL_O">;
defm : SampleRawPatterns<int_SI_image_sample_l_o,       "IMAGE_SAMPLE_L_O">;
defm : SampleRawPatterns<int_SI_image_sample_b_o,       "IMAGE_SAMPLE_B_O">;
defm : SampleRawPatterns<int_SI_image_sample_b_cl_o,    "IMAGE_SAMPLE_B_CL_O">;
defm : SampleRawPatterns<int_SI_image_sample_lz_o,      "IMAGE_SAMPLE_LZ_O">;
defm : SampleRawPatterns<int_SI_image_sample_cd_o,      "IMAGE_SAMPLE_CD_O">;
defm : SampleRawPatterns<int_SI_image_sample_cd_cl_o,   "IMAGE_SAMPLE_CD_CL_O">;

// Sample with comparison and offsets
defm : SampleRawPatterns<int_SI_image_sample_c_o,       "IMAGE_SAMPLE_C_O">;
defm : SampleRawPatterns<int_SI_image_sample_c_cl_o,    "IMAGE_SAMPLE_C_CL_O">;
defm : SampleRawPatterns<int_SI_image_sample_c_d_o,     "IMAGE_SAMPLE_C_D_O">;
defm : SampleRawPatterns<int_SI_image_sample_c_d_cl_o,  "IMAGE_SAMPLE_C_D_CL_O">;
defm : SampleRawPatterns<int_SI_image_sample_c_l_o,     "IMAGE_SAMPLE_C_L_O">;
defm : SampleRawPatterns<int_SI_image_sample_c_b_o,     "IMAGE_SAMPLE_C_B_O">;
defm : SampleRawPatterns<int_SI_image_sample_c_b_cl_o,  "IMAGE_SAMPLE_C_B_CL_O">;
defm : SampleRawPatterns<int_SI_image_sample_c_lz_o,    "IMAGE_SAMPLE_C_LZ_O">;
defm : SampleRawPatterns<int_SI_image_sample_c_cd_o,    "IMAGE_SAMPLE_C_CD_O">;
defm : SampleRawPatterns<int_SI_image_sample_c_cd_cl_o, "IMAGE_SAMPLE_C_CD_CL_O">;

// Gather opcodes
// Only the variants which make sense are defined.
def : SampleRawPattern<int_SI_gather4,           IMAGE_GATHER4_V4_V2,        v2i32>;
def : SampleRawPattern<int_SI_gather4,           IMAGE_GATHER4_V4_V4,        v4i32>;
def : SampleRawPattern<int_SI_gather4_cl,        IMAGE_GATHER4_CL_V4_V4,     v4i32>;
def : SampleRawPattern<int_SI_gather4_l,         IMAGE_GATHER4_L_V4_V4,      v4i32>;
def : SampleRawPattern<int_SI_gather4_b,         IMAGE_GATHER4_B_V4_V4,      v4i32>;
def : SampleRawPattern<int_SI_gather4_b_cl,      IMAGE_GATHER4_B_CL_V4_V4,   v4i32>;
def : SampleRawPattern<int_SI_gather4_b_cl,      IMAGE_GATHER4_B_CL_V4_V8,   v8i32>;
def : SampleRawPattern<int_SI_gather4_lz,        IMAGE_GATHER4_LZ_V4_V2,     v2i32>;
def : SampleRawPattern<int_SI_gather4_lz,        IMAGE_GATHER4_LZ_V4_V4,     v4i32>;

def : SampleRawPattern<int_SI_gather4_c,         IMAGE_GATHER4_C_V4_V4,      v4i32>;
def : SampleRawPattern<int_SI_gather4_c_cl,      IMAGE_GATHER4_C_CL_V4_V4,   v4i32>;
def : SampleRawPattern<int_SI_gather4_c_cl,      IMAGE_GATHER4_C_CL_V4_V8,   v8i32>;
def : SampleRawPattern<int_SI_gather4_c_l,       IMAGE_GATHER4_C_L_V4_V4,    v4i32>;
def : SampleRawPattern<int_SI_gather4_c_l,       IMAGE_GATHER4_C_L_V4_V8,    v8i32>;
def : SampleRawPattern<int_SI_gather4_c_b,       IMAGE_GATHER4_C_B_V4_V4,    v4i32>;
def : SampleRawPattern<int_SI_gather4_c_b,       IMAGE_GATHER4_C_B_V4_V8,    v8i32>;
def : SampleRawPattern<int_SI_gather4_c_b_cl,    IMAGE_GATHER4_C_B_CL_V4_V8, v8i32>;
def : SampleRawPattern<int_SI_gather4_c_lz,      IMAGE_GATHER4_C_LZ_V4_V4,   v4i32>;

def : SampleRawPattern<int_SI_gather4_o,         IMAGE_GATHER4_O_V4_V4,      v4i32>;
def : SampleRawPattern<int_SI_gather4_cl_o,      IMAGE_GATHER4_CL_O_V4_V4,   v4i32>;
def : SampleRawPattern<int_SI_gather4_cl_o,      IMAGE_GATHER4_CL_O_V4_V8,   v8i32>;
def : SampleRawPattern<int_SI_gather4_l_o,       IMAGE_GATHER4_L_O_V4_V4,    v4i32>;
def : SampleRawPattern<int_SI_gather4_l_o,       IMAGE_GATHER4_L_O_V4_V8,    v8i32>;
def : SampleRawPattern<int_SI_gather4_b_o,       IMAGE_GATHER4_B_O_V4_V4,    v4i32>;
def : SampleRawPattern<int_SI_gather4_b_o,       IMAGE_GATHER4_B_O_V4_V8,    v8i32>;
def : SampleRawPattern<int_SI_gather4_b_cl_o,    IMAGE_GATHER4_B_CL_O_V4_V8, v8i32>;
def : SampleRawPattern<int_SI_gather4_lz_o,      IMAGE_GATHER4_LZ_O_V4_V4,   v4i32>;

def : SampleRawPattern<int_SI_gather4_c_o,       IMAGE_GATHER4_C_O_V4_V4,    v4i32>;
def : SampleRawPattern<int_SI_gather4_c_o,       IMAGE_GATHER4_C_O_V4_V8,    v8i32>;
def : SampleRawPattern<int_SI_gather4_c_cl_o,    IMAGE_GATHER4_C_CL_O_V4_V8, v8i32>;
def : SampleRawPattern<int_SI_gather4_c_l_o,     IMAGE_GATHER4_C_L_O_V4_V8,  v8i32>;
def : SampleRawPattern<int_SI_gather4_c_b_o,     IMAGE_GATHER4_C_B_O_V4_V8,  v8i32>;
def : SampleRawPattern<int_SI_gather4_c_b_cl_o,  IMAGE_GATHER4_C_B_CL_O_V4_V8, v8i32>;
def : SampleRawPattern<int_SI_gather4_c_lz_o,    IMAGE_GATHER4_C_LZ_O_V4_V4, v4i32>;
def : SampleRawPattern<int_SI_gather4_c_lz_o,    IMAGE_GATHER4_C_LZ_O_V4_V8, v8i32>;

def : SampleRawPattern<int_SI_getlod, IMAGE_GET_LOD_V4_V1, i32>;
def : SampleRawPattern<int_SI_getlod, IMAGE_GET_LOD_V4_V2, v2i32>;
def : SampleRawPattern<int_SI_getlod, IMAGE_GET_LOD_V4_V4, v4i32>;

def : ImagePattern<int_SI_getresinfo, IMAGE_GET_RESINFO_V4_V1, i32>;
defm : ImagePatterns<int_SI_image_load, "IMAGE_LOAD">;
defm : ImagePatterns<int_SI_image_load_mip, "IMAGE_LOAD_MIP">;

/* SIsample for simple 1D texture lookup */
def : Pat <
  (SIsample i32:$addr, v32i8:$rsrc, v4i32:$sampler, imm),
  (IMAGE_SAMPLE_V4_V1 0xf, 0, 0, 0, 0, 0, 0, 0, $addr, $rsrc, $sampler)
>;

class SamplePattern<SDNode name, MIMG opcode, ValueType vt> : Pat <
    (name vt:$addr, v32i8:$rsrc, v4i32:$sampler, imm),
    (opcode 0xf, 0, 0, 0, 0, 0, 0, 0, $addr, $rsrc, $sampler)
>;

class SampleRectPattern<SDNode name, MIMG opcode, ValueType vt> : Pat <
    (name vt:$addr, v32i8:$rsrc, v4i32:$sampler, TEX_RECT),
    (opcode 0xf, 1, 0, 0, 0, 0, 0, 0, $addr, $rsrc, $sampler)
>;

class SampleArrayPattern<SDNode name, MIMG opcode, ValueType vt> : Pat <
    (name vt:$addr, v32i8:$rsrc, v4i32:$sampler, TEX_ARRAY),
    (opcode 0xf, 0, 0, 1, 0, 0, 0, 0, $addr, $rsrc, $sampler)
>;

class SampleShadowPattern<SDNode name, MIMG opcode,
                          ValueType vt> : Pat <
    (name vt:$addr, v32i8:$rsrc, v4i32:$sampler, TEX_SHADOW),
    (opcode 0xf, 0, 0, 0, 0, 0, 0, 0, $addr, $rsrc, $sampler)
>;

class SampleShadowArrayPattern<SDNode name, MIMG opcode,
                               ValueType vt> : Pat <
    (name vt:$addr, v32i8:$rsrc, v4i32:$sampler, TEX_SHADOW_ARRAY),
    (opcode 0xf, 0, 0, 1, 0, 0, 0, 0, $addr, $rsrc, $sampler)
>;

/* SIsample* for texture lookups consuming more address parameters */
multiclass SamplePatterns<MIMG sample, MIMG sample_c, MIMG sample_l,
                          MIMG sample_c_l, MIMG sample_b, MIMG sample_c_b,
MIMG sample_d, MIMG sample_c_d, ValueType addr_type> {
  def : SamplePattern <SIsample, sample, addr_type>;
  def : SampleRectPattern <SIsample, sample, addr_type>;
  def : SampleArrayPattern <SIsample, sample, addr_type>;
  def : SampleShadowPattern <SIsample, sample_c, addr_type>;
  def : SampleShadowArrayPattern <SIsample, sample_c, addr_type>;

  def : SamplePattern <SIsamplel, sample_l, addr_type>;
  def : SampleArrayPattern <SIsamplel, sample_l, addr_type>;
  def : SampleShadowPattern <SIsamplel, sample_c_l, addr_type>;
  def : SampleShadowArrayPattern <SIsamplel, sample_c_l, addr_type>;

  def : SamplePattern <SIsampleb, sample_b, addr_type>;
  def : SampleArrayPattern <SIsampleb, sample_b, addr_type>;
  def : SampleShadowPattern <SIsampleb, sample_c_b, addr_type>;
  def : SampleShadowArrayPattern <SIsampleb, sample_c_b, addr_type>;

  def : SamplePattern <SIsampled, sample_d, addr_type>;
  def : SampleArrayPattern <SIsampled, sample_d, addr_type>;
  def : SampleShadowPattern <SIsampled, sample_c_d, addr_type>;
  def : SampleShadowArrayPattern <SIsampled, sample_c_d, addr_type>;
}

defm : SamplePatterns<IMAGE_SAMPLE_V4_V2, IMAGE_SAMPLE_C_V4_V2,
                      IMAGE_SAMPLE_L_V4_V2, IMAGE_SAMPLE_C_L_V4_V2,
                      IMAGE_SAMPLE_B_V4_V2, IMAGE_SAMPLE_C_B_V4_V2,
                      IMAGE_SAMPLE_D_V4_V2, IMAGE_SAMPLE_C_D_V4_V2,
                      v2i32>;
defm : SamplePatterns<IMAGE_SAMPLE_V4_V4, IMAGE_SAMPLE_C_V4_V4,
                      IMAGE_SAMPLE_L_V4_V4, IMAGE_SAMPLE_C_L_V4_V4,
                      IMAGE_SAMPLE_B_V4_V4, IMAGE_SAMPLE_C_B_V4_V4,
                      IMAGE_SAMPLE_D_V4_V4, IMAGE_SAMPLE_C_D_V4_V4,
                      v4i32>;
defm : SamplePatterns<IMAGE_SAMPLE_V4_V8, IMAGE_SAMPLE_C_V4_V8,
                      IMAGE_SAMPLE_L_V4_V8, IMAGE_SAMPLE_C_L_V4_V8,
                      IMAGE_SAMPLE_B_V4_V8, IMAGE_SAMPLE_C_B_V4_V8,
                      IMAGE_SAMPLE_D_V4_V8, IMAGE_SAMPLE_C_D_V4_V8,
                      v8i32>;
defm : SamplePatterns<IMAGE_SAMPLE_V4_V16, IMAGE_SAMPLE_C_V4_V16,
                      IMAGE_SAMPLE_L_V4_V16, IMAGE_SAMPLE_C_L_V4_V16,
                      IMAGE_SAMPLE_B_V4_V16, IMAGE_SAMPLE_C_B_V4_V16,
                      IMAGE_SAMPLE_D_V4_V16, IMAGE_SAMPLE_C_D_V4_V16,
                      v16i32>;

/* int_SI_imageload for texture fetches consuming varying address parameters */
class ImageLoadPattern<Intrinsic name, MIMG opcode, ValueType addr_type> : Pat <
    (name addr_type:$addr, v32i8:$rsrc, imm),
    (opcode 0xf, 0, 0, 0, 0, 0, 0, 0, $addr, $rsrc)
>;

class ImageLoadArrayPattern<Intrinsic name, MIMG opcode, ValueType addr_type> : Pat <
    (name addr_type:$addr, v32i8:$rsrc, TEX_ARRAY),
    (opcode 0xf, 0, 0, 1, 0, 0, 0, 0, $addr, $rsrc)
>;

class ImageLoadMSAAPattern<Intrinsic name, MIMG opcode, ValueType addr_type> : Pat <
    (name addr_type:$addr, v32i8:$rsrc, TEX_MSAA),
    (opcode 0xf, 0, 0, 0, 0, 0, 0, 0, $addr, $rsrc)
>;

class ImageLoadArrayMSAAPattern<Intrinsic name, MIMG opcode, ValueType addr_type> : Pat <
    (name addr_type:$addr, v32i8:$rsrc, TEX_ARRAY_MSAA),
    (opcode 0xf, 0, 0, 1, 0, 0, 0, 0, $addr, $rsrc)
>;

multiclass ImageLoadPatterns<MIMG opcode, ValueType addr_type> {
  def : ImageLoadPattern <int_SI_imageload, opcode, addr_type>;
  def : ImageLoadArrayPattern <int_SI_imageload, opcode, addr_type>;
}

multiclass ImageLoadMSAAPatterns<MIMG opcode, ValueType addr_type> {
  def : ImageLoadMSAAPattern <int_SI_imageload, opcode, addr_type>;
  def : ImageLoadArrayMSAAPattern <int_SI_imageload, opcode, addr_type>;
}

defm : ImageLoadPatterns<IMAGE_LOAD_MIP_V4_V2, v2i32>;
defm : ImageLoadPatterns<IMAGE_LOAD_MIP_V4_V4, v4i32>;

defm : ImageLoadMSAAPatterns<IMAGE_LOAD_V4_V2, v2i32>;
defm : ImageLoadMSAAPatterns<IMAGE_LOAD_V4_V4, v4i32>;

/* Image resource information */
def : Pat <
  (int_SI_resinfo i32:$mipid, v32i8:$rsrc, imm),
  (IMAGE_GET_RESINFO_V4_V1 0xf, 0, 0, 0, 0, 0, 0, 0, (V_MOV_B32_e32 $mipid), $rsrc)
>;

def : Pat <
  (int_SI_resinfo i32:$mipid, v32i8:$rsrc, TEX_ARRAY),
  (IMAGE_GET_RESINFO_V4_V1 0xf, 0, 0, 1, 0, 0, 0, 0, (V_MOV_B32_e32 $mipid), $rsrc)
>;

def : Pat <
  (int_SI_resinfo i32:$mipid, v32i8:$rsrc, TEX_ARRAY_MSAA),
  (IMAGE_GET_RESINFO_V4_V1 0xf, 0, 0, 1, 0, 0, 0, 0, (V_MOV_B32_e32 $mipid), $rsrc)
>;

/********** ============================================ **********/
/********** Extraction, Insertion, Building and Casting  **********/
/********** ============================================ **********/

foreach Index = 0-2 in {
  def Extract_Element_v2i32_#Index : Extract_Element <
    i32, v2i32, Index, !cast<SubRegIndex>(sub#Index)
  >;
  def Insert_Element_v2i32_#Index : Insert_Element <
    i32, v2i32, Index, !cast<SubRegIndex>(sub#Index)
  >;

  def Extract_Element_v2f32_#Index : Extract_Element <
    f32, v2f32, Index, !cast<SubRegIndex>(sub#Index)
  >;
  def Insert_Element_v2f32_#Index : Insert_Element <
    f32, v2f32, Index, !cast<SubRegIndex>(sub#Index)
  >;
}

foreach Index = 0-3 in {
  def Extract_Element_v4i32_#Index : Extract_Element <
    i32, v4i32, Index, !cast<SubRegIndex>(sub#Index)
  >;
  def Insert_Element_v4i32_#Index : Insert_Element <
    i32, v4i32, Index, !cast<SubRegIndex>(sub#Index)
  >;

  def Extract_Element_v4f32_#Index : Extract_Element <
    f32, v4f32, Index, !cast<SubRegIndex>(sub#Index)
  >;
  def Insert_Element_v4f32_#Index : Insert_Element <
    f32, v4f32, Index, !cast<SubRegIndex>(sub#Index)
  >;
}

foreach Index = 0-7 in {
  def Extract_Element_v8i32_#Index : Extract_Element <
    i32, v8i32, Index, !cast<SubRegIndex>(sub#Index)
  >;
  def Insert_Element_v8i32_#Index : Insert_Element <
    i32, v8i32, Index, !cast<SubRegIndex>(sub#Index)
  >;

  def Extract_Element_v8f32_#Index : Extract_Element <
    f32, v8f32, Index, !cast<SubRegIndex>(sub#Index)
  >;
  def Insert_Element_v8f32_#Index : Insert_Element <
    f32, v8f32, Index, !cast<SubRegIndex>(sub#Index)
  >;
}

foreach Index = 0-15 in {
  def Extract_Element_v16i32_#Index : Extract_Element <
    i32, v16i32, Index, !cast<SubRegIndex>(sub#Index)
  >;
  def Insert_Element_v16i32_#Index : Insert_Element <
    i32, v16i32, Index, !cast<SubRegIndex>(sub#Index)
  >;

  def Extract_Element_v16f32_#Index : Extract_Element <
    f32, v16f32, Index, !cast<SubRegIndex>(sub#Index)
  >;
  def Insert_Element_v16f32_#Index : Insert_Element <
    f32, v16f32, Index, !cast<SubRegIndex>(sub#Index)
  >;
}

def : BitConvert <i32, f32, SReg_32>;
def : BitConvert <i32, f32, VReg_32>;

def : BitConvert <f32, i32, SReg_32>;
def : BitConvert <f32, i32, VReg_32>;

def : BitConvert <i64, f64, VReg_64>;

def : BitConvert <f64, i64, VReg_64>;

def : BitConvert <v2f32, v2i32, VReg_64>;
def : BitConvert <v2i32, v2f32, VReg_64>;
def : BitConvert <v2i32, i64, VReg_64>;
def : BitConvert <i64, v2i32, VReg_64>;
def : BitConvert <v2f32, i64, VReg_64>;
def : BitConvert <i64, v2f32, VReg_64>;
def : BitConvert <v2i32, f64, VReg_64>;
def : BitConvert <f64, v2i32, VReg_64>;
def : BitConvert <v4f32, v4i32, VReg_128>;
def : BitConvert <v4i32, v4f32, VReg_128>;

def : BitConvert <v8f32, v8i32, SReg_256>;
def : BitConvert <v8i32, v8f32, SReg_256>;
def : BitConvert <v8i32, v32i8, SReg_256>;
def : BitConvert <v32i8, v8i32, SReg_256>;
def : BitConvert <v8i32, v32i8, VReg_256>;
def : BitConvert <v8i32, v8f32, VReg_256>;
def : BitConvert <v8f32, v8i32, VReg_256>;
def : BitConvert <v32i8, v8i32, VReg_256>;

def : BitConvert <v16i32, v16f32, VReg_512>;
def : BitConvert <v16f32, v16i32, VReg_512>;

/********** =================== **********/
/********** Src & Dst modifiers **********/
/********** =================== **********/

def FCLAMP_SI : AMDGPUShaderInst <
  (outs VReg_32:$dst),
  (ins VSrc_32:$src0),
  "FCLAMP_SI $dst, $src0",
  []
> {
  let usesCustomInserter = 1;
}

def : Pat <
  (AMDGPUclamp f32:$src, (f32 FP_ZERO), (f32 FP_ONE)),
  (FCLAMP_SI f32:$src)
>;

/********** ================================ **********/
/********** Floating point absolute/negative **********/
/********** ================================ **********/

// Manipulate the sign bit directly, as e.g. using the source negation modifier
// in V_ADD_F32_e64 $src, 0, [...] does not result in -0.0 for $src == +0.0,
// breaking the piglit *s-floatBitsToInt-neg* tests

// TODO: Look into not implementing isFNegFree/isFAbsFree for SI, and possibly
// removing these patterns

def : Pat <
  (fneg (fabs f32:$src)),
  (V_OR_B32_e32 $src, (V_MOV_B32_e32 0x80000000)) /* Set sign bit */
>;

def FABS_SI : AMDGPUShaderInst <
  (outs VReg_32:$dst),
  (ins VSrc_32:$src0),
  "FABS_SI $dst, $src0",
  []
> {
  let usesCustomInserter = 1;
}

def : Pat <
  (fabs f32:$src),
  (FABS_SI f32:$src)
>;

def FNEG_SI : AMDGPUShaderInst <
  (outs VReg_32:$dst),
  (ins VSrc_32:$src0),
  "FNEG_SI $dst, $src0",
  []
> {
  let usesCustomInserter = 1;
}

def : Pat <
  (fneg f32:$src),
  (FNEG_SI f32:$src)
>;

/********** ================== **********/
/********** Immediate Patterns **********/
/********** ================== **********/

def : Pat <
  (SGPRImm<(i32 imm)>:$imm),
  (S_MOV_B32 imm:$imm)
>;

def : Pat <
  (SGPRImm<(f32 fpimm)>:$imm),
  (S_MOV_B32 fpimm:$imm)
>;

def : Pat <
  (i32 imm:$imm),
  (V_MOV_B32_e32 imm:$imm)
>;

def : Pat <
  (f32 fpimm:$imm),
  (V_MOV_B32_e32 fpimm:$imm)
>;

def : Pat <
  (i64 InlineImm<i64>:$imm),
  (S_MOV_B64 InlineImm<i64>:$imm)
>;

/********** ===================== **********/
/********** Interpolation Paterns **********/
/********** ===================== **********/

def : Pat <
  (int_SI_fs_constant imm:$attr_chan, imm:$attr, i32:$params),
  (V_INTERP_MOV_F32 INTERP.P0, imm:$attr_chan, imm:$attr, $params)
>;

def : Pat <
  (int_SI_fs_interp imm:$attr_chan, imm:$attr, M0Reg:$params, v2i32:$ij),
  (V_INTERP_P2_F32 (V_INTERP_P1_F32 (EXTRACT_SUBREG v2i32:$ij, sub0),
                                    imm:$attr_chan, imm:$attr, i32:$params),
                   (EXTRACT_SUBREG $ij, sub1),
                   imm:$attr_chan, imm:$attr, $params)
>;

/********** ================== **********/
/********** Intrinsic Patterns **********/
/********** ================== **********/

/* llvm.AMDGPU.pow */
def : POW_Common <V_LOG_F32_e32, V_EXP_F32_e32, V_MUL_LEGACY_F32_e32>;

def : Pat <
  (int_AMDGPU_div f32:$src0, f32:$src1),
  (V_MUL_LEGACY_F32_e32 $src0, (V_RCP_LEGACY_F32_e32 $src1))
>;

def : Pat<
  (fdiv f64:$src0, f64:$src1),
  (V_MUL_F64 0 /* src0_modifiers */, $src0,
             0 /* src1_modifiers */, (V_RCP_F64_e32 $src1),
             0 /* clamp */, 0 /* omod */)
>;

def : Pat <
  (int_AMDGPU_cube v4f32:$src),
  (INSERT_SUBREG (INSERT_SUBREG (INSERT_SUBREG (INSERT_SUBREG (v4f32 (IMPLICIT_DEF)),
    (V_CUBETC_F32 0 /* src0_modifiers */, (EXTRACT_SUBREG $src, sub0),
                  0 /* src1_modifiers */, (EXTRACT_SUBREG $src, sub1),
                  0 /* src2_modifiers */, (EXTRACT_SUBREG $src, sub2),
                  0 /* clamp */, 0 /* omod */),
                  sub0),
    (V_CUBESC_F32 0 /* src0_modifiers */, (EXTRACT_SUBREG $src, sub0),
                  0 /* src1_modifiers */,(EXTRACT_SUBREG $src, sub1),
                  0 /* src2_modifiers */,(EXTRACT_SUBREG $src, sub2),
                  0 /* clamp */, 0 /* omod */),
                  sub1),
    (V_CUBEMA_F32 0 /* src1_modifiers */,(EXTRACT_SUBREG $src, sub0),
                  0 /* src1_modifiers */,(EXTRACT_SUBREG $src, sub1),
                  0 /* src1_modifiers */,(EXTRACT_SUBREG $src, sub2),
                  0 /* clamp */, 0 /* omod */),
                  sub2),
    (V_CUBEID_F32 0 /* src1_modifiers */,(EXTRACT_SUBREG $src, sub0),
                  0 /* src1_modifiers */,(EXTRACT_SUBREG $src, sub1),
                  0 /* src1_modifiers */,(EXTRACT_SUBREG $src, sub2),
                  0 /* clamp */, 0 /* omod */),
                  sub3)
>;

def : Pat <
  (i32 (sext i1:$src0)),
  (V_CNDMASK_B32_e64 (i32 0), (i32 -1), $src0)
>;

class Ext32Pat <SDNode ext> : Pat <
  (i32 (ext i1:$src0)),
  (V_CNDMASK_B32_e64 (i32 0), (i32 1), $src0)
>;

def : Ext32Pat <zext>;
def : Ext32Pat <anyext>;

// Offset in an 32Bit VGPR
def : Pat <
  (SIload_constant v4i32:$sbase, i32:$voff),
  (BUFFER_LOAD_DWORD_OFFEN $sbase, $voff, 0, 0, 0, 0, 0)
>;

// The multiplication scales from [0,1] to the unsigned integer range
def : Pat <
  (AMDGPUurecip i32:$src0),
  (V_CVT_U32_F32_e32
    (V_MUL_F32_e32 CONST.FP_UINT_MAX_PLUS_1,
                   (V_RCP_IFLAG_F32_e32 (V_CVT_F32_U32_e32 $src0))))
>;

def : Pat <
  (int_SI_tid),
  (V_MBCNT_HI_U32_B32_e32 0xffffffff,
                          (V_MBCNT_LO_U32_B32_e64 0xffffffff, 0))
>;

//===----------------------------------------------------------------------===//
// VOP3 Patterns
//===----------------------------------------------------------------------===//

def : IMad24Pat<V_MAD_I32_I24>;
def : UMad24Pat<V_MAD_U32_U24>;

def : Pat <
  (mul i32:$src0, i32:$src1),
  (V_MUL_LO_I32 $src0, $src1)
>;

def : Pat <
  (mulhu i32:$src0, i32:$src1),
  (V_MUL_HI_U32 $src0, $src1)
>;

def : Pat <
  (mulhs i32:$src0, i32:$src1),
  (V_MUL_HI_I32 $src0, $src1)
>;

defm : BFIPatterns <V_BFI_B32, S_MOV_B32>;
def : ROTRPattern <V_ALIGNBIT_B32>;

/********** ======================= **********/
/**********   Load/Store Patterns   **********/
/********** ======================= **********/

multiclass DSReadPat <DS inst, ValueType vt, PatFrag frag> {
  def : Pat <
    (vt (frag (add i32:$ptr, (i32 IMM16bit:$offset)))),
    (inst (i1 0), $ptr, (as_i16imm $offset))
  >;

  def : Pat <
    (frag i32:$src0),
    (vt (inst 0, $src0, 0))
  >;
}

defm : DSReadPat <DS_READ_I8,  i32, sextloadi8_local>;
defm : DSReadPat <DS_READ_U8,  i32, az_extloadi8_local>;
defm : DSReadPat <DS_READ_I16, i32, sextloadi16_local>;
defm : DSReadPat <DS_READ_U16, i32, az_extloadi16_local>;
defm : DSReadPat <DS_READ_B32, i32, local_load>;
defm : DSReadPat <DS_READ_B64, v2i32, local_load>;

multiclass DSWritePat <DS inst, ValueType vt, PatFrag frag> {
  def : Pat <
    (frag vt:$value, (add i32:$ptr, (i32 IMM16bit:$offset))),
    (inst (i1 0), $ptr, $value, (as_i16imm $offset))
  >;

  def : Pat <
    (frag vt:$val, i32:$ptr),
    (inst 0, $ptr, $val, 0)
  >;
}

defm : DSWritePat <DS_WRITE_B8, i32, truncstorei8_local>;
defm : DSWritePat <DS_WRITE_B16, i32, truncstorei16_local>;
defm : DSWritePat <DS_WRITE_B32, i32, local_store>;
defm : DSWritePat <DS_WRITE_B64, v2i32, local_store>;

multiclass DSAtomicRetPat<DS inst, ValueType vt, PatFrag frag> {
  def : Pat <
    (frag (add i32:$ptr, (i32 IMM16bit:$offset)), vt:$value),
    (inst (i1 0), $ptr, $value, (as_i16imm $offset))
  >;

  def : Pat <
    (frag i32:$ptr, vt:$val),
    (inst 0, $ptr, $val, 0)
  >;
}

// Special case of DSAtomicRetPat for add / sub 1 -> inc / dec
//
// We need to use something for the data0, so we set a register to
// -1. For the non-rtn variants, the manual says it does
// DS[A] = (DS[A] >= D0) ? 0 : DS[A] + 1, and setting D0 to uint_max
// will always do the increment so I'm assuming it's the same.
//
// We also load this -1 with s_mov_b32 / s_mov_b64 even though this
// needs to be a VGPR. The SGPR copy pass will fix this, and it's
// easier since there is no v_mov_b64.
multiclass DSAtomicIncRetPat<DS inst, ValueType vt,
                             Instruction LoadImm, PatFrag frag> {
  def : Pat <
    (frag (add i32:$ptr, (i32 IMM16bit:$offset)), (vt 1)),
    (inst (i1 0), $ptr, (LoadImm (vt -1)), (as_i16imm $offset))
  >;

  def : Pat <
    (frag i32:$ptr, (vt 1)),
    (inst 0, $ptr, (LoadImm (vt -1)), 0)
  >;
}

multiclass DSAtomicCmpXChg <DS inst, ValueType vt, PatFrag frag> {
  def : Pat <
    (frag (add i32:$ptr, (i32 IMM16bit:$offset)), vt:$cmp, vt:$swap),
    (inst (i1 0), $ptr, $cmp, $swap, (as_i16imm $offset))
  >;

  def : Pat <
    (frag i32:$ptr, vt:$cmp, vt:$swap),
    (inst 0, $ptr, $cmp, $swap, 0)
  >;
}


// 32-bit atomics.
defm : DSAtomicIncRetPat<DS_INC_RTN_U32, i32,
                         S_MOV_B32, atomic_load_add_local>;
defm : DSAtomicIncRetPat<DS_DEC_RTN_U32, i32,
                         S_MOV_B32, atomic_load_sub_local>;

defm : DSAtomicRetPat<DS_WRXCHG_RTN_B32, i32, atomic_swap_local>;
defm : DSAtomicRetPat<DS_ADD_RTN_U32, i32, atomic_load_add_local>;
defm : DSAtomicRetPat<DS_SUB_RTN_U32, i32, atomic_load_sub_local>;
defm : DSAtomicRetPat<DS_AND_RTN_B32, i32, atomic_load_and_local>;
defm : DSAtomicRetPat<DS_OR_RTN_B32, i32, atomic_load_or_local>;
defm : DSAtomicRetPat<DS_XOR_RTN_B32, i32, atomic_load_xor_local>;
defm : DSAtomicRetPat<DS_MIN_RTN_I32, i32, atomic_load_min_local>;
defm : DSAtomicRetPat<DS_MAX_RTN_I32, i32, atomic_load_max_local>;
defm : DSAtomicRetPat<DS_MIN_RTN_U32, i32, atomic_load_umin_local>;
defm : DSAtomicRetPat<DS_MAX_RTN_U32, i32, atomic_load_umax_local>;

defm : DSAtomicCmpXChg<DS_CMPST_RTN_B32, i32, atomic_cmp_swap_32_local>;

// 64-bit atomics.
defm : DSAtomicIncRetPat<DS_INC_RTN_U64, i64,
                         S_MOV_B64, atomic_load_add_local>;
defm : DSAtomicIncRetPat<DS_DEC_RTN_U64, i64,
                         S_MOV_B64, atomic_load_sub_local>;

defm : DSAtomicRetPat<DS_WRXCHG_RTN_B64, i64, atomic_swap_local>;
defm : DSAtomicRetPat<DS_ADD_RTN_U64, i64, atomic_load_add_local>;
defm : DSAtomicRetPat<DS_SUB_RTN_U64, i64, atomic_load_sub_local>;
defm : DSAtomicRetPat<DS_AND_RTN_B64, i64, atomic_load_and_local>;
defm : DSAtomicRetPat<DS_OR_RTN_B64, i64, atomic_load_or_local>;
defm : DSAtomicRetPat<DS_XOR_RTN_B64, i64, atomic_load_xor_local>;
defm : DSAtomicRetPat<DS_MIN_RTN_I64, i64, atomic_load_min_local>;
defm : DSAtomicRetPat<DS_MAX_RTN_I64, i64, atomic_load_max_local>;
defm : DSAtomicRetPat<DS_MIN_RTN_U64, i64, atomic_load_umin_local>;
defm : DSAtomicRetPat<DS_MAX_RTN_U64, i64, atomic_load_umax_local>;

defm : DSAtomicCmpXChg<DS_CMPST_RTN_B64, i64, atomic_cmp_swap_64_local>;


//===----------------------------------------------------------------------===//
// MUBUF Patterns
//===----------------------------------------------------------------------===//

multiclass MUBUFLoad_Pattern <MUBUF Instr_ADDR64, ValueType vt,
                              PatFrag constant_ld> {
  def : Pat <
     (vt (constant_ld (add i64:$ptr, i64:$offset))),
     (Instr_ADDR64 (SI_ADDR64_RSRC $ptr), $offset, 0)
  >;

}

defm : MUBUFLoad_Pattern <BUFFER_LOAD_SBYTE_ADDR64, i32, sextloadi8_constant>;
defm : MUBUFLoad_Pattern <BUFFER_LOAD_UBYTE_ADDR64, i32, az_extloadi8_constant>;
defm : MUBUFLoad_Pattern <BUFFER_LOAD_SSHORT_ADDR64, i32, sextloadi16_constant>;
defm : MUBUFLoad_Pattern <BUFFER_LOAD_USHORT_ADDR64, i32, az_extloadi16_constant>;
defm : MUBUFLoad_Pattern <BUFFER_LOAD_DWORD_ADDR64, i32, constant_load>;
defm : MUBUFLoad_Pattern <BUFFER_LOAD_DWORDX2_ADDR64, v2i32, constant_load>;
defm : MUBUFLoad_Pattern <BUFFER_LOAD_DWORDX4_ADDR64, v4i32, constant_load>;

class MUBUFScratchLoadPat <MUBUF Instr, ValueType vt, PatFrag ld> : Pat <
  (vt (ld (MUBUFScratch v4i32:$srsrc, i32:$vaddr,
                        i32:$soffset, u16imm:$offset))),
  (Instr $srsrc, $vaddr, $soffset, $offset, 0, 0, 0)
>;

def : MUBUFScratchLoadPat <BUFFER_LOAD_SBYTE_OFFEN, i32, sextloadi8_private>;
def : MUBUFScratchLoadPat <BUFFER_LOAD_UBYTE_OFFEN, i32, extloadi8_private>;
def : MUBUFScratchLoadPat <BUFFER_LOAD_SSHORT_OFFEN, i32, sextloadi16_private>;
def : MUBUFScratchLoadPat <BUFFER_LOAD_USHORT_OFFEN, i32, extloadi16_private>;
def : MUBUFScratchLoadPat <BUFFER_LOAD_DWORD_OFFEN, i32, load_private>;
def : MUBUFScratchLoadPat <BUFFER_LOAD_DWORDX2_OFFEN, v2i32, load_private>;
def : MUBUFScratchLoadPat <BUFFER_LOAD_DWORDX4_OFFEN, v4i32, load_private>;

// BUFFER_LOAD_DWORD*, addr64=0
multiclass MUBUF_Load_Dword <ValueType vt, MUBUF offset, MUBUF offen, MUBUF idxen,
                             MUBUF bothen> {

  def : Pat <
    (vt (int_SI_buffer_load_dword v4i32:$rsrc, (i32 imm), i32:$soffset,
                                  imm:$offset, 0, 0, imm:$glc, imm:$slc,
                                  imm:$tfe)),
    (offset $rsrc, (as_i16imm $offset), $soffset, (as_i1imm $glc),
            (as_i1imm $slc), (as_i1imm $tfe))
  >;

  def : Pat <
    (vt (int_SI_buffer_load_dword v4i32:$rsrc, i32:$vaddr, i32:$soffset,
                                  imm:$offset, 1, 0, imm:$glc, imm:$slc,
                                  imm:$tfe)),
    (offen $rsrc, $vaddr, $soffset, (as_i16imm $offset), (as_i1imm $glc), (as_i1imm $slc),
           (as_i1imm $tfe))
  >;

  def : Pat <
    (vt (int_SI_buffer_load_dword v4i32:$rsrc, i32:$vaddr, i32:$soffset,
                                  imm:$offset, 0, 1, imm:$glc, imm:$slc,
                                  imm:$tfe)),
    (idxen $rsrc, $vaddr, (as_i16imm $offset), $soffset, (as_i1imm $glc),
           (as_i1imm $slc), (as_i1imm $tfe))
  >;

  def : Pat <
    (vt (int_SI_buffer_load_dword v4i32:$rsrc, v2i32:$vaddr, i32:$soffset,
                                  imm, 1, 1, imm:$glc, imm:$slc,
                                  imm:$tfe)),
    (bothen $rsrc, $vaddr, $soffset, (as_i1imm $glc), (as_i1imm $slc),
            (as_i1imm $tfe))
  >;
}

defm : MUBUF_Load_Dword <i32, BUFFER_LOAD_DWORD_OFFSET, BUFFER_LOAD_DWORD_OFFEN,
                         BUFFER_LOAD_DWORD_IDXEN, BUFFER_LOAD_DWORD_BOTHEN>;
defm : MUBUF_Load_Dword <v2i32, BUFFER_LOAD_DWORDX2_OFFSET, BUFFER_LOAD_DWORDX2_OFFEN,
                         BUFFER_LOAD_DWORDX2_IDXEN, BUFFER_LOAD_DWORDX2_BOTHEN>;
defm : MUBUF_Load_Dword <v4i32, BUFFER_LOAD_DWORDX4_OFFSET, BUFFER_LOAD_DWORDX4_OFFEN,
                         BUFFER_LOAD_DWORDX4_IDXEN, BUFFER_LOAD_DWORDX4_BOTHEN>;

class MUBUFScratchStorePat <MUBUF Instr, ValueType vt, PatFrag st> : Pat <
  (st vt:$value, (MUBUFAddr32 v4i32:$srsrc, i32:$vaddr, i32:$soffset,
                              u16imm:$offset, i1imm:$offen, i1imm:$idxen,
                              i1imm:$glc, i1imm:$slc, i1imm:$tfe)),
  (Instr $value, $srsrc, $vaddr, $soffset, $offset, $offen, $idxen,
         $glc, $slc, $tfe)
>;

def : MUBUFScratchStorePat <BUFFER_STORE_BYTE, i32, truncstorei8_private>;
def : MUBUFScratchStorePat <BUFFER_STORE_SHORT, i32, truncstorei16_private>;
def : MUBUFScratchStorePat <BUFFER_STORE_DWORD, i32, store_private>;
def : MUBUFScratchStorePat <BUFFER_STORE_DWORDX2, v2i32, store_private>;
def : MUBUFScratchStorePat <BUFFER_STORE_DWORDX4, v4i32, store_private>;

/*
class MUBUFStore_Pattern <MUBUF Instr, ValueType vt, PatFrag st> : Pat <
  (st vt:$value, (MUBUFScratch v4i32:$srsrc, i64:$vaddr, u16imm:$offset)),
  (Instr $value, $srsrc, $vaddr, $offset)
>;

def : MUBUFStore_Pattern <BUFFER_STORE_BYTE_ADDR64, i32, truncstorei8_private>;
def : MUBUFStore_Pattern <BUFFER_STORE_SHORT_ADDR64, i32, truncstorei16_private>;
def : MUBUFStore_Pattern <BUFFER_STORE_DWORD_ADDR64, i32, store_private>;
def : MUBUFStore_Pattern <BUFFER_STORE_DWORDX2_ADDR64, v2i32, store_private>;
def : MUBUFStore_Pattern <BUFFER_STORE_DWORDX4_ADDR64, v4i32, store_private>;

*/

//===----------------------------------------------------------------------===//
// MTBUF Patterns
//===----------------------------------------------------------------------===//

// TBUFFER_STORE_FORMAT_*, addr64=0
class MTBUF_StoreResource <ValueType vt, int num_channels, MTBUF opcode> : Pat<
  (SItbuffer_store v4i32:$rsrc, vt:$vdata, num_channels, i32:$vaddr,
                   i32:$soffset, imm:$inst_offset, imm:$dfmt,
                   imm:$nfmt, imm:$offen, imm:$idxen,
                   imm:$glc, imm:$slc, imm:$tfe),
  (opcode
    $vdata, (as_i16imm $inst_offset), (as_i1imm $offen), (as_i1imm $idxen),
    (as_i1imm $glc), 0, (as_i8imm $dfmt), (as_i8imm $nfmt), $vaddr, $rsrc,
    (as_i1imm $slc), (as_i1imm $tfe), $soffset)
>;

def : MTBUF_StoreResource <i32, 1, TBUFFER_STORE_FORMAT_X>;
def : MTBUF_StoreResource <v2i32, 2, TBUFFER_STORE_FORMAT_XY>;
def : MTBUF_StoreResource <v4i32, 3, TBUFFER_STORE_FORMAT_XYZ>;
def : MTBUF_StoreResource <v4i32, 4, TBUFFER_STORE_FORMAT_XYZW>;

let SubtargetPredicate = isCI in {

// Sea island new arithmetic instructinos
defm V_TRUNC_F64 : VOP1Inst <0x00000017, "V_TRUNC_F64",
  VOP_F64_F64, ftrunc
>;
defm V_CEIL_F64 : VOP1Inst <0x00000018, "V_CEIL_F64",
  VOP_F64_F64, fceil
>;
defm V_FLOOR_F64 : VOP1Inst <0x0000001A, "V_FLOOR_F64",
  VOP_F64_F64, ffloor
>;
defm V_RNDNE_F64 : VOP1Inst <0x00000019, "V_RNDNE_F64",
  VOP_F64_F64, frint
>;

defm V_QSAD_PK_U16_U8 : VOP3Inst <0x00000173, "V_QSAD_PK_U16_U8",
  VOP_I32_I32_I32
>;
defm V_MQSAD_U16_U8 : VOP3Inst <0x000000172, "V_MQSAD_U16_U8",
  VOP_I32_I32_I32
>;
defm V_MQSAD_U32_U8 : VOP3Inst <0x00000175, "V_MQSAD_U32_U8",
  VOP_I32_I32_I32
>;
defm V_MAD_U64_U32 : VOP3Inst <0x00000176, "V_MAD_U64_U32",
  VOP_I64_I32_I32_I64
>;

// XXX - Does this set VCC?
defm V_MAD_I64_I32 : VOP3Inst <0x00000177, "V_MAD_I64_I32",
  VOP_I64_I32_I32_I64
>;

// Remaining instructions:
// FLAT_*
// S_CBRANCH_CDBGUSER
// S_CBRANCH_CDBGSYS
// S_CBRANCH_CDBGSYS_OR_USER
// S_CBRANCH_CDBGSYS_AND_USER
// S_DCACHE_INV_VOL
// V_EXP_LEGACY_F32
// V_LOG_LEGACY_F32
// DS_NOP
// DS_GWS_SEMA_RELEASE_ALL
// DS_WRAP_RTN_B32
// DS_CNDXCHG32_RTN_B64
// DS_WRITE_B96
// DS_WRITE_B128
// DS_CONDXCHG32_RTN_B128
// DS_READ_B96
// DS_READ_B128
// BUFFER_LOAD_DWORDX3
// BUFFER_STORE_DWORDX3

} // End iSCI


/********** ====================== **********/
/**********   Indirect adressing   **********/
/********** ====================== **********/

multiclass SI_INDIRECT_Pattern <ValueType vt, ValueType eltvt, SI_INDIRECT_DST IndDst> {

  // 1. Extract with offset
  def : Pat<
    (vector_extract vt:$vec, (add i32:$idx, imm:$off)),
    (eltvt (SI_INDIRECT_SRC (IMPLICIT_DEF), $vec, $idx, imm:$off))
  >;

  // 2. Extract without offset
  def : Pat<
    (vector_extract vt:$vec, i32:$idx),
    (eltvt (SI_INDIRECT_SRC (IMPLICIT_DEF), $vec, $idx, 0))
  >;

  // 3. Insert with offset
  def : Pat<
    (vector_insert vt:$vec, eltvt:$val, (add i32:$idx, imm:$off)),
    (IndDst (IMPLICIT_DEF), $vec, $idx, imm:$off, $val)
  >;

  // 4. Insert without offset
  def : Pat<
    (vector_insert vt:$vec, eltvt:$val, i32:$idx),
    (IndDst (IMPLICIT_DEF), $vec, $idx, 0, $val)
  >;
}

defm : SI_INDIRECT_Pattern <v2f32, f32, SI_INDIRECT_DST_V2>;
defm : SI_INDIRECT_Pattern <v4f32, f32, SI_INDIRECT_DST_V4>;
defm : SI_INDIRECT_Pattern <v8f32, f32, SI_INDIRECT_DST_V8>;
defm : SI_INDIRECT_Pattern <v16f32, f32, SI_INDIRECT_DST_V16>;

defm : SI_INDIRECT_Pattern <v2i32, i32, SI_INDIRECT_DST_V2>;
defm : SI_INDIRECT_Pattern <v4i32, i32, SI_INDIRECT_DST_V4>;
defm : SI_INDIRECT_Pattern <v8i32, i32, SI_INDIRECT_DST_V8>;
defm : SI_INDIRECT_Pattern <v16i32, i32, SI_INDIRECT_DST_V16>;

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

def : Pat<(i32 (sext_inreg i32:$src, i1)),
  (S_BFE_I32 i32:$src, 65536)>; // 0 | 1 << 16

// TODO: Match 64-bit BFE. SI has a 64-bit BFE, but it's scalar only so it
// might not be worth the effort, and will need to expand to shifts when
// fixing SGPR copies.

// Handle sext_inreg in i64
def : Pat <
  (i64 (sext_inreg i64:$src, i1)),
  (INSERT_SUBREG (INSERT_SUBREG (i64 (IMPLICIT_DEF)),
    (S_BFE_I32 (EXTRACT_SUBREG i64:$src, sub0), 65536), sub0), // 0 | 1 << 16
    (S_MOV_B32 -1), sub1)
>;

def : Pat <
  (i64 (sext_inreg i64:$src, i8)),
  (INSERT_SUBREG (INSERT_SUBREG (i64 (IMPLICIT_DEF)),
    (S_SEXT_I32_I8 (EXTRACT_SUBREG i64:$src, sub0)), sub0),
    (S_MOV_B32 -1), sub1)
>;

def : Pat <
  (i64 (sext_inreg i64:$src, i16)),
  (INSERT_SUBREG (INSERT_SUBREG (i64 (IMPLICIT_DEF)),
    (S_SEXT_I32_I16 (EXTRACT_SUBREG i64:$src, sub0)), sub0),
    (S_MOV_B32 -1), sub1)
>;

class ZExt_i64_i32_Pat <SDNode ext> : Pat <
  (i64 (ext i32:$src)),
  (INSERT_SUBREG (INSERT_SUBREG (i64 (IMPLICIT_DEF)), $src, sub0),
    (S_MOV_B32 0), sub1)
>;

class ZExt_i64_i1_Pat <SDNode ext> : Pat <
  (i64 (ext i1:$src)),
  (INSERT_SUBREG
    (INSERT_SUBREG (i64 (IMPLICIT_DEF)),
      (V_CNDMASK_B32_e64 (i32 0), (i32 1), $src), sub0),
    (S_MOV_B32 0), sub1)
>;


def : ZExt_i64_i32_Pat<zext>;
def : ZExt_i64_i32_Pat<anyext>;
def : ZExt_i64_i1_Pat<zext>;
def : ZExt_i64_i1_Pat<anyext>;

def : Pat <
  (i64 (sext i32:$src)),
    (INSERT_SUBREG
      (INSERT_SUBREG (i64 (IMPLICIT_DEF)), $src, sub0),
      (S_ASHR_I32 $src, 31), sub1)
>;

def : Pat <
  (i64 (sext i1:$src)),
  (INSERT_SUBREG
    (INSERT_SUBREG
      (i64 (IMPLICIT_DEF)),
      (V_CNDMASK_B32_e64 0, -1, $src), sub0),
    (V_CNDMASK_B32_e64 0, -1, $src), sub1)
>;

def : Pat <
  (f32 (sint_to_fp i1:$src)),
  (V_CNDMASK_B32_e64 (i32 0), CONST.FP32_NEG_ONE, $src)
>;

def : Pat <
  (f32 (uint_to_fp i1:$src)),
  (V_CNDMASK_B32_e64 (i32 0), CONST.FP32_ONE, $src)
>;

def : Pat <
  (f64 (sint_to_fp i1:$src)),
    (V_CVT_F64_I32_e32 (V_CNDMASK_B32_e64 (i32 0), (i32 -1), $src))
>;

def : Pat <
  (f64 (uint_to_fp i1:$src)),
  (V_CVT_F64_U32_e32 (V_CNDMASK_B32_e64 (i32 0), (i32 1), $src))
>;

//===----------------------------------------------------------------------===//
// Miscellaneous Patterns
//===----------------------------------------------------------------------===//

def : Pat <
  (i32 (trunc i64:$a)),
  (EXTRACT_SUBREG $a, sub0)
>;

def : Pat <
  (i1 (trunc i32:$a)),
  (V_CMP_EQ_I32_e64 (V_AND_B32_e32 (i32 1), $a), 1)
>;

//============================================================================//
// Miscellaneous Optimization Patterns
//============================================================================//

def : SHA256MaPattern <V_BFI_B32, V_XOR_B32_e32>;

} // End isSI predicate
