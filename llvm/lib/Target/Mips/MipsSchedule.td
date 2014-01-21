//===-- MipsSchedule.td - Mips Scheduling Definitions ------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Functional units across Mips chips sets. Based on GCC/Mips backend files.
//===----------------------------------------------------------------------===//
def ALU     : FuncUnit;
def IMULDIV : FuncUnit;

//===----------------------------------------------------------------------===//
// Instruction Itinerary classes used for Mips
//===----------------------------------------------------------------------===//
def IIAlu              : InstrItinClass;
def IILoad             : InstrItinClass;
def IIStore            : InstrItinClass;
def IIBranch           : InstrItinClass;
def IIFrecipFsqrtStep  : InstrItinClass;
def IIFLoad            : InstrItinClass;
def IIFStore           : InstrItinClass;
def IIFmoveC1          : InstrItinClass;
def IIPseudo           : InstrItinClass;

def II_ABS              : InstrItinClass;
def II_ADDI             : InstrItinClass;
def II_ADDIU            : InstrItinClass;
def II_ADDU             : InstrItinClass;
def II_ADD_D            : InstrItinClass;
def II_ADD_S            : InstrItinClass;
def II_AND              : InstrItinClass;
def II_ANDI             : InstrItinClass;
def II_CEIL             : InstrItinClass;
def II_CFC1             : InstrItinClass;
def II_CLO              : InstrItinClass;
def II_CLZ              : InstrItinClass;
def II_CTC1             : InstrItinClass;
def II_CVT              : InstrItinClass;
def II_C_CC_D           : InstrItinClass; // Any c.<cc>.d instruction
def II_C_CC_S           : InstrItinClass; // Any c.<cc>.s instruction
def II_DADDIU           : InstrItinClass;
def II_DADDU            : InstrItinClass;
def II_DDIV             : InstrItinClass;
def II_DDIVU            : InstrItinClass;
def II_DIV              : InstrItinClass;
def II_DIVU             : InstrItinClass;
def II_DIV_D            : InstrItinClass;
def II_DIV_S            : InstrItinClass;
def II_DMULT            : InstrItinClass;
def II_DMULTU           : InstrItinClass;
def II_DROTR            : InstrItinClass;
def II_DROTR32          : InstrItinClass;
def II_DROTRV           : InstrItinClass;
def II_DSLL             : InstrItinClass;
def II_DSLL32           : InstrItinClass;
def II_DSLLV            : InstrItinClass;
def II_DSRA             : InstrItinClass;
def II_DSRA32           : InstrItinClass;
def II_DSRAV            : InstrItinClass;
def II_DSRL             : InstrItinClass;
def II_DSRL32           : InstrItinClass;
def II_DSRLV            : InstrItinClass;
def II_DSUBU            : InstrItinClass;
def II_FLOOR            : InstrItinClass;
def II_LUI              : InstrItinClass;
def II_MADD             : InstrItinClass;
def II_MADDU            : InstrItinClass;
def II_MADD_D           : InstrItinClass;
def II_MADD_S           : InstrItinClass;
def II_MFHI_MFLO        : InstrItinClass; // mfhi and mflo
def II_MOVF             : InstrItinClass;
def II_MOVF_D           : InstrItinClass;
def II_MOVF_S           : InstrItinClass;
def II_MOVN             : InstrItinClass;
def II_MOVN_D           : InstrItinClass;
def II_MOVN_S           : InstrItinClass;
def II_MOVT             : InstrItinClass;
def II_MOVT_D           : InstrItinClass;
def II_MOVT_S           : InstrItinClass;
def II_MOVZ             : InstrItinClass;
def II_MOVZ_D           : InstrItinClass;
def II_MOVZ_S           : InstrItinClass;
def II_MOV_D            : InstrItinClass;
def II_MOV_S            : InstrItinClass;
def II_MSUB             : InstrItinClass;
def II_MSUBU            : InstrItinClass;
def II_MSUB_D           : InstrItinClass;
def II_MSUB_S           : InstrItinClass;
def II_MTHI_MTLO        : InstrItinClass; // mthi and mtlo
def II_MUL              : InstrItinClass;
def II_MULT             : InstrItinClass;
def II_MULTU            : InstrItinClass;
def II_MUL_D            : InstrItinClass;
def II_MUL_S            : InstrItinClass;
def II_NEG              : InstrItinClass;
def II_NMADD_D          : InstrItinClass;
def II_NMADD_S          : InstrItinClass;
def II_NMSUB_D          : InstrItinClass;
def II_NMSUB_S          : InstrItinClass;
def II_NOR              : InstrItinClass;
def II_OR               : InstrItinClass;
def II_ORI              : InstrItinClass;
def II_RDHWR            : InstrItinClass;
def II_ROTR             : InstrItinClass;
def II_ROTRV            : InstrItinClass;
def II_ROUND            : InstrItinClass;
def II_SEB              : InstrItinClass;
def II_SEH              : InstrItinClass;
def II_SLL              : InstrItinClass;
def II_SLLV             : InstrItinClass;
def II_SLTI_SLTIU       : InstrItinClass; // slti and sltiu
def II_SLT_SLTU         : InstrItinClass; // slt and sltu
def II_SQRT_D           : InstrItinClass;
def II_SQRT_S           : InstrItinClass;
def II_SRA              : InstrItinClass;
def II_SRAV             : InstrItinClass;
def II_SRL              : InstrItinClass;
def II_SRLV             : InstrItinClass;
def II_SUBU             : InstrItinClass;
def II_SUB_D            : InstrItinClass;
def II_SUB_S            : InstrItinClass;
def II_TRUNC            : InstrItinClass;
def II_XOR              : InstrItinClass;
def II_XORI             : InstrItinClass;

//===----------------------------------------------------------------------===//
// Mips Generic instruction itineraries.
//===----------------------------------------------------------------------===//
def MipsGenericItineraries : ProcessorItineraries<[ALU, IMULDIV], [], [
  InstrItinData<IIAlu              , [InstrStage<1,  [ALU]>]>,
  InstrItinData<II_ADDI            , [InstrStage<1,  [ALU]>]>,
  InstrItinData<II_ADDIU           , [InstrStage<1,  [ALU]>]>,
  InstrItinData<II_ADDU            , [InstrStage<1,  [ALU]>]>,
  InstrItinData<II_AND             , [InstrStage<1,  [ALU]>]>,
  InstrItinData<II_SLL             , [InstrStage<1,  [ALU]>]>,
  InstrItinData<II_SRA             , [InstrStage<1,  [ALU]>]>,
  InstrItinData<II_SRL             , [InstrStage<1,  [ALU]>]>,
  InstrItinData<II_ROTR            , [InstrStage<1,  [ALU]>]>,
  InstrItinData<II_SLLV            , [InstrStage<1,  [ALU]>]>,
  InstrItinData<II_SRAV            , [InstrStage<1,  [ALU]>]>,
  InstrItinData<II_SRLV            , [InstrStage<1,  [ALU]>]>,
  InstrItinData<II_ROTRV           , [InstrStage<1,  [ALU]>]>,
  InstrItinData<II_CLO             , [InstrStage<1,  [ALU]>]>,
  InstrItinData<II_CLZ             , [InstrStage<1,  [ALU]>]>,
  InstrItinData<II_DADDIU          , [InstrStage<1,  [ALU]>]>,
  InstrItinData<II_DADDU           , [InstrStage<1,  [ALU]>]>,
  InstrItinData<II_DSLL            , [InstrStage<1,  [ALU]>]>,
  InstrItinData<II_DSRL            , [InstrStage<1,  [ALU]>]>,
  InstrItinData<II_DSRA            , [InstrStage<1,  [ALU]>]>,
  InstrItinData<II_DSLLV           , [InstrStage<1,  [ALU]>]>,
  InstrItinData<II_DSRLV           , [InstrStage<1,  [ALU]>]>,
  InstrItinData<II_DSRAV           , [InstrStage<1,  [ALU]>]>,
  InstrItinData<II_DSUBU           , [InstrStage<1,  [ALU]>]>,
  InstrItinData<II_DROTR           , [InstrStage<1,  [ALU]>]>,
  InstrItinData<II_DROTRV          , [InstrStage<1,  [ALU]>]>,
  InstrItinData<II_LUI             , [InstrStage<1,  [ALU]>]>,
  InstrItinData<II_MOVF            , [InstrStage<1,  [ALU]>]>,
  InstrItinData<II_MOVN            , [InstrStage<1,  [ALU]>]>,
  InstrItinData<II_MOVN_S          , [InstrStage<1,  [ALU]>]>,
  InstrItinData<II_MOVN_D          , [InstrStage<1,  [ALU]>]>,
  InstrItinData<II_MOVT            , [InstrStage<1,  [ALU]>]>,
  InstrItinData<II_MOVZ            , [InstrStage<1,  [ALU]>]>,
  InstrItinData<II_NOR             , [InstrStage<1,  [ALU]>]>,
  InstrItinData<II_OR              , [InstrStage<1,  [ALU]>]>,
  InstrItinData<II_RDHWR           , [InstrStage<1,  [ALU]>]>,
  InstrItinData<II_SUBU            , [InstrStage<1,  [ALU]>]>,
  InstrItinData<II_XOR             , [InstrStage<1,  [ALU]>]>,
  InstrItinData<II_ANDI            , [InstrStage<1,  [ALU]>]>,
  InstrItinData<II_ORI             , [InstrStage<1,  [ALU]>]>,
  InstrItinData<II_XORI            , [InstrStage<1,  [ALU]>]>,
  InstrItinData<IILoad             , [InstrStage<3,  [ALU]>]>,
  InstrItinData<IIStore            , [InstrStage<1,  [ALU]>]>,
  InstrItinData<IIBranch           , [InstrStage<1,  [ALU]>]>,
  InstrItinData<II_DMULT           , [InstrStage<17, [IMULDIV]>]>,
  InstrItinData<II_DMULTU          , [InstrStage<17, [IMULDIV]>]>,
  InstrItinData<II_MADD            , [InstrStage<17, [IMULDIV]>]>,
  InstrItinData<II_MADDU           , [InstrStage<17, [IMULDIV]>]>,
  InstrItinData<II_MFHI_MFLO       , [InstrStage<1,  [IMULDIV]>]>,
  InstrItinData<II_MSUB            , [InstrStage<17, [IMULDIV]>]>,
  InstrItinData<II_MSUBU           , [InstrStage<17, [IMULDIV]>]>,
  InstrItinData<II_MTHI_MTLO       , [InstrStage<1,  [IMULDIV]>]>,
  InstrItinData<II_MUL             , [InstrStage<17, [IMULDIV]>]>,
  InstrItinData<II_MULT            , [InstrStage<17, [IMULDIV]>]>,
  InstrItinData<II_MULTU           , [InstrStage<17, [IMULDIV]>]>,
  InstrItinData<II_MSUB            , [InstrStage<17, [IMULDIV]>]>,
  InstrItinData<II_MSUBU           , [InstrStage<17, [IMULDIV]>]>,
  InstrItinData<II_DIV             , [InstrStage<38, [IMULDIV]>]>,
  InstrItinData<II_DIVU            , [InstrStage<38, [IMULDIV]>]>,
  InstrItinData<II_DDIV            , [InstrStage<38, [IMULDIV]>]>,
  InstrItinData<II_DDIVU           , [InstrStage<38, [IMULDIV]>]>,
  InstrItinData<II_CEIL            , [InstrStage<1,  [ALU]>]>,
  InstrItinData<II_CVT             , [InstrStage<1,  [ALU]>]>,
  InstrItinData<II_ABS             , [InstrStage<1,  [ALU]>]>,
  InstrItinData<II_FLOOR           , [InstrStage<1,  [ALU]>]>,
  InstrItinData<II_NEG             , [InstrStage<1,  [ALU]>]>,
  InstrItinData<II_ROUND           , [InstrStage<1,  [ALU]>]>,
  InstrItinData<II_TRUNC           , [InstrStage<1,  [ALU]>]>,
  InstrItinData<II_MOV_D           , [InstrStage<2,  [ALU]>]>,
  InstrItinData<II_MOV_S           , [InstrStage<2,  [ALU]>]>,
  InstrItinData<II_CFC1            , [InstrStage<2,  [ALU]>]>,
  InstrItinData<II_CTC1            , [InstrStage<2,  [ALU]>]>,
  InstrItinData<II_MOVF_D          , [InstrStage<2,  [ALU]>]>,
  InstrItinData<II_MOVF_S          , [InstrStage<2,  [ALU]>]>,
  InstrItinData<II_MOVT_D          , [InstrStage<2,  [ALU]>]>,
  InstrItinData<II_MOVT_S          , [InstrStage<2,  [ALU]>]>,
  InstrItinData<II_MOVZ_D          , [InstrStage<2,  [ALU]>]>,
  InstrItinData<II_MOVZ_S          , [InstrStage<2,  [ALU]>]>,
  InstrItinData<II_C_CC_S          , [InstrStage<3,  [ALU]>]>,
  InstrItinData<II_C_CC_D          , [InstrStage<3,  [ALU]>]>,
  InstrItinData<II_ADD_D           , [InstrStage<4,  [ALU]>]>,
  InstrItinData<II_ADD_S           , [InstrStage<4,  [ALU]>]>,
  InstrItinData<II_SUB_D           , [InstrStage<4,  [ALU]>]>,
  InstrItinData<II_SUB_S           , [InstrStage<4,  [ALU]>]>,
  InstrItinData<II_MUL_S           , [InstrStage<7,  [ALU]>]>,
  InstrItinData<II_MADD_S          , [InstrStage<7,  [ALU]>]>,
  InstrItinData<II_MSUB_S          , [InstrStage<7,  [ALU]>]>,
  InstrItinData<II_NMADD_S         , [InstrStage<7,  [ALU]>]>,
  InstrItinData<II_NMSUB_S         , [InstrStage<7,  [ALU]>]>,
  InstrItinData<II_MUL_D           , [InstrStage<8,  [ALU]>]>,
  InstrItinData<II_MADD_D          , [InstrStage<8,  [ALU]>]>,
  InstrItinData<II_MSUB_D          , [InstrStage<8,  [ALU]>]>,
  InstrItinData<II_NMADD_D         , [InstrStage<8,  [ALU]>]>,
  InstrItinData<II_NMSUB_D         , [InstrStage<8,  [ALU]>]>,
  InstrItinData<II_DIV_S           , [InstrStage<23, [ALU]>]>,
  InstrItinData<II_DIV_D           , [InstrStage<36, [ALU]>]>,
  InstrItinData<II_SQRT_S          , [InstrStage<54, [ALU]>]>,
  InstrItinData<II_SQRT_D          , [InstrStage<12, [ALU]>]>,
  InstrItinData<IIFrecipFsqrtStep  , [InstrStage<5,  [ALU]>]>,
  InstrItinData<IIFLoad            , [InstrStage<3,  [ALU]>]>,
  InstrItinData<IIFStore           , [InstrStage<1,  [ALU]>]>,
  InstrItinData<IIFmoveC1          , [InstrStage<2,  [ALU]>]>
]>;
