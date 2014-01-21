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
def IIFmove            : InstrItinClass;
def IIFcmp             : InstrItinClass;
def IIFadd             : InstrItinClass;
def IIFmulSingle       : InstrItinClass;
def IIFmulDouble       : InstrItinClass;
def IIFdivSingle       : InstrItinClass;
def IIFdivDouble       : InstrItinClass;
def IIFsqrtSingle      : InstrItinClass;
def IIFsqrtDouble      : InstrItinClass;
def IIFrecipFsqrtStep  : InstrItinClass;
def IIFLoad            : InstrItinClass;
def IIFStore           : InstrItinClass;
def IIFmoveC1          : InstrItinClass;
def IIPseudo           : InstrItinClass;

def II_ABS              : InstrItinClass;
def II_ADDI             : InstrItinClass;
def II_ADDIU            : InstrItinClass;
def II_ADDU             : InstrItinClass;
def II_AND              : InstrItinClass;
def II_ANDI             : InstrItinClass;
def II_CEIL             : InstrItinClass;
def II_CLO              : InstrItinClass;
def II_CLZ              : InstrItinClass;
def II_CVT              : InstrItinClass;
def II_DADDIU           : InstrItinClass;
def II_DADDU            : InstrItinClass;
def II_DDIV             : InstrItinClass;
def II_DDIVU            : InstrItinClass;
def II_DIV              : InstrItinClass;
def II_DIVU             : InstrItinClass;
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
def II_MFHI_MFLO        : InstrItinClass; // mfhi and mflo
def II_MOVF             : InstrItinClass;
def II_MOVN             : InstrItinClass;
def II_MOVT             : InstrItinClass;
def II_MOVZ             : InstrItinClass;
def II_MSUB             : InstrItinClass;
def II_MSUBU            : InstrItinClass;
def II_MTHI_MTLO        : InstrItinClass; // mthi and mtlo
def II_MUL              : InstrItinClass;
def II_MULT             : InstrItinClass;
def II_MULTU            : InstrItinClass;
def II_NEG              : InstrItinClass;
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
def II_SRA              : InstrItinClass;
def II_SRAV             : InstrItinClass;
def II_SRL              : InstrItinClass;
def II_SRLV             : InstrItinClass;
def II_SUBU             : InstrItinClass;
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
  InstrItinData<IIFmove            , [InstrStage<2,  [ALU]>]>,
  InstrItinData<IIFcmp             , [InstrStage<3,  [ALU]>]>,
  InstrItinData<IIFadd             , [InstrStage<4,  [ALU]>]>,
  InstrItinData<IIFmulSingle       , [InstrStage<7,  [ALU]>]>,
  InstrItinData<IIFmulDouble       , [InstrStage<8,  [ALU]>]>,
  InstrItinData<IIFdivSingle       , [InstrStage<23, [ALU]>]>,
  InstrItinData<IIFdivDouble       , [InstrStage<36, [ALU]>]>,
  InstrItinData<IIFsqrtSingle      , [InstrStage<54, [ALU]>]>,
  InstrItinData<IIFsqrtDouble      , [InstrStage<12, [ALU]>]>,
  InstrItinData<IIFrecipFsqrtStep  , [InstrStage<5,  [ALU]>]>,
  InstrItinData<IIFLoad            , [InstrStage<3,  [ALU]>]>,
  InstrItinData<IIFStore           , [InstrStage<1,  [ALU]>]>,
  InstrItinData<IIFmoveC1          , [InstrStage<2,  [ALU]>]>
]>;
