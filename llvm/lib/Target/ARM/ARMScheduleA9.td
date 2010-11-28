//=- ARMScheduleA9.td - ARM Cortex-A9 Scheduling Definitions -*- tablegen -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the itinerary class data for the ARM Cortex A9 processors.
//
//===----------------------------------------------------------------------===//

//
// Ad-hoc scheduling information derived from pretty vague "Cortex-A9 Technical
// Reference Manual".
//
// Functional units
def A9_Issue0  : FuncUnit; // Issue 0
def A9_Issue1  : FuncUnit; // Issue 1
def A9_Branch  : FuncUnit; // Branch
def A9_ALU0    : FuncUnit; // ALU / MUL pipeline 0
def A9_ALU1    : FuncUnit; // ALU pipeline 1
def A9_AGU     : FuncUnit; // Address generation unit for ld / st
def A9_NPipe   : FuncUnit; // NEON pipeline
def A9_MUX0    : FuncUnit; // AGU + NEON/FPU multiplexer
def A9_LSUnit  : FuncUnit; // L/S Unit
def A9_DRegsVFP: FuncUnit; // FP register set, VFP side
def A9_DRegsN  : FuncUnit; // FP register set, NEON side

// Bypasses
def A9_LdBypass : Bypass;

def CortexA9Itineraries : ProcessorItineraries<
  [A9_Issue0, A9_Issue1, A9_Branch, A9_ALU0, A9_ALU1, A9_AGU, A9_NPipe, A9_MUX0,
   A9_LSUnit, A9_DRegsVFP, A9_DRegsN],
  [A9_LdBypass], [
  // Two fully-pipelined integer ALU pipelines

  //
  // Move instructions, unconditional
  InstrItinData<IIC_iMOVi   , [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_ALU0, A9_ALU1]>], [1]>,
  InstrItinData<IIC_iMOVr   , [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_ALU0, A9_ALU1]>], [1, 1]>,
  InstrItinData<IIC_iMOVsi  , [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_ALU0, A9_ALU1]>], [1, 1]>,
  InstrItinData<IIC_iMOVsr  , [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<2, [A9_ALU0, A9_ALU1]>], [2, 1, 1]>,
  InstrItinData<IIC_iMOVix2 , [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_ALU0, A9_ALU1]>,
                               InstrStage<1, [A9_ALU0, A9_ALU1]>], [2]>,
  //
  // MVN instructions
  InstrItinData<IIC_iMVNi   , [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_ALU0, A9_ALU1]>],
                              [1]>,
  InstrItinData<IIC_iMVNr   , [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_ALU0, A9_ALU1]>],
                              [1, 1], [NoBypass, A9_LdBypass]>,
  InstrItinData<IIC_iMVNsi  , [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<2, [A9_ALU0, A9_ALU1]>],
                              [2, 1]>,
  InstrItinData<IIC_iMVNsr  , [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<3, [A9_ALU0, A9_ALU1]>],
                              [3, 1, 1]>,
  //
  // No operand cycles
  InstrItinData<IIC_iALUx   , [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_ALU0, A9_ALU1]>]>,
  //
  // Binary Instructions that produce a result
  InstrItinData<IIC_iALUi , [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                             InstrStage<1, [A9_ALU0, A9_ALU1]>],
                            [1, 1], [NoBypass, A9_LdBypass]>,
  InstrItinData<IIC_iALUr , [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                             InstrStage<1, [A9_ALU0, A9_ALU1]>],
                            [1, 1, 1], [NoBypass, A9_LdBypass, A9_LdBypass]>,
  InstrItinData<IIC_iALUsi, [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                             InstrStage<2, [A9_ALU0, A9_ALU1]>],
                            [2, 1, 1], [NoBypass, A9_LdBypass, NoBypass]>,
  InstrItinData<IIC_iALUsir,[InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                             InstrStage<2, [A9_ALU0, A9_ALU1]>],
                            [2, 1, 1], [NoBypass, NoBypass, A9_LdBypass]>,
  InstrItinData<IIC_iALUsr, [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                             InstrStage<3, [A9_ALU0, A9_ALU1]>],
                            [3, 1, 1, 1],
                            [NoBypass, A9_LdBypass, NoBypass, NoBypass]>,
  //
  // Bitwise Instructions that produce a result
  InstrItinData<IIC_iBITi , [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                             InstrStage<1, [A9_ALU0, A9_ALU1]>], [1, 1]>,
  InstrItinData<IIC_iBITr , [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                             InstrStage<1, [A9_ALU0, A9_ALU1]>], [1, 1, 1]>,
  InstrItinData<IIC_iBITsi, [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                             InstrStage<2, [A9_ALU0, A9_ALU1]>], [2, 1, 1]>,
  InstrItinData<IIC_iBITsr, [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                             InstrStage<3, [A9_ALU0, A9_ALU1]>], [3, 1, 1, 1]>,
  //
  // Unary Instructions that produce a result

  // CLZ, RBIT, etc.
  InstrItinData<IIC_iUNAr , [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                             InstrStage<1, [A9_ALU0, A9_ALU1]>], [1, 1]>,

  // BFC, BFI, UBFX, SBFX
  InstrItinData<IIC_iUNAsi, [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                             InstrStage<2, [A9_ALU0, A9_ALU1]>], [2, 1]>,

  //
  // Zero and sign extension instructions
  InstrItinData<IIC_iEXTr , [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                             InstrStage<1, [A9_ALU0, A9_ALU1]>], [2, 1]>,
  InstrItinData<IIC_iEXTAr, [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                             InstrStage<2, [A9_ALU0, A9_ALU1]>], [3, 1, 1]>,
  InstrItinData<IIC_iEXTAsr,[InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                             InstrStage<3, [A9_ALU0, A9_ALU1]>], [3, 1, 1, 1]>,
  //
  // Compare instructions
  InstrItinData<IIC_iCMPi   , [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_ALU0, A9_ALU1]>],
                               [1], [A9_LdBypass]>,
  InstrItinData<IIC_iCMPr   , [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_ALU0, A9_ALU1]>],
                               [1, 1], [A9_LdBypass, A9_LdBypass]>,
  InstrItinData<IIC_iCMPsi  , [InstrStage<2, [A9_ALU0, A9_ALU1]>],
                                [1, 1], [A9_LdBypass, NoBypass]>,
  InstrItinData<IIC_iCMPsr  , [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<3, [A9_ALU0, A9_ALU1]>],
                              [1, 1, 1], [A9_LdBypass, NoBypass, NoBypass]>,
  //
  // Test instructions
  InstrItinData<IIC_iTSTi   , [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_ALU0, A9_ALU1]>], [1]>,
  InstrItinData<IIC_iTSTr   , [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_ALU0, A9_ALU1]>], [1, 1]>,
  InstrItinData<IIC_iTSTsi  , [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<2, [A9_ALU0, A9_ALU1]>], [1, 1]>,
  InstrItinData<IIC_iTSTsr  , [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<3, [A9_ALU0, A9_ALU1]>], [1, 1, 1]>,
  //
  // Move instructions, conditional
  // FIXME: Correctly model the extra input dep on the destination.
  InstrItinData<IIC_iCMOVi  , [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_ALU0, A9_ALU1]>], [1]>,
  InstrItinData<IIC_iCMOVr  , [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_ALU0, A9_ALU1]>], [1, 1]>,
  InstrItinData<IIC_iCMOVsi , [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_ALU0, A9_ALU1]>], [1, 1]>,
  InstrItinData<IIC_iCMOVsr , [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<2, [A9_ALU0, A9_ALU1]>], [2, 1, 1]>,
  InstrItinData<IIC_iCMOVix2, [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_ALU0, A9_ALU1]>,
                               InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_ALU0, A9_ALU1]>], [2]>,

  // Integer multiply pipeline
  //
  InstrItinData<IIC_iMUL16  , [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<2, [A9_ALU0]>], [3, 1, 1]>,
  InstrItinData<IIC_iMAC16  , [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<2, [A9_ALU0]>],
                              [3, 1, 1, 1]>,
  InstrItinData<IIC_iMUL32  , [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<2, [A9_ALU0]>], [4, 1, 1]>,
  InstrItinData<IIC_iMAC32  , [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<2, [A9_ALU0]>],
                              [4, 1, 1, 1]>,
  InstrItinData<IIC_iMUL64  , [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<3, [A9_ALU0]>], [4, 5, 1, 1]>,
  InstrItinData<IIC_iMAC64  , [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<3, [A9_ALU0]>],
                              [4, 5, 1, 1]>,
  // Integer load pipeline
  // FIXME: The timings are some rough approximations
  //
  // Immediate offset
  InstrItinData<IIC_iLoad_i   , [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                                 InstrStage<1, [A9_MUX0], 0>,
                                 InstrStage<1, [A9_AGU], 0>,
                                 InstrStage<1, [A9_LSUnit]>],
                                [3, 1], [A9_LdBypass]>,
  InstrItinData<IIC_iLoad_bh_i, [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                                 InstrStage<1, [A9_MUX0], 0>,
                                 InstrStage<2, [A9_AGU], 0>,
                                 InstrStage<1, [A9_LSUnit]>],
                                [4, 1], [A9_LdBypass]>,
  // FIXME: If address is 64-bit aligned, AGU cycles is 1.
  InstrItinData<IIC_iLoad_d_i , [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                                 InstrStage<1, [A9_MUX0], 0>,
                                 InstrStage<2, [A9_AGU], 0>,
                                 InstrStage<1, [A9_LSUnit]>],
                                [3, 3, 1], [A9_LdBypass]>,
  //
  // Register offset
  InstrItinData<IIC_iLoad_r   , [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                                 InstrStage<1, [A9_MUX0], 0>,
                                 InstrStage<1, [A9_AGU], 0>,
                                 InstrStage<1, [A9_LSUnit]>],
                                [3, 1, 1], [A9_LdBypass]>,
  InstrItinData<IIC_iLoad_bh_r, [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                                 InstrStage<1, [A9_MUX0], 0>,
                                 InstrStage<2, [A9_AGU], 0>,
                                 InstrStage<1, [A9_LSUnit]>],
                                [4, 1, 1], [A9_LdBypass]>,
  InstrItinData<IIC_iLoad_d_r , [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                                 InstrStage<1, [A9_MUX0], 0>,
                                 InstrStage<2, [A9_AGU], 0>,
                                 InstrStage<1, [A9_LSUnit]>],
                                [3, 3, 1, 1], [A9_LdBypass]>,
  //
  // Scaled register offset
  InstrItinData<IIC_iLoad_si  , [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                                 InstrStage<1, [A9_MUX0], 0>,
                                 InstrStage<1, [A9_AGU], 0>,
                                 InstrStage<1, [A9_LSUnit], 0>],
                                [4, 1, 1], [A9_LdBypass]>,
  InstrItinData<IIC_iLoad_bh_si,[InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                                 InstrStage<1, [A9_MUX0], 0>,
                                 InstrStage<2, [A9_AGU], 0>,
                                 InstrStage<1, [A9_LSUnit]>],
                                [5, 1, 1], [A9_LdBypass]>,
  //
  // Immediate offset with update
  InstrItinData<IIC_iLoad_iu  , [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                                 InstrStage<1, [A9_MUX0], 0>,
                                 InstrStage<1, [A9_AGU], 0>,
                                 InstrStage<1, [A9_LSUnit]>],
                                [3, 2, 1], [A9_LdBypass]>,
  InstrItinData<IIC_iLoad_bh_iu,[InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                                 InstrStage<1, [A9_MUX0], 0>,
                                 InstrStage<2, [A9_AGU], 0>,
                                 InstrStage<1, [A9_LSUnit]>],
                                [4, 3, 1], [A9_LdBypass]>,
  //
  // Register offset with update
  InstrItinData<IIC_iLoad_ru  , [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                                 InstrStage<1, [A9_MUX0], 0>,
                                 InstrStage<1, [A9_AGU], 0>,
                                 InstrStage<1, [A9_LSUnit]>],
                                [3, 2, 1, 1], [A9_LdBypass]>,
  InstrItinData<IIC_iLoad_bh_ru,[InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                                 InstrStage<1, [A9_MUX0], 0>,
                                 InstrStage<2, [A9_AGU], 0>,
                                 InstrStage<1, [A9_LSUnit]>],
                                [4, 3, 1, 1], [A9_LdBypass]>,
  InstrItinData<IIC_iLoad_d_ru, [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                                 InstrStage<1, [A9_MUX0], 0>,
                                 InstrStage<2, [A9_AGU], 0>,
                                 InstrStage<1, [A9_LSUnit]>],
                                [3, 3, 1, 1], [A9_LdBypass]>,
  //
  // Scaled register offset with update
  InstrItinData<IIC_iLoad_siu , [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                                 InstrStage<1, [A9_MUX0], 0>,
                                 InstrStage<1, [A9_AGU], 0>,
                                 InstrStage<1, [A9_LSUnit]>],
                                [4, 3, 1, 1], [A9_LdBypass]>,
  InstrItinData<IIC_iLoad_bh_siu,[InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                                  InstrStage<1, [A9_MUX0], 0>,
                                  InstrStage<2, [A9_AGU], 0>,
                                  InstrStage<1, [A9_LSUnit]>],
                                 [5, 4, 1, 1], [A9_LdBypass]>,
  //
  // Load multiple, def is the 5th operand.
  // FIXME: This assumes 3 to 4 registers.
  InstrItinData<IIC_iLoad_m  , [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                                InstrStage<1, [A9_MUX0], 0>,
                                InstrStage<2, [A9_AGU], 1>,
                                InstrStage<2, [A9_LSUnit]>],
                               [1, 1, 1, 1, 3],
                         [NoBypass, NoBypass, NoBypass, NoBypass, A9_LdBypass]>,
  //
  // Load multiple + update, defs are the 1st and 5th operands.
  InstrItinData<IIC_iLoad_mu , [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                                InstrStage<1, [A9_MUX0], 0>,
                                InstrStage<2, [A9_AGU], 1>,
                                InstrStage<2, [A9_LSUnit]>],
                               [2, 1, 1, 1, 3],
                         [NoBypass, NoBypass, NoBypass, NoBypass, A9_LdBypass]>,
  //
  // Load multiple plus branch
  InstrItinData<IIC_iLoad_mBr, [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                                InstrStage<1, [A9_MUX0], 0>,
                                InstrStage<1, [A9_AGU], 1>,
                                InstrStage<2, [A9_LSUnit]>,
                                InstrStage<1, [A9_Branch]>],
                               [1, 2, 1, 1, 3],
                         [NoBypass, NoBypass, NoBypass, NoBypass, A9_LdBypass]>,
  //
  // Pop, def is the 3rd operand.
  InstrItinData<IIC_iPop  ,    [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                                InstrStage<1, [A9_MUX0], 0>,
                                InstrStage<2, [A9_AGU], 1>,
                                InstrStage<2, [A9_LSUnit]>],
                               [1, 1, 3],
                               [NoBypass, NoBypass, A9_LdBypass]>,
  //
  // Pop + branch, def is the 3rd operand.
  InstrItinData<IIC_iPop_Br,   [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                                InstrStage<1, [A9_MUX0], 0>,
                                InstrStage<2, [A9_AGU], 1>,
                                InstrStage<2, [A9_LSUnit]>,
                                InstrStage<1, [A9_Branch]>],
                               [1, 1, 3],
                               [NoBypass, NoBypass, A9_LdBypass]>,

  //
  // iLoadi + iALUr for t2LDRpci_pic.
  InstrItinData<IIC_iLoadiALU, [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                                InstrStage<1, [A9_MUX0], 0>,
                                InstrStage<1, [A9_AGU], 0>,
                                InstrStage<1, [A9_LSUnit]>,
                                InstrStage<1, [A9_ALU0, A9_ALU1]>],
                               [2, 1]>,

  // Integer store pipeline
  ///
  // Immediate offset
  InstrItinData<IIC_iStore_i  , [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                                 InstrStage<1, [A9_MUX0], 0>,
                                 InstrStage<1, [A9_AGU], 0>,
                                 InstrStage<1, [A9_LSUnit]>], [1, 1]>,
  InstrItinData<IIC_iStore_bh_i,[InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                                 InstrStage<1, [A9_MUX0], 0>,
                                 InstrStage<2, [A9_AGU], 1>,
                                 InstrStage<1, [A9_LSUnit]>], [1, 1]>,
  // FIXME: If address is 64-bit aligned, AGU cycles is 1.
  InstrItinData<IIC_iStore_d_i, [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                                 InstrStage<1, [A9_MUX0], 0>,
                                 InstrStage<2, [A9_AGU], 1>,
                                 InstrStage<1, [A9_LSUnit]>], [1, 1]>,
  //
  // Register offset
  InstrItinData<IIC_iStore_r  , [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                                 InstrStage<1, [A9_MUX0], 0>,
                                 InstrStage<1, [A9_AGU], 0>,
                                 InstrStage<1, [A9_LSUnit]>], [1, 1, 1]>,
  InstrItinData<IIC_iStore_bh_r,[InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                                 InstrStage<1, [A9_MUX0], 0>,
                                 InstrStage<2, [A9_AGU], 1>,
                                 InstrStage<1, [A9_LSUnit]>], [1, 1, 1]>,
  InstrItinData<IIC_iStore_d_r, [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                                 InstrStage<1, [A9_MUX0], 0>,
                                 InstrStage<2, [A9_AGU], 1>,
                                 InstrStage<1, [A9_LSUnit]>], [1, 1, 1]>,
  //
  // Scaled register offset
  InstrItinData<IIC_iStore_si ,  [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                                  InstrStage<1, [A9_MUX0], 0>,
                                  InstrStage<1, [A9_AGU], 0>,
                                  InstrStage<1, [A9_LSUnit]>], [1, 1, 1]>,
  InstrItinData<IIC_iStore_bh_si,[InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                                  InstrStage<1, [A9_MUX0], 0>,
                                  InstrStage<2, [A9_AGU], 1>,
                                  InstrStage<1, [A9_LSUnit]>], [1, 1, 1]>,
  //
  // Immediate offset with update
  InstrItinData<IIC_iStore_iu ,  [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                                  InstrStage<1, [A9_MUX0], 0>,
                                  InstrStage<1, [A9_AGU], 0>,
                                  InstrStage<1, [A9_LSUnit]>], [2, 1, 1]>,
  InstrItinData<IIC_iStore_bh_iu,[InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                                  InstrStage<1, [A9_MUX0], 0>,
                                  InstrStage<2, [A9_AGU], 1>,
                                  InstrStage<1, [A9_LSUnit]>], [3, 1, 1]>,
  //
  // Register offset with update
  InstrItinData<IIC_iStore_ru ,  [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                                  InstrStage<1, [A9_MUX0], 0>,
                                  InstrStage<1, [A9_AGU], 0>,
                                  InstrStage<1, [A9_LSUnit]>],
                                 [2, 1, 1, 1]>,
  InstrItinData<IIC_iStore_bh_ru,[InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                                  InstrStage<1, [A9_MUX0], 0>,
                                  InstrStage<2, [A9_AGU], 1>,
                                  InstrStage<1, [A9_LSUnit]>],
                                 [3, 1, 1, 1]>,
  InstrItinData<IIC_iStore_d_ru, [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                                  InstrStage<1, [A9_MUX0], 0>,
                                  InstrStage<2, [A9_AGU], 1>,
                                  InstrStage<1, [A9_LSUnit]>],
                                 [3, 1, 1, 1]>,
  //
  // Scaled register offset with update
  InstrItinData<IIC_iStore_siu,    [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                                    InstrStage<1, [A9_MUX0], 0>,
                                    InstrStage<1, [A9_AGU], 0>,
                                    InstrStage<1, [A9_LSUnit]>],
                                   [2, 1, 1, 1]>,
  InstrItinData<IIC_iStore_bh_siu, [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                                    InstrStage<1, [A9_MUX0], 0>,
                                    InstrStage<2, [A9_AGU], 1>,
                                    InstrStage<1, [A9_LSUnit]>],
                                   [3, 1, 1, 1]>,
  //
  // Store multiple
  InstrItinData<IIC_iStore_m , [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                                InstrStage<1, [A9_MUX0], 0>,
                                InstrStage<1, [A9_AGU], 0>,
                                InstrStage<2, [A9_LSUnit]>]>,
  //
  // Store multiple + update
  InstrItinData<IIC_iStore_mu, [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                                InstrStage<1, [A9_MUX0], 0>,
                                InstrStage<1, [A9_AGU], 0>,
                                InstrStage<2, [A9_LSUnit]>], [2]>,

  //
  // Preload
  InstrItinData<IIC_Preload,   [InstrStage<1, [A9_Issue0, A9_Issue1]>], [1, 1]>,

  // Branch
  //
  // no delay slots, so the latency of a branch is unimportant
  InstrItinData<IIC_Br       , [InstrStage<1, [A9_Issue0], 0>,
                                InstrStage<1, [A9_Issue1], 0>,
                                InstrStage<1, [A9_Branch]>]>,

  // VFP and NEON shares the same register file. This means that every VFP
  // instruction should wait for full completion of the consecutive NEON
  // instruction and vice-versa. We model this behavior with two artificial FUs:
  // DRegsVFP and DRegsVFP.
  //
  // Every VFP instruction:
  //  - Acquires DRegsVFP resource for 1 cycle
  //  - Reserves DRegsN resource for the whole duration (including time to
  //    register file writeback!).
  // Every NEON instruction does the same but with FUs swapped.
  //
  // Since the reserved FU cannot be acquired, this models precisely
  // "cross-domain" stalls.

  // VFP
  // Issue through integer pipeline, and execute in NEON unit.

  // FP Special Register to Integer Register File Move
  InstrItinData<IIC_fpSTAT , [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                              InstrStage<1, [A9_MUX0], 0>,
                              InstrStage<1, [A9_DRegsVFP], 0, Required>,
                              InstrStage<2, [A9_DRegsN],   0, Reserved>,
                              InstrStage<1, [A9_NPipe]>],
                             [1]>,
  //
  // Single-precision FP Unary
  InstrItinData<IIC_fpUNA32 , [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsVFP], 0, Required>,
                               // Extra latency cycles since wbck is 2 cycles
                               InstrStage<3, [A9_DRegsN],   0, Reserved>,
                               InstrStage<1, [A9_NPipe]>],
                              [1, 1]>,
  //
  // Double-precision FP Unary
  InstrItinData<IIC_fpUNA64 , [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsVFP], 0, Required>,
                               // Extra latency cycles since wbck is 2 cycles
                               InstrStage<3, [A9_DRegsN],   0, Reserved>,
                               InstrStage<1, [A9_NPipe]>],
                              [1, 1]>,

  //
  // Single-precision FP Compare
  InstrItinData<IIC_fpCMP32 , [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsVFP], 0, Required>,
                               // Extra latency cycles since wbck is 4 cycles
                               InstrStage<5, [A9_DRegsN],   0, Reserved>,
                               InstrStage<1, [A9_NPipe]>],
                              [1, 1]>,
  //
  // Double-precision FP Compare
  InstrItinData<IIC_fpCMP64 , [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsVFP], 0, Required>,
                               // Extra latency cycles since wbck is 4 cycles
                               InstrStage<5, [A9_DRegsN],   0, Reserved>,
                               InstrStage<1, [A9_NPipe]>],
                              [1, 1]>,
  //
  // Single to Double FP Convert
  InstrItinData<IIC_fpCVTSD , [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsVFP], 0, Required>,
                               InstrStage<5, [A9_DRegsN],   0, Reserved>,
                               InstrStage<1, [A9_NPipe]>],
                              [4, 1]>,
  //
  // Double to Single FP Convert
  InstrItinData<IIC_fpCVTDS , [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsVFP], 0, Required>,
                               InstrStage<5, [A9_DRegsN],   0, Reserved>,
                               InstrStage<1, [A9_NPipe]>],
                              [4, 1]>,

  //
  // Single to Half FP Convert
  InstrItinData<IIC_fpCVTSH , [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsVFP], 0, Required>,
                               InstrStage<5, [A9_DRegsN],   0, Reserved>,
                               InstrStage<1, [A9_NPipe]>],
                              [4, 1]>,
  //
  // Half to Single FP Convert
  InstrItinData<IIC_fpCVTHS , [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsVFP], 0, Required>,
                               InstrStage<3, [A9_DRegsN],   0, Reserved>,
                               InstrStage<1, [A9_NPipe]>],
                              [2, 1]>,

  //
  // Single-Precision FP to Integer Convert
  InstrItinData<IIC_fpCVTSI , [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsVFP], 0, Required>,
                               InstrStage<5, [A9_DRegsN],   0, Reserved>,
                               InstrStage<1, [A9_NPipe]>],
                              [4, 1]>,
  //
  // Double-Precision FP to Integer Convert
  InstrItinData<IIC_fpCVTDI , [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsVFP], 0, Required>,
                               InstrStage<5, [A9_DRegsN],   0, Reserved>,
                               InstrStage<1, [A9_NPipe]>],
                              [4, 1]>,
  //
  // Integer to Single-Precision FP Convert
  InstrItinData<IIC_fpCVTIS , [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsVFP], 0, Required>,
                               InstrStage<5, [A9_DRegsN],   0, Reserved>,
                               InstrStage<1, [A9_NPipe]>],
                              [4, 1]>,
  //
  // Integer to Double-Precision FP Convert
  InstrItinData<IIC_fpCVTID , [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsVFP], 0, Required>,
                               InstrStage<5, [A9_DRegsN],   0, Reserved>,
                               InstrStage<1, [A9_NPipe]>],
                              [4, 1]>,
  //
  // Single-precision FP ALU
  InstrItinData<IIC_fpALU32 , [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsVFP], 0, Required>,
                               InstrStage<5, [A9_DRegsN],   0, Reserved>,
                               InstrStage<1, [A9_NPipe]>],
                              [4, 1, 1]>,
  //
  // Double-precision FP ALU
  InstrItinData<IIC_fpALU64 , [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsVFP], 0, Required>,
                               InstrStage<5, [A9_DRegsN],   0, Reserved>,
                               InstrStage<1, [A9_NPipe]>],
                              [4, 1, 1]>,
  //
  // Single-precision FP Multiply
  InstrItinData<IIC_fpMUL32 , [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsVFP], 0, Required>,
                               InstrStage<6, [A9_DRegsN],   0, Reserved>,
                               InstrStage<1, [A9_NPipe]>],
                              [5, 1, 1]>,
  //
  // Double-precision FP Multiply
  InstrItinData<IIC_fpMUL64 , [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsVFP], 0, Required>,
                               InstrStage<7, [A9_DRegsN],   0, Reserved>,
                               InstrStage<2, [A9_NPipe]>],
                              [6, 1, 1]>,
  //
  // Single-precision FP MAC
  InstrItinData<IIC_fpMAC32 , [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsVFP], 0, Required>,
                               InstrStage<9, [A9_DRegsN],   0, Reserved>,
                               InstrStage<1, [A9_NPipe]>],
                              [8, 1, 1, 1]>,
  //
  // Double-precision FP MAC
  InstrItinData<IIC_fpMAC64 , [InstrStage<1,  [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1,  [A9_MUX0], 0>,
                               InstrStage<1,  [A9_DRegsVFP], 0, Required>,
                               InstrStage<10, [A9_DRegsN],  0, Reserved>,
                               InstrStage<2,  [A9_NPipe]>],
                              [9, 1, 1, 1]>,
  //
  // Single-precision FP DIV
  InstrItinData<IIC_fpDIV32 , [InstrStage<1,  [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1,  [A9_MUX0], 0>,
                               InstrStage<1,  [A9_DRegsVFP], 0, Required>,
                               InstrStage<16, [A9_DRegsN],  0, Reserved>,
                               InstrStage<10, [A9_NPipe]>],
                              [15, 1, 1]>,
  //
  // Double-precision FP DIV
  InstrItinData<IIC_fpDIV64 , [InstrStage<1,  [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1,  [A9_MUX0], 0>,
                               InstrStage<1,  [A9_DRegsVFP], 0, Required>,
                               InstrStage<26, [A9_DRegsN],  0, Reserved>,
                               InstrStage<20, [A9_NPipe]>],
                              [25, 1, 1]>,
  //
  // Single-precision FP SQRT
  InstrItinData<IIC_fpSQRT32, [InstrStage<1,  [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1,  [A9_MUX0], 0>,
                               InstrStage<1,  [A9_DRegsVFP], 0, Required>,
                               InstrStage<18, [A9_DRegsN],   0, Reserved>,
                               InstrStage<13, [A9_NPipe]>],
                              [17, 1]>,
  //
  // Double-precision FP SQRT
  InstrItinData<IIC_fpSQRT64, [InstrStage<1,  [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1,  [A9_MUX0], 0>,
                               InstrStage<1,  [A9_DRegsVFP], 0, Required>,
                               InstrStage<33, [A9_DRegsN],   0, Reserved>,
                               InstrStage<28, [A9_NPipe]>],
                              [32, 1]>,

  //
  // Integer to Single-precision Move
  InstrItinData<IIC_fpMOVIS,  [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsVFP], 0, Required>,
                               // Extra 1 latency cycle since wbck is 2 cycles
                               InstrStage<3, [A9_DRegsN],   0, Reserved>,
                               InstrStage<1, [A9_NPipe]>],
                              [1, 1]>,
  //
  // Integer to Double-precision Move
  InstrItinData<IIC_fpMOVID,  [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsVFP], 0, Required>,
                               // Extra 1 latency cycle since wbck is 2 cycles
                               InstrStage<3, [A9_DRegsN],   0, Reserved>,
                               InstrStage<1, [A9_NPipe]>],
                              [1, 1, 1]>,
  //
  // Single-precision to Integer Move
  InstrItinData<IIC_fpMOVSI,  [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsVFP], 0, Required>,
                               InstrStage<2, [A9_DRegsN],   0, Reserved>,
                               InstrStage<1, [A9_NPipe]>],
                              [2, 1]>,
  //
  // Double-precision to Integer Move
  InstrItinData<IIC_fpMOVDI,  [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsVFP], 0, Required>,
                               InstrStage<2, [A9_DRegsN],   0, Reserved>,
                               InstrStage<1, [A9_NPipe]>],
                              [2, 1, 1]>,
  //
  // Single-precision FP Load
  InstrItinData<IIC_fpLoad32, [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsVFP], 0, Required>,
                               InstrStage<2, [A9_DRegsN],   0, Reserved>,
                               InstrStage<1, [A9_NPipe], 0>,
                               InstrStage<1, [A9_LSUnit]>],
                              [1, 1]>,
  //
  // Double-precision FP Load
  // FIXME: Result latency is 1 if address is 64-bit aligned.
  InstrItinData<IIC_fpLoad64, [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsVFP], 0, Required>,
                               InstrStage<2, [A9_DRegsN],   0, Reserved>,
                               InstrStage<1, [A9_NPipe], 0>,
                               InstrStage<1, [A9_LSUnit]>],
                              [2, 1]>,
  //
  // FP Load Multiple
  InstrItinData<IIC_fpLoad_m, [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsVFP], 0, Required>,
                               InstrStage<2, [A9_DRegsN],   0, Reserved>,
                               InstrStage<1, [A9_NPipe], 0>,
                               InstrStage<1, [A9_LSUnit]>], [1, 1, 1, 1]>,
  //
  // FP Load Multiple + update
  InstrItinData<IIC_fpLoad_mu,[InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsVFP], 0, Required>,
                               InstrStage<2, [A9_DRegsN],   0, Reserved>,
                               InstrStage<1, [A9_NPipe], 0>,
                               InstrStage<1, [A9_LSUnit]>], [2, 1, 1, 1]>,
  //
  // Single-precision FP Store
  InstrItinData<IIC_fpStore32,[InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsVFP], 0, Required>,
                               InstrStage<2, [A9_DRegsN],   0, Reserved>,
                               InstrStage<1, [A9_NPipe], 0>,
                               InstrStage<1, [A9_LSUnit]>],
                              [1, 1]>,
  //
  // Double-precision FP Store
  InstrItinData<IIC_fpStore64,[InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsVFP], 0, Required>,
                               InstrStage<2, [A9_DRegsN],   0, Reserved>,
                               InstrStage<1, [A9_NPipe], 0>,
                               InstrStage<1, [A9_LSUnit]>],
                              [1, 1]>,
  //
  // FP Store Multiple
  InstrItinData<IIC_fpStore_m,[InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsVFP], 0, Required>,
                               InstrStage<2, [A9_DRegsN],   0, Reserved>,
                               InstrStage<1, [A9_NPipe], 0>,
                               InstrStage<1, [A9_LSUnit]>], [1, 1, 1, 1]>,
  //
  // FP Store Multiple + update
  InstrItinData<IIC_fpStore_mu,[InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                                InstrStage<1, [A9_MUX0], 0>,
                                InstrStage<1, [A9_DRegsVFP], 0, Required>,
                                InstrStage<2, [A9_DRegsN],   0, Reserved>,
                                InstrStage<1, [A9_NPipe], 0>,
                                InstrStage<1, [A9_LSUnit]>], [2, 1, 1, 1]>,
  // NEON
  // VLD1
  // FIXME: Conservatively assume insufficent alignment.
  InstrItinData<IIC_VLD1,     [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               InstrStage<8, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<2, [A9_NPipe], 0>,
                               InstrStage<2, [A9_LSUnit]>],
                              [2, 1]>,
  // VLD1x2
  InstrItinData<IIC_VLD1x2,   [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               InstrStage<8, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<2, [A9_NPipe], 0>,
                               InstrStage<2, [A9_LSUnit]>],
                              [2, 2, 1]>,
  // VLD1x3
  InstrItinData<IIC_VLD1x3,   [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               InstrStage<9, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<3, [A9_NPipe], 0>,
                               InstrStage<3, [A9_LSUnit]>],
                              [2, 2, 3, 1]>,
  // VLD1x4
  InstrItinData<IIC_VLD1x4,   [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               InstrStage<9, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<3, [A9_NPipe], 0>,
                               InstrStage<3, [A9_LSUnit]>],
                              [2, 2, 3, 3, 1]>,
  // VLD1u
  InstrItinData<IIC_VLD1u,    [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               InstrStage<8, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<2, [A9_NPipe], 0>,
                               InstrStage<2, [A9_LSUnit]>],
                              [2, 2, 1]>,
  // VLD1x2u
  InstrItinData<IIC_VLD1x2u,  [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               InstrStage<8, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<2, [A9_NPipe], 0>,
                               InstrStage<2, [A9_LSUnit]>],
                              [2, 2, 2, 1]>,
  // VLD1x3u
  InstrItinData<IIC_VLD1x3u,  [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               InstrStage<9, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<3, [A9_NPipe], 0>,
                               InstrStage<3, [A9_LSUnit]>],
                              [2, 2, 3, 2, 1]>,
  // VLD1x4u
  InstrItinData<IIC_VLD1x4u,  [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               InstrStage<9, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<3, [A9_NPipe], 0>,
                               InstrStage<3, [A9_LSUnit]>],
                              [2, 2, 3, 3, 2, 1]>,
  //
  // VLD1ln
  InstrItinData<IIC_VLD1ln,   [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               InstrStage<9, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<3, [A9_NPipe], 0>,
                               InstrStage<3, [A9_LSUnit]>],
                              [4, 1, 1, 1]>,
  //
  // VLD1lnu
  InstrItinData<IIC_VLD1lnu,  [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               InstrStage<9, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<3, [A9_NPipe], 0>,
                               InstrStage<3, [A9_LSUnit]>],
                              [4, 2, 1, 1, 1, 1]>,
  //
  // VLD1dup
  InstrItinData<IIC_VLD1dup,  [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               InstrStage<8, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<2, [A9_NPipe], 0>,
                               InstrStage<2, [A9_LSUnit]>],
                              [3, 1]>,
  //
  // VLD1dupu
  InstrItinData<IIC_VLD1dupu, [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               InstrStage<8, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<2, [A9_NPipe], 0>,
                               InstrStage<2, [A9_LSUnit]>],
                              [3, 2, 1, 1]>,
  //
  // VLD2
  InstrItinData<IIC_VLD2,     [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 7 cycles
                               InstrStage<8, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<2, [A9_NPipe], 0>,
                               InstrStage<2, [A9_LSUnit]>],
                              [3, 3, 1]>,
  //
  // VLD2x2
  InstrItinData<IIC_VLD2x2,   [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               InstrStage<9, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<3, [A9_NPipe], 0>,
                               InstrStage<3, [A9_LSUnit]>],
                              [3, 4, 3, 4, 1]>,
  //
  // VLD2ln
  InstrItinData<IIC_VLD2ln,   [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               InstrStage<9, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<3, [A9_NPipe], 0>,
                               InstrStage<3, [A9_LSUnit]>],
                              [4, 4, 1, 1, 1, 1]>,
  //
  // VLD2u
  InstrItinData<IIC_VLD2u,    [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 7 cycles
                               InstrStage<8, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<2, [A9_NPipe], 0>,
                               InstrStage<2, [A9_LSUnit]>],
                              [3, 3, 2, 1, 1, 1]>,
  //
  // VLD2x2u
  InstrItinData<IIC_VLD2x2u,  [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               InstrStage<9, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<3, [A9_NPipe], 0>,
                               InstrStage<3, [A9_LSUnit]>],
                              [3, 4, 3, 4, 2, 1]>,
  //
  // VLD2lnu
  InstrItinData<IIC_VLD2lnu,  [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               InstrStage<9, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<3, [A9_NPipe], 0>,
                               InstrStage<3, [A9_LSUnit]>],
                              [4, 4, 2, 1, 1, 1, 1, 1]>,
  //
  // VLD2dup
  InstrItinData<IIC_VLD2dup,  [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               InstrStage<8, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<2, [A9_NPipe], 0>,
                               InstrStage<2, [A9_LSUnit]>],
                              [3, 1]>,
  //
  // VLD2dupu
  InstrItinData<IIC_VLD2dupu, [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               InstrStage<8, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<2, [A9_NPipe], 0>,
                               InstrStage<2, [A9_LSUnit]>],
                              [3, 2, 1, 1]>,
  //
  // VLD3
  InstrItinData<IIC_VLD3,     [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               InstrStage<10,[A9_DRegsVFP], 0, Reserved>,
                               InstrStage<4, [A9_NPipe], 0>,
                               InstrStage<4, [A9_LSUnit]>],
                              [4, 4, 5, 1]>,
  //
  // VLD3ln
  InstrItinData<IIC_VLD3ln,   [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               InstrStage<11,[A9_DRegsVFP], 0, Reserved>,
                               InstrStage<5, [A9_NPipe], 0>,
                               InstrStage<5, [A9_LSUnit]>],
                              [5, 5, 6, 1, 1, 1, 1, 2]>,
  //
  // VLD3u
  InstrItinData<IIC_VLD3u,    [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               InstrStage<10,[A9_DRegsVFP], 0, Reserved>,
                               InstrStage<4, [A9_NPipe], 0>,
                               InstrStage<4, [A9_LSUnit]>],
                              [4, 4, 5, 2, 1]>,
  //
  // VLD3lnu
  InstrItinData<IIC_VLD3lnu,  [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               InstrStage<11,[A9_DRegsVFP], 0, Reserved>,
                               InstrStage<5, [A9_NPipe], 0>,
                               InstrStage<5, [A9_LSUnit]>],
                              [5, 5, 6, 2, 1, 1, 1, 1, 1, 2]>,
  //
  // VLD4
  InstrItinData<IIC_VLD4,     [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               InstrStage<10,[A9_DRegsVFP], 0, Reserved>,
                               InstrStage<4, [A9_NPipe], 0>,
                               InstrStage<4, [A9_LSUnit]>],
                              [4, 4, 5, 5, 1]>,
  //
  // VLD4ln
  InstrItinData<IIC_VLD4ln,   [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               InstrStage<11,[A9_DRegsVFP], 0, Reserved>,
                               InstrStage<5, [A9_NPipe], 0>,
                               InstrStage<5, [A9_LSUnit]>],
                              [5, 5, 6, 6, 1, 1, 1, 1, 2, 2]>,
  //
  // VLD4u
  InstrItinData<IIC_VLD4u,    [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               InstrStage<10,[A9_DRegsVFP], 0, Reserved>,
                               InstrStage<4, [A9_NPipe], 0>,
                               InstrStage<4, [A9_LSUnit]>],
                              [4, 4, 5, 5, 2, 1]>,
  //
  // VLD4lnu
  InstrItinData<IIC_VLD4lnu,  [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               InstrStage<11,[A9_DRegsVFP], 0, Reserved>,
                               InstrStage<5, [A9_NPipe], 0>,
                               InstrStage<5, [A9_LSUnit]>],
                              [5, 5, 6, 6, 2, 1, 1, 1, 1, 1, 2, 2]>,
  //
  // VST1
  InstrItinData<IIC_VST1,     [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               InstrStage<2, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<2, [A9_NPipe], 0>,
                               InstrStage<2, [A9_LSUnit]>],
                              [1, 1, 1]>,
  //
  // VST1x2
  InstrItinData<IIC_VST1x2,   [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               InstrStage<2, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<2, [A9_NPipe], 0>,
                               InstrStage<2, [A9_LSUnit]>],
                              [1, 1, 1, 1]>,
  //
  // VST1x3
  InstrItinData<IIC_VST1x3,   [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               InstrStage<3, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<3, [A9_NPipe], 0>,
                               InstrStage<3, [A9_LSUnit]>],
                              [1, 1, 1, 1, 2]>,
  //
  // VST1x4
  InstrItinData<IIC_VST1x4,   [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               InstrStage<3, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<3, [A9_NPipe], 0>,
                               InstrStage<3, [A9_LSUnit]>],
                              [1, 1, 1, 1, 2, 2]>,
  //
  // VST1u
  InstrItinData<IIC_VST1u,    [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               InstrStage<2, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<2, [A9_NPipe], 0>,
                               InstrStage<2, [A9_LSUnit]>],
                              [2, 1, 1, 1, 1]>,
  //
  // VST1x2u
  InstrItinData<IIC_VST1x2u,  [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               InstrStage<2, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<2, [A9_NPipe], 0>,
                               InstrStage<2, [A9_LSUnit]>],
                              [2, 1, 1, 1, 1, 1]>,
  //
  // VST1x3u
  InstrItinData<IIC_VST1x3u,  [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               InstrStage<2, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<3, [A9_NPipe], 0>,
                               InstrStage<3, [A9_LSUnit]>],
                              [2, 1, 1, 1, 1, 1, 2]>,
  //
  // VST1x4u
  InstrItinData<IIC_VST1x4u,  [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               InstrStage<3, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<3, [A9_NPipe], 0>,
                               InstrStage<3, [A9_LSUnit]>],
                              [2, 1, 1, 1, 1, 1, 2, 2]>,
  //
  // VST1ln
  InstrItinData<IIC_VST1ln,   [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               InstrStage<2, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<2, [A9_NPipe], 0>,
                               InstrStage<2, [A9_LSUnit]>],
                              [1, 1, 1]>,
  //
  // VST1lnu
  InstrItinData<IIC_VST1lnu,  [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               InstrStage<3, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<3, [A9_NPipe], 0>,
                               InstrStage<3, [A9_LSUnit]>],
                              [2, 1, 1, 1, 1]>,
  //
  // VST2
  InstrItinData<IIC_VST2,     [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               InstrStage<2, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<2, [A9_NPipe], 0>,
                               InstrStage<2, [A9_LSUnit]>],
                              [1, 1, 1, 1]>,
  //
  // VST2x2
  InstrItinData<IIC_VST2x2,   [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               InstrStage<3, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<3, [A9_NPipe], 0>,
                               InstrStage<3, [A9_LSUnit]>],
                              [1, 1, 1, 1, 2, 2]>,
  //
  // VST2u
  InstrItinData<IIC_VST2u,    [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               InstrStage<2, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<2, [A9_NPipe], 0>,
                               InstrStage<2, [A9_LSUnit]>],
                              [2, 1, 1, 1, 1, 1]>,
  //
  // VST2x2u
  InstrItinData<IIC_VST2x2u,  [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               InstrStage<3, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<3, [A9_NPipe], 0>,
                               InstrStage<3, [A9_LSUnit]>],
                              [2, 1, 1, 1, 1, 1, 2, 2]>,
  //
  // VST2ln
  InstrItinData<IIC_VST2ln,   [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               InstrStage<2, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<2, [A9_NPipe], 0>,
                               InstrStage<2, [A9_LSUnit]>],
                              [1, 1, 1, 1]>,
  //
  // VST2lnu
  InstrItinData<IIC_VST2lnu,  [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               InstrStage<3, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<3, [A9_NPipe], 0>,
                               InstrStage<3, [A9_LSUnit]>],
                              [2, 1, 1, 1, 1, 1]>,
  //
  // VST3
  InstrItinData<IIC_VST3,     [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               InstrStage<3, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<3, [A9_NPipe], 0>,
                               InstrStage<3, [A9_LSUnit]>],
                              [1, 1, 1, 1, 2]>,
  //
  // VST3u
  InstrItinData<IIC_VST3u,    [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               InstrStage<3, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<3, [A9_NPipe], 0>,
                               InstrStage<3, [A9_LSUnit]>],
                              [2, 1, 1, 1, 1, 1, 2]>,
  //
  // VST3ln
  InstrItinData<IIC_VST3ln,   [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               InstrStage<3, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<3, [A9_NPipe], 0>,
                               InstrStage<3, [A9_LSUnit]>],
                              [1, 1, 1, 1, 2]>,
  //
  // VST3lnu
  InstrItinData<IIC_VST3lnu,  [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               InstrStage<3, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<3, [A9_NPipe], 0>,
                               InstrStage<3, [A9_LSUnit]>],
                              [2, 1, 1, 1, 1, 1, 2]>,
  //
  // VST4
  InstrItinData<IIC_VST4,     [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               InstrStage<3, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<3, [A9_NPipe], 0>,
                               InstrStage<3, [A9_LSUnit]>],
                              [1, 1, 1, 1, 2, 2]>,
  //
  // VST4u
  InstrItinData<IIC_VST4u,    [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               InstrStage<3, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<3, [A9_NPipe], 0>,
                               InstrStage<3, [A9_LSUnit]>],
                              [2, 1, 1, 1, 1, 1, 2, 2]>,
  //
  // VST4ln
  InstrItinData<IIC_VST4ln,   [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               InstrStage<3, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<3, [A9_NPipe], 0>,
                               InstrStage<3, [A9_LSUnit]>],
                              [1, 1, 1, 1, 2, 2]>,
  //
  // VST4lnu
  InstrItinData<IIC_VST4lnu,  [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               InstrStage<3, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<3, [A9_NPipe], 0>,
                               InstrStage<3, [A9_LSUnit]>],
                              [2, 1, 1, 1, 1, 1, 2, 2]>,

  //
  // Double-register Integer Unary
  InstrItinData<IIC_VUNAiD,   [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 6 cycles
                               InstrStage<7, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_NPipe]>],
                              [4, 2]>,
  //
  // Quad-register Integer Unary
  InstrItinData<IIC_VUNAiQ,   [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 6 cycles
                               InstrStage<7, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_NPipe]>],
                              [4, 2]>,
  //
  // Double-register Integer Q-Unary
  InstrItinData<IIC_VQUNAiD,  [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 6 cycles
                               InstrStage<7, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_NPipe]>],
                              [4, 1]>,
  //
  // Quad-register Integer CountQ-Unary
  InstrItinData<IIC_VQUNAiQ,  [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 6 cycles
                               InstrStage<7, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_NPipe]>],
                              [4, 1]>,
  //
  // Double-register Integer Binary
  InstrItinData<IIC_VBINiD,   [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 6 cycles
                               InstrStage<7, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_NPipe]>],
                              [3, 2, 2]>,
  //
  // Quad-register Integer Binary
  InstrItinData<IIC_VBINiQ,   [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 6 cycles
                               InstrStage<7, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_NPipe]>],
                              [3, 2, 2]>,
  //
  // Double-register Integer Subtract
  InstrItinData<IIC_VSUBiD,   [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 6 cycles
                               InstrStage<7, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_NPipe]>],
                              [3, 2, 1]>,
  //
  // Quad-register Integer Subtract
  InstrItinData<IIC_VSUBiQ,   [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 6 cycles
                               InstrStage<7, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_NPipe]>],
                              [3, 2, 1]>,
  //
  // Double-register Integer Shift
  InstrItinData<IIC_VSHLiD,   [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 6 cycles
                               InstrStage<7, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_NPipe]>],
                              [3, 1, 1]>,
  //
  // Quad-register Integer Shift
  InstrItinData<IIC_VSHLiQ,   [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 6 cycles
                               InstrStage<7, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_NPipe]>],
                              [3, 1, 1]>,
  //
  // Double-register Integer Shift (4 cycle)
  InstrItinData<IIC_VSHLi4D,  [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 6 cycles
                               InstrStage<7, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_NPipe]>],
                              [4, 1, 1]>,
  //
  // Quad-register Integer Shift (4 cycle)
  InstrItinData<IIC_VSHLi4Q,  [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 6 cycles
                               InstrStage<7, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_NPipe]>],
                              [4, 1, 1]>,
  //
  // Double-register Integer Binary (4 cycle)
  InstrItinData<IIC_VBINi4D,  [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 6 cycles
                               InstrStage<7, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_NPipe]>],
                              [4, 2, 2]>,
  //
  // Quad-register Integer Binary (4 cycle)
  InstrItinData<IIC_VBINi4Q,  [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 6 cycles
                               InstrStage<7, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_NPipe]>],
                              [4, 2, 2]>,
  //
  // Double-register Integer Subtract (4 cycle)
  InstrItinData<IIC_VSUBi4D,  [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 6 cycles
                               InstrStage<7, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_NPipe]>],
                              [4, 2, 1]>,
  //
  // Quad-register Integer Subtract (4 cycle)
  InstrItinData<IIC_VSUBi4Q,  [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 6 cycles
                               InstrStage<7, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_NPipe]>],
                              [4, 2, 1]>,

  //
  // Double-register Integer Count
  InstrItinData<IIC_VCNTiD,   [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 6 cycles
                               InstrStage<7, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_NPipe]>],
                              [3, 2, 2]>,
  //
  // Quad-register Integer Count
  // Result written in N3, but that is relative to the last cycle of multicycle,
  // so we use 4 for those cases
  InstrItinData<IIC_VCNTiQ,   [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 7 cycles
                               InstrStage<8, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<2, [A9_NPipe]>],
                              [4, 2, 2]>,
  //
  // Double-register Absolute Difference and Accumulate
  InstrItinData<IIC_VABAD,    [InstrStage<7, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 6 cycles
                               InstrStage<1, [A9_NPipe]>],
                              [6, 3, 2, 1]>,
  //
  // Quad-register Absolute Difference and Accumulate
  InstrItinData<IIC_VABAQ,    [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 6 cycles
                               InstrStage<7, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<2, [A9_NPipe]>],
                              [6, 3, 2, 1]>,
  //
  // Double-register Integer Pair Add Long
  InstrItinData<IIC_VPALiD,   [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 6 cycles
                               InstrStage<7, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_NPipe]>],
                              [6, 3, 1]>,
  //
  // Quad-register Integer Pair Add Long
  InstrItinData<IIC_VPALiQ,   [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 6 cycles
                               InstrStage<7, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<2, [A9_NPipe]>],
                              [6, 3, 1]>,

  //
  // Double-register Integer Multiply (.8, .16)
  InstrItinData<IIC_VMULi16D, [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 6 cycles
                               InstrStage<7, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_NPipe]>],
                              [6, 2, 2]>,
  //
  // Quad-register Integer Multiply (.8, .16)
  InstrItinData<IIC_VMULi16Q, [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 7 cycles
                               InstrStage<8, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<2, [A9_NPipe]>],
                              [7, 2, 2]>,

  //
  // Double-register Integer Multiply (.32)
  InstrItinData<IIC_VMULi32D, [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 7 cycles
                               InstrStage<8, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<2, [A9_NPipe]>],
                              [7, 2, 1]>,
  //
  // Quad-register Integer Multiply (.32)
  InstrItinData<IIC_VMULi32Q, [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 9 cycles
                               InstrStage<10, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<4, [A9_NPipe]>],
                              [9, 2, 1]>,
  //
  // Double-register Integer Multiply-Accumulate (.8, .16)
  InstrItinData<IIC_VMACi16D, [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 6 cycles
                               InstrStage<7, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_NPipe]>],
                              [6, 3, 2, 2]>,
  //
  // Double-register Integer Multiply-Accumulate (.32)
  InstrItinData<IIC_VMACi32D, [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 7 cycles
                               InstrStage<8, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<2, [A9_NPipe]>],
                              [7, 3, 2, 1]>,
  //
  // Quad-register Integer Multiply-Accumulate (.8, .16)
  InstrItinData<IIC_VMACi16Q, [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 7 cycles
                               InstrStage<8, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<2, [A9_NPipe]>],
                              [7, 3, 2, 2]>,
  //
  // Quad-register Integer Multiply-Accumulate (.32)
  InstrItinData<IIC_VMACi32Q, [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 9 cycles
                               InstrStage<10, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<4, [A9_NPipe]>],
                              [9, 3, 2, 1]>,

  //
  // Move
  InstrItinData<IIC_VMOV,     [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               InstrStage<1, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_NPipe]>],
                              [1,1]>,
  //
  // Move Immediate
  InstrItinData<IIC_VMOVImm,  [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 6 cycles
                               InstrStage<7, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_NPipe]>],
                              [3]>,
  //
  // Double-register Permute Move
  InstrItinData<IIC_VMOVD,    [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 6 cycles
                               InstrStage<7, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_NPipe]>],
                              [2, 1]>,
  //
  // Quad-register Permute Move
  InstrItinData<IIC_VMOVQ,    [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 6 cycles
                               InstrStage<7, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_NPipe]>],
                              [2, 1]>,
  //
  // Integer to Single-precision Move
  InstrItinData<IIC_VMOVIS ,  [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               InstrStage<3, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_NPipe]>],
                              [1, 1]>,
  //
  // Integer to Double-precision Move
  InstrItinData<IIC_VMOVID ,  [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               InstrStage<3, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_NPipe]>],
                              [1, 1, 1]>,
  //
  // Single-precision to Integer Move
  InstrItinData<IIC_VMOVSI ,  [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               InstrStage<3, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_NPipe]>],
                              [2, 1]>,
  //
  // Double-precision to Integer Move
  InstrItinData<IIC_VMOVDI ,  [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               InstrStage<3, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_NPipe]>],
                              [2, 2, 1]>,
  //
  // Integer to Lane Move
  InstrItinData<IIC_VMOVISL , [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               InstrStage<4, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<2, [A9_NPipe]>],
                              [3, 1, 1]>,

  //
  // Vector narrow move
  InstrItinData<IIC_VMOVN,    [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 6 cycles
                               InstrStage<7, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_NPipe]>],
                              [3, 1]>,
  //
  // Double-register FP Unary
  InstrItinData<IIC_VUNAD,    [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 6 cycles
                               InstrStage<7, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_NPipe]>],
                              [5, 2]>,
  //
  // Quad-register FP Unary
  // Result written in N5, but that is relative to the last cycle of multicycle,
  // so we use 6 for those cases
  InstrItinData<IIC_VUNAQ,    [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 7 cycles
                               InstrStage<8, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<2, [A9_NPipe]>],
                              [6, 2]>,
  //
  // Double-register FP Binary
  // FIXME: We're using this itin for many instructions and [2, 2] here is too
  // optimistic.
  InstrItinData<IIC_VBIND,    [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 6 cycles
                               InstrStage<7, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_NPipe]>],
                              [5, 2, 2]>,

  //
  // VPADD, etc.
  InstrItinData<IIC_VPBIND,   [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 6 cycles
                               InstrStage<7, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_NPipe]>],
                              [5, 1, 1]>,
  //
  // Double-register FP VMUL
  InstrItinData<IIC_VFMULD,   [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 6 cycles
                               InstrStage<7, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_NPipe]>],
                              [5, 2, 1]>,
  //
  // Quad-register FP Binary
  // Result written in N5, but that is relative to the last cycle of multicycle,
  // so we use 6 for those cases
  // FIXME: We're using this itin for many instructions and [2, 2] here is too
  // optimistic.
  InstrItinData<IIC_VBINQ,    [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 7 cycles
                               InstrStage<8, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<2, [A9_NPipe]>],
                              [6, 2, 2]>,
  //
  // Quad-register FP VMUL
  InstrItinData<IIC_VFMULQ,   [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 7 cycles
                               InstrStage<8, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_NPipe]>],
                              [6, 2, 1]>,
  //
  // Double-register FP Multiple-Accumulate
  InstrItinData<IIC_VMACD,    [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 7 cycles
                               InstrStage<8, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<2, [A9_NPipe]>],
                              [6, 3, 2, 1]>,
  //
  // Quad-register FP Multiple-Accumulate
  // Result written in N9, but that is relative to the last cycle of multicycle,
  // so we use 10 for those cases
  InstrItinData<IIC_VMACQ,    [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 9 cycles
                               InstrStage<10, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<4, [A9_NPipe]>],
                              [8, 4, 2, 1]>,
  //
  // Double-register Reciprical Step
  InstrItinData<IIC_VRECSD,   [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 10 cycles
                               InstrStage<11, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_NPipe]>],
                              [9, 2, 2]>,
  //
  // Quad-register Reciprical Step
  InstrItinData<IIC_VRECSQ,   [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 11 cycles
                               InstrStage<12, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<2, [A9_NPipe]>],
                              [10, 2, 2]>,
  //
  // Double-register Permute
  InstrItinData<IIC_VPERMD,   [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 6 cycles
                               InstrStage<7, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_NPipe]>],
                              [2, 2, 1, 1]>,
  //
  // Quad-register Permute
  // Result written in N2, but that is relative to the last cycle of multicycle,
  // so we use 3 for those cases
  InstrItinData<IIC_VPERMQ,   [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 7 cycles
                               InstrStage<8, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<2, [A9_NPipe]>],
                              [3, 3, 1, 1]>,
  //
  // Quad-register Permute (3 cycle issue)
  // Result written in N2, but that is relative to the last cycle of multicycle,
  // so we use 4 for those cases
  InstrItinData<IIC_VPERMQ3,  [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 8 cycles
                               InstrStage<9, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<3, [A9_NPipe]>],
                              [4, 4, 1, 1]>,

  //
  // Double-register VEXT
  InstrItinData<IIC_VEXTD,    [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 6 cycles
                               InstrStage<7, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_NPipe]>],
                              [2, 1, 1]>,
  //
  // Quad-register VEXT
  InstrItinData<IIC_VEXTQ,    [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 7 cycles
                               InstrStage<8, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<2, [A9_NPipe]>],
                              [3, 1, 2]>,
  //
  // VTB
  InstrItinData<IIC_VTB1,     [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 7 cycles
                               InstrStage<8, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<2, [A9_NPipe]>],
                              [3, 2, 1]>,
  InstrItinData<IIC_VTB2,     [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<2, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 7 cycles
                               InstrStage<8, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<2, [A9_NPipe]>],
                              [3, 2, 2, 1]>,
  InstrItinData<IIC_VTB3,     [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<2, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 8 cycles
                               InstrStage<9, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<3, [A9_NPipe]>],
                              [4, 2, 2, 3, 1]>,
  InstrItinData<IIC_VTB4,     [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 8 cycles
                               InstrStage<9, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<3, [A9_NPipe]>],
                              [4, 2, 2, 3, 3, 1]>,
  //
  // VTBX
  InstrItinData<IIC_VTBX1,    [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 7 cycles
                               InstrStage<8, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<2, [A9_NPipe]>],
                              [3, 1, 2, 1]>,
  InstrItinData<IIC_VTBX2,    [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 7 cycles
                               InstrStage<8, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<2, [A9_NPipe]>],
                              [3, 1, 2, 2, 1]>,
  InstrItinData<IIC_VTBX3,    [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 8 cycles
                               InstrStage<9, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<3, [A9_NPipe]>],
                              [4, 1, 2, 2, 3, 1]>,
  InstrItinData<IIC_VTBX4,    [InstrStage<1, [A9_Issue0, A9_Issue1], 0>,
                               InstrStage<1, [A9_MUX0], 0>,
                               InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 8 cycles
                               InstrStage<9, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<2, [A9_NPipe]>],
                              [4, 1, 2, 2, 3, 3, 1]>
]>;
