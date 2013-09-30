//=- ARMScheduleSwift.td - Swift Scheduling Definitions -*- tablegen -*----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the itinerary class data for the Swift processor..
//
//===----------------------------------------------------------------------===//

// ===---------------------------------------------------------------------===//
// This section contains legacy support for itineraries. This is
// required until SD and PostRA schedulers are replaced by MachineScheduler.

def SW_DIS0 : FuncUnit;
def SW_DIS1 : FuncUnit;
def SW_DIS2 : FuncUnit;

def SW_ALU0 : FuncUnit;
def SW_ALU1 : FuncUnit;
def SW_LS   : FuncUnit;
def SW_IDIV : FuncUnit;
def SW_FDIV : FuncUnit;

// FIXME: Need bypasses.
// FIXME: Model the multiple stages of IIC_iMOVix2, IIC_iMOVix2addpc, and
//        IIC_iMOVix2ld better.
// FIXME: Model the special immediate shifts that are not microcoded.
// FIXME: Do we need to model the fact that uses of r15 in a micro-op force it
//        to issue on pipe 1?
// FIXME: Model the pipelined behavior of CMP / TST instructions.
// FIXME: Better model the microcode stages of multiply instructions, especially
//        conditional variants.
// FIXME: Add preload instruction when it is documented.
// FIXME: Model non-pipelined nature of FP div / sqrt unit.

def SwiftItineraries : ProcessorItineraries<
  [SW_DIS0, SW_DIS1, SW_DIS2, SW_ALU0, SW_ALU1, SW_LS, SW_IDIV, SW_FDIV], [], [
  //
  // Move instructions, unconditional
  InstrItinData<IIC_iMOVi   , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0, SW_ALU1]>],
                              [1]>,
  InstrItinData<IIC_iMOVr   , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0, SW_ALU1]>],
                              [1]>,
  InstrItinData<IIC_iMOVsi  , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0, SW_ALU1]>],
                              [1]>,
  InstrItinData<IIC_iMOVsr  , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0, SW_ALU1]>],
                              [1]>,
  InstrItinData<IIC_iMOVix2 , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0, SW_ALU1]>,
                               InstrStage<1, [SW_ALU0, SW_ALU1]>],
                              [2]>,
  InstrItinData<IIC_iMOVix2addpc,[InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                                  InstrStage<1, [SW_ALU0, SW_ALU1]>,
                                  InstrStage<1, [SW_ALU0, SW_ALU1]>,
                                  InstrStage<1, [SW_ALU0, SW_ALU1]>],
                                 [3]>,
  InstrItinData<IIC_iMOVix2ld,[InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0, SW_ALU1]>,
                               InstrStage<1, [SW_ALU0, SW_ALU1]>,
                               InstrStage<1, [SW_LS]>],
                              [5]>,
  //
  // MVN instructions
  InstrItinData<IIC_iMVNi   , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0, SW_ALU1]>],
                              [1]>,
  InstrItinData<IIC_iMVNr   , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0, SW_ALU1]>],
                              [1]>,
  InstrItinData<IIC_iMVNsi  , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0, SW_ALU1]>],
                              [1]>,
  InstrItinData<IIC_iMVNsr  , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0, SW_ALU1]>],
                              [1]>,
  //
  // No operand cycles
  InstrItinData<IIC_iALUx   , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0, SW_ALU1]>]>,
  //
  // Binary Instructions that produce a result
  InstrItinData<IIC_iALUi , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                             InstrStage<1, [SW_ALU0, SW_ALU1]>],
                            [1, 1]>,
  InstrItinData<IIC_iALUr , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                             InstrStage<1, [SW_ALU0, SW_ALU1]>],
                            [1, 1, 1]>,
  InstrItinData<IIC_iALUsi, [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                             InstrStage<1, [SW_ALU0, SW_ALU1]>],
                            [2, 1, 1]>,
  InstrItinData<IIC_iALUsir,[InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                             InstrStage<1, [SW_ALU0, SW_ALU1]>],
                            [2, 1, 1]>,
  InstrItinData<IIC_iALUsr, [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                             InstrStage<1, [SW_ALU0, SW_ALU1]>],
                            [2, 1, 1, 1]>,
  //
  // Bitwise Instructions that produce a result
  InstrItinData<IIC_iBITi , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                             InstrStage<1, [SW_ALU0, SW_ALU1]>],
                            [1, 1]>,
  InstrItinData<IIC_iBITr , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                             InstrStage<1, [SW_ALU0, SW_ALU1]>],
                            [1, 1, 1]>,
  InstrItinData<IIC_iBITsi, [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                             InstrStage<1, [SW_ALU0, SW_ALU1]>],
                            [2, 1, 1]>,
  InstrItinData<IIC_iBITsr, [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                             InstrStage<1, [SW_ALU0, SW_ALU1]>],
                            [2, 1, 1, 1]>,
  //
  // Unary Instructions that produce a result

  // CLZ, RBIT, etc.
  InstrItinData<IIC_iUNAr , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                             InstrStage<1, [SW_ALU0, SW_ALU1]>],
                            [1, 1]>,

  // BFC, BFI, UBFX, SBFX
  InstrItinData<IIC_iUNAsi, [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                             InstrStage<1, [SW_ALU0, SW_ALU1]>],
                            [2, 1]>,

  //
  // Zero and sign extension instructions
  InstrItinData<IIC_iEXTr , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                             InstrStage<1, [SW_ALU0, SW_ALU1]>],
                            [1, 1]>,
  InstrItinData<IIC_iEXTAr, [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                             InstrStage<1, [SW_ALU0, SW_ALU1]>],
                            [1, 1, 1]>,
  InstrItinData<IIC_iEXTAsr,[InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                             InstrStage<1, [SW_ALU0, SW_ALU1]>],
                            [1, 1, 1, 1]>,
  //
  // Compare instructions
  InstrItinData<IIC_iCMPi   , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0, SW_ALU1]>],
                              [1]>,
  InstrItinData<IIC_iCMPr   , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0, SW_ALU1]>],
                              [1, 1]>,
  InstrItinData<IIC_iCMPsi  , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<2, [SW_ALU0, SW_ALU1]>],
                              [1, 1]>,
  InstrItinData<IIC_iCMPsr  , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<2, [SW_ALU0, SW_ALU1]>],
                              [1, 1, 1]>,
  //
  // Test instructions
  InstrItinData<IIC_iTSTi   , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0, SW_ALU1]>],
                              [1]>,
  InstrItinData<IIC_iTSTr   , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0, SW_ALU1]>],
                              [1, 1]>,
  InstrItinData<IIC_iTSTsi  , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<2, [SW_ALU0, SW_ALU1]>],
                              [1, 1]>,
  InstrItinData<IIC_iTSTsr  , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<2, [SW_ALU0, SW_ALU1]>],
                              [1, 1, 1]>,
  //
  // Move instructions, conditional
  // FIXME: Correctly model the extra input dep on the destination.
  InstrItinData<IIC_iCMOVi  , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0, SW_ALU1]>],
                              [1]>,
  InstrItinData<IIC_iCMOVr  , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0, SW_ALU1]>],
                              [1, 1]>,
  InstrItinData<IIC_iCMOVsi , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0, SW_ALU1]>],
                              [1, 1]>,
  InstrItinData<IIC_iCMOVsr , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0, SW_ALU1]>],
                              [2, 1, 1]>,
  InstrItinData<IIC_iCMOVix2, [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0, SW_ALU1]>,
                               InstrStage<1, [SW_ALU0, SW_ALU1]>],
                              [2]>,

  // Integer multiply pipeline
  //
  InstrItinData<IIC_iMUL16  , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [3, 1, 1]>,
  InstrItinData<IIC_iMAC16  , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [3, 1, 1, 1]>,
  InstrItinData<IIC_iMUL32  , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [4, 1, 1]>,
  InstrItinData<IIC_iMAC32  , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [4, 1, 1, 1]>,
  InstrItinData<IIC_iMUL64  , [InstrStage<1, [SW_DIS0], 0>,
                               InstrStage<1, [SW_DIS1], 0>,
                               InstrStage<1, [SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0], 1>,
                               InstrStage<1, [SW_ALU0], 3>,
                               InstrStage<1, [SW_ALU0]>],
                              [5, 5, 1, 1]>,
  InstrItinData<IIC_iMAC64  , [InstrStage<1, [SW_DIS0], 0>,
                               InstrStage<1, [SW_DIS1], 0>,
                               InstrStage<1, [SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0], 1>,
                               InstrStage<1, [SW_ALU0], 1>,
                               InstrStage<1, [SW_ALU0, SW_ALU1], 3>,
                               InstrStage<1, [SW_ALU0, SW_ALU1]>],
                              [5, 6, 1, 1]>,
  //
  // Integer divide
  InstrItinData<IIC_iDIV  , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                             InstrStage<1, [SW_ALU0], 0>,
                             InstrStage<14, [SW_IDIV]>],
                            [14, 1, 1]>,

  // Integer load pipeline
  // FIXME: The timings are some rough approximations
  //
  // Immediate offset
  InstrItinData<IIC_iLoad_i   , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                                 InstrStage<1, [SW_LS]>],
                                [3, 1]>,
  InstrItinData<IIC_iLoad_bh_i, [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                                 InstrStage<1, [SW_LS]>],
                                [3, 1]>,
  InstrItinData<IIC_iLoad_d_i , [InstrStage<1, [SW_DIS0], 0>,
                                 InstrStage<1, [SW_DIS1], 0>,
                                 InstrStage<1, [SW_LS], 1>,
                                 InstrStage<1, [SW_LS]>],
                                [3, 4, 1]>,
  //
  // Register offset
  InstrItinData<IIC_iLoad_r   , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                                 InstrStage<1, [SW_LS]>],
                                [3, 1, 1]>,
  InstrItinData<IIC_iLoad_bh_r, [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                                 InstrStage<1, [SW_LS]>],
                                [3, 1, 1]>,
  InstrItinData<IIC_iLoad_d_r , [InstrStage<1, [SW_DIS0], 0>,
                                 InstrStage<1, [SW_DIS1], 0>,
                                 InstrStage<1, [SW_DIS2], 0>,
                                 InstrStage<1, [SW_LS], 1>,
                                 InstrStage<1, [SW_LS], 3>,
                                 InstrStage<1, [SW_ALU0, SW_ALU1]>],
                                [3, 4, 1, 1]>,
  //
  // Scaled register offset
  InstrItinData<IIC_iLoad_si  , [InstrStage<1, [SW_DIS0], 0>,
                                 InstrStage<1, [SW_DIS1], 0>,
                                 InstrStage<1, [SW_ALU0, SW_ALU1], 2>,
                                 InstrStage<1, [SW_LS]>],
                                [5, 1, 1]>,
  InstrItinData<IIC_iLoad_bh_si,[InstrStage<1, [SW_DIS0], 0>,
                                 InstrStage<1, [SW_DIS1], 0>,
                                 InstrStage<1, [SW_ALU0, SW_ALU1], 2>,
                                 InstrStage<1, [SW_LS]>],
                                [5, 1, 1]>,
  //
  // Immediate offset with update
  InstrItinData<IIC_iLoad_iu  , [InstrStage<1, [SW_DIS0], 0>,
                                 InstrStage<1, [SW_DIS1], 0>,
                                 InstrStage<1, [SW_ALU0, SW_ALU1], 1>,
                                 InstrStage<1, [SW_LS]>],
                                [3, 1, 1]>,
  InstrItinData<IIC_iLoad_bh_iu,[InstrStage<1, [SW_DIS0], 0>,
                                 InstrStage<1, [SW_DIS1], 0>,
                                 InstrStage<1, [SW_ALU0, SW_ALU1], 1>,
                                 InstrStage<1, [SW_LS]>],
                                [3, 1, 1]>,
  //
  // Register offset with update
  InstrItinData<IIC_iLoad_ru  , [InstrStage<1, [SW_DIS0], 0>,
                                 InstrStage<1, [SW_DIS1], 0>,
                                 InstrStage<1, [SW_ALU0], 1>,
                                 InstrStage<1, [SW_LS]>],
                                [3, 1, 1, 1]>,
  InstrItinData<IIC_iLoad_bh_ru,[InstrStage<1, [SW_DIS0], 0>,
                                 InstrStage<1, [SW_DIS1], 0>,
                                 InstrStage<1, [SW_ALU0], 1>,
                                 InstrStage<1, [SW_LS]>],
                                [3, 1, 1, 1]>,
  InstrItinData<IIC_iLoad_d_ru, [InstrStage<1, [SW_DIS0], 0>,
                                 InstrStage<1, [SW_DIS1], 0>,
                                 InstrStage<1, [SW_DIS2], 0>,
                                 InstrStage<1, [SW_ALU0, SW_ALU1], 0>,
                                 InstrStage<1, [SW_LS], 3>,
                                 InstrStage<1, [SW_LS], 0>,
                                 InstrStage<1, [SW_ALU0, SW_ALU1]>],
                                [3, 4, 1, 1]>,
  //
  // Scaled register offset with update
  InstrItinData<IIC_iLoad_siu , [InstrStage<1, [SW_DIS0], 0>,
                                 InstrStage<1, [SW_DIS1], 0>,
                                 InstrStage<1, [SW_DIS2], 0>,
                                 InstrStage<1, [SW_ALU0, SW_ALU1], 2>,
                                 InstrStage<1, [SW_LS], 3>,
                                 InstrStage<1, [SW_ALU0, SW_ALU1]>],
                                [5, 3, 1, 1]>,
  InstrItinData<IIC_iLoad_bh_siu,[InstrStage<1, [SW_DIS0], 0>,
                                  InstrStage<1, [SW_DIS1], 0>,
                                  InstrStage<1, [SW_DIS2], 0>,
                                  InstrStage<1, [SW_ALU0, SW_ALU1], 2>,
                                  InstrStage<1, [SW_LS], 0>,
                                  InstrStage<1, [SW_ALU0, SW_ALU1]>],
                                [5, 3, 1, 1]>,
  //
  // Load multiple, def is the 5th operand.
  // FIXME: This assumes 3 to 4 registers.
  InstrItinData<IIC_iLoad_m  , [InstrStage<1, [SW_DIS0], 0>,
                                InstrStage<1, [SW_DIS1], 0>,
                                InstrStage<1, [SW_DIS2], 0>,
                                InstrStage<1, [SW_ALU0, SW_ALU1], 1>,
                                InstrStage<1, [SW_LS]>],
                               [1, 1, 1, 1, 3], [], -1>, // dynamic uops

  //
  // Load multiple + update, defs are the 1st and 5th operands.
  InstrItinData<IIC_iLoad_mu , [InstrStage<1, [SW_DIS0], 0>,
                                InstrStage<1, [SW_DIS1], 0>,
                                InstrStage<1, [SW_DIS2], 0>,
                                InstrStage<1, [SW_ALU0, SW_ALU1], 0>,
                                InstrStage<1, [SW_LS], 3>,
                                InstrStage<1, [SW_ALU0, SW_ALU1]>],
                               [2, 1, 1, 1, 3], [], -1>, // dynamic uops
  //
  // Load multiple plus branch
  InstrItinData<IIC_iLoad_mBr, [InstrStage<1, [SW_DIS0], 0>,
                                InstrStage<1, [SW_DIS1], 0>,
                                InstrStage<1, [SW_DIS2], 0>,
                                InstrStage<1, [SW_ALU0, SW_ALU1], 1>,
                                InstrStage<1, [SW_LS]>],
                               [1, 1, 1, 1, 3], [], -1>, // dynamic uops
  //
  // Pop, def is the 3rd operand.
  InstrItinData<IIC_iPop  ,    [InstrStage<1, [SW_DIS0], 0>,
                                InstrStage<1, [SW_DIS1], 0>,
                                InstrStage<1, [SW_ALU0, SW_ALU1], 1>,
                                InstrStage<1, [SW_LS]>],
                               [1, 1, 3], [], -1>, // dynamic uops
  //
  // Pop + branch, def is the 3rd operand.
  InstrItinData<IIC_iPop_Br,   [InstrStage<1, [SW_DIS0], 0>,
                                InstrStage<1, [SW_DIS1], 0>,
                                InstrStage<1, [SW_ALU0, SW_ALU1], 1>,
                                InstrStage<1, [SW_LS]>],
                               [1, 1, 3], [], -1>, // dynamic uops

  //
  // iLoadi + iALUr for t2LDRpci_pic.
  InstrItinData<IIC_iLoadiALU, [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                                InstrStage<1, [SW_LS], 3>,
                                InstrStage<1, [SW_ALU0, SW_ALU1]>],
                               [4, 1]>,

  // Integer store pipeline
  ///
  // Immediate offset
  InstrItinData<IIC_iStore_i  , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                                 InstrStage<1, [SW_LS]>],
                                [1, 1]>,
  InstrItinData<IIC_iStore_bh_i,[InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                                 InstrStage<1, [SW_LS]>],
                                [1, 1]>,
  InstrItinData<IIC_iStore_d_i, [InstrStage<1, [SW_DIS0], 0>,
                                 InstrStage<1, [SW_DIS1], 0>,
                                 InstrStage<1, [SW_DIS2], 0>,
                                 InstrStage<1, [SW_LS], 0>,
                                 InstrStage<1, [SW_ALU0, SW_ALU1], 1>,
                                 InstrStage<1, [SW_LS]>],
                                [1, 1]>,
  //
  // Register offset
  InstrItinData<IIC_iStore_r  , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                                 InstrStage<1, [SW_LS]>],
                                [1, 1, 1]>,
  InstrItinData<IIC_iStore_bh_r,[InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                                 InstrStage<1, [SW_LS]>],
                                [1, 1, 1]>,
  InstrItinData<IIC_iStore_d_r, [InstrStage<1, [SW_DIS0], 0>,
                                 InstrStage<1, [SW_DIS1], 0>,
                                 InstrStage<1, [SW_DIS2], 0>,
                                 InstrStage<1, [SW_LS], 0>,
                                 InstrStage<1, [SW_ALU0, SW_ALU1], 1>,
                                 InstrStage<1, [SW_LS]>],
                                [1, 1, 1]>,
  //
  // Scaled register offset
  InstrItinData<IIC_iStore_si ,  [InstrStage<1, [SW_DIS0], 0>,
                                  InstrStage<1, [SW_DIS1], 0>,
                                  InstrStage<1, [SW_ALU0, SW_ALU1], 2>,
                                  InstrStage<1, [SW_LS]>],
                                 [1, 1, 1]>,
  InstrItinData<IIC_iStore_bh_si,[InstrStage<1, [SW_DIS0], 0>,
                                  InstrStage<1, [SW_DIS1], 0>,
                                  InstrStage<1, [SW_ALU0, SW_ALU1], 2>,
                                  InstrStage<1, [SW_LS]>],
                                 [1, 1, 1]>,
  //
  // Immediate offset with update
  InstrItinData<IIC_iStore_iu ,  [InstrStage<1, [SW_DIS0], 0>,
                                  InstrStage<1, [SW_DIS1], 0>,
                                  InstrStage<1, [SW_ALU0, SW_ALU1], 1>,
                                  InstrStage<1, [SW_LS]>],
                                 [1, 1, 1]>,
  InstrItinData<IIC_iStore_bh_iu,[InstrStage<1, [SW_DIS0], 0>,
                                  InstrStage<1, [SW_DIS1], 0>,
                                  InstrStage<1, [SW_ALU0, SW_ALU1], 1>,
                                  InstrStage<1, [SW_LS]>],
                                 [1, 1, 1]>,
  //
  // Register offset with update
  InstrItinData<IIC_iStore_ru ,  [InstrStage<1, [SW_DIS0], 0>,
                                  InstrStage<1, [SW_DIS1], 0>,
                                  InstrStage<1, [SW_ALU0, SW_ALU1], 1>,
                                  InstrStage<1, [SW_LS]>],
                                 [1, 1, 1, 1]>,
  InstrItinData<IIC_iStore_bh_ru,[InstrStage<1, [SW_DIS0], 0>,
                                  InstrStage<1, [SW_DIS1], 0>,
                                  InstrStage<1, [SW_ALU0, SW_ALU1], 1>,
                                  InstrStage<1, [SW_LS]>],
                                 [1, 1, 1, 1]>,
  InstrItinData<IIC_iStore_d_ru, [InstrStage<1, [SW_DIS0], 0>,
                                  InstrStage<1, [SW_DIS1], 0>,
                                  InstrStage<1, [SW_ALU0, SW_ALU1], 1>,
                                  InstrStage<1, [SW_LS]>],
                                 [1, 1, 1, 1]>,
  //
  // Scaled register offset with update
  InstrItinData<IIC_iStore_siu,    [InstrStage<1, [SW_DIS0], 0>,
                                    InstrStage<1, [SW_DIS1], 0>,
                                    InstrStage<1, [SW_ALU0, SW_ALU1], 2>,
                                    InstrStage<1, [SW_LS], 0>,
                                    InstrStage<1, [SW_ALU0, SW_ALU1], 1>],
                                   [3, 1, 1, 1]>,
  InstrItinData<IIC_iStore_bh_siu, [InstrStage<1, [SW_DIS0], 0>,
                                    InstrStage<1, [SW_DIS1], 0>,
                                    InstrStage<1, [SW_ALU0, SW_ALU1], 2>,
                                    InstrStage<1, [SW_LS], 0>,
                                    InstrStage<1, [SW_ALU0, SW_ALU1], 1>],
                                   [3, 1, 1, 1]>,
  //
  // Store multiple
  InstrItinData<IIC_iStore_m , [InstrStage<1, [SW_DIS0], 0>,
                                InstrStage<1, [SW_DIS1], 0>,
                                InstrStage<1, [SW_DIS2], 0>,
                                InstrStage<1, [SW_ALU0, SW_ALU1], 1>,
                                InstrStage<1, [SW_LS], 1>,
                                InstrStage<1, [SW_ALU0, SW_ALU1], 1>,
                                InstrStage<1, [SW_LS], 1>,
                                InstrStage<1, [SW_ALU0, SW_ALU1], 1>,
                                InstrStage<1, [SW_LS]>],
                                [], [], -1>, // dynamic uops
  //
  // Store multiple + update
  InstrItinData<IIC_iStore_mu, [InstrStage<1, [SW_DIS0], 0>,
                                InstrStage<1, [SW_DIS1], 0>,
                                InstrStage<1, [SW_DIS2], 0>,
                                InstrStage<1, [SW_ALU0, SW_ALU1], 1>,
                                InstrStage<1, [SW_LS], 1>,
                                InstrStage<1, [SW_ALU0, SW_ALU1], 1>,
                                InstrStage<1, [SW_LS], 1>,
                                InstrStage<1, [SW_ALU0, SW_ALU1], 1>,
                                InstrStage<1, [SW_LS]>],
                               [2], [], -1>, // dynamic uops

  //
  // Preload
  InstrItinData<IIC_Preload,   [InstrStage<1, [SW_DIS0], 0>], [1, 1]>,

  // Branch
  //
  // no delay slots, so the latency of a branch is unimportant
  InstrItinData<IIC_Br       , [InstrStage<1, [SW_DIS0], 0>]>,

  // FP Special Register to Integer Register File Move
  InstrItinData<IIC_fpSTAT , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                              InstrStage<1, [SW_ALU0, SW_ALU1]>],
                             [1]>,
  //
  // Single-precision FP Unary
  //
  // Most floating-point moves get issued on ALU0.
  InstrItinData<IIC_fpUNA32 , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [2, 1]>,
  //
  // Double-precision FP Unary
  InstrItinData<IIC_fpUNA64 , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [2, 1]>,

  //
  // Single-precision FP Compare
  InstrItinData<IIC_fpCMP32 , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [1, 1]>,
  //
  // Double-precision FP Compare
  InstrItinData<IIC_fpCMP64 , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [1, 1]>,
  //
  // Single to Double FP Convert
  InstrItinData<IIC_fpCVTSD , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [4, 1]>,
  //
  // Double to Single FP Convert
  InstrItinData<IIC_fpCVTDS , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [4, 1]>,

  //
  // Single to Half FP Convert
  InstrItinData<IIC_fpCVTSH , [InstrStage<1, [SW_DIS0], 0>,
                               InstrStage<1, [SW_DIS1], 0>,
                               InstrStage<1, [SW_ALU1], 4>,
                               InstrStage<1, [SW_ALU1]>],
                              [6, 1]>,
  //
  // Half to Single FP Convert
  InstrItinData<IIC_fpCVTHS , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [4, 1]>,

  //
  // Single-Precision FP to Integer Convert
  InstrItinData<IIC_fpCVTSI , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [4, 1]>,
  //
  // Double-Precision FP to Integer Convert
  InstrItinData<IIC_fpCVTDI , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [4, 1]>,
  //
  // Integer to Single-Precision FP Convert
  InstrItinData<IIC_fpCVTIS , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [4, 1]>,
  //
  // Integer to Double-Precision FP Convert
  InstrItinData<IIC_fpCVTID , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [4, 1]>,
  //
  // Single-precision FP ALU
  InstrItinData<IIC_fpALU32 , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [2, 1, 1]>,
  //
  // Double-precision FP ALU
  InstrItinData<IIC_fpALU64 , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [2, 1, 1]>,
  //
  // Single-precision FP Multiply
  InstrItinData<IIC_fpMUL32 , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [4, 1, 1]>,
  //
  // Double-precision FP Multiply
  InstrItinData<IIC_fpMUL64 , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [6, 1, 1]>,
  //
  // Single-precision FP MAC
  InstrItinData<IIC_fpMAC32 , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [8, 1, 1]>,
  //
  // Double-precision FP MAC
  InstrItinData<IIC_fpMAC64 , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [12, 1, 1]>,
  //
  // Single-precision Fused FP MAC
  InstrItinData<IIC_fpFMAC32, [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [8, 1, 1]>,
  //
  // Double-precision Fused FP MAC
  InstrItinData<IIC_fpFMAC64, [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [12, 1, 1]>,
  //
  // Single-precision FP DIV
  InstrItinData<IIC_fpDIV32 , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1], 0>,
                               InstrStage<15, [SW_FDIV]>],
                              [17, 1, 1]>,
  //
  // Double-precision FP DIV
  InstrItinData<IIC_fpDIV64 , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1], 0>,
                               InstrStage<30, [SW_FDIV]>],
                              [32, 1, 1]>,
  //
  // Single-precision FP SQRT
  InstrItinData<IIC_fpSQRT32, [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1], 0>,
                               InstrStage<15, [SW_FDIV]>],
                              [17, 1]>,
  //
  // Double-precision FP SQRT
  InstrItinData<IIC_fpSQRT64, [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1], 0>,
                               InstrStage<30, [SW_FDIV]>],
                              [32, 1, 1]>,

  //
  // Integer to Single-precision Move
  InstrItinData<IIC_fpMOVIS,  [InstrStage<1, [SW_DIS0], 0>,
                               InstrStage<1, [SW_DIS1], 0>,
                               InstrStage<1, [SW_LS], 4>,
                               InstrStage<1, [SW_ALU0]>],
                              [6, 1]>,
  //
  // Integer to Double-precision Move
  InstrItinData<IIC_fpMOVID,  [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_LS]>],
                              [4, 1]>,
  //
  // Single-precision to Integer Move
  InstrItinData<IIC_fpMOVSI,  [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_LS]>],
                              [3, 1]>,
  //
  // Double-precision to Integer Move
  InstrItinData<IIC_fpMOVDI,  [InstrStage<1, [SW_DIS0], 0>,
                               InstrStage<1, [SW_DIS1], 0>,
                               InstrStage<1, [SW_LS], 3>,
                               InstrStage<1, [SW_LS]>],
                              [3, 4, 1]>,
  //
  // Single-precision FP Load
  InstrItinData<IIC_fpLoad32, [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_LS]>],
                              [4, 1]>,
  //
  // Double-precision FP Load
  InstrItinData<IIC_fpLoad64, [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_LS]>],
                              [4, 1]>,
  //
  // FP Load Multiple
  // FIXME: Assumes a single Q register.
  InstrItinData<IIC_fpLoad_m, [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_LS]>],
                              [1, 1, 1, 4], [], -1>, // dynamic uops
  //
  // FP Load Multiple + update
  // FIXME: Assumes a single Q register.
  InstrItinData<IIC_fpLoad_mu,[InstrStage<1, [SW_DIS0], 0>,
                               InstrStage<1, [SW_DIS1], 0>,
                               InstrStage<1, [SW_LS], 4>,
                               InstrStage<1, [SW_ALU0, SW_ALU1]>],
                              [2, 1, 1, 1, 4], [], -1>, // dynamic uops
  //
  // Single-precision FP Store
  InstrItinData<IIC_fpStore32,[InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_LS]>],
                              [1, 1]>,
  //
  // Double-precision FP Store
  InstrItinData<IIC_fpStore64,[InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_LS]>],
                              [1, 1]>,
  //
  // FP Store Multiple
  // FIXME: Assumes a single Q register.
  InstrItinData<IIC_fpStore_m,[InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_LS]>],
                              [1, 1, 1], [], -1>, // dynamic uops
  //
  // FP Store Multiple + update
  // FIXME: Assumes a single Q register.
  InstrItinData<IIC_fpStore_mu,[InstrStage<1, [SW_DIS0], 0>,
                                InstrStage<1, [SW_DIS1], 0>,
                                InstrStage<1, [SW_LS], 4>,
                                InstrStage<1, [SW_ALU0, SW_ALU1]>],
                               [2, 1, 1, 1], [], -1>, // dynamic uops
  // NEON
  //
  // Double-register Integer Unary
  InstrItinData<IIC_VUNAiD,   [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [4, 1]>,
  //
  // Quad-register Integer Unary
  InstrItinData<IIC_VUNAiQ,   [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [4, 1]>,
  //
  // Double-register Integer Q-Unary
  InstrItinData<IIC_VQUNAiD,  [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [4, 1]>,
  //
  // Quad-register Integer CountQ-Unary
  InstrItinData<IIC_VQUNAiQ,  [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [4, 1]>,
  //
  // Double-register Integer Binary
  InstrItinData<IIC_VBINiD,   [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [2, 1, 1]>,
  //
  // Quad-register Integer Binary
  InstrItinData<IIC_VBINiQ,   [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [2, 1, 1]>,
  //
  // Double-register Integer Subtract
  InstrItinData<IIC_VSUBiD,   [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [2, 1, 1]>,
  //
  // Quad-register Integer Subtract
  InstrItinData<IIC_VSUBiQ,   [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [2, 1, 1]>,
  //
  // Double-register Integer Shift
  InstrItinData<IIC_VSHLiD,   [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [2, 1, 1]>,
  //
  // Quad-register Integer Shift
  InstrItinData<IIC_VSHLiQ,   [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [2, 1, 1]>,
  //
  // Double-register Integer Shift (4 cycle)
  InstrItinData<IIC_VSHLi4D,  [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [4, 1, 1]>,
  //
  // Quad-register Integer Shift (4 cycle)
  InstrItinData<IIC_VSHLi4Q,  [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [4, 1, 1]>,
  //
  // Double-register Integer Binary (4 cycle)
  InstrItinData<IIC_VBINi4D,  [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [4, 1, 1]>,
  //
  // Quad-register Integer Binary (4 cycle)
  InstrItinData<IIC_VBINi4Q,  [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [4, 1, 1]>,
  //
  // Double-register Integer Subtract (4 cycle)
  InstrItinData<IIC_VSUBi4D,  [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [4, 1, 1]>,
  //
  // Quad-register Integer Subtract (4 cycle)
  InstrItinData<IIC_VSUBi4Q,  [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [4, 1, 1]>,

  //
  // Double-register Integer Count
  InstrItinData<IIC_VCNTiD,   [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [2, 1, 1]>,
  //
  // Quad-register Integer Count
  InstrItinData<IIC_VCNTiQ,   [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [2, 1, 1]>,
  //
  // Double-register Absolute Difference and Accumulate
  InstrItinData<IIC_VABAD,    [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [4, 1, 1, 1]>,
  //
  // Quad-register Absolute Difference and Accumulate
  InstrItinData<IIC_VABAQ,    [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [4, 1, 1, 1]>,
  //
  // Double-register Integer Pair Add Long
  InstrItinData<IIC_VPALiD,   [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [4, 1, 1]>,
  //
  // Quad-register Integer Pair Add Long
  InstrItinData<IIC_VPALiQ,   [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [4, 1, 1]>,

  //
  // Double-register Integer Multiply (.8, .16)
  InstrItinData<IIC_VMULi16D, [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [4, 1, 1]>,
  //
  // Quad-register Integer Multiply (.8, .16)
  InstrItinData<IIC_VMULi16Q, [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [4, 1, 1]>,

  //
  // Double-register Integer Multiply (.32)
  InstrItinData<IIC_VMULi32D, [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [4, 1, 1]>,
  //
  // Quad-register Integer Multiply (.32)
  InstrItinData<IIC_VMULi32Q, [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [4, 1, 1]>,
  //
  // Double-register Integer Multiply-Accumulate (.8, .16)
  InstrItinData<IIC_VMACi16D, [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [4, 1, 1, 1]>,
  //
  // Double-register Integer Multiply-Accumulate (.32)
  InstrItinData<IIC_VMACi32D, [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [4, 1, 1, 1]>,
  //
  // Quad-register Integer Multiply-Accumulate (.8, .16)
  InstrItinData<IIC_VMACi16Q, [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [4, 1, 1, 1]>,
  //
  // Quad-register Integer Multiply-Accumulate (.32)
  InstrItinData<IIC_VMACi32Q, [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [4, 1, 1, 1]>,

  //
  // Move
  InstrItinData<IIC_VMOV,     [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [2, 1]>,
  //
  // Move Immediate
  InstrItinData<IIC_VMOVImm,  [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [2]>,
  //
  // Double-register Permute Move
  InstrItinData<IIC_VMOVD,    [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [2, 1]>,
  //
  // Quad-register Permute Move
  InstrItinData<IIC_VMOVQ,    [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [2, 1]>,
  //
  // Integer to Single-precision Move
  InstrItinData<IIC_VMOVIS ,  [InstrStage<1, [SW_DIS0], 0>,
                               InstrStage<1, [SW_DIS1], 0>,
                               InstrStage<1, [SW_LS], 4>,
                               InstrStage<1, [SW_ALU0]>],
                              [6, 1]>,
  //
  // Integer to Double-precision Move
  InstrItinData<IIC_VMOVID ,  [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_LS]>],
                              [4, 1, 1]>,
  //
  // Single-precision to Integer Move
  InstrItinData<IIC_VMOVSI ,  [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_LS]>],
                              [3, 1]>,
  //
  // Double-precision to Integer Move
  InstrItinData<IIC_VMOVDI ,  [InstrStage<1, [SW_DIS0], 0>,
                               InstrStage<1, [SW_DIS1], 0>,
                               InstrStage<1, [SW_LS], 3>,
                               InstrStage<1, [SW_LS]>],
                              [3, 4, 1]>,
  //
  // Integer to Lane Move
  // FIXME: I think this is correct, but it is not clear from the tuning guide.
  InstrItinData<IIC_VMOVISL , [InstrStage<1, [SW_DIS0], 0>,
                               InstrStage<1, [SW_DIS1], 0>,
                               InstrStage<1, [SW_LS], 4>,
                               InstrStage<1, [SW_ALU0]>],
                              [6, 1]>,

  //
  // Vector narrow move
  InstrItinData<IIC_VMOVN,    [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [2, 1]>,
  //
  // Double-register FP Unary
  // FIXME: VRECPE / VRSQRTE has a longer latency than VABS, which is used here,
  //        and they issue on a different pipeline.
  InstrItinData<IIC_VUNAD,    [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [2, 1]>,
  //
  // Quad-register FP Unary
  // FIXME: VRECPE / VRSQRTE has a longer latency than VABS, which is used here,
  //        and they issue on a different pipeline.
  InstrItinData<IIC_VUNAQ,    [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [2, 1]>,
  //
  // Double-register FP Binary
  // FIXME: We're using this itin for many instructions.
  InstrItinData<IIC_VBIND,    [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [4, 1, 1]>,

  //
  // VPADD, etc.
  InstrItinData<IIC_VPBIND,   [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [4, 1, 1]>,
  //
  // Double-register FP VMUL
  InstrItinData<IIC_VFMULD,   [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [4, 1, 1]>,
  //
  // Quad-register FP Binary
  InstrItinData<IIC_VBINQ,    [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [4, 1, 1]>,
  //
  // Quad-register FP VMUL
  InstrItinData<IIC_VFMULQ,   [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [4, 1, 1]>,
  //
  // Double-register FP Multiple-Accumulate
  InstrItinData<IIC_VMACD,    [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [8, 1, 1]>,
  //
  // Quad-register FP Multiple-Accumulate
  InstrItinData<IIC_VMACQ,    [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [8, 1, 1]>,
  //
  // Double-register Fused FP Multiple-Accumulate
  InstrItinData<IIC_VFMACD,   [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [8, 1, 1]>,
  //
  // Quad-register FusedF P Multiple-Accumulate
  InstrItinData<IIC_VFMACQ,   [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [8, 1, 1]>,
  //
  // Double-register Reciprical Step
  InstrItinData<IIC_VRECSD,   [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [8, 1, 1]>,
  //
  // Quad-register Reciprical Step
  InstrItinData<IIC_VRECSQ,   [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [8, 1, 1]>,
  //
  // Double-register Permute
  // FIXME: The latencies are unclear from the documentation.
  InstrItinData<IIC_VPERMD,   [InstrStage<1, [SW_DIS0], 0>,
                               InstrStage<1, [SW_DIS1], 0>,
                               InstrStage<1, [SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1], 2>,
                               InstrStage<1, [SW_ALU1], 2>,
                               InstrStage<1, [SW_ALU1]>],
                              [3, 4, 3, 4]>,
  //
  // Quad-register Permute
  // FIXME: The latencies are unclear from the documentation.
  InstrItinData<IIC_VPERMQ,   [InstrStage<1, [SW_DIS0], 0>,
                               InstrStage<1, [SW_DIS1], 0>,
                               InstrStage<1, [SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1], 2>,
                               InstrStage<1, [SW_ALU1], 2>,
                               InstrStage<1, [SW_ALU1]>],
                              [3, 4, 3, 4]>,
  //
  // Quad-register Permute (3 cycle issue on A9)
  InstrItinData<IIC_VPERMQ3,  [InstrStage<1, [SW_DIS0], 0>,
                               InstrStage<1, [SW_DIS1], 0>,
                               InstrStage<1, [SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1], 2>,
                               InstrStage<1, [SW_ALU1], 2>,
                               InstrStage<1, [SW_ALU1]>],
                              [3, 4, 3, 4]>,

  //
  // Double-register VEXT
  InstrItinData<IIC_VEXTD,    [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [2, 1, 1]>,
  //
  // Quad-register VEXT
  InstrItinData<IIC_VEXTQ,    [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [2, 1, 1]>,
  //
  // VTB
  InstrItinData<IIC_VTB1,     [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [2, 1, 1]>,
  InstrItinData<IIC_VTB2,     [InstrStage<1, [SW_DIS0], 0>,
                               InstrStage<1, [SW_DIS1], 0>,
                               InstrStage<1, [SW_ALU1], 2>,
                               InstrStage<1, [SW_ALU1]>],
                              [4, 1, 3, 3]>,
  InstrItinData<IIC_VTB3,     [InstrStage<1, [SW_DIS0], 0>,
                               InstrStage<1, [SW_DIS1], 0>,
                               InstrStage<1, [SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1], 2>,
                               InstrStage<1, [SW_ALU1], 2>,
                               InstrStage<1, [SW_ALU1]>],
                              [6, 1, 3, 5, 5]>,
  InstrItinData<IIC_VTB4,     [InstrStage<1, [SW_DIS0], 0>,
                               InstrStage<1, [SW_DIS1], 0>,
                               InstrStage<1, [SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1], 2>,
                               InstrStage<1, [SW_ALU1], 2>,
                               InstrStage<1, [SW_ALU1], 2>,
                               InstrStage<1, [SW_ALU1]>],
                              [8, 1, 3, 5, 7, 7]>,
  //
  // VTBX
  InstrItinData<IIC_VTBX1,    [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [2, 1, 1]>,
  InstrItinData<IIC_VTBX2,    [InstrStage<1, [SW_DIS0], 0>,
                               InstrStage<1, [SW_DIS1], 0>,
                               InstrStage<1, [SW_ALU1], 2>,
                               InstrStage<1, [SW_ALU1]>],
                              [4, 1, 3, 3]>,
  InstrItinData<IIC_VTBX3,    [InstrStage<1, [SW_DIS0], 0>,
                               InstrStage<1, [SW_DIS1], 0>,
                               InstrStage<1, [SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1], 2>,
                               InstrStage<1, [SW_ALU1], 2>,
                               InstrStage<1, [SW_ALU1]>],
                              [6, 1, 3, 5, 5]>,
  InstrItinData<IIC_VTBX4,    [InstrStage<1, [SW_DIS0], 0>,
                               InstrStage<1, [SW_DIS1], 0>,
                               InstrStage<1, [SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1], 2>,
                               InstrStage<1, [SW_ALU1], 2>,
                               InstrStage<1, [SW_ALU1], 2>,
                               InstrStage<1, [SW_ALU1]>],
                              [8, 1, 3, 5, 7, 7]>
]>;

// ===---------------------------------------------------------------------===//
// This following definitions describe the simple machine model which
// will replace itineraries.

// Swift machine model for scheduling and other instruction cost heuristics.
def SwiftModel : SchedMachineModel {
  let IssueWidth = 3; // 3 micro-ops are dispatched per cycle.
  let MicroOpBufferSize = 45; // Based on NEON renamed registers.
  let LoadLatency = 3;
  let MispredictPenalty = 14; // A branch direction mispredict.

  let Itineraries = SwiftItineraries;
}

// Swift predicates.
def IsFastImmShiftSwiftPred : SchedPredicate<[{TII->isSwiftFastImmShift(MI)}]>;

// Swift resource mapping.
let SchedModel = SwiftModel in {
  // Processor resources.
  def SwiftUnitP01 : ProcResource<2>; // ALU unit.
  def SwiftUnitP0 : ProcResource<1> { let Super = SwiftUnitP01; } // Mul unit.
  def SwiftUnitP1 : ProcResource<1> { let Super = SwiftUnitP01; } // Br unit.
  def SwiftUnitP2 : ProcResource<1>; // LS unit.
  def SwiftUnitDiv : ProcResource<1>;

  // Generic resource requirements.
  def SwiftWriteP0OneCycle : SchedWriteRes<[SwiftUnitP0]>;
  def SwiftWriteP0TwoCycle : SchedWriteRes<[SwiftUnitP0]> { let Latency = 2; }
  def SwiftWriteP0FourCycle : SchedWriteRes<[SwiftUnitP0]> { let Latency = 4; }
  def SwiftWriteP0SixCycle : SchedWriteRes<[SwiftUnitP0]> { let Latency = 6; }
  def SwiftWriteP0P1FourCycle : SchedWriteRes<[SwiftUnitP0, SwiftUnitP1]> {
    let Latency = 4;
  }
  def SwiftWriteP0P1SixCycle : SchedWriteRes<[SwiftUnitP0, SwiftUnitP1]> {
    let Latency = 6;
  }
  def SwiftWriteP01OneCycle : SchedWriteRes<[SwiftUnitP01]>;
  def SwiftWriteP1TwoCycle : SchedWriteRes<[SwiftUnitP1]> { let Latency = 2; }
  def SwiftWriteP1FourCycle : SchedWriteRes<[SwiftUnitP1]> { let Latency = 4; }
  def SwiftWriteP1SixCycle : SchedWriteRes<[SwiftUnitP1]> { let Latency = 6; }
  def SwiftWriteP1EightCycle : SchedWriteRes<[SwiftUnitP1]> { let Latency = 8; }
  def SwiftWriteP1TwelveCyc : SchedWriteRes<[SwiftUnitP1]> { let Latency = 12; }
  def SwiftWriteP01OneCycle2x : WriteSequence<[SwiftWriteP01OneCycle], 2>;
  def SwiftWriteP01OneCycle3x : WriteSequence<[SwiftWriteP01OneCycle], 3>;
  def SwiftWriteP01TwoCycle : SchedWriteRes<[SwiftUnitP01]> { let Latency = 2; }
  def SwiftWriteP01ThreeCycleTwoUops : SchedWriteRes<[SwiftUnitP01,
                                                      SwiftUnitP01]> {
    let Latency = 3;
    let NumMicroOps = 2;
  }
  def SwiftWriteP0ThreeCycleThreeUops : SchedWriteRes<[SwiftUnitP0]> {
    let Latency = 3;
    let NumMicroOps = 3;
    let ResourceCycles = [3];
  }
  // Plain load without writeback.
  def SwiftWriteP2ThreeCycle : SchedWriteRes<[SwiftUnitP2]> {
    let Latency = 3;
  }
  def SwiftWriteP2FourCycle : SchedWriteRes<[SwiftUnitP2]> {
    let Latency = 4;
  }
  // A store does not write to a register.
  def SwiftWriteP2 : SchedWriteRes<[SwiftUnitP2]> {
    let Latency = 0;
  }
  foreach Num = 1-4 in {
    def SwiftWrite#Num#xP2 : WriteSequence<[SwiftWriteP2], Num>;
  }
  def SwiftWriteP01OneCycle2x_load : WriteSequence<[SwiftWriteP01OneCycle,
                                                    SwiftWriteP01OneCycle,
                                                    SwiftWriteP2ThreeCycle]>;
  // 4.2.4 Arithmetic and Logical.
  // ALU operation register shifted by immediate variant.
  def SwiftWriteALUsi : SchedWriteVariant<[
    // lsl #2, lsl #1, or lsr #1.
    SchedVar<IsFastImmShiftSwiftPred, [SwiftWriteP01TwoCycle]>,
    SchedVar<NoSchedPred,             [WriteALU]>
  ]>;
  def SwiftWriteALUsr : SchedWriteVariant<[
    SchedVar<IsPredicatedPred, [SwiftWriteP01ThreeCycleTwoUops]>,
    SchedVar<NoSchedPred,      [SwiftWriteP01TwoCycle]>
  ]>;
  def SwiftWriteALUSsr : SchedWriteVariant<[
    SchedVar<IsPredicatedPred, [SwiftWriteP0ThreeCycleThreeUops]>,
    SchedVar<NoSchedPred,      [SwiftWriteP01TwoCycle]>
  ]>;
  def SwiftReadAdvanceALUsr : SchedReadVariant<[
    SchedVar<IsPredicatedPred, [SchedReadAdvance<2>]>,
    SchedVar<NoSchedPred,      [NoReadAdvance]>
  ]>;
  // ADC,ADD,NEG,RSB,RSC,SBC,SUB,ADR
  // AND,BIC,EOR,ORN,ORR
  // CLZ,RBIT,REV,REV16,REVSH,PKH
  def : WriteRes<WriteALU, [SwiftUnitP01]>;
  def : SchedAlias<WriteALUsi, SwiftWriteALUsi>;
  def : SchedAlias<WriteALUsr, SwiftWriteALUsr>;
  def : SchedAlias<WriteALUSsr, SwiftWriteALUSsr>;
  def : ReadAdvance<ReadALU, 0>;
  def : SchedAlias<ReadALUsr, SwiftReadAdvanceALUsr>;


  def SwiftChooseShiftKindP01OneOrTwoCycle : SchedWriteVariant<[
    SchedVar<IsFastImmShiftSwiftPred, [SwiftWriteP01OneCycle]>,
    SchedVar<NoSchedPred,             [SwiftWriteP01TwoCycle]>
  ]>;

  // 4.2.5 Integer comparison
  def : WriteRes<WriteCMP, [SwiftUnitP01]>;
  def : SchedAlias<WriteCMPsi, SwiftChooseShiftKindP01OneOrTwoCycle>;
  def : SchedAlias<WriteCMPsr, SwiftWriteP01TwoCycle>;

  // 4.2.6 Shift, Move
  // Shift
  //  ASR,LSL,ROR,RRX
  //  MOV(register-shiftedregister)  MVN(register-shiftedregister)
  // Move
  //  MOV,MVN
  //  MOVT
  // Sign/Zero extension
  def : InstRW<[SwiftWriteP01OneCycle],
               (instregex "SXTB", "SXTH", "SXTB16", "UXTB", "UXTH", "UXTB16",
                          "t2SXTB", "t2SXTH", "t2SXTB16", "t2UXTB", "t2UXTH",
                          "t2UXTB16")>;
  // Pseudo instructions.
  def : InstRW<[SwiftWriteP01OneCycle2x],
        (instregex "MOVCCi32imm", "MOVi32imm", "MOV_ga_dyn", "t2MOVCCi32imm",
                   "t2MOVi32imm", "t2MOV_ga_dyn")>;
  def : InstRW<[SwiftWriteP01OneCycle3x],
        (instregex "MOV_ga_pcrel", "t2MOV_ga_pcrel", "t2MOVi16_ga_pcrel")>;
  def : InstRW<[SwiftWriteP01OneCycle2x_load],
        (instregex "MOV_ga_pcrel_ldr", "t2MOV_ga_pcrel_ldr")>;

  def SwiftWriteP0TwoCyleTwoUops : WriteSequence<[SwiftWriteP0OneCycle], 2>;

  def SwiftPredP0OneOrTwoCycle : SchedWriteVariant<[
    SchedVar<IsPredicatedPred, [ SwiftWriteP0TwoCyleTwoUops ]>,
    SchedVar<NoSchedPred,     [ SwiftWriteP0OneCycle ]>
  ]>;

  // 4.2.7 Select
  // SEL
  def : InstRW<[SwiftPredP0OneOrTwoCycle], (instregex "SEL", "t2SEL")>;

  // 4.2.8 Bitfield
  // BFI,BFC, SBFX,UBFX
  def : InstRW< [SwiftWriteP01TwoCycle],
        (instregex "BFC", "BFI", "UBFX", "SBFX", "(t|t2)BFC", "(t|t2)BFI",
        "(t|t2)UBFX", "(t|t2)SBFX")>;

  // 4.2.9 Saturating arithmetic
  def : InstRW< [SwiftWriteP01TwoCycle],
        (instregex "QADD", "QSUB", "QDADD", "QDSUB", "SSAT", "SSAT16", "USAT",
        "USAT16", "QADD8", "QADD16", "QSUB8", "QSUB16", "QASX", "QSAX",
        "UQADD8", "UQADD16","UQSUB8","UQSUB16","UQASX","UQSAX", "t2QADD",
        "t2QSUB", "t2QDADD", "t2QDSUB", "t2SSAT", "t2SSAT16", "t2USAT",
        "t2QADD8", "t2QADD16", "t2QSUB8", "t2QSUB16", "t2QASX", "t2QSAX",
        "t2UQADD8", "t2UQADD16","t2UQSUB8","t2UQSUB16","t2UQASX","t2UQSAX")>;

  // 4.2.10 Parallel Arithmetic
  // Not flag setting.
  def : InstRW< [SwiftWriteALUsr],
        (instregex "SADD8", "SADD16", "SSUB8", "SSUB16", "SASX", "SSAX",
        "UADD8", "UADD16", "USUB8", "USUB16", "UASX", "USAX", "t2SADD8",
        "t2SADD16", "t2SSUB8", "t2SSUB16", "t2SASX", "t2SSAX", "t2UADD8",
        "t2UADD16", "t2USUB8", "t2USUB16", "t2UASX", "t2USAX")>;
  // Flag setting.
  def : InstRW< [SwiftWriteP01TwoCycle],
       (instregex "SHADD8", "SHADD16", "SHSUB8", "SHSUB16", "SHASX", "SHSAX",
       "SXTAB", "SXTAB16", "SXTAH", "UHADD8", "UHADD16", "UHSUB8", "UHSUB16",
       "UHASX", "UHSAX", "UXTAB", "UXTAB16", "UXTAH", "t2SHADD8", "t2SHADD16",
       "t2SHSUB8", "t2SHSUB16", "t2SHASX", "t2SHSAX", "t2SXTAB", "t2SXTAB16",
       "t2SXTAH", "t2UHADD8", "t2UHADD16", "t2UHSUB8", "t2UHSUB16", "t2UHASX",
       "t2UHSAX", "t2UXTAB", "t2UXTAB16", "t2UXTAH")>;

  // 4.2.11 Sum of Absolute Difference
  def : InstRW< [SwiftWriteP0P1FourCycle], (instregex "USAD8") >;
  def : InstRW<[SwiftWriteP0P1FourCycle, ReadALU, ReadALU, SchedReadAdvance<2>],
        (instregex "USADA8")>;

  // 4.2.12 Integer Multiply (32-bit result)
  // Two sources.
  def : InstRW< [SwiftWriteP0FourCycle],
        (instregex "MULS", "MUL", "SMMUL", "SMMULR", "SMULBB", "SMULBT",
        "SMULTB", "SMULTT", "SMULWB", "SMULWT", "SMUSD", "SMUSDXi", "t2MUL",
        "t2SMMUL", "t2SMMULR", "t2SMULBB", "t2SMULBT", "t2SMULTB", "t2SMULTT",
        "t2SMULWB", "t2SMULWT", "t2SMUSD")>;

  def SwiftWriteP0P01FiveCycleTwoUops :
      SchedWriteRes<[SwiftUnitP0, SwiftUnitP01]>  {
    let Latency = 5;
  }

  def SwiftPredP0P01FourFiveCycle : SchedWriteVariant<[
    SchedVar<IsPredicatedPred, [ SwiftWriteP0P01FiveCycleTwoUops ]>,
    SchedVar<NoSchedPred,      [ SwiftWriteP0FourCycle ]>
  ]>;

  def SwiftReadAdvanceFourCyclesPred : SchedReadVariant<[
     SchedVar<IsPredicatedPred, [SchedReadAdvance<4>]>,
     SchedVar<NoSchedPred,      [ReadALU]>
  ]>;

  // Multiply accumulate, three sources
  def : InstRW< [SwiftPredP0P01FourFiveCycle, ReadALU, ReadALU,
                 SwiftReadAdvanceFourCyclesPred],
        (instregex "MLAS", "MLA", "MLS", "SMMLA", "SMMLAR", "SMMLS", "SMMLSR",
        "t2MLA", "t2MLS", "t2MLAS", "t2SMMLA", "t2SMMLAR", "t2SMMLS",
        "t2SMMLSR")>;

  // 4.2.13 Integer Multiply (32-bit result, Q flag)
  def : InstRW< [SwiftWriteP0FourCycle],
        (instregex "SMUAD", "SMUADX", "t2SMUAD", "t2SMUADX")>;
  def : InstRW< [SwiftPredP0P01FourFiveCycle, ReadALU, ReadALU,
                 SwiftReadAdvanceFourCyclesPred],
        (instregex "SMLABB", "SMLABT", "SMLATB", "SMLATT", "SMLSD", "SMLSDX",
        "SMLAWB", "SMLAWT", "t2SMLABB", "t2SMLABT", "t2SMLATB", "t2SMLATT",
        "t2SMLSD", "t2SMLSDX", "t2SMLAWB", "t2SMLAWT")>;
  def : InstRW< [SwiftPredP0P01FourFiveCycle],
        (instregex "SMLAD", "SMLADX", "t2SMLAD", "t2SMLADX")>;

  def SwiftP0P0P01FiveCycle : SchedWriteRes<[SwiftUnitP0, SwiftUnitP01]> {
    let Latency = 5;
    let NumMicroOps = 3;
    let ResourceCycles = [2, 1];
  }
  def SwiftWrite1Cycle : SchedWriteRes<[]> {
    let Latency = 1;
    let NumMicroOps = 0;
  }
  def SwiftWrite5Cycle : SchedWriteRes<[]> {
    let Latency = 5;
    let NumMicroOps = 0;
  }
  def SwiftWrite6Cycle : SchedWriteRes<[]> {
    let Latency = 6;
    let NumMicroOps = 0;
  }

  // 4.2.14 Integer Multiply, Long
  def : InstRW< [SwiftP0P0P01FiveCycle, SwiftWrite5Cycle],
        (instregex "SMULL$", "UMULL$", "t2SMULL$", "t2UMULL$")>;

  def Swift2P03P01FiveCycle : SchedWriteRes<[SwiftUnitP0, SwiftUnitP01]> {
    let Latency = 7;
    let NumMicroOps = 5;
    let ResourceCycles = [2, 3];
  }

  // 4.2.15 Integer Multiply Accumulate, Long
  // 4.2.16 Integer Multiply Accumulate, Dual
  // 4.2.17 Integer Multiply Accumulate Accumulate, Long
  // We are being a bit inaccurate here.
  def : InstRW< [SwiftWrite5Cycle, Swift2P03P01FiveCycle, ReadALU, ReadALU,
                 SchedReadAdvance<4>, SchedReadAdvance<3>],
        (instregex "SMLALS", "UMLALS", "SMLAL", "UMLAL", "MLALBB", "SMLALBT",
        "SMLALTB", "SMLALTT", "SMLALD", "SMLALDX", "SMLSLD", "SMLSLDX",
        "UMAAL", "t2SMLALS", "t2UMLALS", "t2SMLAL", "t2UMLAL", "t2MLALBB", "t2SMLALBT",
        "t2SMLALTB", "t2SMLALTT", "t2SMLALD", "t2SMLALDX", "t2SMLSLD", "t2SMLSLDX",
        "t2UMAAL")>;

  def SwiftDiv : SchedWriteRes<[SwiftUnitP0, SwiftUnitDiv]> {
    let NumMicroOps = 1;
    let Latency = 14;
    let ResourceCycles = [1, 14];
  }
  // 4.2.18 Integer Divide
  def : WriteRes<WriteDiv, [SwiftUnitDiv]>; // Workaround.
  def : InstRW <[SwiftDiv],
        (instregex "SDIV", "UDIV", "t2SDIV", "t2UDIV")>;

  // 4.2.19 Integer Load Single Element
  // 4.2.20 Integer Load Signextended
  def SwiftWriteP2P01ThreeCycle : SchedWriteRes<[SwiftUnitP2, SwiftUnitP01]> {
    let Latency = 3;
    let NumMicroOps = 2;
  }
  def SwiftWriteP2P01FourCyle : SchedWriteRes<[SwiftUnitP2, SwiftUnitP01]> {
    let Latency = 4;
    let NumMicroOps = 2;
  }
  def SwiftWriteP2P01P01FourCycle : SchedWriteRes<[SwiftUnitP2, SwiftUnitP01,
                                                   SwiftUnitP01]> {
    let Latency = 4;
    let NumMicroOps = 3;
  }
  def SwiftWriteP2P2ThreeCycle : SchedWriteRes<[SwiftUnitP2, SwiftUnitP2]> {
    let Latency = 3;
    let NumMicroOps = 2;
  }
  def SwiftWriteP2P2P01ThreeCycle : SchedWriteRes<[SwiftUnitP2, SwiftUnitP2,
                                                   SwiftUnitP01]> {
    let Latency = 3;
    let NumMicroOps = 3;
  }
  def SwiftWrBackOne : SchedWriteRes<[]> {
    let Latency = 1;
    let NumMicroOps = 0;
  }
  def SwiftWriteLdFour : SchedWriteRes<[]> {
    let Latency = 4;
    let NumMicroOps = 0;
  }
   // Not accurate.
  def : InstRW<[SwiftWriteP2ThreeCycle],
        (instregex "LDR(i12|rs)$", "LDRB(i12|rs)$", "t2LDR(i8|i12|s|pci)",
        "t2LDR(H|B)(i8|i12|s|pci)", "LDREX", "tLDR[BH](r|i|spi|pci|pciASM)",
        "tLDR(r|i|spi|pci|pciASM)")>;
  def : InstRW<[SwiftWriteP2ThreeCycle],
        (instregex "LDRH$",  "PICLDR$", "PICLDR(H|B)$", "LDRcp$")>;
  def : InstRW<[SwiftWriteP2P01FourCyle],
        (instregex "PICLDRS(H|B)$", "t2LDRS(H|B)(i|r|p|s)", "LDRS(H|B)$",
        "t2LDRpci_pic", "tLDRS(B|H)")>;
  def : InstRW<[SwiftWriteP2P01ThreeCycle,  SwiftWrBackOne],
        (instregex "LD(RB|R)(_|T_)(POST|PRE)_(IMM|REG)", "LDRH(_PRE|_POST)",
        "LDR(T|BT)_POST_(REG|IMM)", "LDRHT(i|r)",
        "t2LD(R|RB|RH)_(PRE|POST)", "t2LD(R|RB|RH)T")>;
  def : InstRW<[SwiftWriteP2P01P01FourCycle, SwiftWrBackOne],
        (instregex "LDR(SH|SB)(_POST|_PRE)", "t2LDR(SH|SB)(_POST|_PRE)",
        "LDRS(B|H)T(i|r)", "t2LDRS(B|H)T(i|r)", "t2LDRS(B|H)T")>;

  // 4.2.21 Integer Dual Load
  // Not accurate.
  def : InstRW<[SwiftWriteP2P2ThreeCycle, SwiftWriteLdFour],
        (instregex "t2LDRDi8", "LDRD$")>;
  def : InstRW<[SwiftWriteP2P2P01ThreeCycle, SwiftWriteLdFour, SwiftWrBackOne],
        (instregex "LDRD_(POST|PRE)", "t2LDRD_(POST|PRE)")>;

  // 4.2.22 Integer Load, Multiple
  // NumReg = 1 .. 16
  foreach Lat = 3-25 in {
    def SwiftWriteLM#Lat#Cy : SchedWriteRes<[SwiftUnitP2]> {
      let Latency = Lat;
    }
    def SwiftWriteLM#Lat#CyNo : SchedWriteRes<[]> {
      let Latency = Lat;
      let NumMicroOps = 0;
    }
  }
  // Predicate.
  foreach NumAddr = 1-16 in {
    def SwiftLMAddr#NumAddr#Pred : SchedPredicate<"TII->getNumLDMAddresses(MI) == "#NumAddr>;
  }
  def SwiftWriteLDMAddrNoWB : SchedWriteRes<[SwiftUnitP01]> { let Latency = 0; }
  def SwiftWriteLDMAddrWB : SchedWriteRes<[SwiftUnitP01, SwiftUnitP01]>;
  def SwiftWriteLM : SchedWriteVariant<[
    SchedVar<SwiftLMAddr2Pred, [SwiftWriteLM3Cy, SwiftWriteLM4Cy]>,
    SchedVar<SwiftLMAddr3Pred, [SwiftWriteLM3Cy, SwiftWriteLM4Cy,
                                SwiftWriteLM5Cy]>,
    SchedVar<SwiftLMAddr4Pred, [SwiftWriteLM3Cy, SwiftWriteLM4Cy,
                                SwiftWriteLM5Cy, SwiftWriteLM6Cy]>,
    SchedVar<SwiftLMAddr5Pred, [SwiftWriteLM3Cy, SwiftWriteLM4Cy,
                                SwiftWriteLM5Cy, SwiftWriteLM6Cy,
                                SwiftWriteLM7Cy]>,
    SchedVar<SwiftLMAddr6Pred, [SwiftWriteLM3Cy, SwiftWriteLM4Cy,
                                SwiftWriteLM5Cy, SwiftWriteLM6Cy,
                                SwiftWriteLM7Cy, SwiftWriteLM8Cy]>,
    SchedVar<SwiftLMAddr7Pred, [SwiftWriteLM3Cy, SwiftWriteLM4Cy,
                                SwiftWriteLM5Cy, SwiftWriteLM6Cy,
                                SwiftWriteLM7Cy, SwiftWriteLM8Cy,
                                SwiftWriteLM9Cy]>,
    SchedVar<SwiftLMAddr8Pred, [SwiftWriteLM3Cy, SwiftWriteLM4Cy,
                                SwiftWriteLM5Cy, SwiftWriteLM6Cy,
                                SwiftWriteLM7Cy, SwiftWriteLM8Cy,
                                SwiftWriteLM9Cy, SwiftWriteLM10Cy]>,
    SchedVar<SwiftLMAddr9Pred, [SwiftWriteLM3Cy, SwiftWriteLM4Cy,
                                SwiftWriteLM5Cy, SwiftWriteLM6Cy,
                                SwiftWriteLM7Cy, SwiftWriteLM8Cy,
                                SwiftWriteLM9Cy, SwiftWriteLM10Cy,
                                SwiftWriteLM11Cy]>,
    SchedVar<SwiftLMAddr10Pred,[SwiftWriteLM3Cy, SwiftWriteLM4Cy,
                                SwiftWriteLM5Cy, SwiftWriteLM6Cy,
                                SwiftWriteLM7Cy, SwiftWriteLM8Cy,
                                SwiftWriteLM9Cy, SwiftWriteLM10Cy,
                                SwiftWriteLM11Cy, SwiftWriteLM12Cy]>,
    SchedVar<SwiftLMAddr11Pred,[SwiftWriteLM3Cy, SwiftWriteLM4Cy,
                                SwiftWriteLM5Cy, SwiftWriteLM6Cy,
                                SwiftWriteLM7Cy, SwiftWriteLM8Cy,
                                SwiftWriteLM9Cy, SwiftWriteLM10Cy,
                                SwiftWriteLM11Cy, SwiftWriteLM12Cy,
                                SwiftWriteLM13Cy]>,
    SchedVar<SwiftLMAddr12Pred,[SwiftWriteLM3Cy, SwiftWriteLM4Cy,
                                SwiftWriteLM5Cy, SwiftWriteLM6Cy,
                                SwiftWriteLM7Cy, SwiftWriteLM8Cy,
                                SwiftWriteLM9Cy, SwiftWriteLM10Cy,
                                SwiftWriteLM11Cy, SwiftWriteLM12Cy,
                                SwiftWriteLM13Cy, SwiftWriteLM14Cy]>,
    SchedVar<SwiftLMAddr13Pred,[SwiftWriteLM3Cy, SwiftWriteLM4Cy,
                                SwiftWriteLM5Cy, SwiftWriteLM6Cy,
                                SwiftWriteLM7Cy, SwiftWriteLM8Cy,
                                SwiftWriteLM9Cy, SwiftWriteLM10Cy,
                                SwiftWriteLM11Cy, SwiftWriteLM12Cy,
                                SwiftWriteLM13Cy, SwiftWriteLM14Cy,
                                SwiftWriteLM15Cy]>,
    SchedVar<SwiftLMAddr14Pred,[SwiftWriteLM3Cy, SwiftWriteLM4Cy,
                                SwiftWriteLM5Cy, SwiftWriteLM6Cy,
                                SwiftWriteLM7Cy, SwiftWriteLM8Cy,
                                SwiftWriteLM9Cy, SwiftWriteLM10Cy,
                                SwiftWriteLM11Cy, SwiftWriteLM12Cy,
                                SwiftWriteLM13Cy, SwiftWriteLM14Cy,
                                SwiftWriteLM15Cy, SwiftWriteLM16Cy]>,
    SchedVar<SwiftLMAddr15Pred,[SwiftWriteLM3Cy, SwiftWriteLM4Cy,
                                SwiftWriteLM5Cy, SwiftWriteLM6Cy,
                                SwiftWriteLM7Cy, SwiftWriteLM8Cy,
                                SwiftWriteLM9Cy, SwiftWriteLM10Cy,
                                SwiftWriteLM11Cy, SwiftWriteLM12Cy,
                                SwiftWriteLM13Cy, SwiftWriteLM14Cy,
                                SwiftWriteLM15Cy, SwiftWriteLM16Cy,
                                SwiftWriteLM17Cy]>,
    SchedVar<SwiftLMAddr16Pred,[SwiftWriteLM3Cy, SwiftWriteLM4Cy,
                                SwiftWriteLM5Cy, SwiftWriteLM6Cy,
                                SwiftWriteLM7Cy, SwiftWriteLM8Cy,
                                SwiftWriteLM9Cy, SwiftWriteLM10Cy,
                                SwiftWriteLM11Cy, SwiftWriteLM12Cy,
                                SwiftWriteLM13Cy, SwiftWriteLM14Cy,
                                SwiftWriteLM15Cy, SwiftWriteLM16Cy,
                                SwiftWriteLM17Cy, SwiftWriteLM18Cy]>,
    // Unknow number of registers, just use resources for two registers.
    SchedVar<NoSchedPred,      [SwiftWriteLM3Cy, SwiftWriteLM4Cy,
                                SwiftWriteLM5CyNo, SwiftWriteLM6CyNo,
                                SwiftWriteLM7CyNo, SwiftWriteLM8CyNo,
                                SwiftWriteLM9CyNo, SwiftWriteLM10CyNo,
                                SwiftWriteLM11CyNo, SwiftWriteLM12CyNo,
                                SwiftWriteLM13CyNo, SwiftWriteLM14CyNo,
                                SwiftWriteLM15CyNo, SwiftWriteLM16CyNo,
                                SwiftWriteLM17CyNo, SwiftWriteLM18CyNo]>

  ]> { let Variadic=1; }

  def : InstRW<[SwiftWriteLM, SwiftWriteLDMAddrNoWB],
        (instregex "LDM(IA|DA|DB|IB)$", "t2LDM(IA|DA|DB|IB)$",
        "(t|sys)LDM(IA|DA|DB|IB)$")>;
  def : InstRW<[SwiftWriteLDMAddrWB, SwiftWriteLM],
        (instregex /*"t2LDMIA_RET", "tLDMIA_RET", "LDMIA_RET",*/
        "LDM(IA|DA|DB|IB)_UPD", "(t2|sys|t)LDM(IA|DA|DB|IB)_UPD")>;
  def : InstRW<[SwiftWriteLDMAddrWB, SwiftWriteLM, SwiftWriteP1TwoCycle],
        (instregex "LDMIA_RET", "(t|t2)LDMIA_RET", "POP", "tPOP")>;
  // 4.2.23 Integer Store, Single Element
  def : InstRW<[SwiftWriteP2],
        (instregex "PICSTR", "STR(i12|rs)", "STRB(i12|rs)", "STRH$", "STREX",
        "t2STR(i12|i8|s)$", "t2STR[BH](i12|i8|s)$", "tSTR[BH](i|r)", "tSTR(i|r)", "tSTRspi")>;

  def : InstRW<[SwiftWriteP01OneCycle, SwiftWriteP2],
        (instregex "STR(B_|_|BT_|T_)(PRE_IMM|PRE_REG|POST_REG|POST_IMM)",
        "STR(i|r)_preidx", "STRB(i|r)_preidx", "STRH_preidx", "STR(H_|HT_)(PRE|POST)",
        "STR(BT|HT|T)", "t2STR_(PRE|POST)", "t2STR[BH]_(PRE|POST)",
        "t2STR_preidx", "t2STR[BH]_preidx", "t2ST(RB|RH|R)T")>;

  // 4.2.24 Integer Store, Dual
  def : InstRW<[SwiftWriteP2, SwiftWriteP2, SwiftWriteP01OneCycle],
        (instregex "STRD$", "t2STRDi8")>;
  def : InstRW<[SwiftWriteP01OneCycle, SwiftWriteP2, SwiftWriteP2,
                SwiftWriteP01OneCycle],
        (instregex "(t2|t)STRD_(POST|PRE)", "STRD_(POST|PRE)")>;

  // 4.2.25 Integer Store, Multiple
  def SwiftWriteStIncAddr : SchedWriteRes<[SwiftUnitP2, SwiftUnitP01]> {
    let Latency = 0;
    let NumMicroOps = 2;
  }
  foreach NumAddr = 1-16 in {
     def SwiftWriteSTM#NumAddr : WriteSequence<[SwiftWriteStIncAddr], NumAddr>;
  }
  def SwiftWriteSTM : SchedWriteVariant<[
    SchedVar<SwiftLMAddr2Pred, [SwiftWriteSTM2]>,
    SchedVar<SwiftLMAddr3Pred, [SwiftWriteSTM3]>,
    SchedVar<SwiftLMAddr4Pred, [SwiftWriteSTM4]>,
    SchedVar<SwiftLMAddr5Pred, [SwiftWriteSTM5]>,
    SchedVar<SwiftLMAddr6Pred, [SwiftWriteSTM6]>,
    SchedVar<SwiftLMAddr7Pred, [SwiftWriteSTM7]>,
    SchedVar<SwiftLMAddr8Pred, [SwiftWriteSTM8]>,
    SchedVar<SwiftLMAddr9Pred, [SwiftWriteSTM9]>,
    SchedVar<SwiftLMAddr10Pred,[SwiftWriteSTM10]>,
    SchedVar<SwiftLMAddr11Pred,[SwiftWriteSTM11]>,
    SchedVar<SwiftLMAddr12Pred,[SwiftWriteSTM12]>,
    SchedVar<SwiftLMAddr13Pred,[SwiftWriteSTM13]>,
    SchedVar<SwiftLMAddr14Pred,[SwiftWriteSTM14]>,
    SchedVar<SwiftLMAddr15Pred,[SwiftWriteSTM15]>,
    SchedVar<SwiftLMAddr16Pred,[SwiftWriteSTM16]>,
    // Unknow number of registers, just use resources for two registers.
    SchedVar<NoSchedPred,      [SwiftWriteSTM2]>
  ]>;
  def : InstRW<[SwiftWriteSTM],
        (instregex "STM(IB|IA|DB|DA)$", "(t2|sys|t)STM(IB|IA|DB|DA)$")>;
  def : InstRW<[SwiftWriteP01OneCycle, SwiftWriteSTM],
        (instregex "STM(IB|IA|DB|DA)_UPD", "(t2|sys|t)STM(IB|IA|DB|DA)_UPD",
        "PUSH", "tPUSH")>;

  // 4.2.26 Branch
  def : WriteRes<WriteBr, [SwiftUnitP1]> { let Latency = 0; }
  def : WriteRes<WriteBrL, [SwiftUnitP1]> { let Latency = 2; }
  def : WriteRes<WriteBrTbl, [SwiftUnitP1, SwiftUnitP2]> { let Latency = 0; }

  // 4.2.27 Not issued
  def : WriteRes<WriteNoop, []> { let Latency = 0; let NumMicroOps = 0; }
  def : InstRW<[WriteNoop], (instregex "t2IT", "IT", "NOP")>;

  // 4.2.28 Advanced SIMD, Integer, 2 cycle
  def : InstRW<[SwiftWriteP0TwoCycle],
        (instregex "VADDv", "VSUBv", "VNEG(s|f|v)", "VADDL", "VSUBL",
                   "VADDW", "VSUBW", "VHADD", "VHSUB", "VRHADD", "VPADDi",
                   "VPADDL", "VAND", "VBIC", "VEOR", "VORN", "VORR", "VTST",
                   "VSHL", "VSHR(s|u)", "VSHLL", "VQSHL", "VQSHLU", "VBIF",
                   "VBIT", "VBSL", "VSLI", "VSRI", "VCLS", "VCLZ", "VCNT")>;

  def : InstRW<[SwiftWriteP1TwoCycle],
        (instregex "VEXT", "VREV16", "VREV32", "VREV64")>;

  // 4.2.29 Advanced SIMD, Integer, 4 cycle
  // 4.2.30 Advanced SIMD, Integer with Accumulate
  def : InstRW<[SwiftWriteP0FourCycle],
        (instregex "VABA", "VABAL", "VPADAL", "VRSRA", "VSRA", "VACGE", "VACGT",
        "VACLE", "VACLT", "VCEQ", "VCGE", "VCGT", "VCLE", "VCLT", "VRSHL",
        "VQRSHL", "VRSHR(u|s)", "VABS(f|v)", "VQABS", "VQNEG", "VQADD",
        "VQSUB")>;
  def : InstRW<[SwiftWriteP1FourCycle],
        (instregex "VRECPE", "VRSQRTE")>;

  // 4.2.31 Advanced SIMD, Add and Shift with Narrow
  def : InstRW<[SwiftWriteP0P1FourCycle],
        (instregex "VADDHN", "VSUBHN", "VSHRN")>;
  def : InstRW<[SwiftWriteP0P1SixCycle],
        (instregex "VRADDHN", "VRSUBHN", "VRSHRN", "VQSHRN", "VQSHRUN",
                   "VQRSHRN", "VQRSHRUN")>;

  // 4.2.32 Advanced SIMD, Vector Table Lookup
  foreach Num = 1-4 in {
    def SwiftWrite#Num#xP1TwoCycle : WriteSequence<[SwiftWriteP1TwoCycle], Num>;
  }
  def : InstRW<[SwiftWrite1xP1TwoCycle],
        (instregex "VTB(L|X)1")>;
  def : InstRW<[SwiftWrite2xP1TwoCycle],
        (instregex "VTB(L|X)2")>;
  def : InstRW<[SwiftWrite3xP1TwoCycle],
        (instregex "VTB(L|X)3")>;
  def : InstRW<[SwiftWrite4xP1TwoCycle],
        (instregex "VTB(L|X)4")>;

  // 4.2.33 Advanced SIMD, Transpose
  def : InstRW<[SwiftWriteP1FourCycle, SwiftWriteP1FourCycle,
                SwiftWriteP1TwoCycle/*RsrcOnly*/, SchedReadAdvance<2>],
        (instregex "VSWP", "VTRN", "VUZP", "VZIP")>;

  // 4.2.34 Advanced SIMD and VFP, Floating Point
  def : InstRW<[SwiftWriteP0TwoCycle], (instregex "VABS(S|D)$", "VNEG(S|D)$")>;
  def : InstRW<[SwiftWriteP0FourCycle],
        (instregex "VCMP(D|S|ZD|ZS)$", "VCMPE(D|S|ZD|ZS)")>;
  def : InstRW<[SwiftWriteP0FourCycle],
        (instregex "VADD(S|f)", "VSUB(S|f)", "VABD", "VPADDf", "VMAX", "VMIN", "VPMAX",
                   "VPMIN")>;
  def : InstRW<[SwiftWriteP0SixCycle], (instregex "VADDD$", "VSUBD$")>;
  def : InstRW<[SwiftWriteP1EightCycle], (instregex "VRECPS", "VRSQRTS")>;

  // 4.2.35 Advanced SIMD and VFP, Multiply
  def : InstRW<[SwiftWriteP1FourCycle],
        (instregex "VMUL(S|v|p|f|s)", "VNMULS", "VQDMULH", "VQRDMULH",
                   "VMULL", "VQDMULL")>;
  def : InstRW<[SwiftWriteP1SixCycle],
        (instregex "VMULD", "VNMULD")>;
  def : InstRW<[SwiftWriteP1FourCycle],
        (instregex "VMLA", "VMLS", "VNMLA", "VNMLS", "VFMA(S|D)", "VFMS(S|D)",
        "VFNMA", "VFNMS", "VMLAL", "VMLSL","VQDMLAL", "VQDMLSL")>;
  def : InstRW<[SwiftWriteP1EightCycle], (instregex "VFMAfd", "VFMSfd")>;
  def : InstRW<[SwiftWriteP1TwelveCyc], (instregex "VFMAfq", "VFMSfq")>;

  // 4.2.36 Advanced SIMD and VFP, Convert
  def : InstRW<[SwiftWriteP1FourCycle], (instregex "VCVT", "V(S|U)IT", "VTO(S|U)")>;
  // Fixpoint conversions.
  def : WriteRes<WriteCvtFP, [SwiftUnitP1]> { let Latency = 4; }

  // 4.2.37 Advanced SIMD and VFP, Move
  def : InstRW<[SwiftWriteP0TwoCycle],
        (instregex "VMOVv", "VMOV(S|D)$", "VMOV(S|D)cc",
                   "VMVNv", "VMVN(d|q)", "VMVN(S|D)cc",
                   "FCONST(D|S)")>;
  def : InstRW<[SwiftWriteP1TwoCycle], (instregex "VMOVN", "VMOVL")>;
  def : InstRW<[WriteSequence<[SwiftWriteP0FourCycle, SwiftWriteP1TwoCycle]>],
        (instregex "VQMOVN")>;
  def : InstRW<[SwiftWriteP1TwoCycle], (instregex "VDUPLN", "VDUPf")>;
  def : InstRW<[WriteSequence<[SwiftWriteP2FourCycle, SwiftWriteP1TwoCycle]>],
        (instregex "VDUP(8|16|32)")>;
  def : InstRW<[SwiftWriteP2ThreeCycle], (instregex "VMOVRS$")>;
  def : InstRW<[WriteSequence<[SwiftWriteP2FourCycle, SwiftWriteP0TwoCycle]>],
        (instregex "VMOVSR$", "VSETLN")>;
  def : InstRW<[SwiftWriteP2ThreeCycle, SwiftWriteP2FourCycle],
        (instregex "VMOVRR(D|S)$")>;
  def : InstRW<[SwiftWriteP2FourCycle], (instregex "VMOVDRR$")>;
  def : InstRW<[WriteSequence<[SwiftWriteP2FourCycle, SwiftWriteP1TwoCycle]>,
                WriteSequence<[SwiftWrite1Cycle, SwiftWriteP2FourCycle,
                               SwiftWriteP1TwoCycle]>],
                (instregex "VMOVSRR$")>;
  def : InstRW<[WriteSequence<[SwiftWriteP1TwoCycle, SwiftWriteP2ThreeCycle]>],
        (instregex "VGETLN(u|i)")>;
  def : InstRW<[WriteSequence<[SwiftWriteP1TwoCycle, SwiftWriteP2ThreeCycle,
                               SwiftWriteP01OneCycle]>],
        (instregex "VGETLNs")>;

  // 4.2.38 Advanced SIMD and VFP, Move FPSCR
  // Serializing instructions.
  def SwiftWaitP0For15Cy : SchedWriteRes<[SwiftUnitP0]> {
    let Latency = 15;
    let ResourceCycles = [15];
  }
  def SwiftWaitP1For15Cy : SchedWriteRes<[SwiftUnitP1]> {
    let Latency = 15;
    let ResourceCycles = [15];
  }
  def SwiftWaitP2For15Cy : SchedWriteRes<[SwiftUnitP2]> {
    let Latency = 15;
    let ResourceCycles = [15];
  }
  def : InstRW<[SwiftWaitP0For15Cy, SwiftWaitP1For15Cy, SwiftWaitP2For15Cy],
        (instregex "VMRS")>;
  def : InstRW<[SwiftWaitP0For15Cy, SwiftWaitP1For15Cy, SwiftWaitP2For15Cy],
        (instregex "VMSR")>;
  // Not serializing.
  def : InstRW<[SwiftWriteP0TwoCycle], (instregex "FMSTAT")>;

  // 4.2.39 Advanced SIMD and VFP, Load Single Element
  def : InstRW<[SwiftWriteLM4Cy], (instregex "VLDRD$", "VLDRS$")>;

  // 4.2.40 Advanced SIMD and VFP, Store Single Element
  def : InstRW<[SwiftWriteLM4Cy], (instregex "VSTRD$", "VSTRS$")>;

  // 4.2.41 Advanced SIMD and VFP, Load Multiple
  // 4.2.42 Advanced SIMD and VFP, Store Multiple

  // Resource requirement for permuting, just reserves the resources.
  foreach Num = 1-28 in {
    def SwiftVLDMPerm#Num : SchedWriteRes<[SwiftUnitP1]> {
      let Latency = 0;
      let NumMicroOps = Num;
      let ResourceCycles = [Num];
    }
  }

  // Pre RA pseudos - load/store to a Q register as a D register pair.
  def : InstRW<[SwiftWriteLM4Cy], (instregex "VLDMQIA$", "VSTMQIA$")>;

  // Post RA not modelled accurately. We assume that register use of width 64
  // bit maps to a D register, 128 maps to a Q register. Not all different kinds
  // are accurately represented.
  def SwiftWriteVLDM : SchedWriteVariant<[
    // Load of one S register.
    SchedVar<SwiftLMAddr1Pred, [SwiftWriteLM4Cy]>,
    // Load of one D register.
    SchedVar<SwiftLMAddr2Pred, [SwiftWriteLM4Cy, SwiftWriteLM4CyNo]>,
    // Load of 3 S register.
    SchedVar<SwiftLMAddr3Pred, [SwiftWriteLM9Cy, SwiftWriteLM10Cy,
                                SwiftWriteLM13CyNo, SwiftWriteP01OneCycle,
                                SwiftVLDMPerm3]>,
    // Load of a Q register (not neccessarily true). We should not be mapping to
    // 4 S registers, either.
    SchedVar<SwiftLMAddr4Pred, [SwiftWriteLM4Cy, SwiftWriteLM4CyNo,
                                SwiftWriteLM4CyNo, SwiftWriteLM4CyNo]>,
    // Load of 5 S registers.
    SchedVar<SwiftLMAddr5Pred, [SwiftWriteLM9Cy, SwiftWriteLM10Cy,
                                SwiftWriteLM13CyNo, SwiftWriteLM14CyNo,
                                SwiftWriteLM17CyNo,  SwiftWriteP01OneCycle,
                                SwiftVLDMPerm5]>,
    // Load of 3 D registers. (Must also be able to handle s register list -
    // though, not accurate)
    SchedVar<SwiftLMAddr6Pred, [SwiftWriteLM7Cy, SwiftWriteLM8Cy,
                                SwiftWriteLM10Cy, SwiftWriteLM14CyNo,
                                SwiftWriteLM14CyNo, SwiftWriteLM14CyNo,
                                SwiftWriteP01OneCycle, SwiftVLDMPerm5]>,
    // Load of 7 S registers.
    SchedVar<SwiftLMAddr7Pred, [SwiftWriteLM9Cy, SwiftWriteLM10Cy,
                                SwiftWriteLM13Cy, SwiftWriteLM14CyNo,
                                SwiftWriteLM17CyNo, SwiftWriteLM18CyNo,
                                SwiftWriteLM21CyNo, SwiftWriteP01OneCycle,
                                SwiftVLDMPerm7]>,
    // Load of two Q registers.
    SchedVar<SwiftLMAddr8Pred, [SwiftWriteLM7Cy, SwiftWriteLM8Cy,
                                SwiftWriteLM13Cy, SwiftWriteLM13CyNo,
                                SwiftWriteLM13CyNo, SwiftWriteLM13CyNo,
                                SwiftWriteLM13CyNo, SwiftWriteLM13CyNo,
                                SwiftWriteP01OneCycle,  SwiftVLDMPerm2]>,
    // Load of 9 S registers.
    SchedVar<SwiftLMAddr9Pred, [SwiftWriteLM9Cy, SwiftWriteLM10Cy,
                                SwiftWriteLM13Cy, SwiftWriteLM14CyNo,
                                SwiftWriteLM17CyNo, SwiftWriteLM18CyNo,
                                SwiftWriteLM21CyNo, SwiftWriteLM22CyNo,
                                SwiftWriteLM25CyNo, SwiftWriteP01OneCycle,
                                SwiftVLDMPerm9]>,
    // Load of 5 D registers.
    SchedVar<SwiftLMAddr10Pred,[SwiftWriteLM7Cy, SwiftWriteLM8Cy,
                                SwiftWriteLM10Cy, SwiftWriteLM14Cy,
                                SwiftWriteLM14CyNo, SwiftWriteLM14CyNo,
                                SwiftWriteLM14CyNo, SwiftWriteLM14CyNo,
                                SwiftWriteLM14CyNo,  SwiftWriteLM14CyNo,
                                SwiftWriteP01OneCycle, SwiftVLDMPerm5]>,
    // Inaccurate: reuse describtion from 9 S registers.
    SchedVar<SwiftLMAddr11Pred,[SwiftWriteLM9Cy, SwiftWriteLM10Cy,
                                SwiftWriteLM13Cy, SwiftWriteLM14CyNo,
                                SwiftWriteLM17CyNo, SwiftWriteLM18CyNo,
                                SwiftWriteLM21CyNo, SwiftWriteLM22CyNo,
                                SwiftWriteLM21CyNo, SwiftWriteLM22CyNo,
                                SwiftWriteLM25CyNo, SwiftWriteP01OneCycle,
                                SwiftVLDMPerm9]>,
    // Load of three Q registers.
    SchedVar<SwiftLMAddr12Pred,[SwiftWriteLM7Cy, SwiftWriteLM8Cy,
                                SwiftWriteLM11Cy, SwiftWriteLM11Cy,
                                SwiftWriteLM11CyNo, SwiftWriteLM11CyNo,
                                SwiftWriteLM11CyNo, SwiftWriteLM11CyNo,
                                SwiftWriteLM11CyNo, SwiftWriteLM11CyNo,
                                SwiftWriteLM11CyNo, SwiftWriteLM11CyNo,
                                SwiftWriteP01OneCycle, SwiftVLDMPerm3]>,
    // Inaccurate: reuse describtion from 9 S registers.
    SchedVar<SwiftLMAddr13Pred, [SwiftWriteLM9Cy, SwiftWriteLM10Cy,
                                SwiftWriteLM13Cy, SwiftWriteLM14CyNo,
                                SwiftWriteLM17CyNo, SwiftWriteLM18CyNo,
                                SwiftWriteLM21CyNo, SwiftWriteLM22CyNo,
                                SwiftWriteLM21CyNo, SwiftWriteLM22CyNo,
                                SwiftWriteLM21CyNo, SwiftWriteLM22CyNo,
                                SwiftWriteLM25CyNo, SwiftWriteP01OneCycle,
                                SwiftVLDMPerm9]>,
    // Load of 7 D registers inaccurate.
    SchedVar<SwiftLMAddr14Pred,[SwiftWriteLM7Cy, SwiftWriteLM8Cy,
                                SwiftWriteLM10Cy, SwiftWriteLM14Cy,
                                SwiftWriteLM14Cy, SwiftWriteLM14CyNo,
                                SwiftWriteLM14CyNo, SwiftWriteLM14CyNo,
                                SwiftWriteLM14CyNo,  SwiftWriteLM14CyNo,
                                SwiftWriteLM14CyNo,  SwiftWriteLM14CyNo,
                                SwiftWriteP01OneCycle, SwiftVLDMPerm7]>,
    SchedVar<SwiftLMAddr15Pred,[SwiftWriteLM9Cy, SwiftWriteLM10Cy,
                                SwiftWriteLM13Cy, SwiftWriteLM14Cy,
                                SwiftWriteLM17Cy, SwiftWriteLM18CyNo,
                                SwiftWriteLM21CyNo, SwiftWriteLM22CyNo,
                                SwiftWriteLM21CyNo, SwiftWriteLM22CyNo,
                                SwiftWriteLM21CyNo, SwiftWriteLM22CyNo,
                                SwiftWriteLM21CyNo, SwiftWriteLM22CyNo,
                                SwiftWriteLM25CyNo, SwiftWriteP01OneCycle,
                                SwiftVLDMPerm9]>,
    // Load of 4 Q registers.
    SchedVar<SwiftLMAddr16Pred,[SwiftWriteLM7Cy, SwiftWriteLM10Cy,
                                SwiftWriteLM11Cy, SwiftWriteLM14Cy,
                                SwiftWriteLM15Cy, SwiftWriteLM18CyNo,
                                SwiftWriteLM19CyNo, SwiftWriteLM22CyNo,
                                SwiftWriteLM19CyNo, SwiftWriteLM22CyNo,
                                SwiftWriteLM19CyNo, SwiftWriteLM22CyNo,
                                SwiftWriteLM19CyNo, SwiftWriteLM22CyNo,
                                SwiftWriteLM19CyNo, SwiftWriteLM22CyNo,
                                SwiftWriteP01OneCycle, SwiftVLDMPerm4]>,
    // Unknow number of registers, just use resources for two registers.
    SchedVar<NoSchedPred,      [SwiftWriteLM7Cy, SwiftWriteLM8Cy,
                                SwiftWriteLM13Cy, SwiftWriteLM13CyNo,
                                SwiftWriteLM13CyNo, SwiftWriteLM13CyNo,
                                SwiftWriteLM13CyNo, SwiftWriteLM13CyNo,
                                SwiftWriteLM13CyNo, SwiftWriteLM13CyNo,
                                SwiftWriteLM13CyNo, SwiftWriteLM13CyNo,
                                SwiftWriteLM13CyNo, SwiftWriteLM13CyNo,
                                SwiftWriteLM13CyNo, SwiftWriteLM13CyNo,
                                SwiftWriteLM13CyNo, SwiftWriteLM13CyNo,
                                SwiftWriteLM13CyNo, SwiftWriteLM13CyNo,
                                SwiftWriteLM13CyNo, SwiftWriteLM13CyNo,
                                SwiftWriteLM13CyNo, SwiftWriteLM13CyNo,
                                SwiftWriteLM13CyNo, SwiftWriteLM13CyNo,
                                SwiftWriteLM13CyNo, SwiftWriteLM13CyNo,
                                SwiftWriteLM13CyNo, SwiftWriteLM13CyNo,
                                SwiftWriteLM13CyNo, SwiftWriteLM13CyNo,
                                SwiftWriteP01OneCycle,  SwiftVLDMPerm2]>
  ]> { let Variadic = 1; }

  def : InstRW<[SwiftWriteVLDM], (instregex "VLDM[SD](IA|DB)$")>;

  def : InstRW<[SwiftWriteP01OneCycle2x, SwiftWriteVLDM],
        (instregex "VLDM[SD](IA|DB)_UPD$")>;

  def SwiftWriteVSTM : SchedWriteVariant<[
    // One S register.
    SchedVar<SwiftLMAddr1Pred, [SwiftWriteSTM1]>,
    // One D register.
    SchedVar<SwiftLMAddr2Pred, [SwiftWriteSTM1]>,
    // Three S registers.
    SchedVar<SwiftLMAddr3Pred, [SwiftWriteSTM4]>,
    // Assume one Q register.
    SchedVar<SwiftLMAddr4Pred, [SwiftWriteSTM1]>,
    SchedVar<SwiftLMAddr5Pred, [SwiftWriteSTM6]>,
    // Assume three D registers.
    SchedVar<SwiftLMAddr6Pred, [SwiftWriteSTM4]>,
    SchedVar<SwiftLMAddr7Pred, [SwiftWriteSTM8]>,
    // Assume two Q registers.
    SchedVar<SwiftLMAddr8Pred, [SwiftWriteSTM3]>,
    SchedVar<SwiftLMAddr9Pred, [SwiftWriteSTM10]>,
    // Assume 5 D registers.
    SchedVar<SwiftLMAddr10Pred, [SwiftWriteSTM6]>,
    SchedVar<SwiftLMAddr11Pred, [SwiftWriteSTM12]>,
    // Asume three Q registers.
    SchedVar<SwiftLMAddr12Pred, [SwiftWriteSTM4]>,
    SchedVar<SwiftLMAddr13Pred, [SwiftWriteSTM14]>,
    // Assume 7 D registers.
    SchedVar<SwiftLMAddr14Pred, [SwiftWriteSTM8]>,
    SchedVar<SwiftLMAddr15Pred, [SwiftWriteSTM16]>,
    // Assume four Q registers.
    SchedVar<SwiftLMAddr16Pred, [SwiftWriteSTM5]>,
    // Asumme two Q registers.
    SchedVar<NoSchedPred, [SwiftWriteSTM3]>
  ]> { let Variadic = 1; }

  def : InstRW<[SwiftWriteVSTM], (instregex "VSTM[SD](IA|DB)$")>;

  def : InstRW<[SwiftWriteP01OneCycle2x, SwiftWriteVSTM],
        (instregex "VSTM[SD](IA|DB)_UPD")>;

  // 4.2.43 Advanced SIMD, Element or Structure Load and Store
  def SwiftWrite2xP2FourCy : SchedWriteRes<[SwiftUnitP2]> {
      let Latency = 4;
      let ResourceCycles = [2];
  }
  def SwiftWrite3xP2FourCy : SchedWriteRes<[SwiftUnitP2]> {
      let Latency = 4;
      let ResourceCycles = [3];
  }
  foreach Num = 1-2 in {
    def SwiftExt#Num#xP0 : SchedWriteRes<[SwiftUnitP0]> {
      let Latency = 0;
      let NumMicroOps = Num;
      let ResourceCycles = [Num];
    }
  }
  // VLDx
  // Multiple structures.
  // Single element structure loads.
  // We assume aligned.
  // Single/two register.
  def : InstRW<[SwiftWriteLM4Cy], (instregex "VLD1(d|q)(8|16|32|64)$")>;
  def : InstRW<[SwiftWriteLM4Cy, SwiftWriteP01OneCycle],
        (instregex "VLD1(d|q)(8|16|32|64)wb")>;
  // Three register.
  def : InstRW<[SwiftWrite3xP2FourCy],
        (instregex "VLD1(d|q)(8|16|32|64)T$", "VLD1d64TPseudo")>;
  def : InstRW<[SwiftWrite3xP2FourCy, SwiftWriteP01OneCycle],
        (instregex "VLD1(d|q)(8|16|32|64)Twb")>;
  /// Four Register.
  def : InstRW<[SwiftWrite2xP2FourCy],
        (instregex "VLD1(d|q)(8|16|32|64)Q$", "VLD1d64QPseudo")>;
  def : InstRW<[SwiftWrite2xP2FourCy, SwiftWriteP01OneCycle],
        (instregex "VLD1(d|q)(8|16|32|64)Qwb")>;
  // Two element structure loads.
  // Two/four register.
  def : InstRW<[SwiftWriteLM9Cy, SwiftExt2xP0, SwiftVLDMPerm2],
        (instregex "VLD2(d|q|b)(8|16|32)$", "VLD2q(8|16|32)Pseudo$")>;
  def : InstRW<[SwiftWriteLM9Cy, SwiftWriteP01OneCycle, SwiftExt2xP0,
                SwiftVLDMPerm2],
        (instregex "VLD2(d|q|b)(8|16|32)wb", "VLD2q(8|16|32)PseudoWB")>;
  // Three element structure.
  def : InstRW<[SwiftWriteLM9Cy, SwiftWriteLM9CyNo, SwiftWriteLM9CyNo,
                SwiftVLDMPerm3, SwiftWrite3xP2FourCy],
        (instregex "VLD3(d|q)(8|16|32)$")>;
  def : InstRW<[SwiftWriteLM9Cy, SwiftVLDMPerm3, SwiftWrite3xP2FourCy],
        (instregex "VLD3(d|q)(8|16|32)(oddP|P)seudo$")>;

  def : InstRW<[SwiftWriteLM9Cy, SwiftWriteLM9CyNo, SwiftWriteLM9CyNo,
                SwiftWriteP01OneCycle, SwiftVLDMPerm3, SwiftWrite3xP2FourCy],
        (instregex "VLD3(d|q)(8|16|32)_UPD$")>;
  def : InstRW<[SwiftWriteLM9Cy, SwiftWriteP01OneCycle, SwiftVLDMPerm3,
                SwiftWrite3xP2FourCy],
        (instregex "VLD3(d|q)(8|16|32)(oddP|P)seudo_UPD")>;
  // Four element structure loads.
  def : InstRW<[SwiftWriteLM11Cy, SwiftWriteLM11Cy, SwiftWriteLM11Cy,
                SwiftWriteLM11Cy, SwiftExt2xP0, SwiftVLDMPerm4,
                SwiftWrite3xP2FourCy],
        (instregex "VLD4(d|q)(8|16|32)$")>;
  def : InstRW<[SwiftWriteLM11Cy,  SwiftExt2xP0, SwiftVLDMPerm4,
                SwiftWrite3xP2FourCy],
        (instregex "VLD4(d|q)(8|16|32)(oddP|P)seudo$")>;
  def : InstRW<[SwiftWriteLM11Cy, SwiftWriteLM11Cy, SwiftWriteLM11Cy,
                SwiftWriteLM11Cy, SwiftWriteP01OneCycle, SwiftExt2xP0,
                SwiftVLDMPerm4, SwiftWrite3xP2FourCy],
        (instregex "VLD4(d|q)(8|16|32)_UPD")>;
  def : InstRW<[SwiftWriteLM11Cy, SwiftWriteP01OneCycle, SwiftExt2xP0,
                SwiftVLDMPerm4, SwiftWrite3xP2FourCy],
        (instregex  "VLD4(d|q)(8|16|32)(oddP|P)seudo_UPD")>;

  // Single all/lane loads.
  // One element structure.
  def : InstRW<[SwiftWriteLM6Cy, SwiftVLDMPerm2],
        (instregex "VLD1(LN|DUP)(d|q)(8|16|32)$", "VLD1(LN|DUP)(d|q)(8|16|32)Pseudo$")>;
  def : InstRW<[SwiftWriteLM6Cy, SwiftWriteP01OneCycle, SwiftVLDMPerm2],
        (instregex "VLD1(LN|DUP)(d|q)(8|16|32)(wb|_UPD)",
                  "VLD1LNq(8|16|32)Pseudo_UPD")>;
  // Two element structure.
  def : InstRW<[SwiftWriteLM6Cy, SwiftWriteLM6Cy, SwiftExt1xP0, SwiftVLDMPerm2],
        (instregex "VLD2(DUP|LN)(d|q)(8|16|32|8x2|16x2|32x2)$",
                   "VLD2LN(d|q)(8|16|32)Pseudo$")>;
  def : InstRW<[SwiftWriteLM6Cy, SwiftWriteLM6Cy, SwiftWriteP01OneCycle,
                SwiftExt1xP0, SwiftVLDMPerm2],
        (instregex "VLD2LN(d|q)(8|16|32)_UPD$")>;
  def : InstRW<[SwiftWriteLM6Cy, SwiftWriteP01OneCycle, SwiftWriteLM6Cy,
                SwiftExt1xP0, SwiftVLDMPerm2],
        (instregex "VLD2DUPd(8|16|32|8x2|16x2|32x2)wb")>;
  def : InstRW<[SwiftWriteLM6Cy, SwiftWriteP01OneCycle, SwiftWriteLM6Cy,
                SwiftExt1xP0, SwiftVLDMPerm2],
        (instregex "VLD2LN(d|q)(8|16|32)Pseudo_UPD")>;
  // Three element structure.
  def : InstRW<[SwiftWriteLM7Cy, SwiftWriteLM8Cy, SwiftWriteLM8Cy, SwiftExt1xP0,
                SwiftVLDMPerm3],
        (instregex "VLD3(DUP|LN)(d|q)(8|16|32)$",
                   "VLD3(LN|DUP)(d|q)(8|16|32)Pseudo$")>;
  def : InstRW<[SwiftWriteLM7Cy, SwiftWriteLM8Cy, SwiftWriteLM8Cy,
                SwiftWriteP01OneCycle, SwiftExt1xP0, SwiftVLDMPerm3],
        (instregex "VLD3(LN|DUP)(d|q)(8|16|32)_UPD")>;
  def : InstRW<[SwiftWriteLM7Cy, SwiftWriteP01OneCycle, SwiftWriteLM8Cy,
                SwiftWriteLM8Cy, SwiftExt1xP0, SwiftVLDMPerm3],
        (instregex "VLD3(LN|DUP)(d|q)(8|16|32)Pseudo_UPD")>;
  // Four element struture.
  def : InstRW<[SwiftWriteLM8Cy, SwiftWriteLM9Cy, SwiftWriteLM10CyNo,
                SwiftWriteLM10CyNo, SwiftExt1xP0, SwiftVLDMPerm5],
        (instregex "VLD4(LN|DUP)(d|q)(8|16|32)$",
                   "VLD4(LN|DUP)(d|q)(8|16|32)Pseudo$")>;
  def : InstRW<[SwiftWriteLM8Cy, SwiftWriteLM9Cy, SwiftWriteLM10CyNo,
                SwiftWriteLM10CyNo, SwiftWriteP01OneCycle, SwiftExt1xP0,
                SwiftVLDMPerm5],
        (instregex "VLD4(DUP|LN)(d|q)(8|16|32)_UPD")>;
  def : InstRW<[SwiftWriteLM8Cy, SwiftWriteP01OneCycle, SwiftWriteLM9Cy,
                SwiftWriteLM10CyNo, SwiftWriteLM10CyNo, SwiftExt1xP0,
                SwiftVLDMPerm5],
        (instregex "VLD4(DUP|LN)(d|q)(8|16|32)Pseudo_UPD")>;
  // VSTx
  // Multiple structures.
  // Single element structure store.
  def : InstRW<[SwiftWrite1xP2], (instregex "VST1d(8|16|32|64)$")>;
  def : InstRW<[SwiftWrite2xP2], (instregex "VST1q(8|16|32|64)$")>;
  def : InstRW<[SwiftWriteP01OneCycle, SwiftWrite1xP2],
        (instregex "VST1d(8|16|32|64)wb")>;
  def : InstRW<[SwiftWriteP01OneCycle, SwiftWrite2xP2],
        (instregex "VST1q(8|16|32|64)wb")>;
  def : InstRW<[SwiftWrite3xP2],
        (instregex "VST1d(8|16|32|64)T$", "VST1d64TPseudo$")>;
  def : InstRW<[SwiftWriteP01OneCycle, SwiftWrite3xP2],
        (instregex "VST1d(8|16|32|64)Twb", "VST1d64TPseudoWB")>;
  def : InstRW<[SwiftWrite4xP2],
        (instregex "VST1d(8|16|32|64)(Q|QPseudo)$")>;
  def : InstRW<[SwiftWriteP01OneCycle, SwiftWrite4xP2],
        (instregex "VST1d(8|16|32|64)(Qwb|QPseudoWB)")>;
  // Two element structure store.
  def : InstRW<[SwiftWrite1xP2, SwiftVLDMPerm1],
        (instregex "VST2(d|b)(8|16|32)$")>;
  def : InstRW<[SwiftWriteP01OneCycle, SwiftWrite1xP2, SwiftVLDMPerm1],
        (instregex "VST2(b|d)(8|16|32)wb")>;
  def : InstRW<[SwiftWrite2xP2, SwiftVLDMPerm2],
        (instregex "VST2q(8|16|32)$", "VST2q(8|16|32)Pseudo$")>;
  def : InstRW<[SwiftWrite2xP2, SwiftVLDMPerm2],
        (instregex "VST2q(8|16|32)wb", "VST2q(8|16|32)PseudoWB")>;
  // Three element structure store.
  def : InstRW<[SwiftWrite4xP2, SwiftVLDMPerm2],
        (instregex "VST3(d|q)(8|16|32)$", "VST3(d|q)(8|16|32)(oddP|P)seudo$")>;
  def : InstRW<[SwiftWriteP01OneCycle, SwiftWrite4xP2, SwiftVLDMPerm2],
        (instregex "VST3(d|q)(8|16|32)_UPD",
                   "VST3(d|q)(8|16|32)(oddP|P)seudo_UPD$")>;
  // Four element structure store.
  def : InstRW<[SwiftWrite4xP2, SwiftVLDMPerm2],
        (instregex "VST4(d|q)(8|16|32)$", "VST4(d|q)(8|16|32)(oddP|P)seudo$")>;
  def : InstRW<[SwiftWriteP01OneCycle, SwiftWrite4xP2, SwiftVLDMPerm4],
        (instregex "VST4(d|q)(8|16|32)_UPD",
                   "VST4(d|q)(8|16|32)(oddP|P)seudo_UPD$")>;
  // Single/all lane store.
  // One element structure.
  def : InstRW<[SwiftWrite1xP2, SwiftVLDMPerm1],
        (instregex "VST1LNd(8|16|32)$", "VST1LNq(8|16|32)Pseudo$")>;
  def : InstRW<[SwiftWriteP01OneCycle, SwiftWrite1xP2, SwiftVLDMPerm1],
        (instregex "VST1LNd(8|16|32)_UPD", "VST1LNq(8|16|32)Pseudo_UPD")>;
  // Two element structure.
  def : InstRW<[SwiftWrite1xP2, SwiftVLDMPerm2],
        (instregex "VST2LN(d|q)(8|16|32)$", "VST2LN(d|q)(8|16|32)Pseudo$")>;
  def : InstRW<[SwiftWriteP01OneCycle, SwiftWrite1xP2, SwiftVLDMPerm2],
        (instregex "VST2LN(d|q)(8|16|32)_UPD",
                   "VST2LN(d|q)(8|16|32)Pseudo_UPD")>;
  // Three element structure.
  def : InstRW<[SwiftWrite4xP2, SwiftVLDMPerm2],
        (instregex "VST3LN(d|q)(8|16|32)$", "VST3LN(d|q)(8|16|32)Pseudo$")>;
  def : InstRW<[SwiftWriteP01OneCycle, SwiftWrite4xP2, SwiftVLDMPerm2],
        (instregex "VST3LN(d|q)(8|16|32)_UPD",
                   "VST3LN(d|q)(8|16|32)Pseudo_UPD")>;
  // Four element structure.
  def : InstRW<[SwiftWrite2xP2, SwiftVLDMPerm2],
        (instregex "VST4LN(d|q)(8|16|32)$", "VST4LN(d|q)(8|16|32)Pseudo$")>;
  def : InstRW<[SwiftWriteP01OneCycle, SwiftWrite2xP2, SwiftVLDMPerm2],
        (instregex "VST4LN(d|q)(8|16|32)_UPD",
                   "VST4LN(d|q)(8|16|32)Pseudo_UPD")>;

  // 4.2.44 VFP, Divide and Square Root
  def SwiftDiv17 : SchedWriteRes<[SwiftUnitP0, SwiftUnitDiv]> {
    let NumMicroOps = 1;
    let Latency = 17;
    let ResourceCycles = [1, 15];
  }
  def SwiftDiv32 : SchedWriteRes<[SwiftUnitP0, SwiftUnitDiv]> {
    let NumMicroOps = 1;
    let Latency = 32;
    let ResourceCycles = [1, 30];
  }
  def : InstRW<[SwiftDiv17], (instregex "VDIVS", "VSQRTS")>;
  def : InstRW<[SwiftDiv32], (instregex "VDIVD", "VSQRTD")>;

  // Not specified.
  def : InstRW<[SwiftWriteP01OneCycle2x], (instregex "ABS")>;
  // Preload.
  def : WriteRes<WritePreLd, [SwiftUnitP2]> { let Latency = 0;
    let ResourceCycles = [0];
  }

}
