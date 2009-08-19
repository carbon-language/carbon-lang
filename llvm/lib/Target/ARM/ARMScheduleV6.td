//===- ARMScheduleV6.td - ARM v6 Scheduling Definitions ----*- tablegen -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines the itinerary class data for the ARM v6 processors.
//
//===----------------------------------------------------------------------===//

// TODO: this should model an ARM11
// Single issue pipeline so every itinerary starts with FU_pipe0
def V6Itineraries : ProcessorItineraries<[
  InstrItinData<IIC_iALUx   , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iALUi   , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iALUr   , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iALUsi  , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iALUsr  , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iUNAr   , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iUNAsi  , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iUNAsr  , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iCMPi   , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iCMPr   , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iCMPsi  , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iCMPsr  , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iMOVi   , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iMOVr   , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iMOVsi  , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iMOVsr  , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iCMOVi  , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iCMOVr  , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iCMOVsi , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iCMOVsr , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iMUL16  , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iMAC16  , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iMUL32  , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iMAC32  , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iMUL64  , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iMAC64  , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iLoadi  , [InstrStage<1, [FU_Pipe0]>,
                               InstrStage<1, [FU_LdSt0]>]>,
  InstrItinData<IIC_iLoadr  , [InstrStage<1, [FU_Pipe0]>,
                               InstrStage<1, [FU_LdSt0]>]>,
  InstrItinData<IIC_iLoadsi , [InstrStage<1, [FU_Pipe0]>,
                               InstrStage<1, [FU_LdSt0]>]>,
  InstrItinData<IIC_iLoadiu , [InstrStage<1, [FU_Pipe0]>,
                               InstrStage<1, [FU_LdSt0]>]>,
  InstrItinData<IIC_iLoadru , [InstrStage<1, [FU_Pipe0]>,
                               InstrStage<1, [FU_LdSt0]>]>,
  InstrItinData<IIC_iLoadsiu, [InstrStage<1, [FU_Pipe0]>,
                               InstrStage<1, [FU_LdSt0]>]>,
  InstrItinData<IIC_iLoadm  , [InstrStage<2, [FU_Pipe0]>,
                               InstrStage<2, [FU_LdSt0]>]>,
  InstrItinData<IIC_iStorei  , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iStorer  , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iStoresi , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iStoreiu , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iStoreru , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iStoresiu, [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iStorem  , [InstrStage<2, [FU_Pipe0]>]>,
  InstrItinData<IIC_Br      , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_fpALU   , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_fpMPY   , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_fpLoad  , [InstrStage<1, [FU_Pipe0]>,
                               InstrStage<1, [FU_LdSt0]>]>,
  InstrItinData<IIC_fpStore , [InstrStage<1, [FU_Pipe0]>]>
]>;
