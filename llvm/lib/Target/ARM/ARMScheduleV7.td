//===- ARMScheduleV7.td - ARM v7 Scheduling Definitions ----*- tablegen -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines the itinerary class data for the ARM v7 processors.
//
//===----------------------------------------------------------------------===//

def V7Itineraries : ProcessorItineraries<[
  InstrItinData<IIC_iALU    , [InstrStage<1, [FU_iALU]>]>,
  InstrItinData<IIC_iLoad   , [InstrStage<2, [FU_iLdSt]>]>,
  InstrItinData<IIC_iStore  , [InstrStage<1, [FU_iLdSt]>]>,
  InstrItinData<IIC_fpALU   , [InstrStage<6, [FU_FpALU]>]>,
  InstrItinData<IIC_fpLoad  , [InstrStage<2, [FU_FpLdSt]>]>,
  InstrItinData<IIC_fpStore , [InstrStage<1, [FU_FpLdSt]>]>,
  InstrItinData<IIC_Br      , [InstrStage<3, [FU_Br]>]>
]>;


def CortexA8Itineraries : ProcessorItineraries<[
  InstrItinData<IIC_iALU    , [InstrStage<1, [FU_iALU]>]>,
  InstrItinData<IIC_iLoad   , [InstrStage<2, [FU_iLdSt]>]>,
  InstrItinData<IIC_iStore  , [InstrStage<1, [FU_iLdSt]>]>,
  InstrItinData<IIC_fpALU   , [InstrStage<6, [FU_FpALU]>]>,
  InstrItinData<IIC_fpLoad  , [InstrStage<2, [FU_FpLdSt]>]>,
  InstrItinData<IIC_fpStore , [InstrStage<1, [FU_FpLdSt]>]>,
  InstrItinData<IIC_Br      , [InstrStage<3, [FU_Br]>]>
]>;
