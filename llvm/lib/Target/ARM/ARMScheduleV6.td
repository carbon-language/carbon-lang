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
  InstrItinData<IIC_iALU    , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iMPY    , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iLoad   , [InstrStage<1, [FU_Pipe0]>, InstrStage<1, [FU_LdSt0]>]>,
  InstrItinData<IIC_iStore  , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_Br      , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_fpALU   , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_fpMPY   , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_fpLoad  , [InstrStage<1, [FU_Pipe0]>, InstrStage<1, [FU_LdSt0]>]>,
  InstrItinData<IIC_fpStore , [InstrStage<1, [FU_Pipe0]>]>
]>;
