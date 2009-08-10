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

// Single issue pipeline so every itinerary starts with FU_pipe0
def V6Itineraries : ProcessorItineraries<[
  // single-cycle integer ALU
  InstrItinData<IIC_iALU    , [InstrStage<1, [FU_Pipe0]>]>,
  // loads have an extra cycle of latency, but are fully pipelined
  InstrItinData<IIC_iLoad   , [InstrStage<1, [FU_Pipe0]>, InstrStage<1, [FU_LdSt0]>]>,
  InstrItinData<IIC_fpLoad  , [InstrStage<1, [FU_Pipe0]>, InstrStage<1, [FU_LdSt0]>]>,
  // fully-pipelined stores
  InstrItinData<IIC_iStore  , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_fpStore , [InstrStage<1, [FU_Pipe0]>]>,
  // fp ALU is not pipelined
  InstrItinData<IIC_fpALU   , [InstrStage<6, [FU_Pipe0]>]>,
  // no delay slots, so the latency of a branch is unimportant
  InstrItinData<IIC_Br      , [InstrStage<1, [FU_Pipe0]>]>
]>;
