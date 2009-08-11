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

// Dual issue pipeline so every itinerary starts with FU_Pipe0 | FU_Pipe1
def CortexA8Itineraries : ProcessorItineraries<[
  // two fully-pipelined integer ALU pipelines
  InstrItinData<IIC_iALU    , [InstrStage<1, [FU_Pipe0, FU_Pipe1]>]>,
  // one fully-pipelined integer Multiply pipeline
  // function units are used in alpha order, so use FU_Pipe1
  // for the Multiple pipeline
  InstrItinData<IIC_iMPY    , [InstrStage<1, [FU_Pipe1]>]>,
  // loads have an extra cycle of latency, but are fully pipelined
  // use a 0 cycle FU_Issue to enforce the 1 load/store per cycle limit
  InstrItinData<IIC_iLoad   , [InstrStage<0, [FU_Issue]>, 
                               InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<1, [FU_LdSt0]>]>,
  // fully-pipelined stores
  // use a 0 cycle FU_Issue to enforce the 1 load/store per cycle limit
  InstrItinData<IIC_iStore  , [InstrStage<0, [FU_Issue]>, 
                               InstrStage<1, [FU_Pipe0, FU_Pipe1]>]>,
  // no delay slots, so the latency of a branch is unimportant
  InstrItinData<IIC_Br      , [InstrStage<1, [FU_Pipe0, FU_Pipe1]>]>,

  // VFP ALU is not pipelined so stall all issues 
  // FIXME assume NFP pipeline and 7 cycle non-pipelined latency
  InstrItinData<IIC_fpALU   , [InstrStage<7, [FU_Pipe0, FU_Pipe1]>]>,
  // VFP MPY is not pipelined so stall all issues 
  // FIXME assume NFP pipeline and 7 cycle non-pipelined latency
  InstrItinData<IIC_fpMPY   , [InstrStage<7, [FU_Pipe0, FU_Pipe1]>]>,
  // loads have an extra cycle of latency, but are fully pipelined
  // use a 0 cycle FU_Issue to enforce the 1 load/store per cycle limit
  InstrItinData<IIC_fpLoad  , [InstrStage<0, [FU_Issue]>, 
                               InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<1, [FU_LdSt0]>]>,
  // use a 0 cycle FU_Issue to enforce the 1 load/store per cycle limit
  InstrItinData<IIC_fpStore , [InstrStage<0, [FU_Issue]>, 
                               InstrStage<1, [FU_Pipe0, FU_Pipe1]>]>
]>;

// FIXME
def CortexA9Itineraries : ProcessorItineraries<[
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
