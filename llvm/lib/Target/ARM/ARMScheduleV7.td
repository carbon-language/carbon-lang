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
  // integer Multiply pipeline
  InstrItinData<IIC_iMPYh   , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iMPYw   , [InstrStage<1, [FU_Pipe1], 0>, 
                               InstrStage<2, [FU_Pipe0]>]>,
  InstrItinData<IIC_iMPYl   , [InstrStage<2, [FU_Pipe1], 0>, 
                               InstrStage<3, [FU_Pipe0]>]>,
  // loads have an extra cycle of latency, but are fully pipelined
  // use FU_Issue to enforce the 1 load/store per cycle limit
  InstrItinData<IIC_iLoad   , [InstrStage<1, [FU_Issue], 0>, 
                               InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<1, [FU_LdSt0]>]>,
  // fully-pipelined stores
  // use FU_Issue to enforce the 1 load/store per cycle limit
  InstrItinData<IIC_iStore  , [InstrStage<1, [FU_Issue], 0>, 
                               InstrStage<1, [FU_Pipe0, FU_Pipe1]>]>,
  // no delay slots, so the latency of a branch is unimportant
  InstrItinData<IIC_Br      , [InstrStage<1, [FU_Pipe0, FU_Pipe1]>]>,

  // NFP ALU is not pipelined so stall all issues 
  InstrItinData<IIC_fpALU   , [InstrStage<7, [FU_Pipe0], 0>,
                               InstrStage<7, [FU_Pipe1], 0>]>,
  // VFP MPY is not pipelined so stall all issues 
  InstrItinData<IIC_fpMPY   , [InstrStage<7, [FU_Pipe0], 0>,
                               InstrStage<7, [FU_Pipe1], 0>]>,
  // loads have an extra cycle of latency, but are fully pipelined
  // use FU_Issue to enforce the 1 load/store per cycle limit
  InstrItinData<IIC_fpLoad  , [InstrStage<1, [FU_Issue], 0>, 
                               InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<1, [FU_LdSt0]>]>,
  // use FU_Issue to enforce the 1 load/store per cycle limit
  InstrItinData<IIC_fpStore , [InstrStage<1, [FU_Issue], 0>, 
                               InstrStage<1, [FU_Pipe0, FU_Pipe1]>]>
]>;

// FIXME
def CortexA9Itineraries : ProcessorItineraries<[
  InstrItinData<IIC_iALU    , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iMPYh   , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iMPYw   , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iMPYl   , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iLoad   , [InstrStage<1, [FU_Pipe0]>,
                               InstrStage<1, [FU_LdSt0]>]>,
  InstrItinData<IIC_iStore  , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_Br      , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_fpALU   , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_fpMPY   , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_fpLoad  , [InstrStage<1, [FU_Pipe0]>,
                               InstrStage<1, [FU_LdSt0]>]>,
  InstrItinData<IIC_fpStore , [InstrStage<1, [FU_Pipe0]>]>
]>;
