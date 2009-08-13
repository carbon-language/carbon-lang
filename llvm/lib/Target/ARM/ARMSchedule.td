//===- ARMSchedule.td - ARM Scheduling Definitions ---------*- tablegen -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Functional units across ARM processors
//
def FU_Issue   : FuncUnit; // issue
def FU_Pipe0   : FuncUnit; // pipeline 0
def FU_Pipe1   : FuncUnit; // pipeline 1
def FU_LdSt0   : FuncUnit; // pipeline 0 load/store
def FU_LdSt1   : FuncUnit; // pipeline 1 load/store

//===----------------------------------------------------------------------===//
// Instruction Itinerary classes used for ARM
//
def IIC_iALU    : InstrItinClass;
def IIC_iMPYh   : InstrItinClass;
def IIC_iMPYw   : InstrItinClass;
def IIC_iMPYl   : InstrItinClass;
def IIC_iLoad   : InstrItinClass;
def IIC_iStore  : InstrItinClass;
def IIC_fpALU   : InstrItinClass;
def IIC_fpMPY   : InstrItinClass;
def IIC_fpLoad  : InstrItinClass;
def IIC_fpStore : InstrItinClass;
def IIC_Br      : InstrItinClass;

//===----------------------------------------------------------------------===//
// Processor instruction itineraries.

def GenericItineraries : ProcessorItineraries<[
  InstrItinData<IIC_iALU    , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iMPYh   , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iMPYw   , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iMPYl   , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iLoad   , [InstrStage<1, [FU_Pipe0]>, InstrStage<1, [FU_LdSt0]>]>,
  InstrItinData<IIC_iStore  , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_Br      , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_fpALU   , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_fpMPY   , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_fpLoad  , [InstrStage<1, [FU_Pipe0]>, InstrStage<1, [FU_LdSt0]>]>,
  InstrItinData<IIC_fpStore , [InstrStage<1, [FU_Pipe0]>]>
]>;


include "ARMScheduleV6.td"
include "ARMScheduleV7.td"
