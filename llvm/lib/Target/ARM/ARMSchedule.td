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
def FU_iALU   : FuncUnit; // Integer alu unit
def FU_iLdSt  : FuncUnit; // Integer load / store unit
def FU_FpALU  : FuncUnit; // FP alu unit
def FU_FpLdSt : FuncUnit; // FP load / store unit
def FU_Br     : FuncUnit; // Branch unit

//===----------------------------------------------------------------------===//
// Instruction Itinerary classes used for ARM
//
def IIC_iALU    : InstrItinClass;
def IIC_iLoad   : InstrItinClass;
def IIC_iStore  : InstrItinClass;
def IIC_fpALU   : InstrItinClass;
def IIC_fpLoad  : InstrItinClass;
def IIC_fpStore : InstrItinClass;
def IIC_Br      : InstrItinClass;

//===----------------------------------------------------------------------===//
// Processor instruction itineraries.

def GenericItineraries : ProcessorItineraries<[]>;

include "ARMScheduleV6.td"
include "ARMScheduleV7.td"
