//===- AArch64Schedule.td - AArch64 Scheduling Definitions -*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Generic processor itineraries for legacy compatibility.

def GenericItineraries : ProcessorItineraries<[], [], []>;


//===----------------------------------------------------------------------===//
// Base SchedReadWrite types

// Basic ALU
def WriteALU : SchedWrite;  // Generic: may contain shift and/or ALU operation
def WriteALUs : SchedWrite; // Shift only with no ALU operation
def ReadALU : SchedRead;    // Operand not needed for shifting
def ReadALUs : SchedRead;   // Operand needed for shifting

// Multiply with optional accumulate
def WriteMAC : SchedWrite;
def ReadMAC : SchedRead;

// Compares
def WriteCMP : SchedWrite;
def ReadCMP : SchedRead;

// Division
def WriteDiv : SchedWrite;
def ReadDiv : SchedRead;

// Loads
def WriteLd : SchedWrite;
def WritePreLd : SchedWrite;
def WriteVecLd : SchedWrite;
def ReadLd : SchedRead;
def ReadPreLd : SchedRead;
def ReadVecLd : SchedRead;

// Stores
def WriteSt : SchedWrite;
def WriteVecSt : SchedWrite;
def ReadSt : SchedRead;
def ReadVecSt : SchedRead;

// Branches
def WriteBr : SchedWrite;
def WriteBrL : SchedWrite;
def ReadBr : SchedRead;

// Floating Point ALU
def WriteFPALU : SchedWrite;
def ReadFPALU : SchedRead;

// Floating Point MAC, Mul, Div, Sqrt
//   Most processors will simply send all of these down a dedicated pipe, but
//   they're explicitly seperated here for flexibility of modeling later. May
//   consider consolidating them into a single WriteFPXXXX type in the future.
def WriteFPMAC : SchedWrite;
def WriteFPMul : SchedWrite;
def WriteFPDiv : SchedWrite;
def WriteFPSqrt : SchedWrite;
def ReadFPMAC : SchedRead;
def ReadFPMul : SchedRead;
def ReadFPDiv : SchedRead;
def ReadFPSqrt : SchedRead;

// Noop
def WriteNoop : SchedWrite;


//===----------------------------------------------------------------------===//
// Subtarget specific Machine Models.

include "AArch64ScheduleA53.td"
