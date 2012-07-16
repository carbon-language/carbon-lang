//===------------ AMDILInstrInfo.td - AMDIL Target ------*-tablegen-*------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//==-----------------------------------------------------------------------===//
//
// This file describes the AMDIL instructions in TableGen format.
//
//===----------------------------------------------------------------------===//
// AMDIL Instruction Predicate Definitions
// Predicate that is set to true if the hardware supports double precision
// divide
def HasHWDDiv                 : Predicate<"Subtarget.device()"
                           "->getGeneration() > AMDILDeviceInfo::HD4XXX && "
              "Subtarget.device()->usesHardware(AMDILDeviceInfo::DoubleOps)">;

// Predicate that is set to true if the hardware supports double, but not double
// precision divide in hardware
def HasSWDDiv             : Predicate<"Subtarget.device()"
                           "->getGeneration() == AMDILDeviceInfo::HD4XXX &&"
              "Subtarget.device()->usesHardware(AMDILDeviceInfo::DoubleOps)">;

// Predicate that is set to true if the hardware support 24bit signed
// math ops. Otherwise a software expansion to 32bit math ops is used instead.
def HasHWSign24Bit          : Predicate<"Subtarget.device()"
                            "->getGeneration() > AMDILDeviceInfo::HD5XXX">;

// Predicate that is set to true if 64bit operations are supported or not
def HasHW64Bit              : Predicate<"Subtarget.device()"
                            "->usesHardware(AMDILDeviceInfo::LongOps)">;
def HasSW64Bit              : Predicate<"Subtarget.device()"
                            "->usesSoftware(AMDILDeviceInfo::LongOps)">;

// Predicate that is set to true if the timer register is supported
def HasTmrRegister          : Predicate<"Subtarget.device()"
                            "->isSupported(AMDILDeviceInfo::TmrReg)">;
// Predicate that is true if we are at least evergreen series
def HasDeviceIDInst         : Predicate<"Subtarget.device()"
                            "->getGeneration() >= AMDILDeviceInfo::HD5XXX">;

// Predicate that is true if we have region address space.
def hasRegionAS             : Predicate<"Subtarget.device()"
                            "->usesHardware(AMDILDeviceInfo::RegionMem)">;

// Predicate that is false if we don't have region address space.
def noRegionAS             : Predicate<"!Subtarget.device()"
                            "->isSupported(AMDILDeviceInfo::RegionMem)">;


// Predicate that is set to true if 64bit Mul is supported in the IL or not
def HasHW64Mul              : Predicate<"Subtarget.calVersion()" 
                                          ">= CAL_VERSION_SC_139"
                                          "&& Subtarget.device()"
                                          "->getGeneration() >="
                                          "AMDILDeviceInfo::HD5XXX">;
def HasSW64Mul              : Predicate<"Subtarget.calVersion()" 
                                          "< CAL_VERSION_SC_139">;
// Predicate that is set to true if 64bit Div/Mod is supported in the IL or not
def HasHW64DivMod           : Predicate<"Subtarget.device()"
                            "->usesHardware(AMDILDeviceInfo::HW64BitDivMod)">;
def HasSW64DivMod           : Predicate<"Subtarget.device()"
                            "->usesSoftware(AMDILDeviceInfo::HW64BitDivMod)">;

// Predicate that is set to true if 64bit pointer are used.
def Has64BitPtr             : Predicate<"Subtarget.is64bit()">;
def Has32BitPtr             : Predicate<"!Subtarget.is64bit()">;
//===--------------------------------------------------------------------===//
// Custom Operands
//===--------------------------------------------------------------------===//
include "AMDILOperands.td"

//===--------------------------------------------------------------------===//
// Custom Selection DAG Type Profiles
//===--------------------------------------------------------------------===//
include "AMDILProfiles.td"

//===--------------------------------------------------------------------===//
// Custom Selection DAG Nodes
//===--------------------------------------------------------------------===//
include "AMDILNodes.td"

//===--------------------------------------------------------------------===//
// Custom Pattern DAG Nodes
//===--------------------------------------------------------------------===//
include "AMDILPatterns.td"

//===----------------------------------------------------------------------===//
// Instruction format classes
//===----------------------------------------------------------------------===//
include "AMDILFormats.td"

//===--------------------------------------------------------------------===//
// Multiclass Instruction formats
//===--------------------------------------------------------------------===//
include "AMDILMultiClass.td"

//===--------------------------------------------------------------------===//
// Intrinsics support
//===--------------------------------------------------------------------===//
include "AMDILIntrinsics.td"

//===--------------------------------------------------------------------===//
// Instructions support
//===--------------------------------------------------------------------===//
include "AMDILInstructions.td"
