//===-- AMDIL.h - Top-level interface for AMDIL representation --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//==-----------------------------------------------------------------------===//
//
/// This file contains the entry points for global functions defined in the LLVM
/// AMDGPU back-end.
//
//===----------------------------------------------------------------------===//

#ifndef AMDIL_H
#define AMDIL_H

#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/Target/TargetMachine.h"

#define ARENA_SEGMENT_RESERVED_UAVS 12
#define DEFAULT_ARENA_UAV_ID 8
#define DEFAULT_RAW_UAV_ID 7
#define GLOBAL_RETURN_RAW_UAV_ID 11
#define HW_MAX_NUM_CB 8
#define MAX_NUM_UNIQUE_UAVS 8
#define OPENCL_MAX_NUM_ATOMIC_COUNTERS 8
#define OPENCL_MAX_READ_IMAGES 128
#define OPENCL_MAX_WRITE_IMAGES 8
#define OPENCL_MAX_SAMPLERS 16

// The next two values can never be zero, as zero is the ID that is
// used to assert against.
#define DEFAULT_LDS_ID     1
#define DEFAULT_GDS_ID     1
#define DEFAULT_SCRATCH_ID 1
#define DEFAULT_VEC_SLOTS  8

#define OCL_DEVICE_RV710        0x0001
#define OCL_DEVICE_RV730        0x0002
#define OCL_DEVICE_RV770        0x0004
#define OCL_DEVICE_CEDAR        0x0008
#define OCL_DEVICE_REDWOOD      0x0010
#define OCL_DEVICE_JUNIPER      0x0020
#define OCL_DEVICE_CYPRESS      0x0040
#define OCL_DEVICE_CAICOS       0x0080
#define OCL_DEVICE_TURKS        0x0100
#define OCL_DEVICE_BARTS        0x0200
#define OCL_DEVICE_CAYMAN       0x0400
#define OCL_DEVICE_ALL          0x3FFF

/// The number of function ID's that are reserved for 
/// internal compiler usage.
const unsigned int RESERVED_FUNCS = 1024;

namespace llvm {
class AMDGPUInstrPrinter;
class FunctionPass;
class MCAsmInfo;
class raw_ostream;
class Target;
class TargetMachine;

// Instruction selection passes.
FunctionPass*
  createAMDGPUISelDag(TargetMachine &TM);
FunctionPass*
  createAMDGPUPeepholeOpt(TargetMachine &TM);

// Pre emit passes.
FunctionPass*
  createAMDGPUCFGPreparationPass(TargetMachine &TM);
FunctionPass*
  createAMDGPUCFGStructurizerPass(TargetMachine &TM);

extern Target TheAMDGPUTarget;
} // end namespace llvm;

// Include device information enumerations
#include "AMDILDeviceInfo.h"

namespace llvm {
/// OpenCL uses address spaces to differentiate between
/// various memory regions on the hardware. On the CPU
/// all of the address spaces point to the same memory,
/// however on the GPU, each address space points to
/// a seperate piece of memory that is unique from other
/// memory locations.
namespace AMDGPUAS {
enum AddressSpaces {
  PRIVATE_ADDRESS  = 0, ///< Address space for private memory.
  GLOBAL_ADDRESS   = 1, ///< Address space for global memory (RAT0, VTX0).
  CONSTANT_ADDRESS = 2, ///< Address space for constant memory
  LOCAL_ADDRESS    = 3, ///< Address space for local memory.
  REGION_ADDRESS   = 4, ///< Address space for region memory.
  ADDRESS_NONE     = 5, ///< Address space for unknown memory.
  PARAM_D_ADDRESS  = 6, ///< Address space for direct addressible parameter memory (CONST0)
  PARAM_I_ADDRESS  = 7, ///< Address space for indirect addressible parameter memory (VTX1)
  CONSTANT_BUFFER_0 = 8,
  CONSTANT_BUFFER_1 = 9,
  CONSTANT_BUFFER_2 = 10,
  CONSTANT_BUFFER_3 = 11,
  CONSTANT_BUFFER_4 = 12,
  CONSTANT_BUFFER_5 = 13,
  CONSTANT_BUFFER_6 = 14,
  CONSTANT_BUFFER_7 = 15,
  CONSTANT_BUFFER_8 = 16,
  CONSTANT_BUFFER_9 = 17,
  CONSTANT_BUFFER_10 = 18,
  CONSTANT_BUFFER_11 = 19,
  CONSTANT_BUFFER_12 = 20,
  CONSTANT_BUFFER_13 = 21,
  CONSTANT_BUFFER_14 = 22,
  CONSTANT_BUFFER_15 = 23,
  LAST_ADDRESS     = 24
};

} // namespace AMDGPUAS

} // end namespace llvm
#endif // AMDIL_H
