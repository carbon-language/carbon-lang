//===-- SIISelLowering.cpp - SI DAG Lowering Implementation ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Custom DAG lowering for SI
//
//===----------------------------------------------------------------------===//

#include "SIISelLowering.h"
#include "AMDGPU.h"
#include "AMDGPUInstrInfo.h"
#include "AMDGPUTargetMachine.h"
#include "SIMachineFunctionInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/LegacyDivergenceAnalysis.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/CodeGen/Analysis.h"
#include "llvm/CodeGen/FunctionLoweringInfo.h"
#include "llvm/CodeGen/GlobalISel/GISelKnownBits.h"
#include "llvm/CodeGen/GlobalISel/MIPatternMatch.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/IntrinsicsR600.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/KnownBits.h"

using namespace llvm;

#define DEBUG_TYPE "si-lower"

STATISTIC(NumTailCalls, "Number of tail calls");

static cl::opt<bool> DisableLoopAlignment(
  "amdgpu-disable-loop-alignment",
  cl::desc("Do not align and prefetch loops"),
  cl::init(false));

static cl::opt<bool> UseDivergentRegisterIndexing(
  "amdgpu-use-divergent-register-indexing",
  cl::Hidden,
  cl::desc("Use indirect register addressing for divergent indexes"),
  cl::init(false));

static bool hasFP32Denormals(const MachineFunction &MF) {
  const SIMachineFunctionInfo *Info = MF.getInfo<SIMachineFunctionInfo>();
  return Info->getMode().allFP32Denormals();
}

static bool hasFP64FP16Denormals(const MachineFunction &MF) {
  const SIMachineFunctionInfo *Info = MF.getInfo<SIMachineFunctionInfo>();
  return Info->getMode().allFP64FP16Denormals();
}

static unsigned findFirstFreeSGPR(CCState &CCInfo) {
  unsigned NumSGPRs = AMDGPU::SGPR_32RegClass.getNumRegs();
  for (unsigned Reg = 0; Reg < NumSGPRs; ++Reg) {
    if (!CCInfo.isAllocated(AMDGPU::SGPR0 + Reg)) {
      return AMDGPU::SGPR0 + Reg;
    }
  }
  llvm_unreachable("Cannot allocate sgpr");
}

SITargetLowering::SITargetLowering(const TargetMachine &TM,
                                   const GCNSubtarget &STI)
    : AMDGPUTargetLowering(TM, STI),
      Subtarget(&STI) {
  addRegisterClass(MVT::i1, &AMDGPU::VReg_1RegClass);
  addRegisterClass(MVT::i64, &AMDGPU::SReg_64RegClass);

  addRegisterClass(MVT::i32, &AMDGPU::SReg_32RegClass);
  addRegisterClass(MVT::f32, &AMDGPU::VGPR_32RegClass);

  addRegisterClass(MVT::v2i32, &AMDGPU::SReg_64RegClass);

  const SIRegisterInfo *TRI = STI.getRegisterInfo();
  const TargetRegisterClass *V64RegClass = TRI->getVGPR64Class();

  addRegisterClass(MVT::f64, V64RegClass);
  addRegisterClass(MVT::v2f32, V64RegClass);

  addRegisterClass(MVT::v3i32, &AMDGPU::SGPR_96RegClass);
  addRegisterClass(MVT::v3f32, TRI->getVGPRClassForBitWidth(96));

  addRegisterClass(MVT::v2i64, &AMDGPU::SGPR_128RegClass);
  addRegisterClass(MVT::v2f64, &AMDGPU::SGPR_128RegClass);

  addRegisterClass(MVT::v4i32, &AMDGPU::SGPR_128RegClass);
  addRegisterClass(MVT::v4f32, TRI->getVGPRClassForBitWidth(128));

  addRegisterClass(MVT::v5i32, &AMDGPU::SGPR_160RegClass);
  addRegisterClass(MVT::v5f32, TRI->getVGPRClassForBitWidth(160));

  addRegisterClass(MVT::v6i32, &AMDGPU::SGPR_192RegClass);
  addRegisterClass(MVT::v6f32, TRI->getVGPRClassForBitWidth(192));

  addRegisterClass(MVT::v3i64, &AMDGPU::SGPR_192RegClass);
  addRegisterClass(MVT::v3f64, TRI->getVGPRClassForBitWidth(192));

  addRegisterClass(MVT::v7i32, &AMDGPU::SGPR_224RegClass);
  addRegisterClass(MVT::v7f32, TRI->getVGPRClassForBitWidth(224));

  addRegisterClass(MVT::v8i32, &AMDGPU::SGPR_256RegClass);
  addRegisterClass(MVT::v8f32, TRI->getVGPRClassForBitWidth(256));

  addRegisterClass(MVT::v4i64, &AMDGPU::SGPR_256RegClass);
  addRegisterClass(MVT::v4f64, TRI->getVGPRClassForBitWidth(256));

  addRegisterClass(MVT::v16i32, &AMDGPU::SGPR_512RegClass);
  addRegisterClass(MVT::v16f32, TRI->getVGPRClassForBitWidth(512));

  addRegisterClass(MVT::v8i64, &AMDGPU::SGPR_512RegClass);
  addRegisterClass(MVT::v8f64, TRI->getVGPRClassForBitWidth(512));

  addRegisterClass(MVT::v16i64, &AMDGPU::SGPR_1024RegClass);
  addRegisterClass(MVT::v16f64, TRI->getVGPRClassForBitWidth(1024));

  if (Subtarget->has16BitInsts()) {
    addRegisterClass(MVT::i16, &AMDGPU::SReg_32RegClass);
    addRegisterClass(MVT::f16, &AMDGPU::SReg_32RegClass);

    // Unless there are also VOP3P operations, not operations are really legal.
    addRegisterClass(MVT::v2i16, &AMDGPU::SReg_32RegClass);
    addRegisterClass(MVT::v2f16, &AMDGPU::SReg_32RegClass);
    addRegisterClass(MVT::v4i16, &AMDGPU::SReg_64RegClass);
    addRegisterClass(MVT::v4f16, &AMDGPU::SReg_64RegClass);
    addRegisterClass(MVT::v8i16, &AMDGPU::SGPR_128RegClass);
    addRegisterClass(MVT::v8f16, &AMDGPU::SGPR_128RegClass);
  }

  addRegisterClass(MVT::v32i32, &AMDGPU::VReg_1024RegClass);
  addRegisterClass(MVT::v32f32, TRI->getVGPRClassForBitWidth(1024));

  computeRegisterProperties(Subtarget->getRegisterInfo());

  // The boolean content concept here is too inflexible. Compares only ever
  // really produce a 1-bit result. Any copy/extend from these will turn into a
  // select, and zext/1 or sext/-1 are equally cheap. Arbitrarily choose 0/1, as
  // it's what most targets use.
  setBooleanContents(ZeroOrOneBooleanContent);
  setBooleanVectorContents(ZeroOrOneBooleanContent);

  // We need to custom lower vector stores from local memory
  setOperationAction(ISD::LOAD, MVT::v2i32, Custom);
  setOperationAction(ISD::LOAD, MVT::v3i32, Custom);
  setOperationAction(ISD::LOAD, MVT::v4i32, Custom);
  setOperationAction(ISD::LOAD, MVT::v5i32, Custom);
  setOperationAction(ISD::LOAD, MVT::v6i32, Custom);
  setOperationAction(ISD::LOAD, MVT::v7i32, Custom);
  setOperationAction(ISD::LOAD, MVT::v8i32, Custom);
  setOperationAction(ISD::LOAD, MVT::v16i32, Custom);
  setOperationAction(ISD::LOAD, MVT::i1, Custom);
  setOperationAction(ISD::LOAD, MVT::v32i32, Custom);

  setOperationAction(ISD::STORE, MVT::v2i32, Custom);
  setOperationAction(ISD::STORE, MVT::v3i32, Custom);
  setOperationAction(ISD::STORE, MVT::v4i32, Custom);
  setOperationAction(ISD::STORE, MVT::v5i32, Custom);
  setOperationAction(ISD::STORE, MVT::v6i32, Custom);
  setOperationAction(ISD::STORE, MVT::v7i32, Custom);
  setOperationAction(ISD::STORE, MVT::v8i32, Custom);
  setOperationAction(ISD::STORE, MVT::v16i32, Custom);
  setOperationAction(ISD::STORE, MVT::i1, Custom);
  setOperationAction(ISD::STORE, MVT::v32i32, Custom);

  setTruncStoreAction(MVT::v2i32, MVT::v2i16, Expand);
  setTruncStoreAction(MVT::v3i32, MVT::v3i16, Expand);
  setTruncStoreAction(MVT::v4i32, MVT::v4i16, Expand);
  setTruncStoreAction(MVT::v8i32, MVT::v8i16, Expand);
  setTruncStoreAction(MVT::v16i32, MVT::v16i16, Expand);
  setTruncStoreAction(MVT::v32i32, MVT::v32i16, Expand);
  setTruncStoreAction(MVT::v2i32, MVT::v2i8, Expand);
  setTruncStoreAction(MVT::v4i32, MVT::v4i8, Expand);
  setTruncStoreAction(MVT::v8i32, MVT::v8i8, Expand);
  setTruncStoreAction(MVT::v16i32, MVT::v16i8, Expand);
  setTruncStoreAction(MVT::v32i32, MVT::v32i8, Expand);
  setTruncStoreAction(MVT::v2i16, MVT::v2i8, Expand);
  setTruncStoreAction(MVT::v4i16, MVT::v4i8, Expand);
  setTruncStoreAction(MVT::v8i16, MVT::v8i8, Expand);
  setTruncStoreAction(MVT::v16i16, MVT::v16i8, Expand);
  setTruncStoreAction(MVT::v32i16, MVT::v32i8, Expand);

  setTruncStoreAction(MVT::v3i64, MVT::v3i16, Expand);
  setTruncStoreAction(MVT::v3i64, MVT::v3i32, Expand);
  setTruncStoreAction(MVT::v4i64, MVT::v4i8, Expand);
  setTruncStoreAction(MVT::v8i64, MVT::v8i8, Expand);
  setTruncStoreAction(MVT::v8i64, MVT::v8i16, Expand);
  setTruncStoreAction(MVT::v8i64, MVT::v8i32, Expand);
  setTruncStoreAction(MVT::v16i64, MVT::v16i32, Expand);

  setOperationAction(ISD::GlobalAddress, MVT::i32, Custom);
  setOperationAction(ISD::GlobalAddress, MVT::i64, Custom);

  setOperationAction(ISD::SELECT, MVT::i1, Promote);
  setOperationAction(ISD::SELECT, MVT::i64, Custom);
  setOperationAction(ISD::SELECT, MVT::f64, Promote);
  AddPromotedToType(ISD::SELECT, MVT::f64, MVT::i64);

  setOperationAction(ISD::SELECT_CC, MVT::f32, Expand);
  setOperationAction(ISD::SELECT_CC, MVT::i32, Expand);
  setOperationAction(ISD::SELECT_CC, MVT::i64, Expand);
  setOperationAction(ISD::SELECT_CC, MVT::f64, Expand);
  setOperationAction(ISD::SELECT_CC, MVT::i1, Expand);

  setOperationAction(ISD::SETCC, MVT::i1, Promote);
  setOperationAction(ISD::SETCC, MVT::v2i1, Expand);
  setOperationAction(ISD::SETCC, MVT::v4i1, Expand);
  AddPromotedToType(ISD::SETCC, MVT::i1, MVT::i32);

  setOperationAction(ISD::TRUNCATE, MVT::v2i32, Expand);
  setOperationAction(ISD::FP_ROUND, MVT::v2f32, Expand);
  setOperationAction(ISD::TRUNCATE, MVT::v3i32, Expand);
  setOperationAction(ISD::FP_ROUND, MVT::v3f32, Expand);
  setOperationAction(ISD::TRUNCATE, MVT::v4i32, Expand);
  setOperationAction(ISD::FP_ROUND, MVT::v4f32, Expand);
  setOperationAction(ISD::TRUNCATE, MVT::v5i32, Expand);
  setOperationAction(ISD::FP_ROUND, MVT::v5f32, Expand);
  setOperationAction(ISD::TRUNCATE, MVT::v6i32, Expand);
  setOperationAction(ISD::FP_ROUND, MVT::v6f32, Expand);
  setOperationAction(ISD::TRUNCATE, MVT::v7i32, Expand);
  setOperationAction(ISD::FP_ROUND, MVT::v7f32, Expand);
  setOperationAction(ISD::TRUNCATE, MVT::v8i32, Expand);
  setOperationAction(ISD::FP_ROUND, MVT::v8f32, Expand);
  setOperationAction(ISD::TRUNCATE, MVT::v16i32, Expand);
  setOperationAction(ISD::FP_ROUND, MVT::v16f32, Expand);

  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::v2i1, Custom);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::v4i1, Custom);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::v2i8, Custom);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::v4i8, Custom);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::v2i16, Custom);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::v3i16, Custom);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::v4i16, Custom);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::Other, Custom);

  setOperationAction(ISD::BRCOND, MVT::Other, Custom);
  setOperationAction(ISD::BR_CC, MVT::i1, Expand);
  setOperationAction(ISD::BR_CC, MVT::i32, Expand);
  setOperationAction(ISD::BR_CC, MVT::i64, Expand);
  setOperationAction(ISD::BR_CC, MVT::f32, Expand);
  setOperationAction(ISD::BR_CC, MVT::f64, Expand);

  setOperationAction(ISD::UADDO, MVT::i32, Legal);
  setOperationAction(ISD::USUBO, MVT::i32, Legal);

  setOperationAction(ISD::ADDCARRY, MVT::i32, Legal);
  setOperationAction(ISD::SUBCARRY, MVT::i32, Legal);

  setOperationAction(ISD::SHL_PARTS, MVT::i64, Expand);
  setOperationAction(ISD::SRA_PARTS, MVT::i64, Expand);
  setOperationAction(ISD::SRL_PARTS, MVT::i64, Expand);

#if 0
  setOperationAction(ISD::ADDCARRY, MVT::i64, Legal);
  setOperationAction(ISD::SUBCARRY, MVT::i64, Legal);
#endif

  // We only support LOAD/STORE and vector manipulation ops for vectors
  // with > 4 elements.
  for (MVT VT : { MVT::v8i32, MVT::v8f32, MVT::v16i32, MVT::v16f32,
                  MVT::v2i64, MVT::v2f64, MVT::v4i16, MVT::v4f16,
                  MVT::v3i64, MVT::v3f64, MVT::v6i32, MVT::v6f32,
                  MVT::v4i64, MVT::v4f64, MVT::v8i64, MVT::v8f64,
                  MVT::v8i16, MVT::v8f16, MVT::v16i64, MVT::v16f64,
                  MVT::v32i32, MVT::v32f32 }) {
    for (unsigned Op = 0; Op < ISD::BUILTIN_OP_END; ++Op) {
      switch (Op) {
      case ISD::LOAD:
      case ISD::STORE:
      case ISD::BUILD_VECTOR:
      case ISD::BITCAST:
      case ISD::EXTRACT_VECTOR_ELT:
      case ISD::INSERT_VECTOR_ELT:
      case ISD::EXTRACT_SUBVECTOR:
      case ISD::SCALAR_TO_VECTOR:
        break;
      case ISD::INSERT_SUBVECTOR:
      case ISD::CONCAT_VECTORS:
        setOperationAction(Op, VT, Custom);
        break;
      default:
        setOperationAction(Op, VT, Expand);
        break;
      }
    }
  }

  setOperationAction(ISD::FP_EXTEND, MVT::v4f32, Expand);

  // TODO: For dynamic 64-bit vector inserts/extracts, should emit a pseudo that
  // is expanded to avoid having two separate loops in case the index is a VGPR.

  // Most operations are naturally 32-bit vector operations. We only support
  // load and store of i64 vectors, so promote v2i64 vector operations to v4i32.
  for (MVT Vec64 : { MVT::v2i64, MVT::v2f64 }) {
    setOperationAction(ISD::BUILD_VECTOR, Vec64, Promote);
    AddPromotedToType(ISD::BUILD_VECTOR, Vec64, MVT::v4i32);

    setOperationAction(ISD::EXTRACT_VECTOR_ELT, Vec64, Promote);
    AddPromotedToType(ISD::EXTRACT_VECTOR_ELT, Vec64, MVT::v4i32);

    setOperationAction(ISD::INSERT_VECTOR_ELT, Vec64, Promote);
    AddPromotedToType(ISD::INSERT_VECTOR_ELT, Vec64, MVT::v4i32);

    setOperationAction(ISD::SCALAR_TO_VECTOR, Vec64, Promote);
    AddPromotedToType(ISD::SCALAR_TO_VECTOR, Vec64, MVT::v4i32);
  }

  for (MVT Vec64 : { MVT::v3i64, MVT::v3f64 }) {
    setOperationAction(ISD::BUILD_VECTOR, Vec64, Promote);
    AddPromotedToType(ISD::BUILD_VECTOR, Vec64, MVT::v6i32);

    setOperationAction(ISD::EXTRACT_VECTOR_ELT, Vec64, Promote);
    AddPromotedToType(ISD::EXTRACT_VECTOR_ELT, Vec64, MVT::v6i32);

    setOperationAction(ISD::INSERT_VECTOR_ELT, Vec64, Promote);
    AddPromotedToType(ISD::INSERT_VECTOR_ELT, Vec64, MVT::v6i32);

    setOperationAction(ISD::SCALAR_TO_VECTOR, Vec64, Promote);
    AddPromotedToType(ISD::SCALAR_TO_VECTOR, Vec64, MVT::v6i32);
  }

  for (MVT Vec64 : { MVT::v4i64, MVT::v4f64 }) {
    setOperationAction(ISD::BUILD_VECTOR, Vec64, Promote);
    AddPromotedToType(ISD::BUILD_VECTOR, Vec64, MVT::v8i32);

    setOperationAction(ISD::EXTRACT_VECTOR_ELT, Vec64, Promote);
    AddPromotedToType(ISD::EXTRACT_VECTOR_ELT, Vec64, MVT::v8i32);

    setOperationAction(ISD::INSERT_VECTOR_ELT, Vec64, Promote);
    AddPromotedToType(ISD::INSERT_VECTOR_ELT, Vec64, MVT::v8i32);

    setOperationAction(ISD::SCALAR_TO_VECTOR, Vec64, Promote);
    AddPromotedToType(ISD::SCALAR_TO_VECTOR, Vec64, MVT::v8i32);
  }

  for (MVT Vec64 : { MVT::v8i64, MVT::v8f64 }) {
    setOperationAction(ISD::BUILD_VECTOR, Vec64, Promote);
    AddPromotedToType(ISD::BUILD_VECTOR, Vec64, MVT::v16i32);

    setOperationAction(ISD::EXTRACT_VECTOR_ELT, Vec64, Promote);
    AddPromotedToType(ISD::EXTRACT_VECTOR_ELT, Vec64, MVT::v16i32);

    setOperationAction(ISD::INSERT_VECTOR_ELT, Vec64, Promote);
    AddPromotedToType(ISD::INSERT_VECTOR_ELT, Vec64, MVT::v16i32);

    setOperationAction(ISD::SCALAR_TO_VECTOR, Vec64, Promote);
    AddPromotedToType(ISD::SCALAR_TO_VECTOR, Vec64, MVT::v16i32);
  }

  for (MVT Vec64 : { MVT::v16i64, MVT::v16f64 }) {
    setOperationAction(ISD::BUILD_VECTOR, Vec64, Promote);
    AddPromotedToType(ISD::BUILD_VECTOR, Vec64, MVT::v32i32);

    setOperationAction(ISD::EXTRACT_VECTOR_ELT, Vec64, Promote);
    AddPromotedToType(ISD::EXTRACT_VECTOR_ELT, Vec64, MVT::v32i32);

    setOperationAction(ISD::INSERT_VECTOR_ELT, Vec64, Promote);
    AddPromotedToType(ISD::INSERT_VECTOR_ELT, Vec64, MVT::v32i32);

    setOperationAction(ISD::SCALAR_TO_VECTOR, Vec64, Promote);
    AddPromotedToType(ISD::SCALAR_TO_VECTOR, Vec64, MVT::v32i32);
  }

  setOperationAction(ISD::VECTOR_SHUFFLE, MVT::v8i32, Expand);
  setOperationAction(ISD::VECTOR_SHUFFLE, MVT::v8f32, Expand);
  setOperationAction(ISD::VECTOR_SHUFFLE, MVT::v16i32, Expand);
  setOperationAction(ISD::VECTOR_SHUFFLE, MVT::v16f32, Expand);

  setOperationAction(ISD::BUILD_VECTOR, MVT::v4f16, Custom);
  setOperationAction(ISD::BUILD_VECTOR, MVT::v4i16, Custom);

  // Avoid stack access for these.
  // TODO: Generalize to more vector types.
  setOperationAction(ISD::EXTRACT_VECTOR_ELT, MVT::v2i16, Custom);
  setOperationAction(ISD::EXTRACT_VECTOR_ELT, MVT::v2f16, Custom);
  setOperationAction(ISD::INSERT_VECTOR_ELT, MVT::v2i16, Custom);
  setOperationAction(ISD::INSERT_VECTOR_ELT, MVT::v2f16, Custom);

  setOperationAction(ISD::EXTRACT_VECTOR_ELT, MVT::v2i8, Custom);
  setOperationAction(ISD::EXTRACT_VECTOR_ELT, MVT::v4i8, Custom);
  setOperationAction(ISD::EXTRACT_VECTOR_ELT, MVT::v8i8, Custom);
  setOperationAction(ISD::INSERT_VECTOR_ELT, MVT::v2i8, Custom);
  setOperationAction(ISD::INSERT_VECTOR_ELT, MVT::v4i8, Custom);
  setOperationAction(ISD::INSERT_VECTOR_ELT, MVT::v8i8, Custom);

  setOperationAction(ISD::EXTRACT_VECTOR_ELT, MVT::v4i16, Custom);
  setOperationAction(ISD::EXTRACT_VECTOR_ELT, MVT::v4f16, Custom);
  setOperationAction(ISD::INSERT_VECTOR_ELT, MVT::v4i16, Custom);
  setOperationAction(ISD::INSERT_VECTOR_ELT, MVT::v4f16, Custom);

  // Deal with vec3 vector operations when widened to vec4.
  setOperationAction(ISD::INSERT_SUBVECTOR, MVT::v3i32, Custom);
  setOperationAction(ISD::INSERT_SUBVECTOR, MVT::v3f32, Custom);
  setOperationAction(ISD::INSERT_SUBVECTOR, MVT::v4i32, Custom);
  setOperationAction(ISD::INSERT_SUBVECTOR, MVT::v4f32, Custom);

  // Deal with vec5/6/7 vector operations when widened to vec8.
  setOperationAction(ISD::INSERT_SUBVECTOR, MVT::v5i32, Custom);
  setOperationAction(ISD::INSERT_SUBVECTOR, MVT::v5f32, Custom);
  setOperationAction(ISD::INSERT_SUBVECTOR, MVT::v6i32, Custom);
  setOperationAction(ISD::INSERT_SUBVECTOR, MVT::v6f32, Custom);
  setOperationAction(ISD::INSERT_SUBVECTOR, MVT::v7i32, Custom);
  setOperationAction(ISD::INSERT_SUBVECTOR, MVT::v7f32, Custom);
  setOperationAction(ISD::INSERT_SUBVECTOR, MVT::v8i32, Custom);
  setOperationAction(ISD::INSERT_SUBVECTOR, MVT::v8f32, Custom);

  // BUFFER/FLAT_ATOMIC_CMP_SWAP on GCN GPUs needs input marshalling,
  // and output demarshalling
  setOperationAction(ISD::ATOMIC_CMP_SWAP, MVT::i32, Custom);
  setOperationAction(ISD::ATOMIC_CMP_SWAP, MVT::i64, Custom);

  // We can't return success/failure, only the old value,
  // let LLVM add the comparison
  setOperationAction(ISD::ATOMIC_CMP_SWAP_WITH_SUCCESS, MVT::i32, Expand);
  setOperationAction(ISD::ATOMIC_CMP_SWAP_WITH_SUCCESS, MVT::i64, Expand);

  if (Subtarget->hasFlatAddressSpace()) {
    setOperationAction(ISD::ADDRSPACECAST, MVT::i32, Custom);
    setOperationAction(ISD::ADDRSPACECAST, MVT::i64, Custom);
  }

  setOperationAction(ISD::BITREVERSE, MVT::i32, Legal);
  setOperationAction(ISD::BITREVERSE, MVT::i64, Legal);

  // FIXME: This should be narrowed to i32, but that only happens if i64 is
  // illegal.
  // FIXME: Should lower sub-i32 bswaps to bit-ops without v_perm_b32.
  setOperationAction(ISD::BSWAP, MVT::i64, Legal);
  setOperationAction(ISD::BSWAP, MVT::i32, Legal);

  // On SI this is s_memtime and s_memrealtime on VI.
  setOperationAction(ISD::READCYCLECOUNTER, MVT::i64, Legal);
  setOperationAction(ISD::TRAP, MVT::Other, Custom);
  setOperationAction(ISD::DEBUGTRAP, MVT::Other, Custom);

  if (Subtarget->has16BitInsts()) {
    setOperationAction(ISD::FPOW, MVT::f16, Promote);
    setOperationAction(ISD::FPOWI, MVT::f16, Promote);
    setOperationAction(ISD::FLOG, MVT::f16, Custom);
    setOperationAction(ISD::FEXP, MVT::f16, Custom);
    setOperationAction(ISD::FLOG10, MVT::f16, Custom);
  }

  if (Subtarget->hasMadMacF32Insts())
    setOperationAction(ISD::FMAD, MVT::f32, Legal);

  if (!Subtarget->hasBFI()) {
    // fcopysign can be done in a single instruction with BFI.
    setOperationAction(ISD::FCOPYSIGN, MVT::f32, Expand);
    setOperationAction(ISD::FCOPYSIGN, MVT::f64, Expand);
  }

  if (!Subtarget->hasBCNT(32))
    setOperationAction(ISD::CTPOP, MVT::i32, Expand);

  if (!Subtarget->hasBCNT(64))
    setOperationAction(ISD::CTPOP, MVT::i64, Expand);

  if (Subtarget->hasFFBH()) {
    setOperationAction(ISD::CTLZ, MVT::i32, Custom);
    setOperationAction(ISD::CTLZ_ZERO_UNDEF, MVT::i32, Custom);
  }

  if (Subtarget->hasFFBL()) {
    setOperationAction(ISD::CTTZ, MVT::i32, Custom);
    setOperationAction(ISD::CTTZ_ZERO_UNDEF, MVT::i32, Custom);
  }

  // We only really have 32-bit BFE instructions (and 16-bit on VI).
  //
  // On SI+ there are 64-bit BFEs, but they are scalar only and there isn't any
  // effort to match them now. We want this to be false for i64 cases when the
  // extraction isn't restricted to the upper or lower half. Ideally we would
  // have some pass reduce 64-bit extracts to 32-bit if possible. Extracts that
  // span the midpoint are probably relatively rare, so don't worry about them
  // for now.
  if (Subtarget->hasBFE())
    setHasExtractBitsInsn(true);

  // Clamp modifier on add/sub
  if (Subtarget->hasIntClamp()) {
    setOperationAction(ISD::UADDSAT, MVT::i32, Legal);
    setOperationAction(ISD::USUBSAT, MVT::i32, Legal);
  }

  if (Subtarget->hasAddNoCarry()) {
    setOperationAction(ISD::SADDSAT, MVT::i16, Legal);
    setOperationAction(ISD::SSUBSAT, MVT::i16, Legal);
    setOperationAction(ISD::SADDSAT, MVT::i32, Legal);
    setOperationAction(ISD::SSUBSAT, MVT::i32, Legal);
  }

  setOperationAction(ISD::FMINNUM, MVT::f32, Custom);
  setOperationAction(ISD::FMAXNUM, MVT::f32, Custom);
  setOperationAction(ISD::FMINNUM, MVT::f64, Custom);
  setOperationAction(ISD::FMAXNUM, MVT::f64, Custom);


  // These are really only legal for ieee_mode functions. We should be avoiding
  // them for functions that don't have ieee_mode enabled, so just say they are
  // legal.
  setOperationAction(ISD::FMINNUM_IEEE, MVT::f32, Legal);
  setOperationAction(ISD::FMAXNUM_IEEE, MVT::f32, Legal);
  setOperationAction(ISD::FMINNUM_IEEE, MVT::f64, Legal);
  setOperationAction(ISD::FMAXNUM_IEEE, MVT::f64, Legal);


  if (Subtarget->haveRoundOpsF64()) {
    setOperationAction(ISD::FTRUNC, MVT::f64, Legal);
    setOperationAction(ISD::FCEIL, MVT::f64, Legal);
    setOperationAction(ISD::FRINT, MVT::f64, Legal);
  } else {
    setOperationAction(ISD::FCEIL, MVT::f64, Custom);
    setOperationAction(ISD::FTRUNC, MVT::f64, Custom);
    setOperationAction(ISD::FRINT, MVT::f64, Custom);
    setOperationAction(ISD::FFLOOR, MVT::f64, Custom);
  }

  setOperationAction(ISD::FFLOOR, MVT::f64, Legal);

  setOperationAction(ISD::FSIN, MVT::f32, Custom);
  setOperationAction(ISD::FCOS, MVT::f32, Custom);
  setOperationAction(ISD::FDIV, MVT::f32, Custom);
  setOperationAction(ISD::FDIV, MVT::f64, Custom);

  if (Subtarget->has16BitInsts()) {
    setOperationAction(ISD::Constant, MVT::i16, Legal);

    setOperationAction(ISD::SMIN, MVT::i16, Legal);
    setOperationAction(ISD::SMAX, MVT::i16, Legal);

    setOperationAction(ISD::UMIN, MVT::i16, Legal);
    setOperationAction(ISD::UMAX, MVT::i16, Legal);

    setOperationAction(ISD::SIGN_EXTEND, MVT::i16, Promote);
    AddPromotedToType(ISD::SIGN_EXTEND, MVT::i16, MVT::i32);

    setOperationAction(ISD::ROTR, MVT::i16, Expand);
    setOperationAction(ISD::ROTL, MVT::i16, Expand);

    setOperationAction(ISD::SDIV, MVT::i16, Promote);
    setOperationAction(ISD::UDIV, MVT::i16, Promote);
    setOperationAction(ISD::SREM, MVT::i16, Promote);
    setOperationAction(ISD::UREM, MVT::i16, Promote);
    setOperationAction(ISD::UADDSAT, MVT::i16, Legal);
    setOperationAction(ISD::USUBSAT, MVT::i16, Legal);

    setOperationAction(ISD::BITREVERSE, MVT::i16, Promote);

    setOperationAction(ISD::CTTZ, MVT::i16, Promote);
    setOperationAction(ISD::CTTZ_ZERO_UNDEF, MVT::i16, Promote);
    setOperationAction(ISD::CTLZ, MVT::i16, Promote);
    setOperationAction(ISD::CTLZ_ZERO_UNDEF, MVT::i16, Promote);
    setOperationAction(ISD::CTPOP, MVT::i16, Promote);

    setOperationAction(ISD::SELECT_CC, MVT::i16, Expand);

    setOperationAction(ISD::BR_CC, MVT::i16, Expand);

    setOperationAction(ISD::LOAD, MVT::i16, Custom);

    setTruncStoreAction(MVT::i64, MVT::i16, Expand);

    setOperationAction(ISD::FP16_TO_FP, MVT::i16, Promote);
    AddPromotedToType(ISD::FP16_TO_FP, MVT::i16, MVT::i32);
    setOperationAction(ISD::FP_TO_FP16, MVT::i16, Promote);
    AddPromotedToType(ISD::FP_TO_FP16, MVT::i16, MVT::i32);

    setOperationAction(ISD::FP_TO_SINT, MVT::i16, Custom);
    setOperationAction(ISD::FP_TO_UINT, MVT::i16, Custom);

    // F16 - Constant Actions.
    setOperationAction(ISD::ConstantFP, MVT::f16, Legal);

    // F16 - Load/Store Actions.
    setOperationAction(ISD::LOAD, MVT::f16, Promote);
    AddPromotedToType(ISD::LOAD, MVT::f16, MVT::i16);
    setOperationAction(ISD::STORE, MVT::f16, Promote);
    AddPromotedToType(ISD::STORE, MVT::f16, MVT::i16);

    // F16 - VOP1 Actions.
    setOperationAction(ISD::FP_ROUND, MVT::f16, Custom);
    setOperationAction(ISD::FCOS, MVT::f16, Custom);
    setOperationAction(ISD::FSIN, MVT::f16, Custom);

    setOperationAction(ISD::SINT_TO_FP, MVT::i16, Custom);
    setOperationAction(ISD::UINT_TO_FP, MVT::i16, Custom);

    setOperationAction(ISD::FP_TO_SINT, MVT::f16, Promote);
    setOperationAction(ISD::FP_TO_UINT, MVT::f16, Promote);
    setOperationAction(ISD::SINT_TO_FP, MVT::f16, Promote);
    setOperationAction(ISD::UINT_TO_FP, MVT::f16, Promote);
    setOperationAction(ISD::FROUND, MVT::f16, Custom);

    // F16 - VOP2 Actions.
    setOperationAction(ISD::BR_CC, MVT::f16, Expand);
    setOperationAction(ISD::SELECT_CC, MVT::f16, Expand);

    setOperationAction(ISD::FDIV, MVT::f16, Custom);

    // F16 - VOP3 Actions.
    setOperationAction(ISD::FMA, MVT::f16, Legal);
    if (STI.hasMadF16())
      setOperationAction(ISD::FMAD, MVT::f16, Legal);

    for (MVT VT : {MVT::v2i16, MVT::v2f16, MVT::v4i16, MVT::v4f16, MVT::v8i16,
                   MVT::v8f16}) {
      for (unsigned Op = 0; Op < ISD::BUILTIN_OP_END; ++Op) {
        switch (Op) {
        case ISD::LOAD:
        case ISD::STORE:
        case ISD::BUILD_VECTOR:
        case ISD::BITCAST:
        case ISD::EXTRACT_VECTOR_ELT:
        case ISD::INSERT_VECTOR_ELT:
        case ISD::INSERT_SUBVECTOR:
        case ISD::EXTRACT_SUBVECTOR:
        case ISD::SCALAR_TO_VECTOR:
          break;
        case ISD::CONCAT_VECTORS:
          setOperationAction(Op, VT, Custom);
          break;
        default:
          setOperationAction(Op, VT, Expand);
          break;
        }
      }
    }

    // v_perm_b32 can handle either of these.
    setOperationAction(ISD::BSWAP, MVT::i16, Legal);
    setOperationAction(ISD::BSWAP, MVT::v2i16, Legal);
    setOperationAction(ISD::BSWAP, MVT::v4i16, Custom);

    // XXX - Do these do anything? Vector constants turn into build_vector.
    setOperationAction(ISD::Constant, MVT::v2i16, Legal);
    setOperationAction(ISD::ConstantFP, MVT::v2f16, Legal);

    setOperationAction(ISD::UNDEF, MVT::v2i16, Legal);
    setOperationAction(ISD::UNDEF, MVT::v2f16, Legal);

    setOperationAction(ISD::STORE, MVT::v2i16, Promote);
    AddPromotedToType(ISD::STORE, MVT::v2i16, MVT::i32);
    setOperationAction(ISD::STORE, MVT::v2f16, Promote);
    AddPromotedToType(ISD::STORE, MVT::v2f16, MVT::i32);

    setOperationAction(ISD::LOAD, MVT::v2i16, Promote);
    AddPromotedToType(ISD::LOAD, MVT::v2i16, MVT::i32);
    setOperationAction(ISD::LOAD, MVT::v2f16, Promote);
    AddPromotedToType(ISD::LOAD, MVT::v2f16, MVT::i32);

    setOperationAction(ISD::AND, MVT::v2i16, Promote);
    AddPromotedToType(ISD::AND, MVT::v2i16, MVT::i32);
    setOperationAction(ISD::OR, MVT::v2i16, Promote);
    AddPromotedToType(ISD::OR, MVT::v2i16, MVT::i32);
    setOperationAction(ISD::XOR, MVT::v2i16, Promote);
    AddPromotedToType(ISD::XOR, MVT::v2i16, MVT::i32);

    setOperationAction(ISD::LOAD, MVT::v4i16, Promote);
    AddPromotedToType(ISD::LOAD, MVT::v4i16, MVT::v2i32);
    setOperationAction(ISD::LOAD, MVT::v4f16, Promote);
    AddPromotedToType(ISD::LOAD, MVT::v4f16, MVT::v2i32);

    setOperationAction(ISD::STORE, MVT::v4i16, Promote);
    AddPromotedToType(ISD::STORE, MVT::v4i16, MVT::v2i32);
    setOperationAction(ISD::STORE, MVT::v4f16, Promote);
    AddPromotedToType(ISD::STORE, MVT::v4f16, MVT::v2i32);

    setOperationAction(ISD::LOAD, MVT::v8i16, Promote);
    AddPromotedToType(ISD::LOAD, MVT::v8i16, MVT::v4i32);
    setOperationAction(ISD::LOAD, MVT::v8f16, Promote);
    AddPromotedToType(ISD::LOAD, MVT::v8f16, MVT::v4i32);

    setOperationAction(ISD::STORE, MVT::v4i16, Promote);
    AddPromotedToType(ISD::STORE, MVT::v4i16, MVT::v2i32);
    setOperationAction(ISD::STORE, MVT::v4f16, Promote);
    AddPromotedToType(ISD::STORE, MVT::v4f16, MVT::v2i32);

    setOperationAction(ISD::STORE, MVT::v8i16, Promote);
    AddPromotedToType(ISD::STORE, MVT::v8i16, MVT::v4i32);
    setOperationAction(ISD::STORE, MVT::v8f16, Promote);
    AddPromotedToType(ISD::STORE, MVT::v8f16, MVT::v4i32);

    setOperationAction(ISD::ANY_EXTEND, MVT::v2i32, Expand);
    setOperationAction(ISD::ZERO_EXTEND, MVT::v2i32, Expand);
    setOperationAction(ISD::SIGN_EXTEND, MVT::v2i32, Expand);
    setOperationAction(ISD::FP_EXTEND, MVT::v2f32, Expand);

    setOperationAction(ISD::ANY_EXTEND, MVT::v4i32, Expand);
    setOperationAction(ISD::ZERO_EXTEND, MVT::v4i32, Expand);
    setOperationAction(ISD::SIGN_EXTEND, MVT::v4i32, Expand);

    setOperationAction(ISD::ANY_EXTEND, MVT::v8i32, Expand);
    setOperationAction(ISD::ZERO_EXTEND, MVT::v8i32, Expand);
    setOperationAction(ISD::SIGN_EXTEND, MVT::v8i32, Expand);

    if (!Subtarget->hasVOP3PInsts()) {
      setOperationAction(ISD::BUILD_VECTOR, MVT::v2i16, Custom);
      setOperationAction(ISD::BUILD_VECTOR, MVT::v2f16, Custom);
    }

    setOperationAction(ISD::FNEG, MVT::v2f16, Legal);
    // This isn't really legal, but this avoids the legalizer unrolling it (and
    // allows matching fneg (fabs x) patterns)
    setOperationAction(ISD::FABS, MVT::v2f16, Legal);

    setOperationAction(ISD::FMAXNUM, MVT::f16, Custom);
    setOperationAction(ISD::FMINNUM, MVT::f16, Custom);
    setOperationAction(ISD::FMAXNUM_IEEE, MVT::f16, Legal);
    setOperationAction(ISD::FMINNUM_IEEE, MVT::f16, Legal);

    setOperationAction(ISD::FMINNUM_IEEE, MVT::v4f16, Custom);
    setOperationAction(ISD::FMAXNUM_IEEE, MVT::v4f16, Custom);
    setOperationAction(ISD::FMINNUM_IEEE, MVT::v8f16, Custom);
    setOperationAction(ISD::FMAXNUM_IEEE, MVT::v8f16, Custom);

    setOperationAction(ISD::FMINNUM, MVT::v4f16, Expand);
    setOperationAction(ISD::FMAXNUM, MVT::v4f16, Expand);
    setOperationAction(ISD::FMINNUM, MVT::v8f16, Expand);
    setOperationAction(ISD::FMAXNUM, MVT::v8f16, Expand);

    for (MVT Vec16 : { MVT::v8i16, MVT::v8f16 }) {
      setOperationAction(ISD::BUILD_VECTOR, Vec16, Custom);
      setOperationAction(ISD::EXTRACT_VECTOR_ELT, Vec16, Custom);
      setOperationAction(ISD::INSERT_VECTOR_ELT, Vec16, Expand);
      setOperationAction(ISD::SCALAR_TO_VECTOR, Vec16, Expand);
    }
  }

  if (Subtarget->hasVOP3PInsts()) {
    setOperationAction(ISD::ADD, MVT::v2i16, Legal);
    setOperationAction(ISD::SUB, MVT::v2i16, Legal);
    setOperationAction(ISD::MUL, MVT::v2i16, Legal);
    setOperationAction(ISD::SHL, MVT::v2i16, Legal);
    setOperationAction(ISD::SRL, MVT::v2i16, Legal);
    setOperationAction(ISD::SRA, MVT::v2i16, Legal);
    setOperationAction(ISD::SMIN, MVT::v2i16, Legal);
    setOperationAction(ISD::UMIN, MVT::v2i16, Legal);
    setOperationAction(ISD::SMAX, MVT::v2i16, Legal);
    setOperationAction(ISD::UMAX, MVT::v2i16, Legal);

    setOperationAction(ISD::UADDSAT, MVT::v2i16, Legal);
    setOperationAction(ISD::USUBSAT, MVT::v2i16, Legal);
    setOperationAction(ISD::SADDSAT, MVT::v2i16, Legal);
    setOperationAction(ISD::SSUBSAT, MVT::v2i16, Legal);

    setOperationAction(ISD::FADD, MVT::v2f16, Legal);
    setOperationAction(ISD::FMUL, MVT::v2f16, Legal);
    setOperationAction(ISD::FMA, MVT::v2f16, Legal);

    setOperationAction(ISD::FMINNUM_IEEE, MVT::v2f16, Legal);
    setOperationAction(ISD::FMAXNUM_IEEE, MVT::v2f16, Legal);

    setOperationAction(ISD::FCANONICALIZE, MVT::v2f16, Legal);

    setOperationAction(ISD::EXTRACT_VECTOR_ELT, MVT::v2i16, Custom);
    setOperationAction(ISD::EXTRACT_VECTOR_ELT, MVT::v2f16, Custom);

    setOperationAction(ISD::VECTOR_SHUFFLE, MVT::v4f16, Custom);
    setOperationAction(ISD::VECTOR_SHUFFLE, MVT::v4i16, Custom);
    setOperationAction(ISD::VECTOR_SHUFFLE, MVT::v8f16, Custom);
    setOperationAction(ISD::VECTOR_SHUFFLE, MVT::v8i16, Custom);

    for (MVT VT : { MVT::v4i16, MVT::v8i16 }) {
      // Split vector operations.
      setOperationAction(ISD::SHL, VT, Custom);
      setOperationAction(ISD::SRA, VT, Custom);
      setOperationAction(ISD::SRL, VT, Custom);
      setOperationAction(ISD::ADD, VT, Custom);
      setOperationAction(ISD::SUB, VT, Custom);
      setOperationAction(ISD::MUL, VT, Custom);

      setOperationAction(ISD::SMIN, VT, Custom);
      setOperationAction(ISD::SMAX, VT, Custom);
      setOperationAction(ISD::UMIN, VT, Custom);
      setOperationAction(ISD::UMAX, VT, Custom);

      setOperationAction(ISD::UADDSAT, VT, Custom);
      setOperationAction(ISD::SADDSAT, VT, Custom);
      setOperationAction(ISD::USUBSAT, VT, Custom);
      setOperationAction(ISD::SSUBSAT, VT, Custom);
    }

    for (MVT VT : { MVT::v4f16, MVT::v8f16 }) {
      // Split vector operations.
      setOperationAction(ISD::FADD, VT, Custom);
      setOperationAction(ISD::FMUL, VT, Custom);
      setOperationAction(ISD::FMA, VT, Custom);
      setOperationAction(ISD::FCANONICALIZE, VT, Custom);
    }

    setOperationAction(ISD::FMAXNUM, MVT::v2f16, Custom);
    setOperationAction(ISD::FMINNUM, MVT::v2f16, Custom);

    setOperationAction(ISD::FMINNUM, MVT::v4f16, Custom);
    setOperationAction(ISD::FMAXNUM, MVT::v4f16, Custom);

    setOperationAction(ISD::FEXP, MVT::v2f16, Custom);
    setOperationAction(ISD::SELECT, MVT::v4i16, Custom);
    setOperationAction(ISD::SELECT, MVT::v4f16, Custom);

    if (Subtarget->hasPackedFP32Ops()) {
      setOperationAction(ISD::FADD, MVT::v2f32, Legal);
      setOperationAction(ISD::FMUL, MVT::v2f32, Legal);
      setOperationAction(ISD::FMA,  MVT::v2f32, Legal);
      setOperationAction(ISD::FNEG, MVT::v2f32, Legal);

      for (MVT VT : { MVT::v4f32, MVT::v8f32, MVT::v16f32, MVT::v32f32 }) {
        setOperationAction(ISD::FADD, VT, Custom);
        setOperationAction(ISD::FMUL, VT, Custom);
        setOperationAction(ISD::FMA, VT, Custom);
      }
    }
  }

  setOperationAction(ISD::FNEG, MVT::v4f16, Custom);
  setOperationAction(ISD::FABS, MVT::v4f16, Custom);

  if (Subtarget->has16BitInsts()) {
    setOperationAction(ISD::SELECT, MVT::v2i16, Promote);
    AddPromotedToType(ISD::SELECT, MVT::v2i16, MVT::i32);
    setOperationAction(ISD::SELECT, MVT::v2f16, Promote);
    AddPromotedToType(ISD::SELECT, MVT::v2f16, MVT::i32);
  } else {
    // Legalization hack.
    setOperationAction(ISD::SELECT, MVT::v2i16, Custom);
    setOperationAction(ISD::SELECT, MVT::v2f16, Custom);

    setOperationAction(ISD::FNEG, MVT::v2f16, Custom);
    setOperationAction(ISD::FABS, MVT::v2f16, Custom);
  }

  for (MVT VT : { MVT::v4i16, MVT::v4f16, MVT::v2i8, MVT::v4i8, MVT::v8i8,
                  MVT::v8i16, MVT::v8f16 }) {
    setOperationAction(ISD::SELECT, VT, Custom);
  }

  setOperationAction(ISD::SMULO, MVT::i64, Custom);
  setOperationAction(ISD::UMULO, MVT::i64, Custom);

  if (Subtarget->hasMad64_32()) {
    setOperationAction(ISD::SMUL_LOHI, MVT::i32, Custom);
    setOperationAction(ISD::UMUL_LOHI, MVT::i32, Custom);
  }

  setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::Other, Custom);
  setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::f32, Custom);
  setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::v4f32, Custom);
  setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::i16, Custom);
  setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::f16, Custom);
  setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::v2i16, Custom);
  setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::v2f16, Custom);

  setOperationAction(ISD::INTRINSIC_W_CHAIN, MVT::v2f16, Custom);
  setOperationAction(ISD::INTRINSIC_W_CHAIN, MVT::v2i16, Custom);
  setOperationAction(ISD::INTRINSIC_W_CHAIN, MVT::v3f16, Custom);
  setOperationAction(ISD::INTRINSIC_W_CHAIN, MVT::v3i16, Custom);
  setOperationAction(ISD::INTRINSIC_W_CHAIN, MVT::v4f16, Custom);
  setOperationAction(ISD::INTRINSIC_W_CHAIN, MVT::v4i16, Custom);
  setOperationAction(ISD::INTRINSIC_W_CHAIN, MVT::v8f16, Custom);
  setOperationAction(ISD::INTRINSIC_W_CHAIN, MVT::Other, Custom);
  setOperationAction(ISD::INTRINSIC_W_CHAIN, MVT::f16, Custom);
  setOperationAction(ISD::INTRINSIC_W_CHAIN, MVT::i16, Custom);
  setOperationAction(ISD::INTRINSIC_W_CHAIN, MVT::i8, Custom);

  setOperationAction(ISD::INTRINSIC_VOID, MVT::Other, Custom);
  setOperationAction(ISD::INTRINSIC_VOID, MVT::v2i16, Custom);
  setOperationAction(ISD::INTRINSIC_VOID, MVT::v2f16, Custom);
  setOperationAction(ISD::INTRINSIC_VOID, MVT::v3i16, Custom);
  setOperationAction(ISD::INTRINSIC_VOID, MVT::v3f16, Custom);
  setOperationAction(ISD::INTRINSIC_VOID, MVT::v4f16, Custom);
  setOperationAction(ISD::INTRINSIC_VOID, MVT::v4i16, Custom);
  setOperationAction(ISD::INTRINSIC_VOID, MVT::f16, Custom);
  setOperationAction(ISD::INTRINSIC_VOID, MVT::i16, Custom);
  setOperationAction(ISD::INTRINSIC_VOID, MVT::i8, Custom);

  setTargetDAGCombine(ISD::ADD);
  setTargetDAGCombine(ISD::ADDCARRY);
  setTargetDAGCombine(ISD::SUB);
  setTargetDAGCombine(ISD::SUBCARRY);
  setTargetDAGCombine(ISD::FADD);
  setTargetDAGCombine(ISD::FSUB);
  setTargetDAGCombine(ISD::FMINNUM);
  setTargetDAGCombine(ISD::FMAXNUM);
  setTargetDAGCombine(ISD::FMINNUM_IEEE);
  setTargetDAGCombine(ISD::FMAXNUM_IEEE);
  setTargetDAGCombine(ISD::FMA);
  setTargetDAGCombine(ISD::SMIN);
  setTargetDAGCombine(ISD::SMAX);
  setTargetDAGCombine(ISD::UMIN);
  setTargetDAGCombine(ISD::UMAX);
  setTargetDAGCombine(ISD::SETCC);
  setTargetDAGCombine(ISD::AND);
  setTargetDAGCombine(ISD::OR);
  setTargetDAGCombine(ISD::XOR);
  setTargetDAGCombine(ISD::SINT_TO_FP);
  setTargetDAGCombine(ISD::UINT_TO_FP);
  setTargetDAGCombine(ISD::FCANONICALIZE);
  setTargetDAGCombine(ISD::SCALAR_TO_VECTOR);
  setTargetDAGCombine(ISD::ZERO_EXTEND);
  setTargetDAGCombine(ISD::SIGN_EXTEND_INREG);
  setTargetDAGCombine(ISD::EXTRACT_VECTOR_ELT);
  setTargetDAGCombine(ISD::INSERT_VECTOR_ELT);

  // All memory operations. Some folding on the pointer operand is done to help
  // matching the constant offsets in the addressing modes.
  setTargetDAGCombine(ISD::LOAD);
  setTargetDAGCombine(ISD::STORE);
  setTargetDAGCombine(ISD::ATOMIC_LOAD);
  setTargetDAGCombine(ISD::ATOMIC_STORE);
  setTargetDAGCombine(ISD::ATOMIC_CMP_SWAP);
  setTargetDAGCombine(ISD::ATOMIC_CMP_SWAP_WITH_SUCCESS);
  setTargetDAGCombine(ISD::ATOMIC_SWAP);
  setTargetDAGCombine(ISD::ATOMIC_LOAD_ADD);
  setTargetDAGCombine(ISD::ATOMIC_LOAD_SUB);
  setTargetDAGCombine(ISD::ATOMIC_LOAD_AND);
  setTargetDAGCombine(ISD::ATOMIC_LOAD_OR);
  setTargetDAGCombine(ISD::ATOMIC_LOAD_XOR);
  setTargetDAGCombine(ISD::ATOMIC_LOAD_NAND);
  setTargetDAGCombine(ISD::ATOMIC_LOAD_MIN);
  setTargetDAGCombine(ISD::ATOMIC_LOAD_MAX);
  setTargetDAGCombine(ISD::ATOMIC_LOAD_UMIN);
  setTargetDAGCombine(ISD::ATOMIC_LOAD_UMAX);
  setTargetDAGCombine(ISD::ATOMIC_LOAD_FADD);
  setTargetDAGCombine(ISD::INTRINSIC_VOID);
  setTargetDAGCombine(ISD::INTRINSIC_W_CHAIN);

  // FIXME: In other contexts we pretend this is a per-function property.
  setStackPointerRegisterToSaveRestore(AMDGPU::SGPR32);

  setSchedulingPreference(Sched::RegPressure);
}

const GCNSubtarget *SITargetLowering::getSubtarget() const {
  return Subtarget;
}

//===----------------------------------------------------------------------===//
// TargetLowering queries
//===----------------------------------------------------------------------===//

// v_mad_mix* support a conversion from f16 to f32.
//
// There is only one special case when denormals are enabled we don't currently,
// where this is OK to use.
bool SITargetLowering::isFPExtFoldable(const SelectionDAG &DAG, unsigned Opcode,
                                       EVT DestVT, EVT SrcVT) const {
  return ((Opcode == ISD::FMAD && Subtarget->hasMadMixInsts()) ||
          (Opcode == ISD::FMA && Subtarget->hasFmaMixInsts())) &&
    DestVT.getScalarType() == MVT::f32 &&
    SrcVT.getScalarType() == MVT::f16 &&
    // TODO: This probably only requires no input flushing?
    !hasFP32Denormals(DAG.getMachineFunction());
}

bool SITargetLowering::isFPExtFoldable(const MachineInstr &MI, unsigned Opcode,
                                       LLT DestTy, LLT SrcTy) const {
  return ((Opcode == TargetOpcode::G_FMAD && Subtarget->hasMadMixInsts()) ||
          (Opcode == TargetOpcode::G_FMA && Subtarget->hasFmaMixInsts())) &&
         DestTy.getScalarSizeInBits() == 32 &&
         SrcTy.getScalarSizeInBits() == 16 &&
         // TODO: This probably only requires no input flushing?
         !hasFP32Denormals(*MI.getMF());
}

bool SITargetLowering::isShuffleMaskLegal(ArrayRef<int>, EVT) const {
  // SI has some legal vector types, but no legal vector operations. Say no
  // shuffles are legal in order to prefer scalarizing some vector operations.
  return false;
}

MVT SITargetLowering::getRegisterTypeForCallingConv(LLVMContext &Context,
                                                    CallingConv::ID CC,
                                                    EVT VT) const {
  if (CC == CallingConv::AMDGPU_KERNEL)
    return TargetLowering::getRegisterTypeForCallingConv(Context, CC, VT);

  if (VT.isVector()) {
    EVT ScalarVT = VT.getScalarType();
    unsigned Size = ScalarVT.getSizeInBits();
    if (Size == 16) {
      if (Subtarget->has16BitInsts())
        return VT.isInteger() ? MVT::v2i16 : MVT::v2f16;
      return VT.isInteger() ? MVT::i32 : MVT::f32;
    }

    if (Size < 16)
      return Subtarget->has16BitInsts() ? MVT::i16 : MVT::i32;
    return Size == 32 ? ScalarVT.getSimpleVT() : MVT::i32;
  }

  if (VT.getSizeInBits() > 32)
    return MVT::i32;

  return TargetLowering::getRegisterTypeForCallingConv(Context, CC, VT);
}

unsigned SITargetLowering::getNumRegistersForCallingConv(LLVMContext &Context,
                                                         CallingConv::ID CC,
                                                         EVT VT) const {
  if (CC == CallingConv::AMDGPU_KERNEL)
    return TargetLowering::getNumRegistersForCallingConv(Context, CC, VT);

  if (VT.isVector()) {
    unsigned NumElts = VT.getVectorNumElements();
    EVT ScalarVT = VT.getScalarType();
    unsigned Size = ScalarVT.getSizeInBits();

    // FIXME: Should probably promote 8-bit vectors to i16.
    if (Size == 16 && Subtarget->has16BitInsts())
      return (NumElts + 1) / 2;

    if (Size <= 32)
      return NumElts;

    if (Size > 32)
      return NumElts * ((Size + 31) / 32);
  } else if (VT.getSizeInBits() > 32)
    return (VT.getSizeInBits() + 31) / 32;

  return TargetLowering::getNumRegistersForCallingConv(Context, CC, VT);
}

unsigned SITargetLowering::getVectorTypeBreakdownForCallingConv(
  LLVMContext &Context, CallingConv::ID CC,
  EVT VT, EVT &IntermediateVT,
  unsigned &NumIntermediates, MVT &RegisterVT) const {
  if (CC != CallingConv::AMDGPU_KERNEL && VT.isVector()) {
    unsigned NumElts = VT.getVectorNumElements();
    EVT ScalarVT = VT.getScalarType();
    unsigned Size = ScalarVT.getSizeInBits();
    // FIXME: We should fix the ABI to be the same on targets without 16-bit
    // support, but unless we can properly handle 3-vectors, it will be still be
    // inconsistent.
    if (Size == 16 && Subtarget->has16BitInsts()) {
      RegisterVT = VT.isInteger() ? MVT::v2i16 : MVT::v2f16;
      IntermediateVT = RegisterVT;
      NumIntermediates = (NumElts + 1) / 2;
      return NumIntermediates;
    }

    if (Size == 32) {
      RegisterVT = ScalarVT.getSimpleVT();
      IntermediateVT = RegisterVT;
      NumIntermediates = NumElts;
      return NumIntermediates;
    }

    if (Size < 16 && Subtarget->has16BitInsts()) {
      // FIXME: Should probably form v2i16 pieces
      RegisterVT = MVT::i16;
      IntermediateVT = ScalarVT;
      NumIntermediates = NumElts;
      return NumIntermediates;
    }


    if (Size != 16 && Size <= 32) {
      RegisterVT = MVT::i32;
      IntermediateVT = ScalarVT;
      NumIntermediates = NumElts;
      return NumIntermediates;
    }

    if (Size > 32) {
      RegisterVT = MVT::i32;
      IntermediateVT = RegisterVT;
      NumIntermediates = NumElts * ((Size + 31) / 32);
      return NumIntermediates;
    }
  }

  return TargetLowering::getVectorTypeBreakdownForCallingConv(
    Context, CC, VT, IntermediateVT, NumIntermediates, RegisterVT);
}

static EVT memVTFromImageData(Type *Ty, unsigned DMaskLanes) {
  assert(DMaskLanes != 0);

  if (auto *VT = dyn_cast<FixedVectorType>(Ty)) {
    unsigned NumElts = std::min(DMaskLanes, VT->getNumElements());
    return EVT::getVectorVT(Ty->getContext(),
                            EVT::getEVT(VT->getElementType()),
                            NumElts);
  }

  return EVT::getEVT(Ty);
}

// Peek through TFE struct returns to only use the data size.
static EVT memVTFromImageReturn(Type *Ty, unsigned DMaskLanes) {
  auto *ST = dyn_cast<StructType>(Ty);
  if (!ST)
    return memVTFromImageData(Ty, DMaskLanes);

  // Some intrinsics return an aggregate type - special case to work out the
  // correct memVT.
  //
  // Only limited forms of aggregate type currently expected.
  if (ST->getNumContainedTypes() != 2 ||
      !ST->getContainedType(1)->isIntegerTy(32))
    return EVT();
  return memVTFromImageData(ST->getContainedType(0), DMaskLanes);
}

bool SITargetLowering::getTgtMemIntrinsic(IntrinsicInfo &Info,
                                          const CallInst &CI,
                                          MachineFunction &MF,
                                          unsigned IntrID) const {
  if (const AMDGPU::RsrcIntrinsic *RsrcIntr =
          AMDGPU::lookupRsrcIntrinsic(IntrID)) {
    AttributeList Attr = Intrinsic::getAttributes(CI.getContext(),
                                                  (Intrinsic::ID)IntrID);
    if (Attr.hasFnAttr(Attribute::ReadNone))
      return false;

    SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();

    if (RsrcIntr->IsImage) {
      Info.ptrVal =
          MFI->getImagePSV(*MF.getSubtarget<GCNSubtarget>().getInstrInfo());
      Info.align.reset();
    } else {
      Info.ptrVal =
          MFI->getBufferPSV(*MF.getSubtarget<GCNSubtarget>().getInstrInfo());
    }

    Info.flags = MachineMemOperand::MODereferenceable;
    if (Attr.hasFnAttr(Attribute::ReadOnly)) {
      unsigned DMaskLanes = 4;

      if (RsrcIntr->IsImage) {
        const AMDGPU::ImageDimIntrinsicInfo *Intr
          = AMDGPU::getImageDimIntrinsicInfo(IntrID);
        const AMDGPU::MIMGBaseOpcodeInfo *BaseOpcode =
          AMDGPU::getMIMGBaseOpcodeInfo(Intr->BaseOpcode);

        if (!BaseOpcode->Gather4) {
          // If this isn't a gather, we may have excess loaded elements in the
          // IR type. Check the dmask for the real number of elements loaded.
          unsigned DMask
            = cast<ConstantInt>(CI.getArgOperand(0))->getZExtValue();
          DMaskLanes = DMask == 0 ? 1 : countPopulation(DMask);
        }

        Info.memVT = memVTFromImageReturn(CI.getType(), DMaskLanes);
      } else
        Info.memVT = EVT::getEVT(CI.getType());

      // FIXME: What does alignment mean for an image?
      Info.opc = ISD::INTRINSIC_W_CHAIN;
      Info.flags |= MachineMemOperand::MOLoad;
    } else if (Attr.hasFnAttr(Attribute::WriteOnly)) {
      Info.opc = ISD::INTRINSIC_VOID;

      Type *DataTy = CI.getArgOperand(0)->getType();
      if (RsrcIntr->IsImage) {
        unsigned DMask = cast<ConstantInt>(CI.getArgOperand(1))->getZExtValue();
        unsigned DMaskLanes = DMask == 0 ? 1 : countPopulation(DMask);
        Info.memVT = memVTFromImageData(DataTy, DMaskLanes);
      } else
        Info.memVT = EVT::getEVT(DataTy);

      Info.flags |= MachineMemOperand::MOStore;
    } else {
      // Atomic
      Info.opc = CI.getType()->isVoidTy() ? ISD::INTRINSIC_VOID :
                                            ISD::INTRINSIC_W_CHAIN;
      Info.memVT = MVT::getVT(CI.getArgOperand(0)->getType());
      Info.flags = MachineMemOperand::MOLoad |
                   MachineMemOperand::MOStore |
                   MachineMemOperand::MODereferenceable;

      // XXX - Should this be volatile without known ordering?
      Info.flags |= MachineMemOperand::MOVolatile;
    }
    return true;
  }

  switch (IntrID) {
  case Intrinsic::amdgcn_atomic_inc:
  case Intrinsic::amdgcn_atomic_dec:
  case Intrinsic::amdgcn_ds_ordered_add:
  case Intrinsic::amdgcn_ds_ordered_swap:
  case Intrinsic::amdgcn_ds_fadd:
  case Intrinsic::amdgcn_ds_fmin:
  case Intrinsic::amdgcn_ds_fmax: {
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.memVT = MVT::getVT(CI.getType());
    Info.ptrVal = CI.getOperand(0);
    Info.align.reset();
    Info.flags = MachineMemOperand::MOLoad | MachineMemOperand::MOStore;

    const ConstantInt *Vol = cast<ConstantInt>(CI.getOperand(4));
    if (!Vol->isZero())
      Info.flags |= MachineMemOperand::MOVolatile;

    return true;
  }
  case Intrinsic::amdgcn_buffer_atomic_fadd: {
    SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();

    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.memVT = MVT::getVT(CI.getOperand(0)->getType());
    Info.ptrVal =
        MFI->getBufferPSV(*MF.getSubtarget<GCNSubtarget>().getInstrInfo());
    Info.align.reset();
    Info.flags = MachineMemOperand::MOLoad | MachineMemOperand::MOStore;

    const ConstantInt *Vol = dyn_cast<ConstantInt>(CI.getOperand(4));
    if (!Vol || !Vol->isZero())
      Info.flags |= MachineMemOperand::MOVolatile;

    return true;
  }
  case Intrinsic::amdgcn_ds_append:
  case Intrinsic::amdgcn_ds_consume: {
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.memVT = MVT::getVT(CI.getType());
    Info.ptrVal = CI.getOperand(0);
    Info.align.reset();
    Info.flags = MachineMemOperand::MOLoad | MachineMemOperand::MOStore;

    const ConstantInt *Vol = cast<ConstantInt>(CI.getOperand(1));
    if (!Vol->isZero())
      Info.flags |= MachineMemOperand::MOVolatile;

    return true;
  }
  case Intrinsic::amdgcn_global_atomic_csub: {
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.memVT = MVT::getVT(CI.getType());
    Info.ptrVal = CI.getOperand(0);
    Info.align.reset();
    Info.flags = MachineMemOperand::MOLoad |
                 MachineMemOperand::MOStore |
                 MachineMemOperand::MOVolatile;
    return true;
  }
  case Intrinsic::amdgcn_image_bvh_intersect_ray: {
    SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.memVT = MVT::getVT(CI.getType()); // XXX: what is correct VT?
    Info.ptrVal =
        MFI->getImagePSV(*MF.getSubtarget<GCNSubtarget>().getInstrInfo());
    Info.align.reset();
    Info.flags = MachineMemOperand::MOLoad |
                 MachineMemOperand::MODereferenceable;
    return true;
  }
  case Intrinsic::amdgcn_global_atomic_fadd:
  case Intrinsic::amdgcn_global_atomic_fmin:
  case Intrinsic::amdgcn_global_atomic_fmax:
  case Intrinsic::amdgcn_flat_atomic_fadd:
  case Intrinsic::amdgcn_flat_atomic_fmin:
  case Intrinsic::amdgcn_flat_atomic_fmax: {
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.memVT = MVT::getVT(CI.getType());
    Info.ptrVal = CI.getOperand(0);
    Info.align.reset();
    Info.flags = MachineMemOperand::MOLoad |
                 MachineMemOperand::MOStore |
                 MachineMemOperand::MODereferenceable |
                 MachineMemOperand::MOVolatile;
    return true;
  }
  case Intrinsic::amdgcn_ds_gws_init:
  case Intrinsic::amdgcn_ds_gws_barrier:
  case Intrinsic::amdgcn_ds_gws_sema_v:
  case Intrinsic::amdgcn_ds_gws_sema_br:
  case Intrinsic::amdgcn_ds_gws_sema_p:
  case Intrinsic::amdgcn_ds_gws_sema_release_all: {
    Info.opc = ISD::INTRINSIC_VOID;

    SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
    Info.ptrVal =
        MFI->getGWSPSV(*MF.getSubtarget<GCNSubtarget>().getInstrInfo());

    // This is an abstract access, but we need to specify a type and size.
    Info.memVT = MVT::i32;
    Info.size = 4;
    Info.align = Align(4);

    Info.flags = MachineMemOperand::MOStore;
    if (IntrID == Intrinsic::amdgcn_ds_gws_barrier)
      Info.flags = MachineMemOperand::MOLoad;
    return true;
  }
  default:
    return false;
  }
}

bool SITargetLowering::getAddrModeArguments(IntrinsicInst *II,
                                            SmallVectorImpl<Value*> &Ops,
                                            Type *&AccessTy) const {
  switch (II->getIntrinsicID()) {
  case Intrinsic::amdgcn_atomic_inc:
  case Intrinsic::amdgcn_atomic_dec:
  case Intrinsic::amdgcn_ds_ordered_add:
  case Intrinsic::amdgcn_ds_ordered_swap:
  case Intrinsic::amdgcn_ds_append:
  case Intrinsic::amdgcn_ds_consume:
  case Intrinsic::amdgcn_ds_fadd:
  case Intrinsic::amdgcn_ds_fmin:
  case Intrinsic::amdgcn_ds_fmax:
  case Intrinsic::amdgcn_global_atomic_fadd:
  case Intrinsic::amdgcn_flat_atomic_fadd:
  case Intrinsic::amdgcn_flat_atomic_fmin:
  case Intrinsic::amdgcn_flat_atomic_fmax:
  case Intrinsic::amdgcn_global_atomic_csub: {
    Value *Ptr = II->getArgOperand(0);
    AccessTy = II->getType();
    Ops.push_back(Ptr);
    return true;
  }
  default:
    return false;
  }
}

bool SITargetLowering::isLegalFlatAddressingMode(const AddrMode &AM) const {
  if (!Subtarget->hasFlatInstOffsets()) {
    // Flat instructions do not have offsets, and only have the register
    // address.
    return AM.BaseOffs == 0 && AM.Scale == 0;
  }

  return AM.Scale == 0 &&
         (AM.BaseOffs == 0 ||
          Subtarget->getInstrInfo()->isLegalFLATOffset(
              AM.BaseOffs, AMDGPUAS::FLAT_ADDRESS, SIInstrFlags::FLAT));
}

bool SITargetLowering::isLegalGlobalAddressingMode(const AddrMode &AM) const {
  if (Subtarget->hasFlatGlobalInsts())
    return AM.Scale == 0 &&
           (AM.BaseOffs == 0 || Subtarget->getInstrInfo()->isLegalFLATOffset(
                                    AM.BaseOffs, AMDGPUAS::GLOBAL_ADDRESS,
                                    SIInstrFlags::FlatGlobal));

  if (!Subtarget->hasAddr64() || Subtarget->useFlatForGlobal()) {
      // Assume the we will use FLAT for all global memory accesses
      // on VI.
      // FIXME: This assumption is currently wrong.  On VI we still use
      // MUBUF instructions for the r + i addressing mode.  As currently
      // implemented, the MUBUF instructions only work on buffer < 4GB.
      // It may be possible to support > 4GB buffers with MUBUF instructions,
      // by setting the stride value in the resource descriptor which would
      // increase the size limit to (stride * 4GB).  However, this is risky,
      // because it has never been validated.
    return isLegalFlatAddressingMode(AM);
  }

  return isLegalMUBUFAddressingMode(AM);
}

bool SITargetLowering::isLegalMUBUFAddressingMode(const AddrMode &AM) const {
  // MUBUF / MTBUF instructions have a 12-bit unsigned byte offset, and
  // additionally can do r + r + i with addr64. 32-bit has more addressing
  // mode options. Depending on the resource constant, it can also do
  // (i64 r0) + (i32 r1) * (i14 i).
  //
  // Private arrays end up using a scratch buffer most of the time, so also
  // assume those use MUBUF instructions. Scratch loads / stores are currently
  // implemented as mubuf instructions with offen bit set, so slightly
  // different than the normal addr64.
  if (!SIInstrInfo::isLegalMUBUFImmOffset(AM.BaseOffs))
    return false;

  // FIXME: Since we can split immediate into soffset and immediate offset,
  // would it make sense to allow any immediate?

  switch (AM.Scale) {
  case 0: // r + i or just i, depending on HasBaseReg.
    return true;
  case 1:
    return true; // We have r + r or r + i.
  case 2:
    if (AM.HasBaseReg) {
      // Reject 2 * r + r.
      return false;
    }

    // Allow 2 * r as r + r
    // Or  2 * r + i is allowed as r + r + i.
    return true;
  default: // Don't allow n * r
    return false;
  }
}

bool SITargetLowering::isLegalAddressingMode(const DataLayout &DL,
                                             const AddrMode &AM, Type *Ty,
                                             unsigned AS, Instruction *I) const {
  // No global is ever allowed as a base.
  if (AM.BaseGV)
    return false;

  if (AS == AMDGPUAS::GLOBAL_ADDRESS)
    return isLegalGlobalAddressingMode(AM);

  if (AS == AMDGPUAS::CONSTANT_ADDRESS ||
      AS == AMDGPUAS::CONSTANT_ADDRESS_32BIT ||
      AS == AMDGPUAS::BUFFER_FAT_POINTER) {
    // If the offset isn't a multiple of 4, it probably isn't going to be
    // correctly aligned.
    // FIXME: Can we get the real alignment here?
    if (AM.BaseOffs % 4 != 0)
      return isLegalMUBUFAddressingMode(AM);

    // There are no SMRD extloads, so if we have to do a small type access we
    // will use a MUBUF load.
    // FIXME?: We also need to do this if unaligned, but we don't know the
    // alignment here.
    if (Ty->isSized() && DL.getTypeStoreSize(Ty) < 4)
      return isLegalGlobalAddressingMode(AM);

    if (Subtarget->getGeneration() == AMDGPUSubtarget::SOUTHERN_ISLANDS) {
      // SMRD instructions have an 8-bit, dword offset on SI.
      if (!isUInt<8>(AM.BaseOffs / 4))
        return false;
    } else if (Subtarget->getGeneration() == AMDGPUSubtarget::SEA_ISLANDS) {
      // On CI+, this can also be a 32-bit literal constant offset. If it fits
      // in 8-bits, it can use a smaller encoding.
      if (!isUInt<32>(AM.BaseOffs / 4))
        return false;
    } else if (Subtarget->getGeneration() >= AMDGPUSubtarget::VOLCANIC_ISLANDS) {
      // On VI, these use the SMEM format and the offset is 20-bit in bytes.
      if (!isUInt<20>(AM.BaseOffs))
        return false;
    } else
      llvm_unreachable("unhandled generation");

    if (AM.Scale == 0) // r + i or just i, depending on HasBaseReg.
      return true;

    if (AM.Scale == 1 && AM.HasBaseReg)
      return true;

    return false;

  } else if (AS == AMDGPUAS::PRIVATE_ADDRESS) {
    return isLegalMUBUFAddressingMode(AM);
  } else if (AS == AMDGPUAS::LOCAL_ADDRESS ||
             AS == AMDGPUAS::REGION_ADDRESS) {
    // Basic, single offset DS instructions allow a 16-bit unsigned immediate
    // field.
    // XXX - If doing a 4-byte aligned 8-byte type access, we effectively have
    // an 8-bit dword offset but we don't know the alignment here.
    if (!isUInt<16>(AM.BaseOffs))
      return false;

    if (AM.Scale == 0) // r + i or just i, depending on HasBaseReg.
      return true;

    if (AM.Scale == 1 && AM.HasBaseReg)
      return true;

    return false;
  } else if (AS == AMDGPUAS::FLAT_ADDRESS ||
             AS == AMDGPUAS::UNKNOWN_ADDRESS_SPACE) {
    // For an unknown address space, this usually means that this is for some
    // reason being used for pure arithmetic, and not based on some addressing
    // computation. We don't have instructions that compute pointers with any
    // addressing modes, so treat them as having no offset like flat
    // instructions.
    return isLegalFlatAddressingMode(AM);
  }

  // Assume a user alias of global for unknown address spaces.
  return isLegalGlobalAddressingMode(AM);
}

bool SITargetLowering::canMergeStoresTo(unsigned AS, EVT MemVT,
                                        const MachineFunction &MF) const {
  if (AS == AMDGPUAS::GLOBAL_ADDRESS || AS == AMDGPUAS::FLAT_ADDRESS) {
    return (MemVT.getSizeInBits() <= 4 * 32);
  } else if (AS == AMDGPUAS::PRIVATE_ADDRESS) {
    unsigned MaxPrivateBits = 8 * getSubtarget()->getMaxPrivateElementSize();
    return (MemVT.getSizeInBits() <= MaxPrivateBits);
  } else if (AS == AMDGPUAS::LOCAL_ADDRESS || AS == AMDGPUAS::REGION_ADDRESS) {
    return (MemVT.getSizeInBits() <= 2 * 32);
  }
  return true;
}

bool SITargetLowering::allowsMisalignedMemoryAccessesImpl(
    unsigned Size, unsigned AddrSpace, Align Alignment,
    MachineMemOperand::Flags Flags, bool *IsFast) const {
  if (IsFast)
    *IsFast = false;

  if (AddrSpace == AMDGPUAS::LOCAL_ADDRESS ||
      AddrSpace == AMDGPUAS::REGION_ADDRESS) {
    // Check if alignment requirements for ds_read/write instructions are
    // disabled.
    if (Subtarget->hasUnalignedDSAccessEnabled() &&
        !Subtarget->hasLDSMisalignedBug()) {
      if (IsFast)
        *IsFast = Alignment != Align(2);
      return true;
    }

    // Either, the alignment requirements are "enabled", or there is an
    // unaligned LDS access related hardware bug though alignment requirements
    // are "disabled". In either case, we need to check for proper alignment
    // requirements.
    //
    if (Size == 64) {
      // 8 byte accessing via ds_read/write_b64 require 8-byte alignment, but we
      // can do a 4 byte aligned, 8 byte access in a single operation using
      // ds_read2/write2_b32 with adjacent offsets.
      bool AlignedBy4 = Alignment >= Align(4);
      if (IsFast)
        *IsFast = AlignedBy4;

      return AlignedBy4;
    }
    if (Size == 96) {
      // 12 byte accessing via ds_read/write_b96 require 16-byte alignment on
      // gfx8 and older.
      bool AlignedBy16 = Alignment >= Align(16);
      if (IsFast)
        *IsFast = AlignedBy16;

      return AlignedBy16;
    }
    if (Size == 128) {
      // 16 byte accessing via ds_read/write_b128 require 16-byte alignment on
      // gfx8 and older, but  we can do a 8 byte aligned, 16 byte access in a
      // single operation using ds_read2/write2_b64.
      bool AlignedBy8 = Alignment >= Align(8);
      if (IsFast)
        *IsFast = AlignedBy8;

      return AlignedBy8;
    }
  }

  if (AddrSpace == AMDGPUAS::PRIVATE_ADDRESS) {
    bool AlignedBy4 = Alignment >= Align(4);
    if (IsFast)
      *IsFast = AlignedBy4;

    return AlignedBy4 ||
           Subtarget->enableFlatScratch() ||
           Subtarget->hasUnalignedScratchAccess();
  }

  // FIXME: We have to be conservative here and assume that flat operations
  // will access scratch.  If we had access to the IR function, then we
  // could determine if any private memory was used in the function.
  if (AddrSpace == AMDGPUAS::FLAT_ADDRESS &&
      !Subtarget->hasUnalignedScratchAccess()) {
    bool AlignedBy4 = Alignment >= Align(4);
    if (IsFast)
      *IsFast = AlignedBy4;

    return AlignedBy4;
  }

  if (Subtarget->hasUnalignedBufferAccessEnabled() &&
      !(AddrSpace == AMDGPUAS::LOCAL_ADDRESS ||
        AddrSpace == AMDGPUAS::REGION_ADDRESS)) {
    // If we have an uniform constant load, it still requires using a slow
    // buffer instruction if unaligned.
    if (IsFast) {
      // Accesses can really be issued as 1-byte aligned or 4-byte aligned, so
      // 2-byte alignment is worse than 1 unless doing a 2-byte accesss.
      *IsFast = (AddrSpace == AMDGPUAS::CONSTANT_ADDRESS ||
                 AddrSpace == AMDGPUAS::CONSTANT_ADDRESS_32BIT) ?
        Alignment >= Align(4) : Alignment != Align(2);
    }

    return true;
  }

  // Smaller than dword value must be aligned.
  if (Size < 32)
    return false;

  // 8.1.6 - For Dword or larger reads or writes, the two LSBs of the
  // byte-address are ignored, thus forcing Dword alignment.
  // This applies to private, global, and constant memory.
  if (IsFast)
    *IsFast = true;

  return Size >= 32 && Alignment >= Align(4);
}

bool SITargetLowering::allowsMisalignedMemoryAccesses(
    EVT VT, unsigned AddrSpace, Align Alignment, MachineMemOperand::Flags Flags,
    bool *IsFast) const {
  if (IsFast)
    *IsFast = false;

  // TODO: I think v3i32 should allow unaligned accesses on CI with DS_READ_B96,
  // which isn't a simple VT.
  // Until MVT is extended to handle this, simply check for the size and
  // rely on the condition below: allow accesses if the size is a multiple of 4.
  if (VT == MVT::Other || (VT != MVT::Other && VT.getSizeInBits() > 1024 &&
                           VT.getStoreSize() > 16)) {
    return false;
  }

  return allowsMisalignedMemoryAccessesImpl(VT.getSizeInBits(), AddrSpace,
                                            Alignment, Flags, IsFast);
}

EVT SITargetLowering::getOptimalMemOpType(
    const MemOp &Op, const AttributeList &FuncAttributes) const {
  // FIXME: Should account for address space here.

  // The default fallback uses the private pointer size as a guess for a type to
  // use. Make sure we switch these to 64-bit accesses.

  if (Op.size() >= 16 &&
      Op.isDstAligned(Align(4))) // XXX: Should only do for global
    return MVT::v4i32;

  if (Op.size() >= 8 && Op.isDstAligned(Align(4)))
    return MVT::v2i32;

  // Use the default.
  return MVT::Other;
}

bool SITargetLowering::isMemOpHasNoClobberedMemOperand(const SDNode *N) const {
  const MemSDNode *MemNode = cast<MemSDNode>(N);
  const Value *Ptr = MemNode->getMemOperand()->getValue();
  const Instruction *I = dyn_cast_or_null<Instruction>(Ptr);
  return I && I->getMetadata("amdgpu.noclobber");
}

bool SITargetLowering::isNonGlobalAddrSpace(unsigned AS) {
  return AS == AMDGPUAS::LOCAL_ADDRESS || AS == AMDGPUAS::REGION_ADDRESS ||
         AS == AMDGPUAS::PRIVATE_ADDRESS;
}

bool SITargetLowering::isFreeAddrSpaceCast(unsigned SrcAS,
                                           unsigned DestAS) const {
  // Flat -> private/local is a simple truncate.
  // Flat -> global is no-op
  if (SrcAS == AMDGPUAS::FLAT_ADDRESS)
    return true;

  const GCNTargetMachine &TM =
      static_cast<const GCNTargetMachine &>(getTargetMachine());
  return TM.isNoopAddrSpaceCast(SrcAS, DestAS);
}

bool SITargetLowering::isMemOpUniform(const SDNode *N) const {
  const MemSDNode *MemNode = cast<MemSDNode>(N);

  return AMDGPUInstrInfo::isUniformMMO(MemNode->getMemOperand());
}

TargetLoweringBase::LegalizeTypeAction
SITargetLowering::getPreferredVectorAction(MVT VT) const {
  if (!VT.isScalableVector() && VT.getVectorNumElements() != 1 &&
      VT.getScalarType().bitsLE(MVT::i16))
    return VT.isPow2VectorType() ? TypeSplitVector : TypeWidenVector;
  return TargetLoweringBase::getPreferredVectorAction(VT);
}

bool SITargetLowering::shouldConvertConstantLoadToIntImm(const APInt &Imm,
                                                         Type *Ty) const {
  // FIXME: Could be smarter if called for vector constants.
  return true;
}

bool SITargetLowering::isTypeDesirableForOp(unsigned Op, EVT VT) const {
  if (Subtarget->has16BitInsts() && VT == MVT::i16) {
    switch (Op) {
    case ISD::LOAD:
    case ISD::STORE:

    // These operations are done with 32-bit instructions anyway.
    case ISD::AND:
    case ISD::OR:
    case ISD::XOR:
    case ISD::SELECT:
      // TODO: Extensions?
      return true;
    default:
      return false;
    }
  }

  // SimplifySetCC uses this function to determine whether or not it should
  // create setcc with i1 operands.  We don't have instructions for i1 setcc.
  if (VT == MVT::i1 && Op == ISD::SETCC)
    return false;

  return TargetLowering::isTypeDesirableForOp(Op, VT);
}

SDValue SITargetLowering::lowerKernArgParameterPtr(SelectionDAG &DAG,
                                                   const SDLoc &SL,
                                                   SDValue Chain,
                                                   uint64_t Offset) const {
  const DataLayout &DL = DAG.getDataLayout();
  MachineFunction &MF = DAG.getMachineFunction();
  const SIMachineFunctionInfo *Info = MF.getInfo<SIMachineFunctionInfo>();

  const ArgDescriptor *InputPtrReg;
  const TargetRegisterClass *RC;
  LLT ArgTy;
  MVT PtrVT = getPointerTy(DL, AMDGPUAS::CONSTANT_ADDRESS);

  std::tie(InputPtrReg, RC, ArgTy) =
      Info->getPreloadedValue(AMDGPUFunctionArgInfo::KERNARG_SEGMENT_PTR);

  // We may not have the kernarg segment argument if we have no kernel
  // arguments.
  if (!InputPtrReg)
    return DAG.getConstant(0, SL, PtrVT);

  MachineRegisterInfo &MRI = DAG.getMachineFunction().getRegInfo();
  SDValue BasePtr = DAG.getCopyFromReg(Chain, SL,
    MRI.getLiveInVirtReg(InputPtrReg->getRegister()), PtrVT);

  return DAG.getObjectPtrOffset(SL, BasePtr, TypeSize::Fixed(Offset));
}

SDValue SITargetLowering::getImplicitArgPtr(SelectionDAG &DAG,
                                            const SDLoc &SL) const {
  uint64_t Offset = getImplicitParameterOffset(DAG.getMachineFunction(),
                                               FIRST_IMPLICIT);
  return lowerKernArgParameterPtr(DAG, SL, DAG.getEntryNode(), Offset);
}

SDValue SITargetLowering::convertArgType(SelectionDAG &DAG, EVT VT, EVT MemVT,
                                         const SDLoc &SL, SDValue Val,
                                         bool Signed,
                                         const ISD::InputArg *Arg) const {
  // First, if it is a widened vector, narrow it.
  if (VT.isVector() &&
      VT.getVectorNumElements() != MemVT.getVectorNumElements()) {
    EVT NarrowedVT =
        EVT::getVectorVT(*DAG.getContext(), MemVT.getVectorElementType(),
                         VT.getVectorNumElements());
    Val = DAG.getNode(ISD::EXTRACT_SUBVECTOR, SL, NarrowedVT, Val,
                      DAG.getConstant(0, SL, MVT::i32));
  }

  // Then convert the vector elements or scalar value.
  if (Arg && (Arg->Flags.isSExt() || Arg->Flags.isZExt()) &&
      VT.bitsLT(MemVT)) {
    unsigned Opc = Arg->Flags.isZExt() ? ISD::AssertZext : ISD::AssertSext;
    Val = DAG.getNode(Opc, SL, MemVT, Val, DAG.getValueType(VT));
  }

  if (MemVT.isFloatingPoint())
    Val = getFPExtOrFPRound(DAG, Val, SL, VT);
  else if (Signed)
    Val = DAG.getSExtOrTrunc(Val, SL, VT);
  else
    Val = DAG.getZExtOrTrunc(Val, SL, VT);

  return Val;
}

SDValue SITargetLowering::lowerKernargMemParameter(
    SelectionDAG &DAG, EVT VT, EVT MemVT, const SDLoc &SL, SDValue Chain,
    uint64_t Offset, Align Alignment, bool Signed,
    const ISD::InputArg *Arg) const {
  MachinePointerInfo PtrInfo(AMDGPUAS::CONSTANT_ADDRESS);

  // Try to avoid using an extload by loading earlier than the argument address,
  // and extracting the relevant bits. The load should hopefully be merged with
  // the previous argument.
  if (MemVT.getStoreSize() < 4 && Alignment < 4) {
    // TODO: Handle align < 4 and size >= 4 (can happen with packed structs).
    int64_t AlignDownOffset = alignDown(Offset, 4);
    int64_t OffsetDiff = Offset - AlignDownOffset;

    EVT IntVT = MemVT.changeTypeToInteger();

    // TODO: If we passed in the base kernel offset we could have a better
    // alignment than 4, but we don't really need it.
    SDValue Ptr = lowerKernArgParameterPtr(DAG, SL, Chain, AlignDownOffset);
    SDValue Load = DAG.getLoad(MVT::i32, SL, Chain, Ptr, PtrInfo, Align(4),
                               MachineMemOperand::MODereferenceable |
                                   MachineMemOperand::MOInvariant);

    SDValue ShiftAmt = DAG.getConstant(OffsetDiff * 8, SL, MVT::i32);
    SDValue Extract = DAG.getNode(ISD::SRL, SL, MVT::i32, Load, ShiftAmt);

    SDValue ArgVal = DAG.getNode(ISD::TRUNCATE, SL, IntVT, Extract);
    ArgVal = DAG.getNode(ISD::BITCAST, SL, MemVT, ArgVal);
    ArgVal = convertArgType(DAG, VT, MemVT, SL, ArgVal, Signed, Arg);


    return DAG.getMergeValues({ ArgVal, Load.getValue(1) }, SL);
  }

  SDValue Ptr = lowerKernArgParameterPtr(DAG, SL, Chain, Offset);
  SDValue Load = DAG.getLoad(MemVT, SL, Chain, Ptr, PtrInfo, Alignment,
                             MachineMemOperand::MODereferenceable |
                                 MachineMemOperand::MOInvariant);

  SDValue Val = convertArgType(DAG, VT, MemVT, SL, Load, Signed, Arg);
  return DAG.getMergeValues({ Val, Load.getValue(1) }, SL);
}

SDValue SITargetLowering::lowerStackParameter(SelectionDAG &DAG, CCValAssign &VA,
                                              const SDLoc &SL, SDValue Chain,
                                              const ISD::InputArg &Arg) const {
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo &MFI = MF.getFrameInfo();

  if (Arg.Flags.isByVal()) {
    unsigned Size = Arg.Flags.getByValSize();
    int FrameIdx = MFI.CreateFixedObject(Size, VA.getLocMemOffset(), false);
    return DAG.getFrameIndex(FrameIdx, MVT::i32);
  }

  unsigned ArgOffset = VA.getLocMemOffset();
  unsigned ArgSize = VA.getValVT().getStoreSize();

  int FI = MFI.CreateFixedObject(ArgSize, ArgOffset, true);

  // Create load nodes to retrieve arguments from the stack.
  SDValue FIN = DAG.getFrameIndex(FI, MVT::i32);
  SDValue ArgValue;

  // For NON_EXTLOAD, generic code in getLoad assert(ValVT == MemVT)
  ISD::LoadExtType ExtType = ISD::NON_EXTLOAD;
  MVT MemVT = VA.getValVT();

  switch (VA.getLocInfo()) {
  default:
    break;
  case CCValAssign::BCvt:
    MemVT = VA.getLocVT();
    break;
  case CCValAssign::SExt:
    ExtType = ISD::SEXTLOAD;
    break;
  case CCValAssign::ZExt:
    ExtType = ISD::ZEXTLOAD;
    break;
  case CCValAssign::AExt:
    ExtType = ISD::EXTLOAD;
    break;
  }

  ArgValue = DAG.getExtLoad(
    ExtType, SL, VA.getLocVT(), Chain, FIN,
    MachinePointerInfo::getFixedStack(DAG.getMachineFunction(), FI),
    MemVT);
  return ArgValue;
}

SDValue SITargetLowering::getPreloadedValue(SelectionDAG &DAG,
  const SIMachineFunctionInfo &MFI,
  EVT VT,
  AMDGPUFunctionArgInfo::PreloadedValue PVID) const {
  const ArgDescriptor *Reg;
  const TargetRegisterClass *RC;
  LLT Ty;

  std::tie(Reg, RC, Ty) = MFI.getPreloadedValue(PVID);
  if (!Reg) {
    if (PVID == AMDGPUFunctionArgInfo::PreloadedValue::KERNARG_SEGMENT_PTR) {
      // It's possible for a kernarg intrinsic call to appear in a kernel with
      // no allocated segment, in which case we do not add the user sgpr
      // argument, so just return null.
      return DAG.getConstant(0, SDLoc(), VT);
    }

    // It's undefined behavior if a function marked with the amdgpu-no-*
    // attributes uses the corresponding intrinsic.
    return DAG.getUNDEF(VT);
  }

  return CreateLiveInRegister(DAG, RC, Reg->getRegister(), VT);
}

static void processPSInputArgs(SmallVectorImpl<ISD::InputArg> &Splits,
                               CallingConv::ID CallConv,
                               ArrayRef<ISD::InputArg> Ins, BitVector &Skipped,
                               FunctionType *FType,
                               SIMachineFunctionInfo *Info) {
  for (unsigned I = 0, E = Ins.size(), PSInputNum = 0; I != E; ++I) {
    const ISD::InputArg *Arg = &Ins[I];

    assert((!Arg->VT.isVector() || Arg->VT.getScalarSizeInBits() == 16) &&
           "vector type argument should have been split");

    // First check if it's a PS input addr.
    if (CallConv == CallingConv::AMDGPU_PS &&
        !Arg->Flags.isInReg() && PSInputNum <= 15) {
      bool SkipArg = !Arg->Used && !Info->isPSInputAllocated(PSInputNum);

      // Inconveniently only the first part of the split is marked as isSplit,
      // so skip to the end. We only want to increment PSInputNum once for the
      // entire split argument.
      if (Arg->Flags.isSplit()) {
        while (!Arg->Flags.isSplitEnd()) {
          assert((!Arg->VT.isVector() ||
                  Arg->VT.getScalarSizeInBits() == 16) &&
                 "unexpected vector split in ps argument type");
          if (!SkipArg)
            Splits.push_back(*Arg);
          Arg = &Ins[++I];
        }
      }

      if (SkipArg) {
        // We can safely skip PS inputs.
        Skipped.set(Arg->getOrigArgIndex());
        ++PSInputNum;
        continue;
      }

      Info->markPSInputAllocated(PSInputNum);
      if (Arg->Used)
        Info->markPSInputEnabled(PSInputNum);

      ++PSInputNum;
    }

    Splits.push_back(*Arg);
  }
}

// Allocate special inputs passed in VGPRs.
void SITargetLowering::allocateSpecialEntryInputVGPRs(CCState &CCInfo,
                                                      MachineFunction &MF,
                                                      const SIRegisterInfo &TRI,
                                                      SIMachineFunctionInfo &Info) const {
  const LLT S32 = LLT::scalar(32);
  MachineRegisterInfo &MRI = MF.getRegInfo();

  if (Info.hasWorkItemIDX()) {
    Register Reg = AMDGPU::VGPR0;
    MRI.setType(MF.addLiveIn(Reg, &AMDGPU::VGPR_32RegClass), S32);

    CCInfo.AllocateReg(Reg);
    unsigned Mask = (Subtarget->hasPackedTID() &&
                     Info.hasWorkItemIDY()) ? 0x3ff : ~0u;
    Info.setWorkItemIDX(ArgDescriptor::createRegister(Reg, Mask));
  }

  if (Info.hasWorkItemIDY()) {
    assert(Info.hasWorkItemIDX());
    if (Subtarget->hasPackedTID()) {
      Info.setWorkItemIDY(ArgDescriptor::createRegister(AMDGPU::VGPR0,
                                                        0x3ff << 10));
    } else {
      unsigned Reg = AMDGPU::VGPR1;
      MRI.setType(MF.addLiveIn(Reg, &AMDGPU::VGPR_32RegClass), S32);

      CCInfo.AllocateReg(Reg);
      Info.setWorkItemIDY(ArgDescriptor::createRegister(Reg));
    }
  }

  if (Info.hasWorkItemIDZ()) {
    assert(Info.hasWorkItemIDX() && Info.hasWorkItemIDY());
    if (Subtarget->hasPackedTID()) {
      Info.setWorkItemIDZ(ArgDescriptor::createRegister(AMDGPU::VGPR0,
                                                        0x3ff << 20));
    } else {
      unsigned Reg = AMDGPU::VGPR2;
      MRI.setType(MF.addLiveIn(Reg, &AMDGPU::VGPR_32RegClass), S32);

      CCInfo.AllocateReg(Reg);
      Info.setWorkItemIDZ(ArgDescriptor::createRegister(Reg));
    }
  }
}

// Try to allocate a VGPR at the end of the argument list, or if no argument
// VGPRs are left allocating a stack slot.
// If \p Mask is is given it indicates bitfield position in the register.
// If \p Arg is given use it with new ]p Mask instead of allocating new.
static ArgDescriptor allocateVGPR32Input(CCState &CCInfo, unsigned Mask = ~0u,
                                         ArgDescriptor Arg = ArgDescriptor()) {
  if (Arg.isSet())
    return ArgDescriptor::createArg(Arg, Mask);

  ArrayRef<MCPhysReg> ArgVGPRs
    = makeArrayRef(AMDGPU::VGPR_32RegClass.begin(), 32);
  unsigned RegIdx = CCInfo.getFirstUnallocated(ArgVGPRs);
  if (RegIdx == ArgVGPRs.size()) {
    // Spill to stack required.
    int64_t Offset = CCInfo.AllocateStack(4, Align(4));

    return ArgDescriptor::createStack(Offset, Mask);
  }

  unsigned Reg = ArgVGPRs[RegIdx];
  Reg = CCInfo.AllocateReg(Reg);
  assert(Reg != AMDGPU::NoRegister);

  MachineFunction &MF = CCInfo.getMachineFunction();
  Register LiveInVReg = MF.addLiveIn(Reg, &AMDGPU::VGPR_32RegClass);
  MF.getRegInfo().setType(LiveInVReg, LLT::scalar(32));
  return ArgDescriptor::createRegister(Reg, Mask);
}

static ArgDescriptor allocateSGPR32InputImpl(CCState &CCInfo,
                                             const TargetRegisterClass *RC,
                                             unsigned NumArgRegs) {
  ArrayRef<MCPhysReg> ArgSGPRs = makeArrayRef(RC->begin(), 32);
  unsigned RegIdx = CCInfo.getFirstUnallocated(ArgSGPRs);
  if (RegIdx == ArgSGPRs.size())
    report_fatal_error("ran out of SGPRs for arguments");

  unsigned Reg = ArgSGPRs[RegIdx];
  Reg = CCInfo.AllocateReg(Reg);
  assert(Reg != AMDGPU::NoRegister);

  MachineFunction &MF = CCInfo.getMachineFunction();
  MF.addLiveIn(Reg, RC);
  return ArgDescriptor::createRegister(Reg);
}

// If this has a fixed position, we still should allocate the register in the
// CCInfo state. Technically we could get away with this for values passed
// outside of the normal argument range.
static void allocateFixedSGPRInputImpl(CCState &CCInfo,
                                       const TargetRegisterClass *RC,
                                       MCRegister Reg) {
  Reg = CCInfo.AllocateReg(Reg);
  assert(Reg != AMDGPU::NoRegister);
  MachineFunction &MF = CCInfo.getMachineFunction();
  MF.addLiveIn(Reg, RC);
}

static void allocateSGPR32Input(CCState &CCInfo, ArgDescriptor &Arg) {
  if (Arg) {
    allocateFixedSGPRInputImpl(CCInfo, &AMDGPU::SGPR_32RegClass,
                               Arg.getRegister());
  } else
    Arg = allocateSGPR32InputImpl(CCInfo, &AMDGPU::SGPR_32RegClass, 32);
}

static void allocateSGPR64Input(CCState &CCInfo, ArgDescriptor &Arg) {
  if (Arg) {
    allocateFixedSGPRInputImpl(CCInfo, &AMDGPU::SGPR_64RegClass,
                               Arg.getRegister());
  } else
    Arg = allocateSGPR32InputImpl(CCInfo, &AMDGPU::SGPR_64RegClass, 16);
}

/// Allocate implicit function VGPR arguments at the end of allocated user
/// arguments.
void SITargetLowering::allocateSpecialInputVGPRs(
  CCState &CCInfo, MachineFunction &MF,
  const SIRegisterInfo &TRI, SIMachineFunctionInfo &Info) const {
  const unsigned Mask = 0x3ff;
  ArgDescriptor Arg;

  if (Info.hasWorkItemIDX()) {
    Arg = allocateVGPR32Input(CCInfo, Mask);
    Info.setWorkItemIDX(Arg);
  }

  if (Info.hasWorkItemIDY()) {
    Arg = allocateVGPR32Input(CCInfo, Mask << 10, Arg);
    Info.setWorkItemIDY(Arg);
  }

  if (Info.hasWorkItemIDZ())
    Info.setWorkItemIDZ(allocateVGPR32Input(CCInfo, Mask << 20, Arg));
}

/// Allocate implicit function VGPR arguments in fixed registers.
void SITargetLowering::allocateSpecialInputVGPRsFixed(
  CCState &CCInfo, MachineFunction &MF,
  const SIRegisterInfo &TRI, SIMachineFunctionInfo &Info) const {
  Register Reg = CCInfo.AllocateReg(AMDGPU::VGPR31);
  if (!Reg)
    report_fatal_error("failed to allocated VGPR for implicit arguments");

  const unsigned Mask = 0x3ff;
  Info.setWorkItemIDX(ArgDescriptor::createRegister(Reg, Mask));
  Info.setWorkItemIDY(ArgDescriptor::createRegister(Reg, Mask << 10));
  Info.setWorkItemIDZ(ArgDescriptor::createRegister(Reg, Mask << 20));
}

void SITargetLowering::allocateSpecialInputSGPRs(
  CCState &CCInfo,
  MachineFunction &MF,
  const SIRegisterInfo &TRI,
  SIMachineFunctionInfo &Info) const {
  auto &ArgInfo = Info.getArgInfo();

  // TODO: Unify handling with private memory pointers.
  if (Info.hasDispatchPtr())
    allocateSGPR64Input(CCInfo, ArgInfo.DispatchPtr);

  if (Info.hasQueuePtr())
    allocateSGPR64Input(CCInfo, ArgInfo.QueuePtr);

  // Implicit arg ptr takes the place of the kernarg segment pointer. This is a
  // constant offset from the kernarg segment.
  if (Info.hasImplicitArgPtr())
    allocateSGPR64Input(CCInfo, ArgInfo.ImplicitArgPtr);

  if (Info.hasDispatchID())
    allocateSGPR64Input(CCInfo, ArgInfo.DispatchID);

  // flat_scratch_init is not applicable for non-kernel functions.

  if (Info.hasWorkGroupIDX())
    allocateSGPR32Input(CCInfo, ArgInfo.WorkGroupIDX);

  if (Info.hasWorkGroupIDY())
    allocateSGPR32Input(CCInfo, ArgInfo.WorkGroupIDY);

  if (Info.hasWorkGroupIDZ())
    allocateSGPR32Input(CCInfo, ArgInfo.WorkGroupIDZ);
}

// Allocate special inputs passed in user SGPRs.
void SITargetLowering::allocateHSAUserSGPRs(CCState &CCInfo,
                                            MachineFunction &MF,
                                            const SIRegisterInfo &TRI,
                                            SIMachineFunctionInfo &Info) const {
  if (Info.hasImplicitBufferPtr()) {
    Register ImplicitBufferPtrReg = Info.addImplicitBufferPtr(TRI);
    MF.addLiveIn(ImplicitBufferPtrReg, &AMDGPU::SGPR_64RegClass);
    CCInfo.AllocateReg(ImplicitBufferPtrReg);
  }

  // FIXME: How should these inputs interact with inreg / custom SGPR inputs?
  if (Info.hasPrivateSegmentBuffer()) {
    Register PrivateSegmentBufferReg = Info.addPrivateSegmentBuffer(TRI);
    MF.addLiveIn(PrivateSegmentBufferReg, &AMDGPU::SGPR_128RegClass);
    CCInfo.AllocateReg(PrivateSegmentBufferReg);
  }

  if (Info.hasDispatchPtr()) {
    Register DispatchPtrReg = Info.addDispatchPtr(TRI);
    MF.addLiveIn(DispatchPtrReg, &AMDGPU::SGPR_64RegClass);
    CCInfo.AllocateReg(DispatchPtrReg);
  }

  if (Info.hasQueuePtr()) {
    Register QueuePtrReg = Info.addQueuePtr(TRI);
    MF.addLiveIn(QueuePtrReg, &AMDGPU::SGPR_64RegClass);
    CCInfo.AllocateReg(QueuePtrReg);
  }

  if (Info.hasKernargSegmentPtr()) {
    MachineRegisterInfo &MRI = MF.getRegInfo();
    Register InputPtrReg = Info.addKernargSegmentPtr(TRI);
    CCInfo.AllocateReg(InputPtrReg);

    Register VReg = MF.addLiveIn(InputPtrReg, &AMDGPU::SGPR_64RegClass);
    MRI.setType(VReg, LLT::pointer(AMDGPUAS::CONSTANT_ADDRESS, 64));
  }

  if (Info.hasDispatchID()) {
    Register DispatchIDReg = Info.addDispatchID(TRI);
    MF.addLiveIn(DispatchIDReg, &AMDGPU::SGPR_64RegClass);
    CCInfo.AllocateReg(DispatchIDReg);
  }

  if (Info.hasFlatScratchInit() && !getSubtarget()->isAmdPalOS()) {
    Register FlatScratchInitReg = Info.addFlatScratchInit(TRI);
    MF.addLiveIn(FlatScratchInitReg, &AMDGPU::SGPR_64RegClass);
    CCInfo.AllocateReg(FlatScratchInitReg);
  }

  // TODO: Add GridWorkGroupCount user SGPRs when used. For now with HSA we read
  // these from the dispatch pointer.
}

// Allocate special input registers that are initialized per-wave.
void SITargetLowering::allocateSystemSGPRs(CCState &CCInfo,
                                           MachineFunction &MF,
                                           SIMachineFunctionInfo &Info,
                                           CallingConv::ID CallConv,
                                           bool IsShader) const {
  if (Info.hasWorkGroupIDX()) {
    Register Reg = Info.addWorkGroupIDX();
    MF.addLiveIn(Reg, &AMDGPU::SGPR_32RegClass);
    CCInfo.AllocateReg(Reg);
  }

  if (Info.hasWorkGroupIDY()) {
    Register Reg = Info.addWorkGroupIDY();
    MF.addLiveIn(Reg, &AMDGPU::SGPR_32RegClass);
    CCInfo.AllocateReg(Reg);
  }

  if (Info.hasWorkGroupIDZ()) {
    Register Reg = Info.addWorkGroupIDZ();
    MF.addLiveIn(Reg, &AMDGPU::SGPR_32RegClass);
    CCInfo.AllocateReg(Reg);
  }

  if (Info.hasWorkGroupInfo()) {
    Register Reg = Info.addWorkGroupInfo();
    MF.addLiveIn(Reg, &AMDGPU::SGPR_32RegClass);
    CCInfo.AllocateReg(Reg);
  }

  if (Info.hasPrivateSegmentWaveByteOffset()) {
    // Scratch wave offset passed in system SGPR.
    unsigned PrivateSegmentWaveByteOffsetReg;

    if (IsShader) {
      PrivateSegmentWaveByteOffsetReg =
        Info.getPrivateSegmentWaveByteOffsetSystemSGPR();

      // This is true if the scratch wave byte offset doesn't have a fixed
      // location.
      if (PrivateSegmentWaveByteOffsetReg == AMDGPU::NoRegister) {
        PrivateSegmentWaveByteOffsetReg = findFirstFreeSGPR(CCInfo);
        Info.setPrivateSegmentWaveByteOffset(PrivateSegmentWaveByteOffsetReg);
      }
    } else
      PrivateSegmentWaveByteOffsetReg = Info.addPrivateSegmentWaveByteOffset();

    MF.addLiveIn(PrivateSegmentWaveByteOffsetReg, &AMDGPU::SGPR_32RegClass);
    CCInfo.AllocateReg(PrivateSegmentWaveByteOffsetReg);
  }
}

static void reservePrivateMemoryRegs(const TargetMachine &TM,
                                     MachineFunction &MF,
                                     const SIRegisterInfo &TRI,
                                     SIMachineFunctionInfo &Info) {
  // Now that we've figured out where the scratch register inputs are, see if
  // should reserve the arguments and use them directly.
  MachineFrameInfo &MFI = MF.getFrameInfo();
  bool HasStackObjects = MFI.hasStackObjects();
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();

  // Record that we know we have non-spill stack objects so we don't need to
  // check all stack objects later.
  if (HasStackObjects)
    Info.setHasNonSpillStackObjects(true);

  // Everything live out of a block is spilled with fast regalloc, so it's
  // almost certain that spilling will be required.
  if (TM.getOptLevel() == CodeGenOpt::None)
    HasStackObjects = true;

  // For now assume stack access is needed in any callee functions, so we need
  // the scratch registers to pass in.
  bool RequiresStackAccess = HasStackObjects || MFI.hasCalls();

  if (!ST.enableFlatScratch()) {
    if (RequiresStackAccess && ST.isAmdHsaOrMesa(MF.getFunction())) {
      // If we have stack objects, we unquestionably need the private buffer
      // resource. For the Code Object V2 ABI, this will be the first 4 user
      // SGPR inputs. We can reserve those and use them directly.

      Register PrivateSegmentBufferReg =
          Info.getPreloadedReg(AMDGPUFunctionArgInfo::PRIVATE_SEGMENT_BUFFER);
      Info.setScratchRSrcReg(PrivateSegmentBufferReg);
    } else {
      unsigned ReservedBufferReg = TRI.reservedPrivateSegmentBufferReg(MF);
      // We tentatively reserve the last registers (skipping the last registers
      // which may contain VCC, FLAT_SCR, and XNACK). After register allocation,
      // we'll replace these with the ones immediately after those which were
      // really allocated. In the prologue copies will be inserted from the
      // argument to these reserved registers.

      // Without HSA, relocations are used for the scratch pointer and the
      // buffer resource setup is always inserted in the prologue. Scratch wave
      // offset is still in an input SGPR.
      Info.setScratchRSrcReg(ReservedBufferReg);
    }
  }

  MachineRegisterInfo &MRI = MF.getRegInfo();

  // For entry functions we have to set up the stack pointer if we use it,
  // whereas non-entry functions get this "for free". This means there is no
  // intrinsic advantage to using S32 over S34 in cases where we do not have
  // calls but do need a frame pointer (i.e. if we are requested to have one
  // because frame pointer elimination is disabled). To keep things simple we
  // only ever use S32 as the call ABI stack pointer, and so using it does not
  // imply we need a separate frame pointer.
  //
  // Try to use s32 as the SP, but move it if it would interfere with input
  // arguments. This won't work with calls though.
  //
  // FIXME: Move SP to avoid any possible inputs, or find a way to spill input
  // registers.
  if (!MRI.isLiveIn(AMDGPU::SGPR32)) {
    Info.setStackPtrOffsetReg(AMDGPU::SGPR32);
  } else {
    assert(AMDGPU::isShader(MF.getFunction().getCallingConv()));

    if (MFI.hasCalls())
      report_fatal_error("call in graphics shader with too many input SGPRs");

    for (unsigned Reg : AMDGPU::SGPR_32RegClass) {
      if (!MRI.isLiveIn(Reg)) {
        Info.setStackPtrOffsetReg(Reg);
        break;
      }
    }

    if (Info.getStackPtrOffsetReg() == AMDGPU::SP_REG)
      report_fatal_error("failed to find register for SP");
  }

  // hasFP should be accurate for entry functions even before the frame is
  // finalized, because it does not rely on the known stack size, only
  // properties like whether variable sized objects are present.
  if (ST.getFrameLowering()->hasFP(MF)) {
    Info.setFrameOffsetReg(AMDGPU::SGPR33);
  }
}

bool SITargetLowering::supportSplitCSR(MachineFunction *MF) const {
  const SIMachineFunctionInfo *Info = MF->getInfo<SIMachineFunctionInfo>();
  return !Info->isEntryFunction();
}

void SITargetLowering::initializeSplitCSR(MachineBasicBlock *Entry) const {

}

void SITargetLowering::insertCopiesSplitCSR(
  MachineBasicBlock *Entry,
  const SmallVectorImpl<MachineBasicBlock *> &Exits) const {
  const SIRegisterInfo *TRI = getSubtarget()->getRegisterInfo();

  const MCPhysReg *IStart = TRI->getCalleeSavedRegsViaCopy(Entry->getParent());
  if (!IStart)
    return;

  const TargetInstrInfo *TII = Subtarget->getInstrInfo();
  MachineRegisterInfo *MRI = &Entry->getParent()->getRegInfo();
  MachineBasicBlock::iterator MBBI = Entry->begin();
  for (const MCPhysReg *I = IStart; *I; ++I) {
    const TargetRegisterClass *RC = nullptr;
    if (AMDGPU::SReg_64RegClass.contains(*I))
      RC = &AMDGPU::SGPR_64RegClass;
    else if (AMDGPU::SReg_32RegClass.contains(*I))
      RC = &AMDGPU::SGPR_32RegClass;
    else
      llvm_unreachable("Unexpected register class in CSRsViaCopy!");

    Register NewVR = MRI->createVirtualRegister(RC);
    // Create copy from CSR to a virtual register.
    Entry->addLiveIn(*I);
    BuildMI(*Entry, MBBI, DebugLoc(), TII->get(TargetOpcode::COPY), NewVR)
      .addReg(*I);

    // Insert the copy-back instructions right before the terminator.
    for (auto *Exit : Exits)
      BuildMI(*Exit, Exit->getFirstTerminator(), DebugLoc(),
              TII->get(TargetOpcode::COPY), *I)
        .addReg(NewVR);
  }
}

SDValue SITargetLowering::LowerFormalArguments(
    SDValue Chain, CallingConv::ID CallConv, bool isVarArg,
    const SmallVectorImpl<ISD::InputArg> &Ins, const SDLoc &DL,
    SelectionDAG &DAG, SmallVectorImpl<SDValue> &InVals) const {
  const SIRegisterInfo *TRI = getSubtarget()->getRegisterInfo();

  MachineFunction &MF = DAG.getMachineFunction();
  const Function &Fn = MF.getFunction();
  FunctionType *FType = MF.getFunction().getFunctionType();
  SIMachineFunctionInfo *Info = MF.getInfo<SIMachineFunctionInfo>();

  if (Subtarget->isAmdHsaOS() && AMDGPU::isGraphics(CallConv)) {
    DiagnosticInfoUnsupported NoGraphicsHSA(
        Fn, "unsupported non-compute shaders with HSA", DL.getDebugLoc());
    DAG.getContext()->diagnose(NoGraphicsHSA);
    return DAG.getEntryNode();
  }

  Info->allocateModuleLDSGlobal(Fn.getParent());

  SmallVector<ISD::InputArg, 16> Splits;
  SmallVector<CCValAssign, 16> ArgLocs;
  BitVector Skipped(Ins.size());
  CCState CCInfo(CallConv, isVarArg, DAG.getMachineFunction(), ArgLocs,
                 *DAG.getContext());

  bool IsGraphics = AMDGPU::isGraphics(CallConv);
  bool IsKernel = AMDGPU::isKernel(CallConv);
  bool IsEntryFunc = AMDGPU::isEntryFunctionCC(CallConv);

  if (IsGraphics) {
    assert(!Info->hasDispatchPtr() && !Info->hasKernargSegmentPtr() &&
           (!Info->hasFlatScratchInit() || Subtarget->enableFlatScratch()) &&
           !Info->hasWorkGroupIDX() && !Info->hasWorkGroupIDY() &&
           !Info->hasWorkGroupIDZ() && !Info->hasWorkGroupInfo() &&
           !Info->hasWorkItemIDX() && !Info->hasWorkItemIDY() &&
           !Info->hasWorkItemIDZ());
  }

  if (CallConv == CallingConv::AMDGPU_PS) {
    processPSInputArgs(Splits, CallConv, Ins, Skipped, FType, Info);

    // At least one interpolation mode must be enabled or else the GPU will
    // hang.
    //
    // Check PSInputAddr instead of PSInputEnable. The idea is that if the user
    // set PSInputAddr, the user wants to enable some bits after the compilation
    // based on run-time states. Since we can't know what the final PSInputEna
    // will look like, so we shouldn't do anything here and the user should take
    // responsibility for the correct programming.
    //
    // Otherwise, the following restrictions apply:
    // - At least one of PERSP_* (0xF) or LINEAR_* (0x70) must be enabled.
    // - If POS_W_FLOAT (11) is enabled, at least one of PERSP_* must be
    //   enabled too.
    if ((Info->getPSInputAddr() & 0x7F) == 0 ||
        ((Info->getPSInputAddr() & 0xF) == 0 && Info->isPSInputAllocated(11))) {
      CCInfo.AllocateReg(AMDGPU::VGPR0);
      CCInfo.AllocateReg(AMDGPU::VGPR1);
      Info->markPSInputAllocated(0);
      Info->markPSInputEnabled(0);
    }
    if (Subtarget->isAmdPalOS()) {
      // For isAmdPalOS, the user does not enable some bits after compilation
      // based on run-time states; the register values being generated here are
      // the final ones set in hardware. Therefore we need to apply the
      // workaround to PSInputAddr and PSInputEnable together.  (The case where
      // a bit is set in PSInputAddr but not PSInputEnable is where the
      // frontend set up an input arg for a particular interpolation mode, but
      // nothing uses that input arg. Really we should have an earlier pass
      // that removes such an arg.)
      unsigned PsInputBits = Info->getPSInputAddr() & Info->getPSInputEnable();
      if ((PsInputBits & 0x7F) == 0 ||
          ((PsInputBits & 0xF) == 0 && (PsInputBits >> 11 & 1)))
        Info->markPSInputEnabled(
            countTrailingZeros(Info->getPSInputAddr(), ZB_Undefined));
    }
  } else if (IsKernel) {
    assert(Info->hasWorkGroupIDX() && Info->hasWorkItemIDX());
  } else {
    Splits.append(Ins.begin(), Ins.end());
  }

  if (IsEntryFunc) {
    allocateSpecialEntryInputVGPRs(CCInfo, MF, *TRI, *Info);
    allocateHSAUserSGPRs(CCInfo, MF, *TRI, *Info);
  } else if (!IsGraphics) {
    // For the fixed ABI, pass workitem IDs in the last argument register.
    allocateSpecialInputVGPRsFixed(CCInfo, MF, *TRI, *Info);
  }

  if (IsKernel) {
    analyzeFormalArgumentsCompute(CCInfo, Ins);
  } else {
    CCAssignFn *AssignFn = CCAssignFnForCall(CallConv, isVarArg);
    CCInfo.AnalyzeFormalArguments(Splits, AssignFn);
  }

  SmallVector<SDValue, 16> Chains;

  // FIXME: This is the minimum kernel argument alignment. We should improve
  // this to the maximum alignment of the arguments.
  //
  // FIXME: Alignment of explicit arguments totally broken with non-0 explicit
  // kern arg offset.
  const Align KernelArgBaseAlign = Align(16);

  for (unsigned i = 0, e = Ins.size(), ArgIdx = 0; i != e; ++i) {
    const ISD::InputArg &Arg = Ins[i];
    if (Arg.isOrigArg() && Skipped[Arg.getOrigArgIndex()]) {
      InVals.push_back(DAG.getUNDEF(Arg.VT));
      continue;
    }

    CCValAssign &VA = ArgLocs[ArgIdx++];
    MVT VT = VA.getLocVT();

    if (IsEntryFunc && VA.isMemLoc()) {
      VT = Ins[i].VT;
      EVT MemVT = VA.getLocVT();

      const uint64_t Offset = VA.getLocMemOffset();
      Align Alignment = commonAlignment(KernelArgBaseAlign, Offset);

      if (Arg.Flags.isByRef()) {
        SDValue Ptr = lowerKernArgParameterPtr(DAG, DL, Chain, Offset);

        const GCNTargetMachine &TM =
            static_cast<const GCNTargetMachine &>(getTargetMachine());
        if (!TM.isNoopAddrSpaceCast(AMDGPUAS::CONSTANT_ADDRESS,
                                    Arg.Flags.getPointerAddrSpace())) {
          Ptr = DAG.getAddrSpaceCast(DL, VT, Ptr, AMDGPUAS::CONSTANT_ADDRESS,
                                     Arg.Flags.getPointerAddrSpace());
        }

        InVals.push_back(Ptr);
        continue;
      }

      SDValue Arg = lowerKernargMemParameter(
        DAG, VT, MemVT, DL, Chain, Offset, Alignment, Ins[i].Flags.isSExt(), &Ins[i]);
      Chains.push_back(Arg.getValue(1));

      auto *ParamTy =
        dyn_cast<PointerType>(FType->getParamType(Ins[i].getOrigArgIndex()));
      if (Subtarget->getGeneration() == AMDGPUSubtarget::SOUTHERN_ISLANDS &&
          ParamTy && (ParamTy->getAddressSpace() == AMDGPUAS::LOCAL_ADDRESS ||
                      ParamTy->getAddressSpace() == AMDGPUAS::REGION_ADDRESS)) {
        // On SI local pointers are just offsets into LDS, so they are always
        // less than 16-bits.  On CI and newer they could potentially be
        // real pointers, so we can't guarantee their size.
        Arg = DAG.getNode(ISD::AssertZext, DL, Arg.getValueType(), Arg,
                          DAG.getValueType(MVT::i16));
      }

      InVals.push_back(Arg);
      continue;
    } else if (!IsEntryFunc && VA.isMemLoc()) {
      SDValue Val = lowerStackParameter(DAG, VA, DL, Chain, Arg);
      InVals.push_back(Val);
      if (!Arg.Flags.isByVal())
        Chains.push_back(Val.getValue(1));
      continue;
    }

    assert(VA.isRegLoc() && "Parameter must be in a register!");

    Register Reg = VA.getLocReg();
    const TargetRegisterClass *RC = TRI->getMinimalPhysRegClass(Reg, VT);
    EVT ValVT = VA.getValVT();

    Reg = MF.addLiveIn(Reg, RC);
    SDValue Val = DAG.getCopyFromReg(Chain, DL, Reg, VT);

    if (Arg.Flags.isSRet()) {
      // The return object should be reasonably addressable.

      // FIXME: This helps when the return is a real sret. If it is a
      // automatically inserted sret (i.e. CanLowerReturn returns false), an
      // extra copy is inserted in SelectionDAGBuilder which obscures this.
      unsigned NumBits
        = 32 - getSubtarget()->getKnownHighZeroBitsForFrameIndex();
      Val = DAG.getNode(ISD::AssertZext, DL, VT, Val,
        DAG.getValueType(EVT::getIntegerVT(*DAG.getContext(), NumBits)));
    }

    // If this is an 8 or 16-bit value, it is really passed promoted
    // to 32 bits. Insert an assert[sz]ext to capture this, then
    // truncate to the right size.
    switch (VA.getLocInfo()) {
    case CCValAssign::Full:
      break;
    case CCValAssign::BCvt:
      Val = DAG.getNode(ISD::BITCAST, DL, ValVT, Val);
      break;
    case CCValAssign::SExt:
      Val = DAG.getNode(ISD::AssertSext, DL, VT, Val,
                        DAG.getValueType(ValVT));
      Val = DAG.getNode(ISD::TRUNCATE, DL, ValVT, Val);
      break;
    case CCValAssign::ZExt:
      Val = DAG.getNode(ISD::AssertZext, DL, VT, Val,
                        DAG.getValueType(ValVT));
      Val = DAG.getNode(ISD::TRUNCATE, DL, ValVT, Val);
      break;
    case CCValAssign::AExt:
      Val = DAG.getNode(ISD::TRUNCATE, DL, ValVT, Val);
      break;
    default:
      llvm_unreachable("Unknown loc info!");
    }

    InVals.push_back(Val);
  }

  // Start adding system SGPRs.
  if (IsEntryFunc) {
    allocateSystemSGPRs(CCInfo, MF, *Info, CallConv, IsGraphics);
  } else {
    CCInfo.AllocateReg(Info->getScratchRSrcReg());
    if (!IsGraphics)
      allocateSpecialInputSGPRs(CCInfo, MF, *TRI, *Info);
  }

  auto &ArgUsageInfo =
    DAG.getPass()->getAnalysis<AMDGPUArgumentUsageInfo>();
  ArgUsageInfo.setFuncArgInfo(Fn, Info->getArgInfo());

  unsigned StackArgSize = CCInfo.getNextStackOffset();
  Info->setBytesInStackArgArea(StackArgSize);

  return Chains.empty() ? Chain :
    DAG.getNode(ISD::TokenFactor, DL, MVT::Other, Chains);
}

// TODO: If return values can't fit in registers, we should return as many as
// possible in registers before passing on stack.
bool SITargetLowering::CanLowerReturn(
  CallingConv::ID CallConv,
  MachineFunction &MF, bool IsVarArg,
  const SmallVectorImpl<ISD::OutputArg> &Outs,
  LLVMContext &Context) const {
  // Replacing returns with sret/stack usage doesn't make sense for shaders.
  // FIXME: Also sort of a workaround for custom vector splitting in LowerReturn
  // for shaders. Vector types should be explicitly handled by CC.
  if (AMDGPU::isEntryFunctionCC(CallConv))
    return true;

  SmallVector<CCValAssign, 16> RVLocs;
  CCState CCInfo(CallConv, IsVarArg, MF, RVLocs, Context);
  return CCInfo.CheckReturn(Outs, CCAssignFnForReturn(CallConv, IsVarArg));
}

SDValue
SITargetLowering::LowerReturn(SDValue Chain, CallingConv::ID CallConv,
                              bool isVarArg,
                              const SmallVectorImpl<ISD::OutputArg> &Outs,
                              const SmallVectorImpl<SDValue> &OutVals,
                              const SDLoc &DL, SelectionDAG &DAG) const {
  MachineFunction &MF = DAG.getMachineFunction();
  SIMachineFunctionInfo *Info = MF.getInfo<SIMachineFunctionInfo>();

  if (AMDGPU::isKernel(CallConv)) {
    return AMDGPUTargetLowering::LowerReturn(Chain, CallConv, isVarArg, Outs,
                                             OutVals, DL, DAG);
  }

  bool IsShader = AMDGPU::isShader(CallConv);

  Info->setIfReturnsVoid(Outs.empty());
  bool IsWaveEnd = Info->returnsVoid() && IsShader;

  // CCValAssign - represent the assignment of the return value to a location.
  SmallVector<CCValAssign, 48> RVLocs;
  SmallVector<ISD::OutputArg, 48> Splits;

  // CCState - Info about the registers and stack slots.
  CCState CCInfo(CallConv, isVarArg, DAG.getMachineFunction(), RVLocs,
                 *DAG.getContext());

  // Analyze outgoing return values.
  CCInfo.AnalyzeReturn(Outs, CCAssignFnForReturn(CallConv, isVarArg));

  SDValue Flag;
  SmallVector<SDValue, 48> RetOps;
  RetOps.push_back(Chain); // Operand #0 = Chain (updated below)

  // Add return address for callable functions.
  if (!Info->isEntryFunction()) {
    const SIRegisterInfo *TRI = getSubtarget()->getRegisterInfo();
    SDValue ReturnAddrReg = CreateLiveInRegister(
      DAG, &AMDGPU::SReg_64RegClass, TRI->getReturnAddressReg(MF), MVT::i64);

    SDValue ReturnAddrVirtualReg =
        DAG.getRegister(MF.getRegInfo().createVirtualRegister(
                            CallConv != CallingConv::AMDGPU_Gfx
                                ? &AMDGPU::CCR_SGPR_64RegClass
                                : &AMDGPU::Gfx_CCR_SGPR_64RegClass),
                        MVT::i64);
    Chain =
        DAG.getCopyToReg(Chain, DL, ReturnAddrVirtualReg, ReturnAddrReg, Flag);
    Flag = Chain.getValue(1);
    RetOps.push_back(ReturnAddrVirtualReg);
  }

  // Copy the result values into the output registers.
  for (unsigned I = 0, RealRVLocIdx = 0, E = RVLocs.size(); I != E;
       ++I, ++RealRVLocIdx) {
    CCValAssign &VA = RVLocs[I];
    assert(VA.isRegLoc() && "Can only return in registers!");
    // TODO: Partially return in registers if return values don't fit.
    SDValue Arg = OutVals[RealRVLocIdx];

    // Copied from other backends.
    switch (VA.getLocInfo()) {
    case CCValAssign::Full:
      break;
    case CCValAssign::BCvt:
      Arg = DAG.getNode(ISD::BITCAST, DL, VA.getLocVT(), Arg);
      break;
    case CCValAssign::SExt:
      Arg = DAG.getNode(ISD::SIGN_EXTEND, DL, VA.getLocVT(), Arg);
      break;
    case CCValAssign::ZExt:
      Arg = DAG.getNode(ISD::ZERO_EXTEND, DL, VA.getLocVT(), Arg);
      break;
    case CCValAssign::AExt:
      Arg = DAG.getNode(ISD::ANY_EXTEND, DL, VA.getLocVT(), Arg);
      break;
    default:
      llvm_unreachable("Unknown loc info!");
    }

    Chain = DAG.getCopyToReg(Chain, DL, VA.getLocReg(), Arg, Flag);
    Flag = Chain.getValue(1);
    RetOps.push_back(DAG.getRegister(VA.getLocReg(), VA.getLocVT()));
  }

  // FIXME: Does sret work properly?
  if (!Info->isEntryFunction()) {
    const SIRegisterInfo *TRI = Subtarget->getRegisterInfo();
    const MCPhysReg *I =
      TRI->getCalleeSavedRegsViaCopy(&DAG.getMachineFunction());
    if (I) {
      for (; *I; ++I) {
        if (AMDGPU::SReg_64RegClass.contains(*I))
          RetOps.push_back(DAG.getRegister(*I, MVT::i64));
        else if (AMDGPU::SReg_32RegClass.contains(*I))
          RetOps.push_back(DAG.getRegister(*I, MVT::i32));
        else
          llvm_unreachable("Unexpected register class in CSRsViaCopy!");
      }
    }
  }

  // Update chain and glue.
  RetOps[0] = Chain;
  if (Flag.getNode())
    RetOps.push_back(Flag);

  unsigned Opc = AMDGPUISD::ENDPGM;
  if (!IsWaveEnd) {
    if (IsShader)
      Opc = AMDGPUISD::RETURN_TO_EPILOG;
    else if (CallConv == CallingConv::AMDGPU_Gfx)
      Opc = AMDGPUISD::RET_GFX_FLAG;
    else
      Opc = AMDGPUISD::RET_FLAG;
  }

  return DAG.getNode(Opc, DL, MVT::Other, RetOps);
}

SDValue SITargetLowering::LowerCallResult(
    SDValue Chain, SDValue InFlag, CallingConv::ID CallConv, bool IsVarArg,
    const SmallVectorImpl<ISD::InputArg> &Ins, const SDLoc &DL,
    SelectionDAG &DAG, SmallVectorImpl<SDValue> &InVals, bool IsThisReturn,
    SDValue ThisVal) const {
  CCAssignFn *RetCC = CCAssignFnForReturn(CallConv, IsVarArg);

  // Assign locations to each value returned by this call.
  SmallVector<CCValAssign, 16> RVLocs;
  CCState CCInfo(CallConv, IsVarArg, DAG.getMachineFunction(), RVLocs,
                 *DAG.getContext());
  CCInfo.AnalyzeCallResult(Ins, RetCC);

  // Copy all of the result registers out of their specified physreg.
  for (unsigned i = 0; i != RVLocs.size(); ++i) {
    CCValAssign VA = RVLocs[i];
    SDValue Val;

    if (VA.isRegLoc()) {
      Val = DAG.getCopyFromReg(Chain, DL, VA.getLocReg(), VA.getLocVT(), InFlag);
      Chain = Val.getValue(1);
      InFlag = Val.getValue(2);
    } else if (VA.isMemLoc()) {
      report_fatal_error("TODO: return values in memory");
    } else
      llvm_unreachable("unknown argument location type");

    switch (VA.getLocInfo()) {
    case CCValAssign::Full:
      break;
    case CCValAssign::BCvt:
      Val = DAG.getNode(ISD::BITCAST, DL, VA.getValVT(), Val);
      break;
    case CCValAssign::ZExt:
      Val = DAG.getNode(ISD::AssertZext, DL, VA.getLocVT(), Val,
                        DAG.getValueType(VA.getValVT()));
      Val = DAG.getNode(ISD::TRUNCATE, DL, VA.getValVT(), Val);
      break;
    case CCValAssign::SExt:
      Val = DAG.getNode(ISD::AssertSext, DL, VA.getLocVT(), Val,
                        DAG.getValueType(VA.getValVT()));
      Val = DAG.getNode(ISD::TRUNCATE, DL, VA.getValVT(), Val);
      break;
    case CCValAssign::AExt:
      Val = DAG.getNode(ISD::TRUNCATE, DL, VA.getValVT(), Val);
      break;
    default:
      llvm_unreachable("Unknown loc info!");
    }

    InVals.push_back(Val);
  }

  return Chain;
}

// Add code to pass special inputs required depending on used features separate
// from the explicit user arguments present in the IR.
void SITargetLowering::passSpecialInputs(
    CallLoweringInfo &CLI,
    CCState &CCInfo,
    const SIMachineFunctionInfo &Info,
    SmallVectorImpl<std::pair<unsigned, SDValue>> &RegsToPass,
    SmallVectorImpl<SDValue> &MemOpChains,
    SDValue Chain) const {
  // If we don't have a call site, this was a call inserted by
  // legalization. These can never use special inputs.
  if (!CLI.CB)
    return;

  SelectionDAG &DAG = CLI.DAG;
  const SDLoc &DL = CLI.DL;
  const Function &F = DAG.getMachineFunction().getFunction();

  const SIRegisterInfo *TRI = Subtarget->getRegisterInfo();
  const AMDGPUFunctionArgInfo &CallerArgInfo = Info.getArgInfo();

  const AMDGPUFunctionArgInfo *CalleeArgInfo
    = &AMDGPUArgumentUsageInfo::FixedABIFunctionInfo;
  if (const Function *CalleeFunc = CLI.CB->getCalledFunction()) {
    auto &ArgUsageInfo =
      DAG.getPass()->getAnalysis<AMDGPUArgumentUsageInfo>();
    CalleeArgInfo = &ArgUsageInfo.lookupFuncArgInfo(*CalleeFunc);
  }

  // TODO: Unify with private memory register handling. This is complicated by
  // the fact that at least in kernels, the input argument is not necessarily
  // in the same location as the input.
  static constexpr std::pair<AMDGPUFunctionArgInfo::PreloadedValue,
                             StringLiteral> ImplicitAttrs[] = {
    {AMDGPUFunctionArgInfo::DISPATCH_PTR, "amdgpu-no-dispatch-ptr"},
    {AMDGPUFunctionArgInfo::QUEUE_PTR, "amdgpu-no-queue-ptr" },
    {AMDGPUFunctionArgInfo::IMPLICIT_ARG_PTR, "amdgpu-no-implicitarg-ptr"},
    {AMDGPUFunctionArgInfo::DISPATCH_ID, "amdgpu-no-dispatch-id"},
    {AMDGPUFunctionArgInfo::WORKGROUP_ID_X, "amdgpu-no-workgroup-id-x"},
    {AMDGPUFunctionArgInfo::WORKGROUP_ID_Y,"amdgpu-no-workgroup-id-y"},
    {AMDGPUFunctionArgInfo::WORKGROUP_ID_Z,"amdgpu-no-workgroup-id-z"}
  };

  for (auto Attr : ImplicitAttrs) {
    const ArgDescriptor *OutgoingArg;
    const TargetRegisterClass *ArgRC;
    LLT ArgTy;

    AMDGPUFunctionArgInfo::PreloadedValue InputID = Attr.first;

    // If the callee does not use the attribute value, skip copying the value.
    if (CLI.CB->hasFnAttr(Attr.second))
      continue;

    std::tie(OutgoingArg, ArgRC, ArgTy) =
        CalleeArgInfo->getPreloadedValue(InputID);
    if (!OutgoingArg)
      continue;

    const ArgDescriptor *IncomingArg;
    const TargetRegisterClass *IncomingArgRC;
    LLT Ty;
    std::tie(IncomingArg, IncomingArgRC, Ty) =
        CallerArgInfo.getPreloadedValue(InputID);
    assert(IncomingArgRC == ArgRC);

    // All special arguments are ints for now.
    EVT ArgVT = TRI->getSpillSize(*ArgRC) == 8 ? MVT::i64 : MVT::i32;
    SDValue InputReg;

    if (IncomingArg) {
      InputReg = loadInputValue(DAG, ArgRC, ArgVT, DL, *IncomingArg);
    } else if (InputID == AMDGPUFunctionArgInfo::IMPLICIT_ARG_PTR) {
      // The implicit arg ptr is special because it doesn't have a corresponding
      // input for kernels, and is computed from the kernarg segment pointer.
      InputReg = getImplicitArgPtr(DAG, DL);
    } else {
      // We may have proven the input wasn't needed, although the ABI is
      // requiring it. We just need to allocate the register appropriately.
      InputReg = DAG.getUNDEF(ArgVT);
    }

    if (OutgoingArg->isRegister()) {
      RegsToPass.emplace_back(OutgoingArg->getRegister(), InputReg);
      if (!CCInfo.AllocateReg(OutgoingArg->getRegister()))
        report_fatal_error("failed to allocate implicit input argument");
    } else {
      unsigned SpecialArgOffset =
          CCInfo.AllocateStack(ArgVT.getStoreSize(), Align(4));
      SDValue ArgStore = storeStackInputValue(DAG, DL, Chain, InputReg,
                                              SpecialArgOffset);
      MemOpChains.push_back(ArgStore);
    }
  }

  // Pack workitem IDs into a single register or pass it as is if already
  // packed.
  const ArgDescriptor *OutgoingArg;
  const TargetRegisterClass *ArgRC;
  LLT Ty;

  std::tie(OutgoingArg, ArgRC, Ty) =
      CalleeArgInfo->getPreloadedValue(AMDGPUFunctionArgInfo::WORKITEM_ID_X);
  if (!OutgoingArg)
    std::tie(OutgoingArg, ArgRC, Ty) =
        CalleeArgInfo->getPreloadedValue(AMDGPUFunctionArgInfo::WORKITEM_ID_Y);
  if (!OutgoingArg)
    std::tie(OutgoingArg, ArgRC, Ty) =
        CalleeArgInfo->getPreloadedValue(AMDGPUFunctionArgInfo::WORKITEM_ID_Z);
  if (!OutgoingArg)
    return;

  const ArgDescriptor *IncomingArgX = std::get<0>(
      CallerArgInfo.getPreloadedValue(AMDGPUFunctionArgInfo::WORKITEM_ID_X));
  const ArgDescriptor *IncomingArgY = std::get<0>(
      CallerArgInfo.getPreloadedValue(AMDGPUFunctionArgInfo::WORKITEM_ID_Y));
  const ArgDescriptor *IncomingArgZ = std::get<0>(
      CallerArgInfo.getPreloadedValue(AMDGPUFunctionArgInfo::WORKITEM_ID_Z));

  SDValue InputReg;
  SDLoc SL;

  const bool NeedWorkItemIDX = !CLI.CB->hasFnAttr("amdgpu-no-workitem-id-x");
  const bool NeedWorkItemIDY = !CLI.CB->hasFnAttr("amdgpu-no-workitem-id-y");
  const bool NeedWorkItemIDZ = !CLI.CB->hasFnAttr("amdgpu-no-workitem-id-z");

  // If incoming ids are not packed we need to pack them.
  if (IncomingArgX && !IncomingArgX->isMasked() && CalleeArgInfo->WorkItemIDX &&
      NeedWorkItemIDX) {
    if (Subtarget->getMaxWorkitemID(F, 0) != 0) {
      InputReg = loadInputValue(DAG, ArgRC, MVT::i32, DL, *IncomingArgX);
    } else {
      InputReg = DAG.getConstant(0, DL, MVT::i32);
    }
  }

  if (IncomingArgY && !IncomingArgY->isMasked() && CalleeArgInfo->WorkItemIDY &&
      NeedWorkItemIDY && Subtarget->getMaxWorkitemID(F, 1) != 0) {
    SDValue Y = loadInputValue(DAG, ArgRC, MVT::i32, DL, *IncomingArgY);
    Y = DAG.getNode(ISD::SHL, SL, MVT::i32, Y,
                    DAG.getShiftAmountConstant(10, MVT::i32, SL));
    InputReg = InputReg.getNode() ?
                 DAG.getNode(ISD::OR, SL, MVT::i32, InputReg, Y) : Y;
  }

  if (IncomingArgZ && !IncomingArgZ->isMasked() && CalleeArgInfo->WorkItemIDZ &&
      NeedWorkItemIDZ && Subtarget->getMaxWorkitemID(F, 2) != 0) {
    SDValue Z = loadInputValue(DAG, ArgRC, MVT::i32, DL, *IncomingArgZ);
    Z = DAG.getNode(ISD::SHL, SL, MVT::i32, Z,
                    DAG.getShiftAmountConstant(20, MVT::i32, SL));
    InputReg = InputReg.getNode() ?
                 DAG.getNode(ISD::OR, SL, MVT::i32, InputReg, Z) : Z;
  }

  if (!InputReg && (NeedWorkItemIDX || NeedWorkItemIDY || NeedWorkItemIDZ)) {
    if (!IncomingArgX && !IncomingArgY && !IncomingArgZ) {
      // We're in a situation where the outgoing function requires the workitem
      // ID, but the calling function does not have it (e.g a graphics function
      // calling a C calling convention function). This is illegal, but we need
      // to produce something.
      InputReg = DAG.getUNDEF(MVT::i32);
    } else {
      // Workitem ids are already packed, any of present incoming arguments
      // will carry all required fields.
      ArgDescriptor IncomingArg = ArgDescriptor::createArg(
        IncomingArgX ? *IncomingArgX :
        IncomingArgY ? *IncomingArgY :
        *IncomingArgZ, ~0u);
      InputReg = loadInputValue(DAG, ArgRC, MVT::i32, DL, IncomingArg);
    }
  }

  if (OutgoingArg->isRegister()) {
    if (InputReg)
      RegsToPass.emplace_back(OutgoingArg->getRegister(), InputReg);

    CCInfo.AllocateReg(OutgoingArg->getRegister());
  } else {
    unsigned SpecialArgOffset = CCInfo.AllocateStack(4, Align(4));
    if (InputReg) {
      SDValue ArgStore = storeStackInputValue(DAG, DL, Chain, InputReg,
                                              SpecialArgOffset);
      MemOpChains.push_back(ArgStore);
    }
  }
}

static bool canGuaranteeTCO(CallingConv::ID CC) {
  return CC == CallingConv::Fast;
}

/// Return true if we might ever do TCO for calls with this calling convention.
static bool mayTailCallThisCC(CallingConv::ID CC) {
  switch (CC) {
  case CallingConv::C:
  case CallingConv::AMDGPU_Gfx:
    return true;
  default:
    return canGuaranteeTCO(CC);
  }
}

bool SITargetLowering::isEligibleForTailCallOptimization(
    SDValue Callee, CallingConv::ID CalleeCC, bool IsVarArg,
    const SmallVectorImpl<ISD::OutputArg> &Outs,
    const SmallVectorImpl<SDValue> &OutVals,
    const SmallVectorImpl<ISD::InputArg> &Ins, SelectionDAG &DAG) const {
  if (!mayTailCallThisCC(CalleeCC))
    return false;

  // For a divergent call target, we need to do a waterfall loop over the
  // possible callees which precludes us from using a simple jump.
  if (Callee->isDivergent())
    return false;

  MachineFunction &MF = DAG.getMachineFunction();
  const Function &CallerF = MF.getFunction();
  CallingConv::ID CallerCC = CallerF.getCallingConv();
  const SIRegisterInfo *TRI = getSubtarget()->getRegisterInfo();
  const uint32_t *CallerPreserved = TRI->getCallPreservedMask(MF, CallerCC);

  // Kernels aren't callable, and don't have a live in return address so it
  // doesn't make sense to do a tail call with entry functions.
  if (!CallerPreserved)
    return false;

  bool CCMatch = CallerCC == CalleeCC;

  if (DAG.getTarget().Options.GuaranteedTailCallOpt) {
    if (canGuaranteeTCO(CalleeCC) && CCMatch)
      return true;
    return false;
  }

  // TODO: Can we handle var args?
  if (IsVarArg)
    return false;

  for (const Argument &Arg : CallerF.args()) {
    if (Arg.hasByValAttr())
      return false;
  }

  LLVMContext &Ctx = *DAG.getContext();

  // Check that the call results are passed in the same way.
  if (!CCState::resultsCompatible(CalleeCC, CallerCC, MF, Ctx, Ins,
                                  CCAssignFnForCall(CalleeCC, IsVarArg),
                                  CCAssignFnForCall(CallerCC, IsVarArg)))
    return false;

  // The callee has to preserve all registers the caller needs to preserve.
  if (!CCMatch) {
    const uint32_t *CalleePreserved = TRI->getCallPreservedMask(MF, CalleeCC);
    if (!TRI->regmaskSubsetEqual(CallerPreserved, CalleePreserved))
      return false;
  }

  // Nothing more to check if the callee is taking no arguments.
  if (Outs.empty())
    return true;

  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CalleeCC, IsVarArg, MF, ArgLocs, Ctx);

  CCInfo.AnalyzeCallOperands(Outs, CCAssignFnForCall(CalleeCC, IsVarArg));

  const SIMachineFunctionInfo *FuncInfo = MF.getInfo<SIMachineFunctionInfo>();
  // If the stack arguments for this call do not fit into our own save area then
  // the call cannot be made tail.
  // TODO: Is this really necessary?
  if (CCInfo.getNextStackOffset() > FuncInfo->getBytesInStackArgArea())
    return false;

  const MachineRegisterInfo &MRI = MF.getRegInfo();
  return parametersInCSRMatch(MRI, CallerPreserved, ArgLocs, OutVals);
}

bool SITargetLowering::mayBeEmittedAsTailCall(const CallInst *CI) const {
  if (!CI->isTailCall())
    return false;

  const Function *ParentFn = CI->getParent()->getParent();
  if (AMDGPU::isEntryFunctionCC(ParentFn->getCallingConv()))
    return false;
  return true;
}

// The wave scratch offset register is used as the global base pointer.
SDValue SITargetLowering::LowerCall(CallLoweringInfo &CLI,
                                    SmallVectorImpl<SDValue> &InVals) const {
  SelectionDAG &DAG = CLI.DAG;
  const SDLoc &DL = CLI.DL;
  SmallVector<ISD::OutputArg, 32> &Outs = CLI.Outs;
  SmallVector<SDValue, 32> &OutVals = CLI.OutVals;
  SmallVector<ISD::InputArg, 32> &Ins = CLI.Ins;
  SDValue Chain = CLI.Chain;
  SDValue Callee = CLI.Callee;
  bool &IsTailCall = CLI.IsTailCall;
  CallingConv::ID CallConv = CLI.CallConv;
  bool IsVarArg = CLI.IsVarArg;
  bool IsSibCall = false;
  bool IsThisReturn = false;
  MachineFunction &MF = DAG.getMachineFunction();

  if (Callee.isUndef() || isNullConstant(Callee)) {
    if (!CLI.IsTailCall) {
      for (unsigned I = 0, E = CLI.Ins.size(); I != E; ++I)
        InVals.push_back(DAG.getUNDEF(CLI.Ins[I].VT));
    }

    return Chain;
  }

  if (IsVarArg) {
    return lowerUnhandledCall(CLI, InVals,
                              "unsupported call to variadic function ");
  }

  if (!CLI.CB)
    report_fatal_error("unsupported libcall legalization");

  if (IsTailCall && MF.getTarget().Options.GuaranteedTailCallOpt) {
    return lowerUnhandledCall(CLI, InVals,
                              "unsupported required tail call to function ");
  }

  if (AMDGPU::isShader(CallConv)) {
    // Note the issue is with the CC of the called function, not of the call
    // itself.
    return lowerUnhandledCall(CLI, InVals,
                              "unsupported call to a shader function ");
  }

  if (AMDGPU::isShader(MF.getFunction().getCallingConv()) &&
      CallConv != CallingConv::AMDGPU_Gfx) {
    // Only allow calls with specific calling conventions.
    return lowerUnhandledCall(CLI, InVals,
                              "unsupported calling convention for call from "
                              "graphics shader of function ");
  }

  if (IsTailCall) {
    IsTailCall = isEligibleForTailCallOptimization(
      Callee, CallConv, IsVarArg, Outs, OutVals, Ins, DAG);
    if (!IsTailCall && CLI.CB && CLI.CB->isMustTailCall()) {
      report_fatal_error("failed to perform tail call elimination on a call "
                         "site marked musttail");
    }

    bool TailCallOpt = MF.getTarget().Options.GuaranteedTailCallOpt;

    // A sibling call is one where we're under the usual C ABI and not planning
    // to change that but can still do a tail call:
    if (!TailCallOpt && IsTailCall)
      IsSibCall = true;

    if (IsTailCall)
      ++NumTailCalls;
  }

  const SIMachineFunctionInfo *Info = MF.getInfo<SIMachineFunctionInfo>();
  SmallVector<std::pair<unsigned, SDValue>, 8> RegsToPass;
  SmallVector<SDValue, 8> MemOpChains;

  // Analyze operands of the call, assigning locations to each operand.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, IsVarArg, MF, ArgLocs, *DAG.getContext());
  CCAssignFn *AssignFn = CCAssignFnForCall(CallConv, IsVarArg);

  if (CallConv != CallingConv::AMDGPU_Gfx) {
    // With a fixed ABI, allocate fixed registers before user arguments.
    passSpecialInputs(CLI, CCInfo, *Info, RegsToPass, MemOpChains, Chain);
  }

  CCInfo.AnalyzeCallOperands(Outs, AssignFn);

  // Get a count of how many bytes are to be pushed on the stack.
  unsigned NumBytes = CCInfo.getNextStackOffset();

  if (IsSibCall) {
    // Since we're not changing the ABI to make this a tail call, the memory
    // operands are already available in the caller's incoming argument space.
    NumBytes = 0;
  }

  // FPDiff is the byte offset of the call's argument area from the callee's.
  // Stores to callee stack arguments will be placed in FixedStackSlots offset
  // by this amount for a tail call. In a sibling call it must be 0 because the
  // caller will deallocate the entire stack and the callee still expects its
  // arguments to begin at SP+0. Completely unused for non-tail calls.
  int32_t FPDiff = 0;
  MachineFrameInfo &MFI = MF.getFrameInfo();

  // Adjust the stack pointer for the new arguments...
  // These operations are automatically eliminated by the prolog/epilog pass
  if (!IsSibCall) {
    Chain = DAG.getCALLSEQ_START(Chain, 0, 0, DL);

    if (!Subtarget->enableFlatScratch()) {
      SmallVector<SDValue, 4> CopyFromChains;

      // In the HSA case, this should be an identity copy.
      SDValue ScratchRSrcReg
        = DAG.getCopyFromReg(Chain, DL, Info->getScratchRSrcReg(), MVT::v4i32);
      RegsToPass.emplace_back(AMDGPU::SGPR0_SGPR1_SGPR2_SGPR3, ScratchRSrcReg);
      CopyFromChains.push_back(ScratchRSrcReg.getValue(1));
      Chain = DAG.getTokenFactor(DL, CopyFromChains);
    }
  }

  MVT PtrVT = MVT::i32;

  // Walk the register/memloc assignments, inserting copies/loads.
  for (unsigned i = 0, e = ArgLocs.size(); i != e; ++i) {
    CCValAssign &VA = ArgLocs[i];
    SDValue Arg = OutVals[i];

    // Promote the value if needed.
    switch (VA.getLocInfo()) {
    case CCValAssign::Full:
      break;
    case CCValAssign::BCvt:
      Arg = DAG.getNode(ISD::BITCAST, DL, VA.getLocVT(), Arg);
      break;
    case CCValAssign::ZExt:
      Arg = DAG.getNode(ISD::ZERO_EXTEND, DL, VA.getLocVT(), Arg);
      break;
    case CCValAssign::SExt:
      Arg = DAG.getNode(ISD::SIGN_EXTEND, DL, VA.getLocVT(), Arg);
      break;
    case CCValAssign::AExt:
      Arg = DAG.getNode(ISD::ANY_EXTEND, DL, VA.getLocVT(), Arg);
      break;
    case CCValAssign::FPExt:
      Arg = DAG.getNode(ISD::FP_EXTEND, DL, VA.getLocVT(), Arg);
      break;
    default:
      llvm_unreachable("Unknown loc info!");
    }

    if (VA.isRegLoc()) {
      RegsToPass.push_back(std::make_pair(VA.getLocReg(), Arg));
    } else {
      assert(VA.isMemLoc());

      SDValue DstAddr;
      MachinePointerInfo DstInfo;

      unsigned LocMemOffset = VA.getLocMemOffset();
      int32_t Offset = LocMemOffset;

      SDValue PtrOff = DAG.getConstant(Offset, DL, PtrVT);
      MaybeAlign Alignment;

      if (IsTailCall) {
        ISD::ArgFlagsTy Flags = Outs[i].Flags;
        unsigned OpSize = Flags.isByVal() ?
          Flags.getByValSize() : VA.getValVT().getStoreSize();

        // FIXME: We can have better than the minimum byval required alignment.
        Alignment =
            Flags.isByVal()
                ? Flags.getNonZeroByValAlign()
                : commonAlignment(Subtarget->getStackAlignment(), Offset);

        Offset = Offset + FPDiff;
        int FI = MFI.CreateFixedObject(OpSize, Offset, true);

        DstAddr = DAG.getFrameIndex(FI, PtrVT);
        DstInfo = MachinePointerInfo::getFixedStack(MF, FI);

        // Make sure any stack arguments overlapping with where we're storing
        // are loaded before this eventual operation. Otherwise they'll be
        // clobbered.

        // FIXME: Why is this really necessary? This seems to just result in a
        // lot of code to copy the stack and write them back to the same
        // locations, which are supposed to be immutable?
        Chain = addTokenForArgument(Chain, DAG, MFI, FI);
      } else {
        // Stores to the argument stack area are relative to the stack pointer.
        SDValue SP = DAG.getCopyFromReg(Chain, DL, Info->getStackPtrOffsetReg(),
                                        MVT::i32);
        DstAddr = DAG.getNode(ISD::ADD, DL, MVT::i32, SP, PtrOff);
        DstInfo = MachinePointerInfo::getStack(MF, LocMemOffset);
        Alignment =
            commonAlignment(Subtarget->getStackAlignment(), LocMemOffset);
      }

      if (Outs[i].Flags.isByVal()) {
        SDValue SizeNode =
            DAG.getConstant(Outs[i].Flags.getByValSize(), DL, MVT::i32);
        SDValue Cpy =
            DAG.getMemcpy(Chain, DL, DstAddr, Arg, SizeNode,
                          Outs[i].Flags.getNonZeroByValAlign(),
                          /*isVol = */ false, /*AlwaysInline = */ true,
                          /*isTailCall = */ false, DstInfo,
                          MachinePointerInfo(AMDGPUAS::PRIVATE_ADDRESS));

        MemOpChains.push_back(Cpy);
      } else {
        SDValue Store =
            DAG.getStore(Chain, DL, Arg, DstAddr, DstInfo, Alignment);
        MemOpChains.push_back(Store);
      }
    }
  }

  if (!MemOpChains.empty())
    Chain = DAG.getNode(ISD::TokenFactor, DL, MVT::Other, MemOpChains);

  // Build a sequence of copy-to-reg nodes chained together with token chain
  // and flag operands which copy the outgoing args into the appropriate regs.
  SDValue InFlag;
  for (auto &RegToPass : RegsToPass) {
    Chain = DAG.getCopyToReg(Chain, DL, RegToPass.first,
                             RegToPass.second, InFlag);
    InFlag = Chain.getValue(1);
  }


  SDValue PhysReturnAddrReg;
  if (IsTailCall) {
    // Since the return is being combined with the call, we need to pass on the
    // return address.

    const SIRegisterInfo *TRI = getSubtarget()->getRegisterInfo();
    SDValue ReturnAddrReg = CreateLiveInRegister(
      DAG, &AMDGPU::SReg_64RegClass, TRI->getReturnAddressReg(MF), MVT::i64);

    PhysReturnAddrReg = DAG.getRegister(TRI->getReturnAddressReg(MF),
                                        MVT::i64);
    Chain = DAG.getCopyToReg(Chain, DL, PhysReturnAddrReg, ReturnAddrReg, InFlag);
    InFlag = Chain.getValue(1);
  }

  // We don't usually want to end the call-sequence here because we would tidy
  // the frame up *after* the call, however in the ABI-changing tail-call case
  // we've carefully laid out the parameters so that when sp is reset they'll be
  // in the correct location.
  if (IsTailCall && !IsSibCall) {
    Chain = DAG.getCALLSEQ_END(Chain,
                               DAG.getTargetConstant(NumBytes, DL, MVT::i32),
                               DAG.getTargetConstant(0, DL, MVT::i32),
                               InFlag, DL);
    InFlag = Chain.getValue(1);
  }

  std::vector<SDValue> Ops;
  Ops.push_back(Chain);
  Ops.push_back(Callee);
  // Add a redundant copy of the callee global which will not be legalized, as
  // we need direct access to the callee later.
  if (GlobalAddressSDNode *GSD = dyn_cast<GlobalAddressSDNode>(Callee)) {
    const GlobalValue *GV = GSD->getGlobal();
    Ops.push_back(DAG.getTargetGlobalAddress(GV, DL, MVT::i64));
  } else {
    Ops.push_back(DAG.getTargetConstant(0, DL, MVT::i64));
  }

  if (IsTailCall) {
    // Each tail call may have to adjust the stack by a different amount, so
    // this information must travel along with the operation for eventual
    // consumption by emitEpilogue.
    Ops.push_back(DAG.getTargetConstant(FPDiff, DL, MVT::i32));

    Ops.push_back(PhysReturnAddrReg);
  }

  // Add argument registers to the end of the list so that they are known live
  // into the call.
  for (auto &RegToPass : RegsToPass) {
    Ops.push_back(DAG.getRegister(RegToPass.first,
                                  RegToPass.second.getValueType()));
  }

  // Add a register mask operand representing the call-preserved registers.

  auto *TRI = static_cast<const SIRegisterInfo*>(Subtarget->getRegisterInfo());
  const uint32_t *Mask = TRI->getCallPreservedMask(MF, CallConv);
  assert(Mask && "Missing call preserved mask for calling convention");
  Ops.push_back(DAG.getRegisterMask(Mask));

  if (InFlag.getNode())
    Ops.push_back(InFlag);

  SDVTList NodeTys = DAG.getVTList(MVT::Other, MVT::Glue);

  // If we're doing a tall call, use a TC_RETURN here rather than an
  // actual call instruction.
  if (IsTailCall) {
    MFI.setHasTailCall();
    return DAG.getNode(AMDGPUISD::TC_RETURN, DL, NodeTys, Ops);
  }

  // Returns a chain and a flag for retval copy to use.
  SDValue Call = DAG.getNode(AMDGPUISD::CALL, DL, NodeTys, Ops);
  Chain = Call.getValue(0);
  InFlag = Call.getValue(1);

  uint64_t CalleePopBytes = NumBytes;
  Chain = DAG.getCALLSEQ_END(Chain, DAG.getTargetConstant(0, DL, MVT::i32),
                             DAG.getTargetConstant(CalleePopBytes, DL, MVT::i32),
                             InFlag, DL);
  if (!Ins.empty())
    InFlag = Chain.getValue(1);

  // Handle result values, copying them out of physregs into vregs that we
  // return.
  return LowerCallResult(Chain, InFlag, CallConv, IsVarArg, Ins, DL, DAG,
                         InVals, IsThisReturn,
                         IsThisReturn ? OutVals[0] : SDValue());
}

// This is identical to the default implementation in ExpandDYNAMIC_STACKALLOC,
// except for applying the wave size scale to the increment amount.
SDValue SITargetLowering::lowerDYNAMIC_STACKALLOCImpl(
    SDValue Op, SelectionDAG &DAG) const {
  const MachineFunction &MF = DAG.getMachineFunction();
  const SIMachineFunctionInfo *Info = MF.getInfo<SIMachineFunctionInfo>();

  SDLoc dl(Op);
  EVT VT = Op.getValueType();
  SDValue Tmp1 = Op;
  SDValue Tmp2 = Op.getValue(1);
  SDValue Tmp3 = Op.getOperand(2);
  SDValue Chain = Tmp1.getOperand(0);

  Register SPReg = Info->getStackPtrOffsetReg();

  // Chain the dynamic stack allocation so that it doesn't modify the stack
  // pointer when other instructions are using the stack.
  Chain = DAG.getCALLSEQ_START(Chain, 0, 0, dl);

  SDValue Size  = Tmp2.getOperand(1);
  SDValue SP = DAG.getCopyFromReg(Chain, dl, SPReg, VT);
  Chain = SP.getValue(1);
  MaybeAlign Alignment = cast<ConstantSDNode>(Tmp3)->getMaybeAlignValue();
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  const TargetFrameLowering *TFL = ST.getFrameLowering();
  unsigned Opc =
    TFL->getStackGrowthDirection() == TargetFrameLowering::StackGrowsUp ?
    ISD::ADD : ISD::SUB;

  SDValue ScaledSize = DAG.getNode(
      ISD::SHL, dl, VT, Size,
      DAG.getConstant(ST.getWavefrontSizeLog2(), dl, MVT::i32));

  Align StackAlign = TFL->getStackAlign();
  Tmp1 = DAG.getNode(Opc, dl, VT, SP, ScaledSize); // Value
  if (Alignment && *Alignment > StackAlign) {
    Tmp1 = DAG.getNode(ISD::AND, dl, VT, Tmp1,
                       DAG.getConstant(-(uint64_t)Alignment->value()
                                           << ST.getWavefrontSizeLog2(),
                                       dl, VT));
  }

  Chain = DAG.getCopyToReg(Chain, dl, SPReg, Tmp1);    // Output chain
  Tmp2 = DAG.getCALLSEQ_END(
      Chain, DAG.getIntPtrConstant(0, dl, true),
      DAG.getIntPtrConstant(0, dl, true), SDValue(), dl);

  return DAG.getMergeValues({Tmp1, Tmp2}, dl);
}

SDValue SITargetLowering::LowerDYNAMIC_STACKALLOC(SDValue Op,
                                                  SelectionDAG &DAG) const {
  // We only handle constant sizes here to allow non-entry block, static sized
  // allocas. A truly dynamic value is more difficult to support because we
  // don't know if the size value is uniform or not. If the size isn't uniform,
  // we would need to do a wave reduction to get the maximum size to know how
  // much to increment the uniform stack pointer.
  SDValue Size = Op.getOperand(1);
  if (isa<ConstantSDNode>(Size))
      return lowerDYNAMIC_STACKALLOCImpl(Op, DAG); // Use "generic" expansion.

  return AMDGPUTargetLowering::LowerDYNAMIC_STACKALLOC(Op, DAG);
}

Register SITargetLowering::getRegisterByName(const char* RegName, LLT VT,
                                             const MachineFunction &MF) const {
  Register Reg = StringSwitch<Register>(RegName)
    .Case("m0", AMDGPU::M0)
    .Case("exec", AMDGPU::EXEC)
    .Case("exec_lo", AMDGPU::EXEC_LO)
    .Case("exec_hi", AMDGPU::EXEC_HI)
    .Case("flat_scratch", AMDGPU::FLAT_SCR)
    .Case("flat_scratch_lo", AMDGPU::FLAT_SCR_LO)
    .Case("flat_scratch_hi", AMDGPU::FLAT_SCR_HI)
    .Default(Register());

  if (Reg == AMDGPU::NoRegister) {
    report_fatal_error(Twine("invalid register name \""
                             + StringRef(RegName)  + "\"."));

  }

  if (!Subtarget->hasFlatScrRegister() &&
       Subtarget->getRegisterInfo()->regsOverlap(Reg, AMDGPU::FLAT_SCR)) {
    report_fatal_error(Twine("invalid register \""
                             + StringRef(RegName)  + "\" for subtarget."));
  }

  switch (Reg) {
  case AMDGPU::M0:
  case AMDGPU::EXEC_LO:
  case AMDGPU::EXEC_HI:
  case AMDGPU::FLAT_SCR_LO:
  case AMDGPU::FLAT_SCR_HI:
    if (VT.getSizeInBits() == 32)
      return Reg;
    break;
  case AMDGPU::EXEC:
  case AMDGPU::FLAT_SCR:
    if (VT.getSizeInBits() == 64)
      return Reg;
    break;
  default:
    llvm_unreachable("missing register type checking");
  }

  report_fatal_error(Twine("invalid type for register \""
                           + StringRef(RegName) + "\"."));
}

// If kill is not the last instruction, split the block so kill is always a
// proper terminator.
MachineBasicBlock *
SITargetLowering::splitKillBlock(MachineInstr &MI,
                                 MachineBasicBlock *BB) const {
  MachineBasicBlock *SplitBB = BB->splitAt(MI, false /*UpdateLiveIns*/);
  const SIInstrInfo *TII = getSubtarget()->getInstrInfo();
  MI.setDesc(TII->getKillTerminatorFromPseudo(MI.getOpcode()));
  return SplitBB;
}

// Split block \p MBB at \p MI, as to insert a loop. If \p InstInLoop is true,
// \p MI will be the only instruction in the loop body block. Otherwise, it will
// be the first instruction in the remainder block.
//
/// \returns { LoopBody, Remainder }
static std::pair<MachineBasicBlock *, MachineBasicBlock *>
splitBlockForLoop(MachineInstr &MI, MachineBasicBlock &MBB, bool InstInLoop) {
  MachineFunction *MF = MBB.getParent();
  MachineBasicBlock::iterator I(&MI);

  // To insert the loop we need to split the block. Move everything after this
  // point to a new block, and insert a new empty block between the two.
  MachineBasicBlock *LoopBB = MF->CreateMachineBasicBlock();
  MachineBasicBlock *RemainderBB = MF->CreateMachineBasicBlock();
  MachineFunction::iterator MBBI(MBB);
  ++MBBI;

  MF->insert(MBBI, LoopBB);
  MF->insert(MBBI, RemainderBB);

  LoopBB->addSuccessor(LoopBB);
  LoopBB->addSuccessor(RemainderBB);

  // Move the rest of the block into a new block.
  RemainderBB->transferSuccessorsAndUpdatePHIs(&MBB);

  if (InstInLoop) {
    auto Next = std::next(I);

    // Move instruction to loop body.
    LoopBB->splice(LoopBB->begin(), &MBB, I, Next);

    // Move the rest of the block.
    RemainderBB->splice(RemainderBB->begin(), &MBB, Next, MBB.end());
  } else {
    RemainderBB->splice(RemainderBB->begin(), &MBB, I, MBB.end());
  }

  MBB.addSuccessor(LoopBB);

  return std::make_pair(LoopBB, RemainderBB);
}

/// Insert \p MI into a BUNDLE with an S_WAITCNT 0 immediately following it.
void SITargetLowering::bundleInstWithWaitcnt(MachineInstr &MI) const {
  MachineBasicBlock *MBB = MI.getParent();
  const SIInstrInfo *TII = getSubtarget()->getInstrInfo();
  auto I = MI.getIterator();
  auto E = std::next(I);

  BuildMI(*MBB, E, MI.getDebugLoc(), TII->get(AMDGPU::S_WAITCNT))
    .addImm(0);

  MIBundleBuilder Bundler(*MBB, I, E);
  finalizeBundle(*MBB, Bundler.begin());
}

MachineBasicBlock *
SITargetLowering::emitGWSMemViolTestLoop(MachineInstr &MI,
                                         MachineBasicBlock *BB) const {
  const DebugLoc &DL = MI.getDebugLoc();

  MachineRegisterInfo &MRI = BB->getParent()->getRegInfo();

  MachineBasicBlock *LoopBB;
  MachineBasicBlock *RemainderBB;
  const SIInstrInfo *TII = getSubtarget()->getInstrInfo();

  // Apparently kill flags are only valid if the def is in the same block?
  if (MachineOperand *Src = TII->getNamedOperand(MI, AMDGPU::OpName::data0))
    Src->setIsKill(false);

  std::tie(LoopBB, RemainderBB) = splitBlockForLoop(MI, *BB, true);

  MachineBasicBlock::iterator I = LoopBB->end();

  const unsigned EncodedReg = AMDGPU::Hwreg::encodeHwreg(
    AMDGPU::Hwreg::ID_TRAPSTS, AMDGPU::Hwreg::OFFSET_MEM_VIOL, 1);

  // Clear TRAP_STS.MEM_VIOL
  BuildMI(*LoopBB, LoopBB->begin(), DL, TII->get(AMDGPU::S_SETREG_IMM32_B32))
    .addImm(0)
    .addImm(EncodedReg);

  bundleInstWithWaitcnt(MI);

  Register Reg = MRI.createVirtualRegister(&AMDGPU::SReg_32_XM0RegClass);

  // Load and check TRAP_STS.MEM_VIOL
  BuildMI(*LoopBB, I, DL, TII->get(AMDGPU::S_GETREG_B32), Reg)
    .addImm(EncodedReg);

  // FIXME: Do we need to use an isel pseudo that may clobber scc?
  BuildMI(*LoopBB, I, DL, TII->get(AMDGPU::S_CMP_LG_U32))
    .addReg(Reg, RegState::Kill)
    .addImm(0);
  BuildMI(*LoopBB, I, DL, TII->get(AMDGPU::S_CBRANCH_SCC1))
    .addMBB(LoopBB);

  return RemainderBB;
}

// Do a v_movrels_b32 or v_movreld_b32 for each unique value of \p IdxReg in the
// wavefront. If the value is uniform and just happens to be in a VGPR, this
// will only do one iteration. In the worst case, this will loop 64 times.
//
// TODO: Just use v_readlane_b32 if we know the VGPR has a uniform value.
static MachineBasicBlock::iterator
emitLoadM0FromVGPRLoop(const SIInstrInfo *TII, MachineRegisterInfo &MRI,
                       MachineBasicBlock &OrigBB, MachineBasicBlock &LoopBB,
                       const DebugLoc &DL, const MachineOperand &Idx,
                       unsigned InitReg, unsigned ResultReg, unsigned PhiReg,
                       unsigned InitSaveExecReg, int Offset, bool UseGPRIdxMode,
                       Register &SGPRIdxReg) {

  MachineFunction *MF = OrigBB.getParent();
  const GCNSubtarget &ST = MF->getSubtarget<GCNSubtarget>();
  const SIRegisterInfo *TRI = ST.getRegisterInfo();
  MachineBasicBlock::iterator I = LoopBB.begin();

  const TargetRegisterClass *BoolRC = TRI->getBoolRC();
  Register PhiExec = MRI.createVirtualRegister(BoolRC);
  Register NewExec = MRI.createVirtualRegister(BoolRC);
  Register CurrentIdxReg = MRI.createVirtualRegister(&AMDGPU::SGPR_32RegClass);
  Register CondReg = MRI.createVirtualRegister(BoolRC);

  BuildMI(LoopBB, I, DL, TII->get(TargetOpcode::PHI), PhiReg)
    .addReg(InitReg)
    .addMBB(&OrigBB)
    .addReg(ResultReg)
    .addMBB(&LoopBB);

  BuildMI(LoopBB, I, DL, TII->get(TargetOpcode::PHI), PhiExec)
    .addReg(InitSaveExecReg)
    .addMBB(&OrigBB)
    .addReg(NewExec)
    .addMBB(&LoopBB);

  // Read the next variant <- also loop target.
  BuildMI(LoopBB, I, DL, TII->get(AMDGPU::V_READFIRSTLANE_B32), CurrentIdxReg)
      .addReg(Idx.getReg(), getUndefRegState(Idx.isUndef()));

  // Compare the just read M0 value to all possible Idx values.
  BuildMI(LoopBB, I, DL, TII->get(AMDGPU::V_CMP_EQ_U32_e64), CondReg)
      .addReg(CurrentIdxReg)
      .addReg(Idx.getReg(), 0, Idx.getSubReg());

  // Update EXEC, save the original EXEC value to VCC.
  BuildMI(LoopBB, I, DL, TII->get(ST.isWave32() ? AMDGPU::S_AND_SAVEEXEC_B32
                                                : AMDGPU::S_AND_SAVEEXEC_B64),
          NewExec)
    .addReg(CondReg, RegState::Kill);

  MRI.setSimpleHint(NewExec, CondReg);

  if (UseGPRIdxMode) {
    if (Offset == 0) {
      SGPRIdxReg = CurrentIdxReg;
    } else {
      SGPRIdxReg = MRI.createVirtualRegister(&AMDGPU::SGPR_32RegClass);
      BuildMI(LoopBB, I, DL, TII->get(AMDGPU::S_ADD_I32), SGPRIdxReg)
          .addReg(CurrentIdxReg, RegState::Kill)
          .addImm(Offset);
    }
  } else {
    // Move index from VCC into M0
    if (Offset == 0) {
      BuildMI(LoopBB, I, DL, TII->get(AMDGPU::S_MOV_B32), AMDGPU::M0)
        .addReg(CurrentIdxReg, RegState::Kill);
    } else {
      BuildMI(LoopBB, I, DL, TII->get(AMDGPU::S_ADD_I32), AMDGPU::M0)
        .addReg(CurrentIdxReg, RegState::Kill)
        .addImm(Offset);
    }
  }

  // Update EXEC, switch all done bits to 0 and all todo bits to 1.
  unsigned Exec = ST.isWave32() ? AMDGPU::EXEC_LO : AMDGPU::EXEC;
  MachineInstr *InsertPt =
    BuildMI(LoopBB, I, DL, TII->get(ST.isWave32() ? AMDGPU::S_XOR_B32_term
                                                  : AMDGPU::S_XOR_B64_term), Exec)
      .addReg(Exec)
      .addReg(NewExec);

  // XXX - s_xor_b64 sets scc to 1 if the result is nonzero, so can we use
  // s_cbranch_scc0?

  // Loop back to V_READFIRSTLANE_B32 if there are still variants to cover.
  BuildMI(LoopBB, I, DL, TII->get(AMDGPU::S_CBRANCH_EXECNZ))
    .addMBB(&LoopBB);

  return InsertPt->getIterator();
}

// This has slightly sub-optimal regalloc when the source vector is killed by
// the read. The register allocator does not understand that the kill is
// per-workitem, so is kept alive for the whole loop so we end up not re-using a
// subregister from it, using 1 more VGPR than necessary. This was saved when
// this was expanded after register allocation.
static MachineBasicBlock::iterator
loadM0FromVGPR(const SIInstrInfo *TII, MachineBasicBlock &MBB, MachineInstr &MI,
               unsigned InitResultReg, unsigned PhiReg, int Offset,
               bool UseGPRIdxMode, Register &SGPRIdxReg) {
  MachineFunction *MF = MBB.getParent();
  const GCNSubtarget &ST = MF->getSubtarget<GCNSubtarget>();
  const SIRegisterInfo *TRI = ST.getRegisterInfo();
  MachineRegisterInfo &MRI = MF->getRegInfo();
  const DebugLoc &DL = MI.getDebugLoc();
  MachineBasicBlock::iterator I(&MI);

  const auto *BoolXExecRC = TRI->getRegClass(AMDGPU::SReg_1_XEXECRegClassID);
  Register DstReg = MI.getOperand(0).getReg();
  Register SaveExec = MRI.createVirtualRegister(BoolXExecRC);
  Register TmpExec = MRI.createVirtualRegister(BoolXExecRC);
  unsigned Exec = ST.isWave32() ? AMDGPU::EXEC_LO : AMDGPU::EXEC;
  unsigned MovExecOpc = ST.isWave32() ? AMDGPU::S_MOV_B32 : AMDGPU::S_MOV_B64;

  BuildMI(MBB, I, DL, TII->get(TargetOpcode::IMPLICIT_DEF), TmpExec);

  // Save the EXEC mask
  BuildMI(MBB, I, DL, TII->get(MovExecOpc), SaveExec)
    .addReg(Exec);

  MachineBasicBlock *LoopBB;
  MachineBasicBlock *RemainderBB;
  std::tie(LoopBB, RemainderBB) = splitBlockForLoop(MI, MBB, false);

  const MachineOperand *Idx = TII->getNamedOperand(MI, AMDGPU::OpName::idx);

  auto InsPt = emitLoadM0FromVGPRLoop(TII, MRI, MBB, *LoopBB, DL, *Idx,
                                      InitResultReg, DstReg, PhiReg, TmpExec,
                                      Offset, UseGPRIdxMode, SGPRIdxReg);

  MachineBasicBlock* LandingPad = MF->CreateMachineBasicBlock();
  MachineFunction::iterator MBBI(LoopBB);
  ++MBBI;
  MF->insert(MBBI, LandingPad);
  LoopBB->removeSuccessor(RemainderBB);
  LandingPad->addSuccessor(RemainderBB);
  LoopBB->addSuccessor(LandingPad);
  MachineBasicBlock::iterator First = LandingPad->begin();
  BuildMI(*LandingPad, First, DL, TII->get(MovExecOpc), Exec)
    .addReg(SaveExec);

  return InsPt;
}

// Returns subreg index, offset
static std::pair<unsigned, int>
computeIndirectRegAndOffset(const SIRegisterInfo &TRI,
                            const TargetRegisterClass *SuperRC,
                            unsigned VecReg,
                            int Offset) {
  int NumElts = TRI.getRegSizeInBits(*SuperRC) / 32;

  // Skip out of bounds offsets, or else we would end up using an undefined
  // register.
  if (Offset >= NumElts || Offset < 0)
    return std::make_pair(AMDGPU::sub0, Offset);

  return std::make_pair(SIRegisterInfo::getSubRegFromChannel(Offset), 0);
}

static void setM0ToIndexFromSGPR(const SIInstrInfo *TII,
                                 MachineRegisterInfo &MRI, MachineInstr &MI,
                                 int Offset) {
  MachineBasicBlock *MBB = MI.getParent();
  const DebugLoc &DL = MI.getDebugLoc();
  MachineBasicBlock::iterator I(&MI);

  const MachineOperand *Idx = TII->getNamedOperand(MI, AMDGPU::OpName::idx);

  assert(Idx->getReg() != AMDGPU::NoRegister);

  if (Offset == 0) {
    BuildMI(*MBB, I, DL, TII->get(AMDGPU::S_MOV_B32), AMDGPU::M0).add(*Idx);
  } else {
    BuildMI(*MBB, I, DL, TII->get(AMDGPU::S_ADD_I32), AMDGPU::M0)
        .add(*Idx)
        .addImm(Offset);
  }
}

static Register getIndirectSGPRIdx(const SIInstrInfo *TII,
                                   MachineRegisterInfo &MRI, MachineInstr &MI,
                                   int Offset) {
  MachineBasicBlock *MBB = MI.getParent();
  const DebugLoc &DL = MI.getDebugLoc();
  MachineBasicBlock::iterator I(&MI);

  const MachineOperand *Idx = TII->getNamedOperand(MI, AMDGPU::OpName::idx);

  if (Offset == 0)
    return Idx->getReg();

  Register Tmp = MRI.createVirtualRegister(&AMDGPU::SReg_32_XM0RegClass);
  BuildMI(*MBB, I, DL, TII->get(AMDGPU::S_ADD_I32), Tmp)
      .add(*Idx)
      .addImm(Offset);
  return Tmp;
}

static MachineBasicBlock *emitIndirectSrc(MachineInstr &MI,
                                          MachineBasicBlock &MBB,
                                          const GCNSubtarget &ST) {
  const SIInstrInfo *TII = ST.getInstrInfo();
  const SIRegisterInfo &TRI = TII->getRegisterInfo();
  MachineFunction *MF = MBB.getParent();
  MachineRegisterInfo &MRI = MF->getRegInfo();

  Register Dst = MI.getOperand(0).getReg();
  const MachineOperand *Idx = TII->getNamedOperand(MI, AMDGPU::OpName::idx);
  Register SrcReg = TII->getNamedOperand(MI, AMDGPU::OpName::src)->getReg();
  int Offset = TII->getNamedOperand(MI, AMDGPU::OpName::offset)->getImm();

  const TargetRegisterClass *VecRC = MRI.getRegClass(SrcReg);
  const TargetRegisterClass *IdxRC = MRI.getRegClass(Idx->getReg());

  unsigned SubReg;
  std::tie(SubReg, Offset)
    = computeIndirectRegAndOffset(TRI, VecRC, SrcReg, Offset);

  const bool UseGPRIdxMode = ST.useVGPRIndexMode();

  // Check for a SGPR index.
  if (TII->getRegisterInfo().isSGPRClass(IdxRC)) {
    MachineBasicBlock::iterator I(&MI);
    const DebugLoc &DL = MI.getDebugLoc();

    if (UseGPRIdxMode) {
      // TODO: Look at the uses to avoid the copy. This may require rescheduling
      // to avoid interfering with other uses, so probably requires a new
      // optimization pass.
      Register Idx = getIndirectSGPRIdx(TII, MRI, MI, Offset);

      const MCInstrDesc &GPRIDXDesc =
          TII->getIndirectGPRIDXPseudo(TRI.getRegSizeInBits(*VecRC), true);
      BuildMI(MBB, I, DL, GPRIDXDesc, Dst)
          .addReg(SrcReg)
          .addReg(Idx)
          .addImm(SubReg);
    } else {
      setM0ToIndexFromSGPR(TII, MRI, MI, Offset);

      BuildMI(MBB, I, DL, TII->get(AMDGPU::V_MOVRELS_B32_e32), Dst)
        .addReg(SrcReg, 0, SubReg)
        .addReg(SrcReg, RegState::Implicit);
    }

    MI.eraseFromParent();

    return &MBB;
  }

  // Control flow needs to be inserted if indexing with a VGPR.
  const DebugLoc &DL = MI.getDebugLoc();
  MachineBasicBlock::iterator I(&MI);

  Register PhiReg = MRI.createVirtualRegister(&AMDGPU::VGPR_32RegClass);
  Register InitReg = MRI.createVirtualRegister(&AMDGPU::VGPR_32RegClass);

  BuildMI(MBB, I, DL, TII->get(TargetOpcode::IMPLICIT_DEF), InitReg);

  Register SGPRIdxReg;
  auto InsPt = loadM0FromVGPR(TII, MBB, MI, InitReg, PhiReg, Offset,
                              UseGPRIdxMode, SGPRIdxReg);

  MachineBasicBlock *LoopBB = InsPt->getParent();

  if (UseGPRIdxMode) {
    const MCInstrDesc &GPRIDXDesc =
        TII->getIndirectGPRIDXPseudo(TRI.getRegSizeInBits(*VecRC), true);

    BuildMI(*LoopBB, InsPt, DL, GPRIDXDesc, Dst)
        .addReg(SrcReg)
        .addReg(SGPRIdxReg)
        .addImm(SubReg);
  } else {
    BuildMI(*LoopBB, InsPt, DL, TII->get(AMDGPU::V_MOVRELS_B32_e32), Dst)
      .addReg(SrcReg, 0, SubReg)
      .addReg(SrcReg, RegState::Implicit);
  }

  MI.eraseFromParent();

  return LoopBB;
}

static MachineBasicBlock *emitIndirectDst(MachineInstr &MI,
                                          MachineBasicBlock &MBB,
                                          const GCNSubtarget &ST) {
  const SIInstrInfo *TII = ST.getInstrInfo();
  const SIRegisterInfo &TRI = TII->getRegisterInfo();
  MachineFunction *MF = MBB.getParent();
  MachineRegisterInfo &MRI = MF->getRegInfo();

  Register Dst = MI.getOperand(0).getReg();
  const MachineOperand *SrcVec = TII->getNamedOperand(MI, AMDGPU::OpName::src);
  const MachineOperand *Idx = TII->getNamedOperand(MI, AMDGPU::OpName::idx);
  const MachineOperand *Val = TII->getNamedOperand(MI, AMDGPU::OpName::val);
  int Offset = TII->getNamedOperand(MI, AMDGPU::OpName::offset)->getImm();
  const TargetRegisterClass *VecRC = MRI.getRegClass(SrcVec->getReg());
  const TargetRegisterClass *IdxRC = MRI.getRegClass(Idx->getReg());

  // This can be an immediate, but will be folded later.
  assert(Val->getReg());

  unsigned SubReg;
  std::tie(SubReg, Offset) = computeIndirectRegAndOffset(TRI, VecRC,
                                                         SrcVec->getReg(),
                                                         Offset);
  const bool UseGPRIdxMode = ST.useVGPRIndexMode();

  if (Idx->getReg() == AMDGPU::NoRegister) {
    MachineBasicBlock::iterator I(&MI);
    const DebugLoc &DL = MI.getDebugLoc();

    assert(Offset == 0);

    BuildMI(MBB, I, DL, TII->get(TargetOpcode::INSERT_SUBREG), Dst)
        .add(*SrcVec)
        .add(*Val)
        .addImm(SubReg);

    MI.eraseFromParent();
    return &MBB;
  }

  // Check for a SGPR index.
  if (TII->getRegisterInfo().isSGPRClass(IdxRC)) {
    MachineBasicBlock::iterator I(&MI);
    const DebugLoc &DL = MI.getDebugLoc();

    if (UseGPRIdxMode) {
      Register Idx = getIndirectSGPRIdx(TII, MRI, MI, Offset);

      const MCInstrDesc &GPRIDXDesc =
          TII->getIndirectGPRIDXPseudo(TRI.getRegSizeInBits(*VecRC), false);
      BuildMI(MBB, I, DL, GPRIDXDesc, Dst)
          .addReg(SrcVec->getReg())
          .add(*Val)
          .addReg(Idx)
          .addImm(SubReg);
    } else {
      setM0ToIndexFromSGPR(TII, MRI, MI, Offset);

      const MCInstrDesc &MovRelDesc = TII->getIndirectRegWriteMovRelPseudo(
          TRI.getRegSizeInBits(*VecRC), 32, false);
      BuildMI(MBB, I, DL, MovRelDesc, Dst)
          .addReg(SrcVec->getReg())
          .add(*Val)
          .addImm(SubReg);
    }
    MI.eraseFromParent();
    return &MBB;
  }

  // Control flow needs to be inserted if indexing with a VGPR.
  if (Val->isReg())
    MRI.clearKillFlags(Val->getReg());

  const DebugLoc &DL = MI.getDebugLoc();

  Register PhiReg = MRI.createVirtualRegister(VecRC);

  Register SGPRIdxReg;
  auto InsPt = loadM0FromVGPR(TII, MBB, MI, SrcVec->getReg(), PhiReg, Offset,
                              UseGPRIdxMode, SGPRIdxReg);
  MachineBasicBlock *LoopBB = InsPt->getParent();

  if (UseGPRIdxMode) {
    const MCInstrDesc &GPRIDXDesc =
        TII->getIndirectGPRIDXPseudo(TRI.getRegSizeInBits(*VecRC), false);

    BuildMI(*LoopBB, InsPt, DL, GPRIDXDesc, Dst)
        .addReg(PhiReg)
        .add(*Val)
        .addReg(SGPRIdxReg)
        .addImm(AMDGPU::sub0);
  } else {
    const MCInstrDesc &MovRelDesc = TII->getIndirectRegWriteMovRelPseudo(
        TRI.getRegSizeInBits(*VecRC), 32, false);
    BuildMI(*LoopBB, InsPt, DL, MovRelDesc, Dst)
        .addReg(PhiReg)
        .add(*Val)
        .addImm(AMDGPU::sub0);
  }

  MI.eraseFromParent();
  return LoopBB;
}

MachineBasicBlock *SITargetLowering::EmitInstrWithCustomInserter(
  MachineInstr &MI, MachineBasicBlock *BB) const {

  const SIInstrInfo *TII = getSubtarget()->getInstrInfo();
  MachineFunction *MF = BB->getParent();
  SIMachineFunctionInfo *MFI = MF->getInfo<SIMachineFunctionInfo>();

  switch (MI.getOpcode()) {
  case AMDGPU::S_UADDO_PSEUDO:
  case AMDGPU::S_USUBO_PSEUDO: {
    const DebugLoc &DL = MI.getDebugLoc();
    MachineOperand &Dest0 = MI.getOperand(0);
    MachineOperand &Dest1 = MI.getOperand(1);
    MachineOperand &Src0 = MI.getOperand(2);
    MachineOperand &Src1 = MI.getOperand(3);

    unsigned Opc = (MI.getOpcode() == AMDGPU::S_UADDO_PSEUDO)
                       ? AMDGPU::S_ADD_I32
                       : AMDGPU::S_SUB_I32;
    BuildMI(*BB, MI, DL, TII->get(Opc), Dest0.getReg()).add(Src0).add(Src1);

    BuildMI(*BB, MI, DL, TII->get(AMDGPU::S_CSELECT_B64), Dest1.getReg())
        .addImm(1)
        .addImm(0);

    MI.eraseFromParent();
    return BB;
  }
  case AMDGPU::S_ADD_U64_PSEUDO:
  case AMDGPU::S_SUB_U64_PSEUDO: {
    MachineRegisterInfo &MRI = BB->getParent()->getRegInfo();
    const GCNSubtarget &ST = MF->getSubtarget<GCNSubtarget>();
    const SIRegisterInfo *TRI = ST.getRegisterInfo();
    const TargetRegisterClass *BoolRC = TRI->getBoolRC();
    const DebugLoc &DL = MI.getDebugLoc();

    MachineOperand &Dest = MI.getOperand(0);
    MachineOperand &Src0 = MI.getOperand(1);
    MachineOperand &Src1 = MI.getOperand(2);

    Register DestSub0 = MRI.createVirtualRegister(&AMDGPU::SReg_32RegClass);
    Register DestSub1 = MRI.createVirtualRegister(&AMDGPU::SReg_32RegClass);

    MachineOperand Src0Sub0 = TII->buildExtractSubRegOrImm(
        MI, MRI, Src0, BoolRC, AMDGPU::sub0, &AMDGPU::SReg_32RegClass);
    MachineOperand Src0Sub1 = TII->buildExtractSubRegOrImm(
        MI, MRI, Src0, BoolRC, AMDGPU::sub1, &AMDGPU::SReg_32RegClass);

    MachineOperand Src1Sub0 = TII->buildExtractSubRegOrImm(
        MI, MRI, Src1, BoolRC, AMDGPU::sub0, &AMDGPU::SReg_32RegClass);
    MachineOperand Src1Sub1 = TII->buildExtractSubRegOrImm(
        MI, MRI, Src1, BoolRC, AMDGPU::sub1, &AMDGPU::SReg_32RegClass);

    bool IsAdd = (MI.getOpcode() == AMDGPU::S_ADD_U64_PSEUDO);

    unsigned LoOpc = IsAdd ? AMDGPU::S_ADD_U32 : AMDGPU::S_SUB_U32;
    unsigned HiOpc = IsAdd ? AMDGPU::S_ADDC_U32 : AMDGPU::S_SUBB_U32;
    BuildMI(*BB, MI, DL, TII->get(LoOpc), DestSub0).add(Src0Sub0).add(Src1Sub0);
    BuildMI(*BB, MI, DL, TII->get(HiOpc), DestSub1).add(Src0Sub1).add(Src1Sub1);
    BuildMI(*BB, MI, DL, TII->get(TargetOpcode::REG_SEQUENCE), Dest.getReg())
        .addReg(DestSub0)
        .addImm(AMDGPU::sub0)
        .addReg(DestSub1)
        .addImm(AMDGPU::sub1);
    MI.eraseFromParent();
    return BB;
  }
  case AMDGPU::V_ADD_U64_PSEUDO:
  case AMDGPU::V_SUB_U64_PSEUDO: {
    MachineRegisterInfo &MRI = BB->getParent()->getRegInfo();
    const GCNSubtarget &ST = MF->getSubtarget<GCNSubtarget>();
    const SIRegisterInfo *TRI = ST.getRegisterInfo();
    const DebugLoc &DL = MI.getDebugLoc();

    bool IsAdd = (MI.getOpcode() == AMDGPU::V_ADD_U64_PSEUDO);

    const auto *CarryRC = TRI->getRegClass(AMDGPU::SReg_1_XEXECRegClassID);

    Register DestSub0 = MRI.createVirtualRegister(&AMDGPU::VGPR_32RegClass);
    Register DestSub1 = MRI.createVirtualRegister(&AMDGPU::VGPR_32RegClass);

    Register CarryReg = MRI.createVirtualRegister(CarryRC);
    Register DeadCarryReg = MRI.createVirtualRegister(CarryRC);

    MachineOperand &Dest = MI.getOperand(0);
    MachineOperand &Src0 = MI.getOperand(1);
    MachineOperand &Src1 = MI.getOperand(2);

    const TargetRegisterClass *Src0RC = Src0.isReg()
                                            ? MRI.getRegClass(Src0.getReg())
                                            : &AMDGPU::VReg_64RegClass;
    const TargetRegisterClass *Src1RC = Src1.isReg()
                                            ? MRI.getRegClass(Src1.getReg())
                                            : &AMDGPU::VReg_64RegClass;

    const TargetRegisterClass *Src0SubRC =
        TRI->getSubRegClass(Src0RC, AMDGPU::sub0);
    const TargetRegisterClass *Src1SubRC =
        TRI->getSubRegClass(Src1RC, AMDGPU::sub1);

    MachineOperand SrcReg0Sub0 = TII->buildExtractSubRegOrImm(
        MI, MRI, Src0, Src0RC, AMDGPU::sub0, Src0SubRC);
    MachineOperand SrcReg1Sub0 = TII->buildExtractSubRegOrImm(
        MI, MRI, Src1, Src1RC, AMDGPU::sub0, Src1SubRC);

    MachineOperand SrcReg0Sub1 = TII->buildExtractSubRegOrImm(
        MI, MRI, Src0, Src0RC, AMDGPU::sub1, Src0SubRC);
    MachineOperand SrcReg1Sub1 = TII->buildExtractSubRegOrImm(
        MI, MRI, Src1, Src1RC, AMDGPU::sub1, Src1SubRC);

    unsigned LoOpc = IsAdd ? AMDGPU::V_ADD_CO_U32_e64 : AMDGPU::V_SUB_CO_U32_e64;
    MachineInstr *LoHalf = BuildMI(*BB, MI, DL, TII->get(LoOpc), DestSub0)
                               .addReg(CarryReg, RegState::Define)
                               .add(SrcReg0Sub0)
                               .add(SrcReg1Sub0)
                               .addImm(0); // clamp bit

    unsigned HiOpc = IsAdd ? AMDGPU::V_ADDC_U32_e64 : AMDGPU::V_SUBB_U32_e64;
    MachineInstr *HiHalf =
        BuildMI(*BB, MI, DL, TII->get(HiOpc), DestSub1)
            .addReg(DeadCarryReg, RegState::Define | RegState::Dead)
            .add(SrcReg0Sub1)
            .add(SrcReg1Sub1)
            .addReg(CarryReg, RegState::Kill)
            .addImm(0); // clamp bit

    BuildMI(*BB, MI, DL, TII->get(TargetOpcode::REG_SEQUENCE), Dest.getReg())
        .addReg(DestSub0)
        .addImm(AMDGPU::sub0)
        .addReg(DestSub1)
        .addImm(AMDGPU::sub1);
    TII->legalizeOperands(*LoHalf);
    TII->legalizeOperands(*HiHalf);
    MI.eraseFromParent();
    return BB;
  }
  case AMDGPU::S_ADD_CO_PSEUDO:
  case AMDGPU::S_SUB_CO_PSEUDO: {
    // This pseudo has a chance to be selected
    // only from uniform add/subcarry node. All the VGPR operands
    // therefore assumed to be splat vectors.
    MachineRegisterInfo &MRI = BB->getParent()->getRegInfo();
    const GCNSubtarget &ST = MF->getSubtarget<GCNSubtarget>();
    const SIRegisterInfo *TRI = ST.getRegisterInfo();
    MachineBasicBlock::iterator MII = MI;
    const DebugLoc &DL = MI.getDebugLoc();
    MachineOperand &Dest = MI.getOperand(0);
    MachineOperand &CarryDest = MI.getOperand(1);
    MachineOperand &Src0 = MI.getOperand(2);
    MachineOperand &Src1 = MI.getOperand(3);
    MachineOperand &Src2 = MI.getOperand(4);
    unsigned Opc = (MI.getOpcode() == AMDGPU::S_ADD_CO_PSEUDO)
                       ? AMDGPU::S_ADDC_U32
                       : AMDGPU::S_SUBB_U32;
    if (Src0.isReg() && TRI->isVectorRegister(MRI, Src0.getReg())) {
      Register RegOp0 = MRI.createVirtualRegister(&AMDGPU::SReg_32RegClass);
      BuildMI(*BB, MII, DL, TII->get(AMDGPU::V_READFIRSTLANE_B32), RegOp0)
          .addReg(Src0.getReg());
      Src0.setReg(RegOp0);
    }
    if (Src1.isReg() && TRI->isVectorRegister(MRI, Src1.getReg())) {
      Register RegOp1 = MRI.createVirtualRegister(&AMDGPU::SReg_32RegClass);
      BuildMI(*BB, MII, DL, TII->get(AMDGPU::V_READFIRSTLANE_B32), RegOp1)
          .addReg(Src1.getReg());
      Src1.setReg(RegOp1);
    }
    Register RegOp2 = MRI.createVirtualRegister(&AMDGPU::SReg_32RegClass);
    if (TRI->isVectorRegister(MRI, Src2.getReg())) {
      BuildMI(*BB, MII, DL, TII->get(AMDGPU::V_READFIRSTLANE_B32), RegOp2)
          .addReg(Src2.getReg());
      Src2.setReg(RegOp2);
    }

    const TargetRegisterClass *Src2RC = MRI.getRegClass(Src2.getReg());
    unsigned WaveSize = TRI->getRegSizeInBits(*Src2RC);
    assert(WaveSize == 64 || WaveSize == 32);

    if (WaveSize == 64) {
      if (ST.hasScalarCompareEq64()) {
        BuildMI(*BB, MII, DL, TII->get(AMDGPU::S_CMP_LG_U64))
            .addReg(Src2.getReg())
            .addImm(0);
      } else {
        const TargetRegisterClass *SubRC =
            TRI->getSubRegClass(Src2RC, AMDGPU::sub0);
        MachineOperand Src2Sub0 = TII->buildExtractSubRegOrImm(
            MII, MRI, Src2, Src2RC, AMDGPU::sub0, SubRC);
        MachineOperand Src2Sub1 = TII->buildExtractSubRegOrImm(
            MII, MRI, Src2, Src2RC, AMDGPU::sub1, SubRC);
        Register Src2_32 = MRI.createVirtualRegister(&AMDGPU::SReg_32RegClass);

        BuildMI(*BB, MII, DL, TII->get(AMDGPU::S_OR_B32), Src2_32)
            .add(Src2Sub0)
            .add(Src2Sub1);

        BuildMI(*BB, MII, DL, TII->get(AMDGPU::S_CMP_LG_U32))
            .addReg(Src2_32, RegState::Kill)
            .addImm(0);
      }
    } else {
      BuildMI(*BB, MII, DL, TII->get(AMDGPU::S_CMPK_LG_U32))
          .addReg(Src2.getReg())
          .addImm(0);
    }

    BuildMI(*BB, MII, DL, TII->get(Opc), Dest.getReg()).add(Src0).add(Src1);

    unsigned SelOpc =
        (WaveSize == 64) ? AMDGPU::S_CSELECT_B64 : AMDGPU::S_CSELECT_B32;

    BuildMI(*BB, MII, DL, TII->get(SelOpc), CarryDest.getReg())
        .addImm(-1)
        .addImm(0);

    MI.eraseFromParent();
    return BB;
  }
  case AMDGPU::SI_INIT_M0: {
    BuildMI(*BB, MI.getIterator(), MI.getDebugLoc(),
            TII->get(AMDGPU::S_MOV_B32), AMDGPU::M0)
        .add(MI.getOperand(0));
    MI.eraseFromParent();
    return BB;
  }
  case AMDGPU::GET_GROUPSTATICSIZE: {
    assert(getTargetMachine().getTargetTriple().getOS() == Triple::AMDHSA ||
           getTargetMachine().getTargetTriple().getOS() == Triple::AMDPAL);
    DebugLoc DL = MI.getDebugLoc();
    BuildMI(*BB, MI, DL, TII->get(AMDGPU::S_MOV_B32))
        .add(MI.getOperand(0))
        .addImm(MFI->getLDSSize());
    MI.eraseFromParent();
    return BB;
  }
  case AMDGPU::SI_INDIRECT_SRC_V1:
  case AMDGPU::SI_INDIRECT_SRC_V2:
  case AMDGPU::SI_INDIRECT_SRC_V4:
  case AMDGPU::SI_INDIRECT_SRC_V8:
  case AMDGPU::SI_INDIRECT_SRC_V16:
  case AMDGPU::SI_INDIRECT_SRC_V32:
    return emitIndirectSrc(MI, *BB, *getSubtarget());
  case AMDGPU::SI_INDIRECT_DST_V1:
  case AMDGPU::SI_INDIRECT_DST_V2:
  case AMDGPU::SI_INDIRECT_DST_V4:
  case AMDGPU::SI_INDIRECT_DST_V8:
  case AMDGPU::SI_INDIRECT_DST_V16:
  case AMDGPU::SI_INDIRECT_DST_V32:
    return emitIndirectDst(MI, *BB, *getSubtarget());
  case AMDGPU::SI_KILL_F32_COND_IMM_PSEUDO:
  case AMDGPU::SI_KILL_I1_PSEUDO:
    return splitKillBlock(MI, BB);
  case AMDGPU::V_CNDMASK_B64_PSEUDO: {
    MachineRegisterInfo &MRI = BB->getParent()->getRegInfo();
    const GCNSubtarget &ST = MF->getSubtarget<GCNSubtarget>();
    const SIRegisterInfo *TRI = ST.getRegisterInfo();

    Register Dst = MI.getOperand(0).getReg();
    Register Src0 = MI.getOperand(1).getReg();
    Register Src1 = MI.getOperand(2).getReg();
    const DebugLoc &DL = MI.getDebugLoc();
    Register SrcCond = MI.getOperand(3).getReg();

    Register DstLo = MRI.createVirtualRegister(&AMDGPU::VGPR_32RegClass);
    Register DstHi = MRI.createVirtualRegister(&AMDGPU::VGPR_32RegClass);
    const auto *CondRC = TRI->getRegClass(AMDGPU::SReg_1_XEXECRegClassID);
    Register SrcCondCopy = MRI.createVirtualRegister(CondRC);

    BuildMI(*BB, MI, DL, TII->get(AMDGPU::COPY), SrcCondCopy)
      .addReg(SrcCond);
    BuildMI(*BB, MI, DL, TII->get(AMDGPU::V_CNDMASK_B32_e64), DstLo)
      .addImm(0)
      .addReg(Src0, 0, AMDGPU::sub0)
      .addImm(0)
      .addReg(Src1, 0, AMDGPU::sub0)
      .addReg(SrcCondCopy);
    BuildMI(*BB, MI, DL, TII->get(AMDGPU::V_CNDMASK_B32_e64), DstHi)
      .addImm(0)
      .addReg(Src0, 0, AMDGPU::sub1)
      .addImm(0)
      .addReg(Src1, 0, AMDGPU::sub1)
      .addReg(SrcCondCopy);

    BuildMI(*BB, MI, DL, TII->get(AMDGPU::REG_SEQUENCE), Dst)
      .addReg(DstLo)
      .addImm(AMDGPU::sub0)
      .addReg(DstHi)
      .addImm(AMDGPU::sub1);
    MI.eraseFromParent();
    return BB;
  }
  case AMDGPU::SI_BR_UNDEF: {
    const SIInstrInfo *TII = getSubtarget()->getInstrInfo();
    const DebugLoc &DL = MI.getDebugLoc();
    MachineInstr *Br = BuildMI(*BB, MI, DL, TII->get(AMDGPU::S_CBRANCH_SCC1))
                           .add(MI.getOperand(0));
    Br->getOperand(1).setIsUndef(true); // read undef SCC
    MI.eraseFromParent();
    return BB;
  }
  case AMDGPU::ADJCALLSTACKUP:
  case AMDGPU::ADJCALLSTACKDOWN: {
    const SIMachineFunctionInfo *Info = MF->getInfo<SIMachineFunctionInfo>();
    MachineInstrBuilder MIB(*MF, &MI);
    MIB.addReg(Info->getStackPtrOffsetReg(), RegState::ImplicitDefine)
       .addReg(Info->getStackPtrOffsetReg(), RegState::Implicit);
    return BB;
  }
  case AMDGPU::SI_CALL_ISEL: {
    const SIInstrInfo *TII = getSubtarget()->getInstrInfo();
    const DebugLoc &DL = MI.getDebugLoc();

    unsigned ReturnAddrReg = TII->getRegisterInfo().getReturnAddressReg(*MF);

    MachineInstrBuilder MIB;
    MIB = BuildMI(*BB, MI, DL, TII->get(AMDGPU::SI_CALL), ReturnAddrReg);

    for (const MachineOperand &MO : MI.operands())
      MIB.add(MO);

    MIB.cloneMemRefs(MI);
    MI.eraseFromParent();
    return BB;
  }
  case AMDGPU::V_ADD_CO_U32_e32:
  case AMDGPU::V_SUB_CO_U32_e32:
  case AMDGPU::V_SUBREV_CO_U32_e32: {
    // TODO: Define distinct V_*_I32_Pseudo instructions instead.
    const DebugLoc &DL = MI.getDebugLoc();
    unsigned Opc = MI.getOpcode();

    bool NeedClampOperand = false;
    if (TII->pseudoToMCOpcode(Opc) == -1) {
      Opc = AMDGPU::getVOPe64(Opc);
      NeedClampOperand = true;
    }

    auto I = BuildMI(*BB, MI, DL, TII->get(Opc), MI.getOperand(0).getReg());
    if (TII->isVOP3(*I)) {
      const GCNSubtarget &ST = MF->getSubtarget<GCNSubtarget>();
      const SIRegisterInfo *TRI = ST.getRegisterInfo();
      I.addReg(TRI->getVCC(), RegState::Define);
    }
    I.add(MI.getOperand(1))
     .add(MI.getOperand(2));
    if (NeedClampOperand)
      I.addImm(0); // clamp bit for e64 encoding

    TII->legalizeOperands(*I);

    MI.eraseFromParent();
    return BB;
  }
  case AMDGPU::V_ADDC_U32_e32:
  case AMDGPU::V_SUBB_U32_e32:
  case AMDGPU::V_SUBBREV_U32_e32:
    // These instructions have an implicit use of vcc which counts towards the
    // constant bus limit.
    TII->legalizeOperands(MI);
    return BB;
  case AMDGPU::DS_GWS_INIT:
  case AMDGPU::DS_GWS_SEMA_BR:
  case AMDGPU::DS_GWS_BARRIER:
    if (Subtarget->needsAlignedVGPRs()) {
      // Add implicit aligned super-reg to force alignment on the data operand.
      const DebugLoc &DL = MI.getDebugLoc();
      MachineRegisterInfo &MRI = BB->getParent()->getRegInfo();
      const SIRegisterInfo *TRI = Subtarget->getRegisterInfo();
      MachineOperand *Op = TII->getNamedOperand(MI, AMDGPU::OpName::data0);
      Register DataReg = Op->getReg();
      bool IsAGPR = TRI->isAGPR(MRI, DataReg);
      Register Undef = MRI.createVirtualRegister(
          IsAGPR ? &AMDGPU::AGPR_32RegClass : &AMDGPU::VGPR_32RegClass);
      BuildMI(*BB, MI, DL, TII->get(AMDGPU::IMPLICIT_DEF), Undef);
      Register NewVR =
          MRI.createVirtualRegister(IsAGPR ? &AMDGPU::AReg_64_Align2RegClass
                                           : &AMDGPU::VReg_64_Align2RegClass);
      BuildMI(*BB, MI, DL, TII->get(AMDGPU::REG_SEQUENCE), NewVR)
          .addReg(DataReg, 0, Op->getSubReg())
          .addImm(AMDGPU::sub0)
          .addReg(Undef)
          .addImm(AMDGPU::sub1);
      Op->setReg(NewVR);
      Op->setSubReg(AMDGPU::sub0);
      MI.addOperand(MachineOperand::CreateReg(NewVR, false, true));
    }
    LLVM_FALLTHROUGH;
  case AMDGPU::DS_GWS_SEMA_V:
  case AMDGPU::DS_GWS_SEMA_P:
  case AMDGPU::DS_GWS_SEMA_RELEASE_ALL:
    // A s_waitcnt 0 is required to be the instruction immediately following.
    if (getSubtarget()->hasGWSAutoReplay()) {
      bundleInstWithWaitcnt(MI);
      return BB;
    }

    return emitGWSMemViolTestLoop(MI, BB);
  case AMDGPU::S_SETREG_B32: {
    // Try to optimize cases that only set the denormal mode or rounding mode.
    //
    // If the s_setreg_b32 fully sets all of the bits in the rounding mode or
    // denormal mode to a constant, we can use s_round_mode or s_denorm_mode
    // instead.
    //
    // FIXME: This could be predicates on the immediate, but tablegen doesn't
    // allow you to have a no side effect instruction in the output of a
    // sideeffecting pattern.
    unsigned ID, Offset, Width;
    AMDGPU::Hwreg::decodeHwreg(MI.getOperand(1).getImm(), ID, Offset, Width);
    if (ID != AMDGPU::Hwreg::ID_MODE)
      return BB;

    const unsigned WidthMask = maskTrailingOnes<unsigned>(Width);
    const unsigned SetMask = WidthMask << Offset;

    if (getSubtarget()->hasDenormModeInst()) {
      unsigned SetDenormOp = 0;
      unsigned SetRoundOp = 0;

      // The dedicated instructions can only set the whole denorm or round mode
      // at once, not a subset of bits in either.
      if (SetMask ==
          (AMDGPU::Hwreg::FP_ROUND_MASK | AMDGPU::Hwreg::FP_DENORM_MASK)) {
        // If this fully sets both the round and denorm mode, emit the two
        // dedicated instructions for these.
        SetRoundOp = AMDGPU::S_ROUND_MODE;
        SetDenormOp = AMDGPU::S_DENORM_MODE;
      } else if (SetMask == AMDGPU::Hwreg::FP_ROUND_MASK) {
        SetRoundOp = AMDGPU::S_ROUND_MODE;
      } else if (SetMask == AMDGPU::Hwreg::FP_DENORM_MASK) {
        SetDenormOp = AMDGPU::S_DENORM_MODE;
      }

      if (SetRoundOp || SetDenormOp) {
        MachineRegisterInfo &MRI = BB->getParent()->getRegInfo();
        MachineInstr *Def = MRI.getVRegDef(MI.getOperand(0).getReg());
        if (Def && Def->isMoveImmediate() && Def->getOperand(1).isImm()) {
          unsigned ImmVal = Def->getOperand(1).getImm();
          if (SetRoundOp) {
            BuildMI(*BB, MI, MI.getDebugLoc(), TII->get(SetRoundOp))
                .addImm(ImmVal & 0xf);

            // If we also have the denorm mode, get just the denorm mode bits.
            ImmVal >>= 4;
          }

          if (SetDenormOp) {
            BuildMI(*BB, MI, MI.getDebugLoc(), TII->get(SetDenormOp))
                .addImm(ImmVal & 0xf);
          }

          MI.eraseFromParent();
          return BB;
        }
      }
    }

    // If only FP bits are touched, used the no side effects pseudo.
    if ((SetMask & (AMDGPU::Hwreg::FP_ROUND_MASK |
                    AMDGPU::Hwreg::FP_DENORM_MASK)) == SetMask)
      MI.setDesc(TII->get(AMDGPU::S_SETREG_B32_mode));

    return BB;
  }
  default:
    return AMDGPUTargetLowering::EmitInstrWithCustomInserter(MI, BB);
  }
}

bool SITargetLowering::hasBitPreservingFPLogic(EVT VT) const {
  return isTypeLegal(VT.getScalarType());
}

bool SITargetLowering::enableAggressiveFMAFusion(EVT VT) const {
  // This currently forces unfolding various combinations of fsub into fma with
  // free fneg'd operands. As long as we have fast FMA (controlled by
  // isFMAFasterThanFMulAndFAdd), we should perform these.

  // When fma is quarter rate, for f64 where add / sub are at best half rate,
  // most of these combines appear to be cycle neutral but save on instruction
  // count / code size.
  return true;
}

bool SITargetLowering::enableAggressiveFMAFusion(LLT Ty) const { return true; }

EVT SITargetLowering::getSetCCResultType(const DataLayout &DL, LLVMContext &Ctx,
                                         EVT VT) const {
  if (!VT.isVector()) {
    return MVT::i1;
  }
  return EVT::getVectorVT(Ctx, MVT::i1, VT.getVectorNumElements());
}

MVT SITargetLowering::getScalarShiftAmountTy(const DataLayout &, EVT VT) const {
  // TODO: Should i16 be used always if legal? For now it would force VALU
  // shifts.
  return (VT == MVT::i16) ? MVT::i16 : MVT::i32;
}

LLT SITargetLowering::getPreferredShiftAmountTy(LLT Ty) const {
  return (Ty.getScalarSizeInBits() <= 16 && Subtarget->has16BitInsts())
             ? Ty.changeElementSize(16)
             : Ty.changeElementSize(32);
}

// Answering this is somewhat tricky and depends on the specific device which
// have different rates for fma or all f64 operations.
//
// v_fma_f64 and v_mul_f64 always take the same number of cycles as each other
// regardless of which device (although the number of cycles differs between
// devices), so it is always profitable for f64.
//
// v_fma_f32 takes 4 or 16 cycles depending on the device, so it is profitable
// only on full rate devices. Normally, we should prefer selecting v_mad_f32
// which we can always do even without fused FP ops since it returns the same
// result as the separate operations and since it is always full
// rate. Therefore, we lie and report that it is not faster for f32. v_mad_f32
// however does not support denormals, so we do report fma as faster if we have
// a fast fma device and require denormals.
//
bool SITargetLowering::isFMAFasterThanFMulAndFAdd(const MachineFunction &MF,
                                                  EVT VT) const {
  VT = VT.getScalarType();

  switch (VT.getSimpleVT().SimpleTy) {
  case MVT::f32: {
    // If mad is not available this depends only on if f32 fma is full rate.
    if (!Subtarget->hasMadMacF32Insts())
      return Subtarget->hasFastFMAF32();

    // Otherwise f32 mad is always full rate and returns the same result as
    // the separate operations so should be preferred over fma.
    // However does not support denomals.
    if (hasFP32Denormals(MF))
      return Subtarget->hasFastFMAF32() || Subtarget->hasDLInsts();

    // If the subtarget has v_fmac_f32, that's just as good as v_mac_f32.
    return Subtarget->hasFastFMAF32() && Subtarget->hasDLInsts();
  }
  case MVT::f64:
    return true;
  case MVT::f16:
    return Subtarget->has16BitInsts() && hasFP64FP16Denormals(MF);
  default:
    break;
  }

  return false;
}

bool SITargetLowering::isFMAFasterThanFMulAndFAdd(const MachineFunction &MF,
                                                  LLT Ty) const {
  switch (Ty.getScalarSizeInBits()) {
  case 16:
    return isFMAFasterThanFMulAndFAdd(MF, MVT::f16);
  case 32:
    return isFMAFasterThanFMulAndFAdd(MF, MVT::f32);
  case 64:
    return isFMAFasterThanFMulAndFAdd(MF, MVT::f64);
  default:
    break;
  }

  return false;
}

bool SITargetLowering::isFMADLegal(const MachineInstr &MI, LLT Ty) const {
  if (!Ty.isScalar())
    return false;

  if (Ty.getScalarSizeInBits() == 16)
    return Subtarget->hasMadF16() && !hasFP64FP16Denormals(*MI.getMF());
  if (Ty.getScalarSizeInBits() == 32)
    return Subtarget->hasMadMacF32Insts() && !hasFP32Denormals(*MI.getMF());

  return false;
}

bool SITargetLowering::isFMADLegal(const SelectionDAG &DAG,
                                   const SDNode *N) const {
  // TODO: Check future ftz flag
  // v_mad_f32/v_mac_f32 do not support denormals.
  EVT VT = N->getValueType(0);
  if (VT == MVT::f32)
    return Subtarget->hasMadMacF32Insts() &&
           !hasFP32Denormals(DAG.getMachineFunction());
  if (VT == MVT::f16) {
    return Subtarget->hasMadF16() &&
           !hasFP64FP16Denormals(DAG.getMachineFunction());
  }

  return false;
}

//===----------------------------------------------------------------------===//
// Custom DAG Lowering Operations
//===----------------------------------------------------------------------===//

// Work around LegalizeDAG doing the wrong thing and fully scalarizing if the
// wider vector type is legal.
SDValue SITargetLowering::splitUnaryVectorOp(SDValue Op,
                                             SelectionDAG &DAG) const {
  unsigned Opc = Op.getOpcode();
  EVT VT = Op.getValueType();
  assert(VT == MVT::v4f16 || VT == MVT::v4i16);

  SDValue Lo, Hi;
  std::tie(Lo, Hi) = DAG.SplitVectorOperand(Op.getNode(), 0);

  SDLoc SL(Op);
  SDValue OpLo = DAG.getNode(Opc, SL, Lo.getValueType(), Lo,
                             Op->getFlags());
  SDValue OpHi = DAG.getNode(Opc, SL, Hi.getValueType(), Hi,
                             Op->getFlags());

  return DAG.getNode(ISD::CONCAT_VECTORS, SDLoc(Op), VT, OpLo, OpHi);
}

// Work around LegalizeDAG doing the wrong thing and fully scalarizing if the
// wider vector type is legal.
SDValue SITargetLowering::splitBinaryVectorOp(SDValue Op,
                                              SelectionDAG &DAG) const {
  unsigned Opc = Op.getOpcode();
  EVT VT = Op.getValueType();
  assert(VT == MVT::v4i16 || VT == MVT::v4f16 || VT == MVT::v4f32 ||
         VT == MVT::v8i16 || VT == MVT::v8f16 || VT == MVT::v8f32 ||
         VT == MVT::v16f32 || VT == MVT::v32f32);

  SDValue Lo0, Hi0;
  std::tie(Lo0, Hi0) = DAG.SplitVectorOperand(Op.getNode(), 0);
  SDValue Lo1, Hi1;
  std::tie(Lo1, Hi1) = DAG.SplitVectorOperand(Op.getNode(), 1);

  SDLoc SL(Op);

  SDValue OpLo = DAG.getNode(Opc, SL, Lo0.getValueType(), Lo0, Lo1,
                             Op->getFlags());
  SDValue OpHi = DAG.getNode(Opc, SL, Hi0.getValueType(), Hi0, Hi1,
                             Op->getFlags());

  return DAG.getNode(ISD::CONCAT_VECTORS, SDLoc(Op), VT, OpLo, OpHi);
}

SDValue SITargetLowering::splitTernaryVectorOp(SDValue Op,
                                              SelectionDAG &DAG) const {
  unsigned Opc = Op.getOpcode();
  EVT VT = Op.getValueType();
  assert(VT == MVT::v4i16 || VT == MVT::v4f16 || VT == MVT::v8i16 ||
         VT == MVT::v8f16 || VT == MVT::v4f32 || VT == MVT::v8f32 ||
         VT == MVT::v16f32 || VT == MVT::v32f32);

  SDValue Lo0, Hi0;
  SDValue Op0 = Op.getOperand(0);
  std::tie(Lo0, Hi0) = Op0.getValueType().isVector()
                         ? DAG.SplitVectorOperand(Op.getNode(), 0)
                         : std::make_pair(Op0, Op0);
  SDValue Lo1, Hi1;
  std::tie(Lo1, Hi1) = DAG.SplitVectorOperand(Op.getNode(), 1);
  SDValue Lo2, Hi2;
  std::tie(Lo2, Hi2) = DAG.SplitVectorOperand(Op.getNode(), 2);

  SDLoc SL(Op);
  auto ResVT = DAG.GetSplitDestVTs(VT);

  SDValue OpLo = DAG.getNode(Opc, SL, ResVT.first, Lo0, Lo1, Lo2,
                             Op->getFlags());
  SDValue OpHi = DAG.getNode(Opc, SL, ResVT.second, Hi0, Hi1, Hi2,
                             Op->getFlags());

  return DAG.getNode(ISD::CONCAT_VECTORS, SDLoc(Op), VT, OpLo, OpHi);
}


SDValue SITargetLowering::LowerOperation(SDValue Op, SelectionDAG &DAG) const {
  switch (Op.getOpcode()) {
  default: return AMDGPUTargetLowering::LowerOperation(Op, DAG);
  case ISD::BRCOND: return LowerBRCOND(Op, DAG);
  case ISD::RETURNADDR: return LowerRETURNADDR(Op, DAG);
  case ISD::LOAD: {
    SDValue Result = LowerLOAD(Op, DAG);
    assert((!Result.getNode() ||
            Result.getNode()->getNumValues() == 2) &&
           "Load should return a value and a chain");
    return Result;
  }

  case ISD::FSIN:
  case ISD::FCOS:
    return LowerTrig(Op, DAG);
  case ISD::SELECT: return LowerSELECT(Op, DAG);
  case ISD::FDIV: return LowerFDIV(Op, DAG);
  case ISD::ATOMIC_CMP_SWAP: return LowerATOMIC_CMP_SWAP(Op, DAG);
  case ISD::STORE: return LowerSTORE(Op, DAG);
  case ISD::GlobalAddress: {
    MachineFunction &MF = DAG.getMachineFunction();
    SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
    return LowerGlobalAddress(MFI, Op, DAG);
  }
  case ISD::INTRINSIC_WO_CHAIN: return LowerINTRINSIC_WO_CHAIN(Op, DAG);
  case ISD::INTRINSIC_W_CHAIN: return LowerINTRINSIC_W_CHAIN(Op, DAG);
  case ISD::INTRINSIC_VOID: return LowerINTRINSIC_VOID(Op, DAG);
  case ISD::ADDRSPACECAST: return lowerADDRSPACECAST(Op, DAG);
  case ISD::INSERT_SUBVECTOR:
    return lowerINSERT_SUBVECTOR(Op, DAG);
  case ISD::INSERT_VECTOR_ELT:
    return lowerINSERT_VECTOR_ELT(Op, DAG);
  case ISD::EXTRACT_VECTOR_ELT:
    return lowerEXTRACT_VECTOR_ELT(Op, DAG);
  case ISD::VECTOR_SHUFFLE:
    return lowerVECTOR_SHUFFLE(Op, DAG);
  case ISD::BUILD_VECTOR:
    return lowerBUILD_VECTOR(Op, DAG);
  case ISD::FP_ROUND:
    return lowerFP_ROUND(Op, DAG);
  case ISD::TRAP:
    return lowerTRAP(Op, DAG);
  case ISD::DEBUGTRAP:
    return lowerDEBUGTRAP(Op, DAG);
  case ISD::FABS:
  case ISD::FNEG:
  case ISD::FCANONICALIZE:
  case ISD::BSWAP:
    return splitUnaryVectorOp(Op, DAG);
  case ISD::FMINNUM:
  case ISD::FMAXNUM:
    return lowerFMINNUM_FMAXNUM(Op, DAG);
  case ISD::FMA:
    return splitTernaryVectorOp(Op, DAG);
  case ISD::FP_TO_SINT:
  case ISD::FP_TO_UINT:
    return LowerFP_TO_INT(Op, DAG);
  case ISD::SHL:
  case ISD::SRA:
  case ISD::SRL:
  case ISD::ADD:
  case ISD::SUB:
  case ISD::MUL:
  case ISD::SMIN:
  case ISD::SMAX:
  case ISD::UMIN:
  case ISD::UMAX:
  case ISD::FADD:
  case ISD::FMUL:
  case ISD::FMINNUM_IEEE:
  case ISD::FMAXNUM_IEEE:
  case ISD::UADDSAT:
  case ISD::USUBSAT:
  case ISD::SADDSAT:
  case ISD::SSUBSAT:
    return splitBinaryVectorOp(Op, DAG);
  case ISD::SMULO:
  case ISD::UMULO:
    return lowerXMULO(Op, DAG);
  case ISD::SMUL_LOHI:
  case ISD::UMUL_LOHI:
    return lowerXMUL_LOHI(Op, DAG);
  case ISD::DYNAMIC_STACKALLOC:
    return LowerDYNAMIC_STACKALLOC(Op, DAG);
  }
  return SDValue();
}

// Used for D16: Casts the result of an instruction into the right vector,
// packs values if loads return unpacked values.
static SDValue adjustLoadValueTypeImpl(SDValue Result, EVT LoadVT,
                                       const SDLoc &DL,
                                       SelectionDAG &DAG, bool Unpacked) {
  if (!LoadVT.isVector())
    return Result;

  // Cast back to the original packed type or to a larger type that is a
  // multiple of 32 bit for D16. Widening the return type is a required for
  // legalization.
  EVT FittingLoadVT = LoadVT;
  if ((LoadVT.getVectorNumElements() % 2) == 1) {
    FittingLoadVT =
        EVT::getVectorVT(*DAG.getContext(), LoadVT.getVectorElementType(),
                         LoadVT.getVectorNumElements() + 1);
  }

  if (Unpacked) { // From v2i32/v4i32 back to v2f16/v4f16.
    // Truncate to v2i16/v4i16.
    EVT IntLoadVT = FittingLoadVT.changeTypeToInteger();

    // Workaround legalizer not scalarizing truncate after vector op
    // legalization but not creating intermediate vector trunc.
    SmallVector<SDValue, 4> Elts;
    DAG.ExtractVectorElements(Result, Elts);
    for (SDValue &Elt : Elts)
      Elt = DAG.getNode(ISD::TRUNCATE, DL, MVT::i16, Elt);

    // Pad illegal v1i16/v3fi6 to v4i16
    if ((LoadVT.getVectorNumElements() % 2) == 1)
      Elts.push_back(DAG.getUNDEF(MVT::i16));

    Result = DAG.getBuildVector(IntLoadVT, DL, Elts);

    // Bitcast to original type (v2f16/v4f16).
    return DAG.getNode(ISD::BITCAST, DL, FittingLoadVT, Result);
  }

  // Cast back to the original packed type.
  return DAG.getNode(ISD::BITCAST, DL, FittingLoadVT, Result);
}

SDValue SITargetLowering::adjustLoadValueType(unsigned Opcode,
                                              MemSDNode *M,
                                              SelectionDAG &DAG,
                                              ArrayRef<SDValue> Ops,
                                              bool IsIntrinsic) const {
  SDLoc DL(M);

  bool Unpacked = Subtarget->hasUnpackedD16VMem();
  EVT LoadVT = M->getValueType(0);

  EVT EquivLoadVT = LoadVT;
  if (LoadVT.isVector()) {
    if (Unpacked) {
      EquivLoadVT = EVT::getVectorVT(*DAG.getContext(), MVT::i32,
                                     LoadVT.getVectorNumElements());
    } else if ((LoadVT.getVectorNumElements() % 2) == 1) {
      // Widen v3f16 to legal type
      EquivLoadVT =
          EVT::getVectorVT(*DAG.getContext(), LoadVT.getVectorElementType(),
                           LoadVT.getVectorNumElements() + 1);
    }
  }

  // Change from v4f16/v2f16 to EquivLoadVT.
  SDVTList VTList = DAG.getVTList(EquivLoadVT, MVT::Other);

  SDValue Load
    = DAG.getMemIntrinsicNode(
      IsIntrinsic ? (unsigned)ISD::INTRINSIC_W_CHAIN : Opcode, DL,
      VTList, Ops, M->getMemoryVT(),
      M->getMemOperand());

  SDValue Adjusted = adjustLoadValueTypeImpl(Load, LoadVT, DL, DAG, Unpacked);

  return DAG.getMergeValues({ Adjusted, Load.getValue(1) }, DL);
}

SDValue SITargetLowering::lowerIntrinsicLoad(MemSDNode *M, bool IsFormat,
                                             SelectionDAG &DAG,
                                             ArrayRef<SDValue> Ops) const {
  SDLoc DL(M);
  EVT LoadVT = M->getValueType(0);
  EVT EltType = LoadVT.getScalarType();
  EVT IntVT = LoadVT.changeTypeToInteger();

  bool IsD16 = IsFormat && (EltType.getSizeInBits() == 16);

  unsigned Opc =
      IsFormat ? AMDGPUISD::BUFFER_LOAD_FORMAT : AMDGPUISD::BUFFER_LOAD;

  if (IsD16) {
    return adjustLoadValueType(AMDGPUISD::BUFFER_LOAD_FORMAT_D16, M, DAG, Ops);
  }

  // Handle BUFFER_LOAD_BYTE/UBYTE/SHORT/USHORT overloaded intrinsics
  if (!IsD16 && !LoadVT.isVector() && EltType.getSizeInBits() < 32)
    return handleByteShortBufferLoads(DAG, LoadVT, DL, Ops, M);

  if (isTypeLegal(LoadVT)) {
    return getMemIntrinsicNode(Opc, DL, M->getVTList(), Ops, IntVT,
                               M->getMemOperand(), DAG);
  }

  EVT CastVT = getEquivalentMemType(*DAG.getContext(), LoadVT);
  SDVTList VTList = DAG.getVTList(CastVT, MVT::Other);
  SDValue MemNode = getMemIntrinsicNode(Opc, DL, VTList, Ops, CastVT,
                                        M->getMemOperand(), DAG);
  return DAG.getMergeValues(
      {DAG.getNode(ISD::BITCAST, DL, LoadVT, MemNode), MemNode.getValue(1)},
      DL);
}

static SDValue lowerICMPIntrinsic(const SITargetLowering &TLI,
                                  SDNode *N, SelectionDAG &DAG) {
  EVT VT = N->getValueType(0);
  const auto *CD = cast<ConstantSDNode>(N->getOperand(3));
  unsigned CondCode = CD->getZExtValue();
  if (!ICmpInst::isIntPredicate(static_cast<ICmpInst::Predicate>(CondCode)))
    return DAG.getUNDEF(VT);

  ICmpInst::Predicate IcInput = static_cast<ICmpInst::Predicate>(CondCode);

  SDValue LHS = N->getOperand(1);
  SDValue RHS = N->getOperand(2);

  SDLoc DL(N);

  EVT CmpVT = LHS.getValueType();
  if (CmpVT == MVT::i16 && !TLI.isTypeLegal(MVT::i16)) {
    unsigned PromoteOp = ICmpInst::isSigned(IcInput) ?
      ISD::SIGN_EXTEND : ISD::ZERO_EXTEND;
    LHS = DAG.getNode(PromoteOp, DL, MVT::i32, LHS);
    RHS = DAG.getNode(PromoteOp, DL, MVT::i32, RHS);
  }

  ISD::CondCode CCOpcode = getICmpCondCode(IcInput);

  unsigned WavefrontSize = TLI.getSubtarget()->getWavefrontSize();
  EVT CCVT = EVT::getIntegerVT(*DAG.getContext(), WavefrontSize);

  SDValue SetCC = DAG.getNode(AMDGPUISD::SETCC, DL, CCVT, LHS, RHS,
                              DAG.getCondCode(CCOpcode));
  if (VT.bitsEq(CCVT))
    return SetCC;
  return DAG.getZExtOrTrunc(SetCC, DL, VT);
}

static SDValue lowerFCMPIntrinsic(const SITargetLowering &TLI,
                                  SDNode *N, SelectionDAG &DAG) {
  EVT VT = N->getValueType(0);
  const auto *CD = cast<ConstantSDNode>(N->getOperand(3));

  unsigned CondCode = CD->getZExtValue();
  if (!FCmpInst::isFPPredicate(static_cast<FCmpInst::Predicate>(CondCode)))
    return DAG.getUNDEF(VT);

  SDValue Src0 = N->getOperand(1);
  SDValue Src1 = N->getOperand(2);
  EVT CmpVT = Src0.getValueType();
  SDLoc SL(N);

  if (CmpVT == MVT::f16 && !TLI.isTypeLegal(CmpVT)) {
    Src0 = DAG.getNode(ISD::FP_EXTEND, SL, MVT::f32, Src0);
    Src1 = DAG.getNode(ISD::FP_EXTEND, SL, MVT::f32, Src1);
  }

  FCmpInst::Predicate IcInput = static_cast<FCmpInst::Predicate>(CondCode);
  ISD::CondCode CCOpcode = getFCmpCondCode(IcInput);
  unsigned WavefrontSize = TLI.getSubtarget()->getWavefrontSize();
  EVT CCVT = EVT::getIntegerVT(*DAG.getContext(), WavefrontSize);
  SDValue SetCC = DAG.getNode(AMDGPUISD::SETCC, SL, CCVT, Src0,
                              Src1, DAG.getCondCode(CCOpcode));
  if (VT.bitsEq(CCVT))
    return SetCC;
  return DAG.getZExtOrTrunc(SetCC, SL, VT);
}

static SDValue lowerBALLOTIntrinsic(const SITargetLowering &TLI, SDNode *N,
                                    SelectionDAG &DAG) {
  EVT VT = N->getValueType(0);
  SDValue Src = N->getOperand(1);
  SDLoc SL(N);

  if (Src.getOpcode() == ISD::SETCC) {
    // (ballot (ISD::SETCC ...)) -> (AMDGPUISD::SETCC ...)
    return DAG.getNode(AMDGPUISD::SETCC, SL, VT, Src.getOperand(0),
                       Src.getOperand(1), Src.getOperand(2));
  }
  if (const ConstantSDNode *Arg = dyn_cast<ConstantSDNode>(Src)) {
    // (ballot 0) -> 0
    if (Arg->isZero())
      return DAG.getConstant(0, SL, VT);

    // (ballot 1) -> EXEC/EXEC_LO
    if (Arg->isOne()) {
      Register Exec;
      if (VT.getScalarSizeInBits() == 32)
        Exec = AMDGPU::EXEC_LO;
      else if (VT.getScalarSizeInBits() == 64)
        Exec = AMDGPU::EXEC;
      else
        return SDValue();

      return DAG.getCopyFromReg(DAG.getEntryNode(), SL, Exec, VT);
    }
  }

  // (ballot (i1 $src)) -> (AMDGPUISD::SETCC (i32 (zext $src)) (i32 0)
  // ISD::SETNE)
  return DAG.getNode(
      AMDGPUISD::SETCC, SL, VT, DAG.getZExtOrTrunc(Src, SL, MVT::i32),
      DAG.getConstant(0, SL, MVT::i32), DAG.getCondCode(ISD::SETNE));
}

void SITargetLowering::ReplaceNodeResults(SDNode *N,
                                          SmallVectorImpl<SDValue> &Results,
                                          SelectionDAG &DAG) const {
  switch (N->getOpcode()) {
  case ISD::INSERT_VECTOR_ELT: {
    if (SDValue Res = lowerINSERT_VECTOR_ELT(SDValue(N, 0), DAG))
      Results.push_back(Res);
    return;
  }
  case ISD::EXTRACT_VECTOR_ELT: {
    if (SDValue Res = lowerEXTRACT_VECTOR_ELT(SDValue(N, 0), DAG))
      Results.push_back(Res);
    return;
  }
  case ISD::INTRINSIC_WO_CHAIN: {
    unsigned IID = cast<ConstantSDNode>(N->getOperand(0))->getZExtValue();
    switch (IID) {
    case Intrinsic::amdgcn_cvt_pkrtz: {
      SDValue Src0 = N->getOperand(1);
      SDValue Src1 = N->getOperand(2);
      SDLoc SL(N);
      SDValue Cvt = DAG.getNode(AMDGPUISD::CVT_PKRTZ_F16_F32, SL, MVT::i32,
                                Src0, Src1);
      Results.push_back(DAG.getNode(ISD::BITCAST, SL, MVT::v2f16, Cvt));
      return;
    }
    case Intrinsic::amdgcn_cvt_pknorm_i16:
    case Intrinsic::amdgcn_cvt_pknorm_u16:
    case Intrinsic::amdgcn_cvt_pk_i16:
    case Intrinsic::amdgcn_cvt_pk_u16: {
      SDValue Src0 = N->getOperand(1);
      SDValue Src1 = N->getOperand(2);
      SDLoc SL(N);
      unsigned Opcode;

      if (IID == Intrinsic::amdgcn_cvt_pknorm_i16)
        Opcode = AMDGPUISD::CVT_PKNORM_I16_F32;
      else if (IID == Intrinsic::amdgcn_cvt_pknorm_u16)
        Opcode = AMDGPUISD::CVT_PKNORM_U16_F32;
      else if (IID == Intrinsic::amdgcn_cvt_pk_i16)
        Opcode = AMDGPUISD::CVT_PK_I16_I32;
      else
        Opcode = AMDGPUISD::CVT_PK_U16_U32;

      EVT VT = N->getValueType(0);
      if (isTypeLegal(VT))
        Results.push_back(DAG.getNode(Opcode, SL, VT, Src0, Src1));
      else {
        SDValue Cvt = DAG.getNode(Opcode, SL, MVT::i32, Src0, Src1);
        Results.push_back(DAG.getNode(ISD::BITCAST, SL, MVT::v2i16, Cvt));
      }
      return;
    }
    }
    break;
  }
  case ISD::INTRINSIC_W_CHAIN: {
    if (SDValue Res = LowerINTRINSIC_W_CHAIN(SDValue(N, 0), DAG)) {
      if (Res.getOpcode() == ISD::MERGE_VALUES) {
        // FIXME: Hacky
        for (unsigned I = 0; I < Res.getNumOperands(); I++) {
          Results.push_back(Res.getOperand(I));
        }
      } else {
        Results.push_back(Res);
        Results.push_back(Res.getValue(1));
      }
      return;
    }

    break;
  }
  case ISD::SELECT: {
    SDLoc SL(N);
    EVT VT = N->getValueType(0);
    EVT NewVT = getEquivalentMemType(*DAG.getContext(), VT);
    SDValue LHS = DAG.getNode(ISD::BITCAST, SL, NewVT, N->getOperand(1));
    SDValue RHS = DAG.getNode(ISD::BITCAST, SL, NewVT, N->getOperand(2));

    EVT SelectVT = NewVT;
    if (NewVT.bitsLT(MVT::i32)) {
      LHS = DAG.getNode(ISD::ANY_EXTEND, SL, MVT::i32, LHS);
      RHS = DAG.getNode(ISD::ANY_EXTEND, SL, MVT::i32, RHS);
      SelectVT = MVT::i32;
    }

    SDValue NewSelect = DAG.getNode(ISD::SELECT, SL, SelectVT,
                                    N->getOperand(0), LHS, RHS);

    if (NewVT != SelectVT)
      NewSelect = DAG.getNode(ISD::TRUNCATE, SL, NewVT, NewSelect);
    Results.push_back(DAG.getNode(ISD::BITCAST, SL, VT, NewSelect));
    return;
  }
  case ISD::FNEG: {
    if (N->getValueType(0) != MVT::v2f16)
      break;

    SDLoc SL(N);
    SDValue BC = DAG.getNode(ISD::BITCAST, SL, MVT::i32, N->getOperand(0));

    SDValue Op = DAG.getNode(ISD::XOR, SL, MVT::i32,
                             BC,
                             DAG.getConstant(0x80008000, SL, MVT::i32));
    Results.push_back(DAG.getNode(ISD::BITCAST, SL, MVT::v2f16, Op));
    return;
  }
  case ISD::FABS: {
    if (N->getValueType(0) != MVT::v2f16)
      break;

    SDLoc SL(N);
    SDValue BC = DAG.getNode(ISD::BITCAST, SL, MVT::i32, N->getOperand(0));

    SDValue Op = DAG.getNode(ISD::AND, SL, MVT::i32,
                             BC,
                             DAG.getConstant(0x7fff7fff, SL, MVT::i32));
    Results.push_back(DAG.getNode(ISD::BITCAST, SL, MVT::v2f16, Op));
    return;
  }
  default:
    break;
  }
}

/// Helper function for LowerBRCOND
static SDNode *findUser(SDValue Value, unsigned Opcode) {

  SDNode *Parent = Value.getNode();
  for (SDNode::use_iterator I = Parent->use_begin(), E = Parent->use_end();
       I != E; ++I) {

    if (I.getUse().get() != Value)
      continue;

    if (I->getOpcode() == Opcode)
      return *I;
  }
  return nullptr;
}

unsigned SITargetLowering::isCFIntrinsic(const SDNode *Intr) const {
  if (Intr->getOpcode() == ISD::INTRINSIC_W_CHAIN) {
    switch (cast<ConstantSDNode>(Intr->getOperand(1))->getZExtValue()) {
    case Intrinsic::amdgcn_if:
      return AMDGPUISD::IF;
    case Intrinsic::amdgcn_else:
      return AMDGPUISD::ELSE;
    case Intrinsic::amdgcn_loop:
      return AMDGPUISD::LOOP;
    case Intrinsic::amdgcn_end_cf:
      llvm_unreachable("should not occur");
    default:
      return 0;
    }
  }

  // break, if_break, else_break are all only used as inputs to loop, not
  // directly as branch conditions.
  return 0;
}

bool SITargetLowering::shouldEmitFixup(const GlobalValue *GV) const {
  const Triple &TT = getTargetMachine().getTargetTriple();
  return (GV->getAddressSpace() == AMDGPUAS::CONSTANT_ADDRESS ||
          GV->getAddressSpace() == AMDGPUAS::CONSTANT_ADDRESS_32BIT) &&
         AMDGPU::shouldEmitConstantsToTextSection(TT);
}

bool SITargetLowering::shouldEmitGOTReloc(const GlobalValue *GV) const {
  // FIXME: Either avoid relying on address space here or change the default
  // address space for functions to avoid the explicit check.
  return (GV->getValueType()->isFunctionTy() ||
          !isNonGlobalAddrSpace(GV->getAddressSpace())) &&
         !shouldEmitFixup(GV) &&
         !getTargetMachine().shouldAssumeDSOLocal(*GV->getParent(), GV);
}

bool SITargetLowering::shouldEmitPCReloc(const GlobalValue *GV) const {
  return !shouldEmitFixup(GV) && !shouldEmitGOTReloc(GV);
}

bool SITargetLowering::shouldUseLDSConstAddress(const GlobalValue *GV) const {
  if (!GV->hasExternalLinkage())
    return true;

  const auto OS = getTargetMachine().getTargetTriple().getOS();
  return OS == Triple::AMDHSA || OS == Triple::AMDPAL;
}

/// This transforms the control flow intrinsics to get the branch destination as
/// last parameter, also switches branch target with BR if the need arise
SDValue SITargetLowering::LowerBRCOND(SDValue BRCOND,
                                      SelectionDAG &DAG) const {
  SDLoc DL(BRCOND);

  SDNode *Intr = BRCOND.getOperand(1).getNode();
  SDValue Target = BRCOND.getOperand(2);
  SDNode *BR = nullptr;
  SDNode *SetCC = nullptr;

  if (Intr->getOpcode() == ISD::SETCC) {
    // As long as we negate the condition everything is fine
    SetCC = Intr;
    Intr = SetCC->getOperand(0).getNode();

  } else {
    // Get the target from BR if we don't negate the condition
    BR = findUser(BRCOND, ISD::BR);
    assert(BR && "brcond missing unconditional branch user");
    Target = BR->getOperand(1);
  }

  unsigned CFNode = isCFIntrinsic(Intr);
  if (CFNode == 0) {
    // This is a uniform branch so we don't need to legalize.
    return BRCOND;
  }

  bool HaveChain = Intr->getOpcode() == ISD::INTRINSIC_VOID ||
                   Intr->getOpcode() == ISD::INTRINSIC_W_CHAIN;

  assert(!SetCC ||
        (SetCC->getConstantOperandVal(1) == 1 &&
         cast<CondCodeSDNode>(SetCC->getOperand(2).getNode())->get() ==
                                                             ISD::SETNE));

  // operands of the new intrinsic call
  SmallVector<SDValue, 4> Ops;
  if (HaveChain)
    Ops.push_back(BRCOND.getOperand(0));

  Ops.append(Intr->op_begin() + (HaveChain ?  2 : 1), Intr->op_end());
  Ops.push_back(Target);

  ArrayRef<EVT> Res(Intr->value_begin() + 1, Intr->value_end());

  // build the new intrinsic call
  SDNode *Result = DAG.getNode(CFNode, DL, DAG.getVTList(Res), Ops).getNode();

  if (!HaveChain) {
    SDValue Ops[] =  {
      SDValue(Result, 0),
      BRCOND.getOperand(0)
    };

    Result = DAG.getMergeValues(Ops, DL).getNode();
  }

  if (BR) {
    // Give the branch instruction our target
    SDValue Ops[] = {
      BR->getOperand(0),
      BRCOND.getOperand(2)
    };
    SDValue NewBR = DAG.getNode(ISD::BR, DL, BR->getVTList(), Ops);
    DAG.ReplaceAllUsesWith(BR, NewBR.getNode());
  }

  SDValue Chain = SDValue(Result, Result->getNumValues() - 1);

  // Copy the intrinsic results to registers
  for (unsigned i = 1, e = Intr->getNumValues() - 1; i != e; ++i) {
    SDNode *CopyToReg = findUser(SDValue(Intr, i), ISD::CopyToReg);
    if (!CopyToReg)
      continue;

    Chain = DAG.getCopyToReg(
      Chain, DL,
      CopyToReg->getOperand(1),
      SDValue(Result, i - 1),
      SDValue());

    DAG.ReplaceAllUsesWith(SDValue(CopyToReg, 0), CopyToReg->getOperand(0));
  }

  // Remove the old intrinsic from the chain
  DAG.ReplaceAllUsesOfValueWith(
    SDValue(Intr, Intr->getNumValues() - 1),
    Intr->getOperand(0));

  return Chain;
}

SDValue SITargetLowering::LowerRETURNADDR(SDValue Op,
                                          SelectionDAG &DAG) const {
  MVT VT = Op.getSimpleValueType();
  SDLoc DL(Op);
  // Checking the depth
  if (cast<ConstantSDNode>(Op.getOperand(0))->getZExtValue() != 0)
    return DAG.getConstant(0, DL, VT);

  MachineFunction &MF = DAG.getMachineFunction();
  const SIMachineFunctionInfo *Info = MF.getInfo<SIMachineFunctionInfo>();
  // Check for kernel and shader functions
  if (Info->isEntryFunction())
    return DAG.getConstant(0, DL, VT);

  MachineFrameInfo &MFI = MF.getFrameInfo();
  // There is a call to @llvm.returnaddress in this function
  MFI.setReturnAddressIsTaken(true);

  const SIRegisterInfo *TRI = getSubtarget()->getRegisterInfo();
  // Get the return address reg and mark it as an implicit live-in
  Register Reg = MF.addLiveIn(TRI->getReturnAddressReg(MF), getRegClassFor(VT, Op.getNode()->isDivergent()));

  return DAG.getCopyFromReg(DAG.getEntryNode(), DL, Reg, VT);
}

SDValue SITargetLowering::getFPExtOrFPRound(SelectionDAG &DAG,
                                            SDValue Op,
                                            const SDLoc &DL,
                                            EVT VT) const {
  return Op.getValueType().bitsLE(VT) ?
      DAG.getNode(ISD::FP_EXTEND, DL, VT, Op) :
    DAG.getNode(ISD::FP_ROUND, DL, VT, Op,
                DAG.getTargetConstant(0, DL, MVT::i32));
}

SDValue SITargetLowering::lowerFP_ROUND(SDValue Op, SelectionDAG &DAG) const {
  assert(Op.getValueType() == MVT::f16 &&
         "Do not know how to custom lower FP_ROUND for non-f16 type");

  SDValue Src = Op.getOperand(0);
  EVT SrcVT = Src.getValueType();
  if (SrcVT != MVT::f64)
    return Op;

  SDLoc DL(Op);

  SDValue FpToFp16 = DAG.getNode(ISD::FP_TO_FP16, DL, MVT::i32, Src);
  SDValue Trunc = DAG.getNode(ISD::TRUNCATE, DL, MVT::i16, FpToFp16);
  return DAG.getNode(ISD::BITCAST, DL, MVT::f16, Trunc);
}

SDValue SITargetLowering::lowerFMINNUM_FMAXNUM(SDValue Op,
                                               SelectionDAG &DAG) const {
  EVT VT = Op.getValueType();
  const MachineFunction &MF = DAG.getMachineFunction();
  const SIMachineFunctionInfo *Info = MF.getInfo<SIMachineFunctionInfo>();
  bool IsIEEEMode = Info->getMode().IEEE;

  // FIXME: Assert during selection that this is only selected for
  // ieee_mode. Currently a combine can produce the ieee version for non-ieee
  // mode functions, but this happens to be OK since it's only done in cases
  // where there is known no sNaN.
  if (IsIEEEMode)
    return expandFMINNUM_FMAXNUM(Op.getNode(), DAG);

  if (VT == MVT::v4f16 || VT == MVT::v8f16)
    return splitBinaryVectorOp(Op, DAG);
  return Op;
}

SDValue SITargetLowering::lowerXMULO(SDValue Op, SelectionDAG &DAG) const {
  EVT VT = Op.getValueType();
  SDLoc SL(Op);
  SDValue LHS = Op.getOperand(0);
  SDValue RHS = Op.getOperand(1);
  bool isSigned = Op.getOpcode() == ISD::SMULO;

  if (ConstantSDNode *RHSC = isConstOrConstSplat(RHS)) {
    const APInt &C = RHSC->getAPIntValue();
    // mulo(X, 1 << S) -> { X << S, (X << S) >> S != X }
    if (C.isPowerOf2()) {
      // smulo(x, signed_min) is same as umulo(x, signed_min).
      bool UseArithShift = isSigned && !C.isMinSignedValue();
      SDValue ShiftAmt = DAG.getConstant(C.logBase2(), SL, MVT::i32);
      SDValue Result = DAG.getNode(ISD::SHL, SL, VT, LHS, ShiftAmt);
      SDValue Overflow = DAG.getSetCC(SL, MVT::i1,
          DAG.getNode(UseArithShift ? ISD::SRA : ISD::SRL,
                      SL, VT, Result, ShiftAmt),
          LHS, ISD::SETNE);
      return DAG.getMergeValues({ Result, Overflow }, SL);
    }
  }

  SDValue Result = DAG.getNode(ISD::MUL, SL, VT, LHS, RHS);
  SDValue Top = DAG.getNode(isSigned ? ISD::MULHS : ISD::MULHU,
                            SL, VT, LHS, RHS);

  SDValue Sign = isSigned
    ? DAG.getNode(ISD::SRA, SL, VT, Result,
                  DAG.getConstant(VT.getScalarSizeInBits() - 1, SL, MVT::i32))
    : DAG.getConstant(0, SL, VT);
  SDValue Overflow = DAG.getSetCC(SL, MVT::i1, Top, Sign, ISD::SETNE);

  return DAG.getMergeValues({ Result, Overflow }, SL);
}

SDValue SITargetLowering::lowerXMUL_LOHI(SDValue Op, SelectionDAG &DAG) const {
  if (Op->isDivergent()) {
    // Select to V_MAD_[IU]64_[IU]32.
    return Op;
  }
  if (Subtarget->hasSMulHi()) {
    // Expand to S_MUL_I32 + S_MUL_HI_[IU]32.
    return SDValue();
  }
  // The multiply is uniform but we would have to use V_MUL_HI_[IU]32 to
  // calculate the high part, so we might as well do the whole thing with
  // V_MAD_[IU]64_[IU]32.
  return Op;
}

SDValue SITargetLowering::lowerTRAP(SDValue Op, SelectionDAG &DAG) const {
  if (!Subtarget->isTrapHandlerEnabled() ||
      Subtarget->getTrapHandlerAbi() != GCNSubtarget::TrapHandlerAbi::AMDHSA)
    return lowerTrapEndpgm(Op, DAG);

  if (Optional<uint8_t> HsaAbiVer = AMDGPU::getHsaAbiVersion(Subtarget)) {
    switch (*HsaAbiVer) {
    case ELF::ELFABIVERSION_AMDGPU_HSA_V2:
    case ELF::ELFABIVERSION_AMDGPU_HSA_V3:
      return lowerTrapHsaQueuePtr(Op, DAG);
    case ELF::ELFABIVERSION_AMDGPU_HSA_V4:
      return Subtarget->supportsGetDoorbellID() ?
          lowerTrapHsa(Op, DAG) : lowerTrapHsaQueuePtr(Op, DAG);
    }
  }

  llvm_unreachable("Unknown trap handler");
}

SDValue SITargetLowering::lowerTrapEndpgm(
    SDValue Op, SelectionDAG &DAG) const {
  SDLoc SL(Op);
  SDValue Chain = Op.getOperand(0);
  return DAG.getNode(AMDGPUISD::ENDPGM, SL, MVT::Other, Chain);
}

SDValue SITargetLowering::lowerTrapHsaQueuePtr(
    SDValue Op, SelectionDAG &DAG) const {
  SDLoc SL(Op);
  SDValue Chain = Op.getOperand(0);

  MachineFunction &MF = DAG.getMachineFunction();
  SIMachineFunctionInfo *Info = MF.getInfo<SIMachineFunctionInfo>();
  Register UserSGPR = Info->getQueuePtrUserSGPR();

  SDValue QueuePtr;
  if (UserSGPR == AMDGPU::NoRegister) {
    // We probably are in a function incorrectly marked with
    // amdgpu-no-queue-ptr. This is undefined. We don't want to delete the trap,
    // so just use a null pointer.
    QueuePtr = DAG.getConstant(0, SL, MVT::i64);
  } else {
    QueuePtr = CreateLiveInRegister(
      DAG, &AMDGPU::SReg_64RegClass, UserSGPR, MVT::i64);
  }

  SDValue SGPR01 = DAG.getRegister(AMDGPU::SGPR0_SGPR1, MVT::i64);
  SDValue ToReg = DAG.getCopyToReg(Chain, SL, SGPR01,
                                   QueuePtr, SDValue());

  uint64_t TrapID = static_cast<uint64_t>(GCNSubtarget::TrapID::LLVMAMDHSATrap);
  SDValue Ops[] = {
    ToReg,
    DAG.getTargetConstant(TrapID, SL, MVT::i16),
    SGPR01,
    ToReg.getValue(1)
  };
  return DAG.getNode(AMDGPUISD::TRAP, SL, MVT::Other, Ops);
}

SDValue SITargetLowering::lowerTrapHsa(
    SDValue Op, SelectionDAG &DAG) const {
  SDLoc SL(Op);
  SDValue Chain = Op.getOperand(0);

  uint64_t TrapID = static_cast<uint64_t>(GCNSubtarget::TrapID::LLVMAMDHSATrap);
  SDValue Ops[] = {
    Chain,
    DAG.getTargetConstant(TrapID, SL, MVT::i16)
  };
  return DAG.getNode(AMDGPUISD::TRAP, SL, MVT::Other, Ops);
}

SDValue SITargetLowering::lowerDEBUGTRAP(SDValue Op, SelectionDAG &DAG) const {
  SDLoc SL(Op);
  SDValue Chain = Op.getOperand(0);
  MachineFunction &MF = DAG.getMachineFunction();

  if (!Subtarget->isTrapHandlerEnabled() ||
      Subtarget->getTrapHandlerAbi() != GCNSubtarget::TrapHandlerAbi::AMDHSA) {
    DiagnosticInfoUnsupported NoTrap(MF.getFunction(),
                                     "debugtrap handler not supported",
                                     Op.getDebugLoc(),
                                     DS_Warning);
    LLVMContext &Ctx = MF.getFunction().getContext();
    Ctx.diagnose(NoTrap);
    return Chain;
  }

  uint64_t TrapID = static_cast<uint64_t>(GCNSubtarget::TrapID::LLVMAMDHSADebugTrap);
  SDValue Ops[] = {
    Chain,
    DAG.getTargetConstant(TrapID, SL, MVT::i16)
  };
  return DAG.getNode(AMDGPUISD::TRAP, SL, MVT::Other, Ops);
}

SDValue SITargetLowering::getSegmentAperture(unsigned AS, const SDLoc &DL,
                                             SelectionDAG &DAG) const {
  // FIXME: Use inline constants (src_{shared, private}_base) instead.
  if (Subtarget->hasApertureRegs()) {
    unsigned Offset = AS == AMDGPUAS::LOCAL_ADDRESS ?
        AMDGPU::Hwreg::OFFSET_SRC_SHARED_BASE :
        AMDGPU::Hwreg::OFFSET_SRC_PRIVATE_BASE;
    unsigned WidthM1 = AS == AMDGPUAS::LOCAL_ADDRESS ?
        AMDGPU::Hwreg::WIDTH_M1_SRC_SHARED_BASE :
        AMDGPU::Hwreg::WIDTH_M1_SRC_PRIVATE_BASE;
    unsigned Encoding =
        AMDGPU::Hwreg::ID_MEM_BASES << AMDGPU::Hwreg::ID_SHIFT_ |
        Offset << AMDGPU::Hwreg::OFFSET_SHIFT_ |
        WidthM1 << AMDGPU::Hwreg::WIDTH_M1_SHIFT_;

    SDValue EncodingImm = DAG.getTargetConstant(Encoding, DL, MVT::i16);
    SDValue ApertureReg = SDValue(
        DAG.getMachineNode(AMDGPU::S_GETREG_B32, DL, MVT::i32, EncodingImm), 0);
    SDValue ShiftAmount = DAG.getTargetConstant(WidthM1 + 1, DL, MVT::i32);
    return DAG.getNode(ISD::SHL, DL, MVT::i32, ApertureReg, ShiftAmount);
  }

  MachineFunction &MF = DAG.getMachineFunction();
  SIMachineFunctionInfo *Info = MF.getInfo<SIMachineFunctionInfo>();
  Register UserSGPR = Info->getQueuePtrUserSGPR();
  if (UserSGPR == AMDGPU::NoRegister) {
    // We probably are in a function incorrectly marked with
    // amdgpu-no-queue-ptr. This is undefined.
    return DAG.getUNDEF(MVT::i32);
  }

  SDValue QueuePtr = CreateLiveInRegister(
    DAG, &AMDGPU::SReg_64RegClass, UserSGPR, MVT::i64);

  // Offset into amd_queue_t for group_segment_aperture_base_hi /
  // private_segment_aperture_base_hi.
  uint32_t StructOffset = (AS == AMDGPUAS::LOCAL_ADDRESS) ? 0x40 : 0x44;

  SDValue Ptr =
      DAG.getObjectPtrOffset(DL, QueuePtr, TypeSize::Fixed(StructOffset));

  // TODO: Use custom target PseudoSourceValue.
  // TODO: We should use the value from the IR intrinsic call, but it might not
  // be available and how do we get it?
  MachinePointerInfo PtrInfo(AMDGPUAS::CONSTANT_ADDRESS);
  return DAG.getLoad(MVT::i32, DL, QueuePtr.getValue(1), Ptr, PtrInfo,
                     commonAlignment(Align(64), StructOffset),
                     MachineMemOperand::MODereferenceable |
                         MachineMemOperand::MOInvariant);
}

/// Return true if the value is a known valid address, such that a null check is
/// not necessary.
static bool isKnownNonNull(SDValue Val, SelectionDAG &DAG,
                           const AMDGPUTargetMachine &TM, unsigned AddrSpace) {
  if (isa<FrameIndexSDNode>(Val) || isa<GlobalAddressSDNode>(Val) ||
      isa<BasicBlockSDNode>(Val))
    return true;

  if (auto *ConstVal = dyn_cast<ConstantSDNode>(Val))
    return ConstVal->getSExtValue() != TM.getNullPointerValue(AddrSpace);

  // TODO: Search through arithmetic, handle arguments and loads
  // marked nonnull.
  return false;
}

SDValue SITargetLowering::lowerADDRSPACECAST(SDValue Op,
                                             SelectionDAG &DAG) const {
  SDLoc SL(Op);
  const AddrSpaceCastSDNode *ASC = cast<AddrSpaceCastSDNode>(Op);

  SDValue Src = ASC->getOperand(0);
  SDValue FlatNullPtr = DAG.getConstant(0, SL, MVT::i64);
  unsigned SrcAS = ASC->getSrcAddressSpace();

  const AMDGPUTargetMachine &TM =
    static_cast<const AMDGPUTargetMachine &>(getTargetMachine());

  // flat -> local/private
  if (SrcAS == AMDGPUAS::FLAT_ADDRESS) {
    unsigned DestAS = ASC->getDestAddressSpace();

    if (DestAS == AMDGPUAS::LOCAL_ADDRESS ||
        DestAS == AMDGPUAS::PRIVATE_ADDRESS) {
      SDValue Ptr = DAG.getNode(ISD::TRUNCATE, SL, MVT::i32, Src);

      if (isKnownNonNull(Src, DAG, TM, SrcAS))
        return Ptr;

      unsigned NullVal = TM.getNullPointerValue(DestAS);
      SDValue SegmentNullPtr = DAG.getConstant(NullVal, SL, MVT::i32);
      SDValue NonNull = DAG.getSetCC(SL, MVT::i1, Src, FlatNullPtr, ISD::SETNE);

      return DAG.getNode(ISD::SELECT, SL, MVT::i32, NonNull, Ptr,
                         SegmentNullPtr);
    }
  }

  // local/private -> flat
  if (ASC->getDestAddressSpace() == AMDGPUAS::FLAT_ADDRESS) {
    if (SrcAS == AMDGPUAS::LOCAL_ADDRESS ||
        SrcAS == AMDGPUAS::PRIVATE_ADDRESS) {

      SDValue Aperture = getSegmentAperture(ASC->getSrcAddressSpace(), SL, DAG);
      SDValue CvtPtr =
          DAG.getNode(ISD::BUILD_VECTOR, SL, MVT::v2i32, Src, Aperture);
      CvtPtr = DAG.getNode(ISD::BITCAST, SL, MVT::i64, CvtPtr);

      if (isKnownNonNull(Src, DAG, TM, SrcAS))
        return CvtPtr;

      unsigned NullVal = TM.getNullPointerValue(SrcAS);
      SDValue SegmentNullPtr = DAG.getConstant(NullVal, SL, MVT::i32);

      SDValue NonNull
        = DAG.getSetCC(SL, MVT::i1, Src, SegmentNullPtr, ISD::SETNE);

      return DAG.getNode(ISD::SELECT, SL, MVT::i64, NonNull, CvtPtr,
                         FlatNullPtr);
    }
  }

  if (ASC->getDestAddressSpace() == AMDGPUAS::CONSTANT_ADDRESS_32BIT &&
      Src.getValueType() == MVT::i64)
    return DAG.getNode(ISD::TRUNCATE, SL, MVT::i32, Src);

  // global <-> flat are no-ops and never emitted.

  const MachineFunction &MF = DAG.getMachineFunction();
  DiagnosticInfoUnsupported InvalidAddrSpaceCast(
    MF.getFunction(), "invalid addrspacecast", SL.getDebugLoc());
  DAG.getContext()->diagnose(InvalidAddrSpaceCast);

  return DAG.getUNDEF(ASC->getValueType(0));
}

// This lowers an INSERT_SUBVECTOR by extracting the individual elements from
// the small vector and inserting them into the big vector. That is better than
// the default expansion of doing it via a stack slot. Even though the use of
// the stack slot would be optimized away afterwards, the stack slot itself
// remains.
SDValue SITargetLowering::lowerINSERT_SUBVECTOR(SDValue Op,
                                                SelectionDAG &DAG) const {
  SDValue Vec = Op.getOperand(0);
  SDValue Ins = Op.getOperand(1);
  SDValue Idx = Op.getOperand(2);
  EVT VecVT = Vec.getValueType();
  EVT InsVT = Ins.getValueType();
  EVT EltVT = VecVT.getVectorElementType();
  unsigned InsNumElts = InsVT.getVectorNumElements();
  unsigned IdxVal = cast<ConstantSDNode>(Idx)->getZExtValue();
  SDLoc SL(Op);

  for (unsigned I = 0; I != InsNumElts; ++I) {
    SDValue Elt = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, SL, EltVT, Ins,
                              DAG.getConstant(I, SL, MVT::i32));
    Vec = DAG.getNode(ISD::INSERT_VECTOR_ELT, SL, VecVT, Vec, Elt,
                      DAG.getConstant(IdxVal + I, SL, MVT::i32));
  }
  return Vec;
}

SDValue SITargetLowering::lowerINSERT_VECTOR_ELT(SDValue Op,
                                                 SelectionDAG &DAG) const {
  SDValue Vec = Op.getOperand(0);
  SDValue InsVal = Op.getOperand(1);
  SDValue Idx = Op.getOperand(2);
  EVT VecVT = Vec.getValueType();
  EVT EltVT = VecVT.getVectorElementType();
  unsigned VecSize = VecVT.getSizeInBits();
  unsigned EltSize = EltVT.getSizeInBits();


  assert(VecSize <= 64);

  unsigned NumElts = VecVT.getVectorNumElements();
  SDLoc SL(Op);
  auto KIdx = dyn_cast<ConstantSDNode>(Idx);

  if (NumElts == 4 && EltSize == 16 && KIdx) {
    SDValue BCVec = DAG.getNode(ISD::BITCAST, SL, MVT::v2i32, Vec);

    SDValue LoHalf = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, SL, MVT::i32, BCVec,
                                 DAG.getConstant(0, SL, MVT::i32));
    SDValue HiHalf = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, SL, MVT::i32, BCVec,
                                 DAG.getConstant(1, SL, MVT::i32));

    SDValue LoVec = DAG.getNode(ISD::BITCAST, SL, MVT::v2i16, LoHalf);
    SDValue HiVec = DAG.getNode(ISD::BITCAST, SL, MVT::v2i16, HiHalf);

    unsigned Idx = KIdx->getZExtValue();
    bool InsertLo = Idx < 2;
    SDValue InsHalf = DAG.getNode(ISD::INSERT_VECTOR_ELT, SL, MVT::v2i16,
      InsertLo ? LoVec : HiVec,
      DAG.getNode(ISD::BITCAST, SL, MVT::i16, InsVal),
      DAG.getConstant(InsertLo ? Idx : (Idx - 2), SL, MVT::i32));

    InsHalf = DAG.getNode(ISD::BITCAST, SL, MVT::i32, InsHalf);

    SDValue Concat = InsertLo ?
      DAG.getBuildVector(MVT::v2i32, SL, { InsHalf, HiHalf }) :
      DAG.getBuildVector(MVT::v2i32, SL, { LoHalf, InsHalf });

    return DAG.getNode(ISD::BITCAST, SL, VecVT, Concat);
  }

  if (isa<ConstantSDNode>(Idx))
    return SDValue();

  MVT IntVT = MVT::getIntegerVT(VecSize);

  // Avoid stack access for dynamic indexing.
  // v_bfi_b32 (v_bfm_b32 16, (shl idx, 16)), val, vec

  // Create a congruent vector with the target value in each element so that
  // the required element can be masked and ORed into the target vector.
  SDValue ExtVal = DAG.getNode(ISD::BITCAST, SL, IntVT,
                               DAG.getSplatBuildVector(VecVT, SL, InsVal));

  assert(isPowerOf2_32(EltSize));
  SDValue ScaleFactor = DAG.getConstant(Log2_32(EltSize), SL, MVT::i32);

  // Convert vector index to bit-index.
  SDValue ScaledIdx = DAG.getNode(ISD::SHL, SL, MVT::i32, Idx, ScaleFactor);

  SDValue BCVec = DAG.getNode(ISD::BITCAST, SL, IntVT, Vec);
  SDValue BFM = DAG.getNode(ISD::SHL, SL, IntVT,
                            DAG.getConstant(0xffff, SL, IntVT),
                            ScaledIdx);

  SDValue LHS = DAG.getNode(ISD::AND, SL, IntVT, BFM, ExtVal);
  SDValue RHS = DAG.getNode(ISD::AND, SL, IntVT,
                            DAG.getNOT(SL, BFM, IntVT), BCVec);

  SDValue BFI = DAG.getNode(ISD::OR, SL, IntVT, LHS, RHS);
  return DAG.getNode(ISD::BITCAST, SL, VecVT, BFI);
}

SDValue SITargetLowering::lowerEXTRACT_VECTOR_ELT(SDValue Op,
                                                  SelectionDAG &DAG) const {
  SDLoc SL(Op);

  EVT ResultVT = Op.getValueType();
  SDValue Vec = Op.getOperand(0);
  SDValue Idx = Op.getOperand(1);
  EVT VecVT = Vec.getValueType();
  unsigned VecSize = VecVT.getSizeInBits();
  EVT EltVT = VecVT.getVectorElementType();

  DAGCombinerInfo DCI(DAG, AfterLegalizeVectorOps, true, nullptr);

  // Make sure we do any optimizations that will make it easier to fold
  // source modifiers before obscuring it with bit operations.

  // XXX - Why doesn't this get called when vector_shuffle is expanded?
  if (SDValue Combined = performExtractVectorEltCombine(Op.getNode(), DCI))
    return Combined;

  if (VecSize == 128) {
    SDValue Lo, Hi;
    EVT LoVT, HiVT;
    SDValue V2 = DAG.getBitcast(MVT::v2i64, Vec);
    std::tie(LoVT, HiVT) = DAG.GetSplitDestVTs(VecVT);
    Lo =
        DAG.getBitcast(LoVT, DAG.getNode(ISD::EXTRACT_VECTOR_ELT, SL, MVT::i64,
                                         V2, DAG.getConstant(0, SL, MVT::i32)));
    Hi =
        DAG.getBitcast(HiVT, DAG.getNode(ISD::EXTRACT_VECTOR_ELT, SL, MVT::i64,
                                         V2, DAG.getConstant(1, SL, MVT::i32)));
    EVT IdxVT = Idx.getValueType();
    unsigned NElem = VecVT.getVectorNumElements();
    assert(isPowerOf2_32(NElem));
    SDValue IdxMask = DAG.getConstant(NElem / 2 - 1, SL, IdxVT);
    SDValue NewIdx = DAG.getNode(ISD::AND, SL, IdxVT, Idx, IdxMask);
    SDValue Half = DAG.getSelectCC(SL, Idx, IdxMask, Hi, Lo, ISD::SETUGT);
    return DAG.getNode(ISD::EXTRACT_VECTOR_ELT, SL, EltVT, Half, NewIdx);
  }

  assert(VecSize <= 64);

  unsigned EltSize = EltVT.getSizeInBits();
  assert(isPowerOf2_32(EltSize));

  MVT IntVT = MVT::getIntegerVT(VecSize);
  SDValue ScaleFactor = DAG.getConstant(Log2_32(EltSize), SL, MVT::i32);

  // Convert vector index to bit-index (* EltSize)
  SDValue ScaledIdx = DAG.getNode(ISD::SHL, SL, MVT::i32, Idx, ScaleFactor);

  SDValue BC = DAG.getNode(ISD::BITCAST, SL, IntVT, Vec);
  SDValue Elt = DAG.getNode(ISD::SRL, SL, IntVT, BC, ScaledIdx);

  if (ResultVT == MVT::f16) {
    SDValue Result = DAG.getNode(ISD::TRUNCATE, SL, MVT::i16, Elt);
    return DAG.getNode(ISD::BITCAST, SL, ResultVT, Result);
  }

  return DAG.getAnyExtOrTrunc(Elt, SL, ResultVT);
}

static bool elementPairIsContiguous(ArrayRef<int> Mask, int Elt) {
  assert(Elt % 2 == 0);
  return Mask[Elt + 1] == Mask[Elt] + 1 && (Mask[Elt] % 2 == 0);
}

SDValue SITargetLowering::lowerVECTOR_SHUFFLE(SDValue Op,
                                              SelectionDAG &DAG) const {
  SDLoc SL(Op);
  EVT ResultVT = Op.getValueType();
  ShuffleVectorSDNode *SVN = cast<ShuffleVectorSDNode>(Op);

  EVT PackVT = ResultVT.isInteger() ? MVT::v2i16 : MVT::v2f16;
  EVT EltVT = PackVT.getVectorElementType();
  int SrcNumElts = Op.getOperand(0).getValueType().getVectorNumElements();

  // vector_shuffle <0,1,6,7> lhs, rhs
  // -> concat_vectors (extract_subvector lhs, 0), (extract_subvector rhs, 2)
  //
  // vector_shuffle <6,7,2,3> lhs, rhs
  // -> concat_vectors (extract_subvector rhs, 2), (extract_subvector lhs, 2)
  //
  // vector_shuffle <6,7,0,1> lhs, rhs
  // -> concat_vectors (extract_subvector rhs, 2), (extract_subvector lhs, 0)

  // Avoid scalarizing when both halves are reading from consecutive elements.
  SmallVector<SDValue, 4> Pieces;
  for (int I = 0, N = ResultVT.getVectorNumElements(); I != N; I += 2) {
    if (elementPairIsContiguous(SVN->getMask(), I)) {
      const int Idx = SVN->getMaskElt(I);
      int VecIdx = Idx < SrcNumElts ? 0 : 1;
      int EltIdx = Idx < SrcNumElts ? Idx : Idx - SrcNumElts;
      SDValue SubVec = DAG.getNode(ISD::EXTRACT_SUBVECTOR, SL,
                                    PackVT, SVN->getOperand(VecIdx),
                                    DAG.getConstant(EltIdx, SL, MVT::i32));
      Pieces.push_back(SubVec);
    } else {
      const int Idx0 = SVN->getMaskElt(I);
      const int Idx1 = SVN->getMaskElt(I + 1);
      int VecIdx0 = Idx0 < SrcNumElts ? 0 : 1;
      int VecIdx1 = Idx1 < SrcNumElts ? 0 : 1;
      int EltIdx0 = Idx0 < SrcNumElts ? Idx0 : Idx0 - SrcNumElts;
      int EltIdx1 = Idx1 < SrcNumElts ? Idx1 : Idx1 - SrcNumElts;

      SDValue Vec0 = SVN->getOperand(VecIdx0);
      SDValue Elt0 = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, SL, EltVT,
                                 Vec0, DAG.getConstant(EltIdx0, SL, MVT::i32));

      SDValue Vec1 = SVN->getOperand(VecIdx1);
      SDValue Elt1 = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, SL, EltVT,
                                 Vec1, DAG.getConstant(EltIdx1, SL, MVT::i32));
      Pieces.push_back(DAG.getBuildVector(PackVT, SL, { Elt0, Elt1 }));
    }
  }

  return DAG.getNode(ISD::CONCAT_VECTORS, SL, ResultVT, Pieces);
}

SDValue SITargetLowering::lowerBUILD_VECTOR(SDValue Op,
                                            SelectionDAG &DAG) const {
  SDLoc SL(Op);
  EVT VT = Op.getValueType();

  if (VT == MVT::v4i16 || VT == MVT::v4f16 ||
      VT == MVT::v8i16 || VT == MVT::v8f16) {
    EVT HalfVT = MVT::getVectorVT(VT.getVectorElementType().getSimpleVT(),
                                  VT.getVectorNumElements() / 2);
    MVT HalfIntVT = MVT::getIntegerVT(HalfVT.getSizeInBits());

    // Turn into pair of packed build_vectors.
    // TODO: Special case for constants that can be materialized with s_mov_b64.
    SmallVector<SDValue, 4> LoOps, HiOps;
    for (unsigned I = 0, E = VT.getVectorNumElements() / 2; I != E; ++I) {
      LoOps.push_back(Op.getOperand(I));
      HiOps.push_back(Op.getOperand(I + E));
    }
    SDValue Lo = DAG.getBuildVector(HalfVT, SL, LoOps);
    SDValue Hi = DAG.getBuildVector(HalfVT, SL, HiOps);

    SDValue CastLo = DAG.getNode(ISD::BITCAST, SL, HalfIntVT, Lo);
    SDValue CastHi = DAG.getNode(ISD::BITCAST, SL, HalfIntVT, Hi);

    SDValue Blend = DAG.getBuildVector(MVT::getVectorVT(HalfIntVT, 2), SL,
                                       { CastLo, CastHi });
    return DAG.getNode(ISD::BITCAST, SL, VT, Blend);
  }

  assert(VT == MVT::v2f16 || VT == MVT::v2i16);
  assert(!Subtarget->hasVOP3PInsts() && "this should be legal");

  SDValue Lo = Op.getOperand(0);
  SDValue Hi = Op.getOperand(1);

  // Avoid adding defined bits with the zero_extend.
  if (Hi.isUndef()) {
    Lo = DAG.getNode(ISD::BITCAST, SL, MVT::i16, Lo);
    SDValue ExtLo = DAG.getNode(ISD::ANY_EXTEND, SL, MVT::i32, Lo);
    return DAG.getNode(ISD::BITCAST, SL, VT, ExtLo);
  }

  Hi = DAG.getNode(ISD::BITCAST, SL, MVT::i16, Hi);
  Hi = DAG.getNode(ISD::ZERO_EXTEND, SL, MVT::i32, Hi);

  SDValue ShlHi = DAG.getNode(ISD::SHL, SL, MVT::i32, Hi,
                              DAG.getConstant(16, SL, MVT::i32));
  if (Lo.isUndef())
    return DAG.getNode(ISD::BITCAST, SL, VT, ShlHi);

  Lo = DAG.getNode(ISD::BITCAST, SL, MVT::i16, Lo);
  Lo = DAG.getNode(ISD::ZERO_EXTEND, SL, MVT::i32, Lo);

  SDValue Or = DAG.getNode(ISD::OR, SL, MVT::i32, Lo, ShlHi);
  return DAG.getNode(ISD::BITCAST, SL, VT, Or);
}

bool
SITargetLowering::isOffsetFoldingLegal(const GlobalAddressSDNode *GA) const {
  // We can fold offsets for anything that doesn't require a GOT relocation.
  return (GA->getAddressSpace() == AMDGPUAS::GLOBAL_ADDRESS ||
          GA->getAddressSpace() == AMDGPUAS::CONSTANT_ADDRESS ||
          GA->getAddressSpace() == AMDGPUAS::CONSTANT_ADDRESS_32BIT) &&
         !shouldEmitGOTReloc(GA->getGlobal());
}

static SDValue
buildPCRelGlobalAddress(SelectionDAG &DAG, const GlobalValue *GV,
                        const SDLoc &DL, int64_t Offset, EVT PtrVT,
                        unsigned GAFlags = SIInstrInfo::MO_NONE) {
  assert(isInt<32>(Offset + 4) && "32-bit offset is expected!");
  // In order to support pc-relative addressing, the PC_ADD_REL_OFFSET SDNode is
  // lowered to the following code sequence:
  //
  // For constant address space:
  //   s_getpc_b64 s[0:1]
  //   s_add_u32 s0, s0, $symbol
  //   s_addc_u32 s1, s1, 0
  //
  //   s_getpc_b64 returns the address of the s_add_u32 instruction and then
  //   a fixup or relocation is emitted to replace $symbol with a literal
  //   constant, which is a pc-relative offset from the encoding of the $symbol
  //   operand to the global variable.
  //
  // For global address space:
  //   s_getpc_b64 s[0:1]
  //   s_add_u32 s0, s0, $symbol@{gotpc}rel32@lo
  //   s_addc_u32 s1, s1, $symbol@{gotpc}rel32@hi
  //
  //   s_getpc_b64 returns the address of the s_add_u32 instruction and then
  //   fixups or relocations are emitted to replace $symbol@*@lo and
  //   $symbol@*@hi with lower 32 bits and higher 32 bits of a literal constant,
  //   which is a 64-bit pc-relative offset from the encoding of the $symbol
  //   operand to the global variable.
  //
  // What we want here is an offset from the value returned by s_getpc
  // (which is the address of the s_add_u32 instruction) to the global
  // variable, but since the encoding of $symbol starts 4 bytes after the start
  // of the s_add_u32 instruction, we end up with an offset that is 4 bytes too
  // small. This requires us to add 4 to the global variable offset in order to
  // compute the correct address. Similarly for the s_addc_u32 instruction, the
  // encoding of $symbol starts 12 bytes after the start of the s_add_u32
  // instruction.
  SDValue PtrLo =
      DAG.getTargetGlobalAddress(GV, DL, MVT::i32, Offset + 4, GAFlags);
  SDValue PtrHi;
  if (GAFlags == SIInstrInfo::MO_NONE) {
    PtrHi = DAG.getTargetConstant(0, DL, MVT::i32);
  } else {
    PtrHi =
        DAG.getTargetGlobalAddress(GV, DL, MVT::i32, Offset + 12, GAFlags + 1);
  }
  return DAG.getNode(AMDGPUISD::PC_ADD_REL_OFFSET, DL, PtrVT, PtrLo, PtrHi);
}

SDValue SITargetLowering::LowerGlobalAddress(AMDGPUMachineFunction *MFI,
                                             SDValue Op,
                                             SelectionDAG &DAG) const {
  GlobalAddressSDNode *GSD = cast<GlobalAddressSDNode>(Op);
  SDLoc DL(GSD);
  EVT PtrVT = Op.getValueType();

  const GlobalValue *GV = GSD->getGlobal();
  if ((GSD->getAddressSpace() == AMDGPUAS::LOCAL_ADDRESS &&
       shouldUseLDSConstAddress(GV)) ||
      GSD->getAddressSpace() == AMDGPUAS::REGION_ADDRESS ||
      GSD->getAddressSpace() == AMDGPUAS::PRIVATE_ADDRESS) {
    if (GSD->getAddressSpace() == AMDGPUAS::LOCAL_ADDRESS &&
        GV->hasExternalLinkage()) {
      Type *Ty = GV->getValueType();
      // HIP uses an unsized array `extern __shared__ T s[]` or similar
      // zero-sized type in other languages to declare the dynamic shared
      // memory which size is not known at the compile time. They will be
      // allocated by the runtime and placed directly after the static
      // allocated ones. They all share the same offset.
      if (DAG.getDataLayout().getTypeAllocSize(Ty).isZero()) {
        assert(PtrVT == MVT::i32 && "32-bit pointer is expected.");
        // Adjust alignment for that dynamic shared memory array.
        MFI->setDynLDSAlign(DAG.getDataLayout(), *cast<GlobalVariable>(GV));
        return SDValue(
            DAG.getMachineNode(AMDGPU::GET_GROUPSTATICSIZE, DL, PtrVT), 0);
      }
    }
    return AMDGPUTargetLowering::LowerGlobalAddress(MFI, Op, DAG);
  }

  if (GSD->getAddressSpace() == AMDGPUAS::LOCAL_ADDRESS) {
    SDValue GA = DAG.getTargetGlobalAddress(GV, DL, MVT::i32, GSD->getOffset(),
                                            SIInstrInfo::MO_ABS32_LO);
    return DAG.getNode(AMDGPUISD::LDS, DL, MVT::i32, GA);
  }

  if (shouldEmitFixup(GV))
    return buildPCRelGlobalAddress(DAG, GV, DL, GSD->getOffset(), PtrVT);
  else if (shouldEmitPCReloc(GV))
    return buildPCRelGlobalAddress(DAG, GV, DL, GSD->getOffset(), PtrVT,
                                   SIInstrInfo::MO_REL32);

  SDValue GOTAddr = buildPCRelGlobalAddress(DAG, GV, DL, 0, PtrVT,
                                            SIInstrInfo::MO_GOTPCREL32);

  Type *Ty = PtrVT.getTypeForEVT(*DAG.getContext());
  PointerType *PtrTy = PointerType::get(Ty, AMDGPUAS::CONSTANT_ADDRESS);
  const DataLayout &DataLayout = DAG.getDataLayout();
  Align Alignment = DataLayout.getABITypeAlign(PtrTy);
  MachinePointerInfo PtrInfo
    = MachinePointerInfo::getGOT(DAG.getMachineFunction());

  return DAG.getLoad(PtrVT, DL, DAG.getEntryNode(), GOTAddr, PtrInfo, Alignment,
                     MachineMemOperand::MODereferenceable |
                         MachineMemOperand::MOInvariant);
}

SDValue SITargetLowering::copyToM0(SelectionDAG &DAG, SDValue Chain,
                                   const SDLoc &DL, SDValue V) const {
  // We can't use S_MOV_B32 directly, because there is no way to specify m0 as
  // the destination register.
  //
  // We can't use CopyToReg, because MachineCSE won't combine COPY instructions,
  // so we will end up with redundant moves to m0.
  //
  // We use a pseudo to ensure we emit s_mov_b32 with m0 as the direct result.

  // A Null SDValue creates a glue result.
  SDNode *M0 = DAG.getMachineNode(AMDGPU::SI_INIT_M0, DL, MVT::Other, MVT::Glue,
                                  V, Chain);
  return SDValue(M0, 0);
}

SDValue SITargetLowering::lowerImplicitZextParam(SelectionDAG &DAG,
                                                 SDValue Op,
                                                 MVT VT,
                                                 unsigned Offset) const {
  SDLoc SL(Op);
  SDValue Param = lowerKernargMemParameter(
      DAG, MVT::i32, MVT::i32, SL, DAG.getEntryNode(), Offset, Align(4), false);
  // The local size values will have the hi 16-bits as zero.
  return DAG.getNode(ISD::AssertZext, SL, MVT::i32, Param,
                     DAG.getValueType(VT));
}

static SDValue emitNonHSAIntrinsicError(SelectionDAG &DAG, const SDLoc &DL,
                                        EVT VT) {
  DiagnosticInfoUnsupported BadIntrin(DAG.getMachineFunction().getFunction(),
                                      "non-hsa intrinsic with hsa target",
                                      DL.getDebugLoc());
  DAG.getContext()->diagnose(BadIntrin);
  return DAG.getUNDEF(VT);
}

static SDValue emitRemovedIntrinsicError(SelectionDAG &DAG, const SDLoc &DL,
                                         EVT VT) {
  DiagnosticInfoUnsupported BadIntrin(DAG.getMachineFunction().getFunction(),
                                      "intrinsic not supported on subtarget",
                                      DL.getDebugLoc());
  DAG.getContext()->diagnose(BadIntrin);
  return DAG.getUNDEF(VT);
}

static SDValue getBuildDwordsVector(SelectionDAG &DAG, SDLoc DL,
                                    ArrayRef<SDValue> Elts) {
  assert(!Elts.empty());
  MVT Type;
  unsigned NumElts = Elts.size();

  if (NumElts <= 8) {
    Type = MVT::getVectorVT(MVT::f32, NumElts);
  } else {
    assert(Elts.size() <= 16);
    Type = MVT::v16f32;
    NumElts = 16;
  }

  SmallVector<SDValue, 16> VecElts(NumElts);
  for (unsigned i = 0; i < Elts.size(); ++i) {
    SDValue Elt = Elts[i];
    if (Elt.getValueType() != MVT::f32)
      Elt = DAG.getBitcast(MVT::f32, Elt);
    VecElts[i] = Elt;
  }
  for (unsigned i = Elts.size(); i < NumElts; ++i)
    VecElts[i] = DAG.getUNDEF(MVT::f32);

  if (NumElts == 1)
    return VecElts[0];
  return DAG.getBuildVector(Type, DL, VecElts);
}

static SDValue padEltsToUndef(SelectionDAG &DAG, const SDLoc &DL, EVT CastVT,
                              SDValue Src, int ExtraElts) {
  EVT SrcVT = Src.getValueType();

  SmallVector<SDValue, 8> Elts;

  if (SrcVT.isVector())
    DAG.ExtractVectorElements(Src, Elts);
  else
    Elts.push_back(Src);

  SDValue Undef = DAG.getUNDEF(SrcVT.getScalarType());
  while (ExtraElts--)
    Elts.push_back(Undef);

  return DAG.getBuildVector(CastVT, DL, Elts);
}

// Re-construct the required return value for a image load intrinsic.
// This is more complicated due to the optional use TexFailCtrl which means the required
// return type is an aggregate
static SDValue constructRetValue(SelectionDAG &DAG,
                                 MachineSDNode *Result,
                                 ArrayRef<EVT> ResultTypes,
                                 bool IsTexFail, bool Unpacked, bool IsD16,
                                 int DMaskPop, int NumVDataDwords,
                                 const SDLoc &DL) {
  // Determine the required return type. This is the same regardless of IsTexFail flag
  EVT ReqRetVT = ResultTypes[0];
  int ReqRetNumElts = ReqRetVT.isVector() ? ReqRetVT.getVectorNumElements() : 1;
  int NumDataDwords = (!IsD16 || (IsD16 && Unpacked)) ?
    ReqRetNumElts : (ReqRetNumElts + 1) / 2;

  int MaskPopDwords = (!IsD16 || (IsD16 && Unpacked)) ?
    DMaskPop : (DMaskPop + 1) / 2;

  MVT DataDwordVT = NumDataDwords == 1 ?
    MVT::i32 : MVT::getVectorVT(MVT::i32, NumDataDwords);

  MVT MaskPopVT = MaskPopDwords == 1 ?
    MVT::i32 : MVT::getVectorVT(MVT::i32, MaskPopDwords);

  SDValue Data(Result, 0);
  SDValue TexFail;

  if (DMaskPop > 0 && Data.getValueType() != MaskPopVT) {
    SDValue ZeroIdx = DAG.getConstant(0, DL, MVT::i32);
    if (MaskPopVT.isVector()) {
      Data = DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, MaskPopVT,
                         SDValue(Result, 0), ZeroIdx);
    } else {
      Data = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, MaskPopVT,
                         SDValue(Result, 0), ZeroIdx);
    }
  }

  if (DataDwordVT.isVector())
    Data = padEltsToUndef(DAG, DL, DataDwordVT, Data,
                          NumDataDwords - MaskPopDwords);

  if (IsD16)
    Data = adjustLoadValueTypeImpl(Data, ReqRetVT, DL, DAG, Unpacked);

  EVT LegalReqRetVT = ReqRetVT;
  if (!ReqRetVT.isVector()) {
    if (!Data.getValueType().isInteger())
      Data = DAG.getNode(ISD::BITCAST, DL,
                         Data.getValueType().changeTypeToInteger(), Data);
    Data = DAG.getNode(ISD::TRUNCATE, DL, ReqRetVT.changeTypeToInteger(), Data);
  } else {
    // We need to widen the return vector to a legal type
    if ((ReqRetVT.getVectorNumElements() % 2) == 1 &&
        ReqRetVT.getVectorElementType().getSizeInBits() == 16) {
      LegalReqRetVT =
          EVT::getVectorVT(*DAG.getContext(), ReqRetVT.getVectorElementType(),
                           ReqRetVT.getVectorNumElements() + 1);
    }
  }
  Data = DAG.getNode(ISD::BITCAST, DL, LegalReqRetVT, Data);

  if (IsTexFail) {
    TexFail =
        DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, MVT::i32, SDValue(Result, 0),
                    DAG.getConstant(MaskPopDwords, DL, MVT::i32));

    return DAG.getMergeValues({Data, TexFail, SDValue(Result, 1)}, DL);
  }

  if (Result->getNumValues() == 1)
    return Data;

  return DAG.getMergeValues({Data, SDValue(Result, 1)}, DL);
}

static bool parseTexFail(SDValue TexFailCtrl, SelectionDAG &DAG, SDValue *TFE,
                         SDValue *LWE, bool &IsTexFail) {
  auto TexFailCtrlConst = cast<ConstantSDNode>(TexFailCtrl.getNode());

  uint64_t Value = TexFailCtrlConst->getZExtValue();
  if (Value) {
    IsTexFail = true;
  }

  SDLoc DL(TexFailCtrlConst);
  *TFE = DAG.getTargetConstant((Value & 0x1) ? 1 : 0, DL, MVT::i32);
  Value &= ~(uint64_t)0x1;
  *LWE = DAG.getTargetConstant((Value & 0x2) ? 1 : 0, DL, MVT::i32);
  Value &= ~(uint64_t)0x2;

  return Value == 0;
}

static void packImage16bitOpsToDwords(SelectionDAG &DAG, SDValue Op,
                                      MVT PackVectorVT,
                                      SmallVectorImpl<SDValue> &PackedAddrs,
                                      unsigned DimIdx, unsigned EndIdx,
                                      unsigned NumGradients) {
  SDLoc DL(Op);
  for (unsigned I = DimIdx; I < EndIdx; I++) {
    SDValue Addr = Op.getOperand(I);

    // Gradients are packed with undef for each coordinate.
    // In <hi 16 bit>,<lo 16 bit> notation, the registers look like this:
    // 1D: undef,dx/dh; undef,dx/dv
    // 2D: dy/dh,dx/dh; dy/dv,dx/dv
    // 3D: dy/dh,dx/dh; undef,dz/dh; dy/dv,dx/dv; undef,dz/dv
    if (((I + 1) >= EndIdx) ||
        ((NumGradients / 2) % 2 == 1 && (I == DimIdx + (NumGradients / 2) - 1 ||
                                         I == DimIdx + NumGradients - 1))) {
      if (Addr.getValueType() != MVT::i16)
        Addr = DAG.getBitcast(MVT::i16, Addr);
      Addr = DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i32, Addr);
    } else {
      Addr = DAG.getBuildVector(PackVectorVT, DL, {Addr, Op.getOperand(I + 1)});
      I++;
    }
    Addr = DAG.getBitcast(MVT::f32, Addr);
    PackedAddrs.push_back(Addr);
  }
}

SDValue SITargetLowering::lowerImage(SDValue Op,
                                     const AMDGPU::ImageDimIntrinsicInfo *Intr,
                                     SelectionDAG &DAG, bool WithChain) const {
  SDLoc DL(Op);
  MachineFunction &MF = DAG.getMachineFunction();
  const GCNSubtarget* ST = &MF.getSubtarget<GCNSubtarget>();
  const AMDGPU::MIMGBaseOpcodeInfo *BaseOpcode =
      AMDGPU::getMIMGBaseOpcodeInfo(Intr->BaseOpcode);
  const AMDGPU::MIMGDimInfo *DimInfo = AMDGPU::getMIMGDimInfo(Intr->Dim);
  unsigned IntrOpcode = Intr->BaseOpcode;
  bool IsGFX10Plus = AMDGPU::isGFX10Plus(*Subtarget);

  SmallVector<EVT, 3> ResultTypes(Op->values());
  SmallVector<EVT, 3> OrigResultTypes(Op->values());
  bool IsD16 = false;
  bool IsG16 = false;
  bool IsA16 = false;
  SDValue VData;
  int NumVDataDwords;
  bool AdjustRetType = false;

  // Offset of intrinsic arguments
  const unsigned ArgOffset = WithChain ? 2 : 1;

  unsigned DMask;
  unsigned DMaskLanes = 0;

  if (BaseOpcode->Atomic) {
    VData = Op.getOperand(2);

    bool Is64Bit = VData.getValueType() == MVT::i64;
    if (BaseOpcode->AtomicX2) {
      SDValue VData2 = Op.getOperand(3);
      VData = DAG.getBuildVector(Is64Bit ? MVT::v2i64 : MVT::v2i32, DL,
                                 {VData, VData2});
      if (Is64Bit)
        VData = DAG.getBitcast(MVT::v4i32, VData);

      ResultTypes[0] = Is64Bit ? MVT::v2i64 : MVT::v2i32;
      DMask = Is64Bit ? 0xf : 0x3;
      NumVDataDwords = Is64Bit ? 4 : 2;
    } else {
      DMask = Is64Bit ? 0x3 : 0x1;
      NumVDataDwords = Is64Bit ? 2 : 1;
    }
  } else {
    auto *DMaskConst =
        cast<ConstantSDNode>(Op.getOperand(ArgOffset + Intr->DMaskIndex));
    DMask = DMaskConst->getZExtValue();
    DMaskLanes = BaseOpcode->Gather4 ? 4 : countPopulation(DMask);

    if (BaseOpcode->Store) {
      VData = Op.getOperand(2);

      MVT StoreVT = VData.getSimpleValueType();
      if (StoreVT.getScalarType() == MVT::f16) {
        if (!Subtarget->hasD16Images() || !BaseOpcode->HasD16)
          return Op; // D16 is unsupported for this instruction

        IsD16 = true;
        VData = handleD16VData(VData, DAG, true);
      }

      NumVDataDwords = (VData.getValueType().getSizeInBits() + 31) / 32;
    } else {
      // Work out the num dwords based on the dmask popcount and underlying type
      // and whether packing is supported.
      MVT LoadVT = ResultTypes[0].getSimpleVT();
      if (LoadVT.getScalarType() == MVT::f16) {
        if (!Subtarget->hasD16Images() || !BaseOpcode->HasD16)
          return Op; // D16 is unsupported for this instruction

        IsD16 = true;
      }

      // Confirm that the return type is large enough for the dmask specified
      if ((LoadVT.isVector() && LoadVT.getVectorNumElements() < DMaskLanes) ||
          (!LoadVT.isVector() && DMaskLanes > 1))
          return Op;

      // The sq block of gfx8 and gfx9 do not estimate register use correctly
      // for d16 image_gather4, image_gather4_l, and image_gather4_lz
      // instructions.
      if (IsD16 && !Subtarget->hasUnpackedD16VMem() &&
          !(BaseOpcode->Gather4 && Subtarget->hasImageGather4D16Bug()))
        NumVDataDwords = (DMaskLanes + 1) / 2;
      else
        NumVDataDwords = DMaskLanes;

      AdjustRetType = true;
    }
  }

  unsigned VAddrEnd = ArgOffset + Intr->VAddrEnd;
  SmallVector<SDValue, 4> VAddrs;

  // Check for 16 bit addresses or derivatives and pack if true.
  MVT VAddrVT =
      Op.getOperand(ArgOffset + Intr->GradientStart).getSimpleValueType();
  MVT VAddrScalarVT = VAddrVT.getScalarType();
  MVT GradPackVectorVT = VAddrScalarVT == MVT::f16 ? MVT::v2f16 : MVT::v2i16;
  IsG16 = VAddrScalarVT == MVT::f16 || VAddrScalarVT == MVT::i16;

  VAddrVT = Op.getOperand(ArgOffset + Intr->CoordStart).getSimpleValueType();
  VAddrScalarVT = VAddrVT.getScalarType();
  MVT AddrPackVectorVT = VAddrScalarVT == MVT::f16 ? MVT::v2f16 : MVT::v2i16;
  IsA16 = VAddrScalarVT == MVT::f16 || VAddrScalarVT == MVT::i16;

  // Push back extra arguments.
  for (unsigned I = Intr->VAddrStart; I < Intr->GradientStart; I++) {
    if (IsA16 && (Op.getOperand(ArgOffset + I).getValueType() == MVT::f16)) {
      assert(I == Intr->BiasIndex && "Got unexpected 16-bit extra argument");
      // Special handling of bias when A16 is on. Bias is of type half but
      // occupies full 32-bit.
      SDValue Bias = DAG.getBuildVector(
          MVT::v2f16, DL,
          {Op.getOperand(ArgOffset + I), DAG.getUNDEF(MVT::f16)});
      VAddrs.push_back(Bias);
    } else {
      assert((!IsA16 || Intr->NumBiasArgs == 0 || I != Intr->BiasIndex) &&
             "Bias needs to be converted to 16 bit in A16 mode");
      VAddrs.push_back(Op.getOperand(ArgOffset + I));
    }
  }

  if (BaseOpcode->Gradients && !ST->hasG16() && (IsA16 != IsG16)) {
    // 16 bit gradients are supported, but are tied to the A16 control
    // so both gradients and addresses must be 16 bit
    LLVM_DEBUG(
        dbgs() << "Failed to lower image intrinsic: 16 bit addresses "
                  "require 16 bit args for both gradients and addresses");
    return Op;
  }

  if (IsA16) {
    if (!ST->hasA16()) {
      LLVM_DEBUG(dbgs() << "Failed to lower image intrinsic: Target does not "
                           "support 16 bit addresses\n");
      return Op;
    }
  }

  // We've dealt with incorrect input so we know that if IsA16, IsG16
  // are set then we have to compress/pack operands (either address,
  // gradient or both)
  // In the case where a16 and gradients are tied (no G16 support) then we
  // have already verified that both IsA16 and IsG16 are true
  if (BaseOpcode->Gradients && IsG16 && ST->hasG16()) {
    // Activate g16
    const AMDGPU::MIMGG16MappingInfo *G16MappingInfo =
        AMDGPU::getMIMGG16MappingInfo(Intr->BaseOpcode);
    IntrOpcode = G16MappingInfo->G16; // set new opcode to variant with _g16
  }

  // Add gradients (packed or unpacked)
  if (IsG16) {
    // Pack the gradients
    // const int PackEndIdx = IsA16 ? VAddrEnd : (ArgOffset + Intr->CoordStart);
    packImage16bitOpsToDwords(DAG, Op, GradPackVectorVT, VAddrs,
                              ArgOffset + Intr->GradientStart,
                              ArgOffset + Intr->CoordStart, Intr->NumGradients);
  } else {
    for (unsigned I = ArgOffset + Intr->GradientStart;
         I < ArgOffset + Intr->CoordStart; I++)
      VAddrs.push_back(Op.getOperand(I));
  }

  // Add addresses (packed or unpacked)
  if (IsA16) {
    packImage16bitOpsToDwords(DAG, Op, AddrPackVectorVT, VAddrs,
                              ArgOffset + Intr->CoordStart, VAddrEnd,
                              0 /* No gradients */);
  } else {
    // Add uncompressed address
    for (unsigned I = ArgOffset + Intr->CoordStart; I < VAddrEnd; I++)
      VAddrs.push_back(Op.getOperand(I));
  }

  // If the register allocator cannot place the address registers contiguously
  // without introducing moves, then using the non-sequential address encoding
  // is always preferable, since it saves VALU instructions and is usually a
  // wash in terms of code size or even better.
  //
  // However, we currently have no way of hinting to the register allocator that
  // MIMG addresses should be placed contiguously when it is possible to do so,
  // so force non-NSA for the common 2-address case as a heuristic.
  //
  // SIShrinkInstructions will convert NSA encodings to non-NSA after register
  // allocation when possible.
  bool UseNSA = ST->hasFeature(AMDGPU::FeatureNSAEncoding) &&
                VAddrs.size() >= 3 &&
                VAddrs.size() <= (unsigned)ST->getNSAMaxSize();
  SDValue VAddr;
  if (!UseNSA)
    VAddr = getBuildDwordsVector(DAG, DL, VAddrs);

  SDValue True = DAG.getTargetConstant(1, DL, MVT::i1);
  SDValue False = DAG.getTargetConstant(0, DL, MVT::i1);
  SDValue Unorm;
  if (!BaseOpcode->Sampler) {
    Unorm = True;
  } else {
    auto UnormConst =
        cast<ConstantSDNode>(Op.getOperand(ArgOffset + Intr->UnormIndex));

    Unorm = UnormConst->getZExtValue() ? True : False;
  }

  SDValue TFE;
  SDValue LWE;
  SDValue TexFail = Op.getOperand(ArgOffset + Intr->TexFailCtrlIndex);
  bool IsTexFail = false;
  if (!parseTexFail(TexFail, DAG, &TFE, &LWE, IsTexFail))
    return Op;

  if (IsTexFail) {
    if (!DMaskLanes) {
      // Expecting to get an error flag since TFC is on - and dmask is 0
      // Force dmask to be at least 1 otherwise the instruction will fail
      DMask = 0x1;
      DMaskLanes = 1;
      NumVDataDwords = 1;
    }
    NumVDataDwords += 1;
    AdjustRetType = true;
  }

  // Has something earlier tagged that the return type needs adjusting
  // This happens if the instruction is a load or has set TexFailCtrl flags
  if (AdjustRetType) {
    // NumVDataDwords reflects the true number of dwords required in the return type
    if (DMaskLanes == 0 && !BaseOpcode->Store) {
      // This is a no-op load. This can be eliminated
      SDValue Undef = DAG.getUNDEF(Op.getValueType());
      if (isa<MemSDNode>(Op))
        return DAG.getMergeValues({Undef, Op.getOperand(0)}, DL);
      return Undef;
    }

    EVT NewVT = NumVDataDwords > 1 ?
                  EVT::getVectorVT(*DAG.getContext(), MVT::i32, NumVDataDwords)
                : MVT::i32;

    ResultTypes[0] = NewVT;
    if (ResultTypes.size() == 3) {
      // Original result was aggregate type used for TexFailCtrl results
      // The actual instruction returns as a vector type which has now been
      // created. Remove the aggregate result.
      ResultTypes.erase(&ResultTypes[1]);
    }
  }

  unsigned CPol = cast<ConstantSDNode>(
      Op.getOperand(ArgOffset + Intr->CachePolicyIndex))->getZExtValue();
  if (BaseOpcode->Atomic)
    CPol |= AMDGPU::CPol::GLC; // TODO no-return optimization
  if (CPol & ~AMDGPU::CPol::ALL)
    return Op;

  SmallVector<SDValue, 26> Ops;
  if (BaseOpcode->Store || BaseOpcode->Atomic)
    Ops.push_back(VData); // vdata
  if (UseNSA)
    append_range(Ops, VAddrs);
  else
    Ops.push_back(VAddr);
  Ops.push_back(Op.getOperand(ArgOffset + Intr->RsrcIndex));
  if (BaseOpcode->Sampler)
    Ops.push_back(Op.getOperand(ArgOffset + Intr->SampIndex));
  Ops.push_back(DAG.getTargetConstant(DMask, DL, MVT::i32));
  if (IsGFX10Plus)
    Ops.push_back(DAG.getTargetConstant(DimInfo->Encoding, DL, MVT::i32));
  Ops.push_back(Unorm);
  Ops.push_back(DAG.getTargetConstant(CPol, DL, MVT::i32));
  Ops.push_back(IsA16 &&  // r128, a16 for gfx9
                ST->hasFeature(AMDGPU::FeatureR128A16) ? True : False);
  if (IsGFX10Plus)
    Ops.push_back(IsA16 ? True : False);
  if (!Subtarget->hasGFX90AInsts()) {
    Ops.push_back(TFE); //tfe
  } else if (cast<ConstantSDNode>(TFE)->getZExtValue()) {
    report_fatal_error("TFE is not supported on this GPU");
  }
  Ops.push_back(LWE); // lwe
  if (!IsGFX10Plus)
    Ops.push_back(DimInfo->DA ? True : False);
  if (BaseOpcode->HasD16)
    Ops.push_back(IsD16 ? True : False);
  if (isa<MemSDNode>(Op))
    Ops.push_back(Op.getOperand(0)); // chain

  int NumVAddrDwords =
      UseNSA ? VAddrs.size() : VAddr.getValueType().getSizeInBits() / 32;
  int Opcode = -1;

  if (IsGFX10Plus) {
    Opcode = AMDGPU::getMIMGOpcode(IntrOpcode,
                                   UseNSA ? AMDGPU::MIMGEncGfx10NSA
                                          : AMDGPU::MIMGEncGfx10Default,
                                   NumVDataDwords, NumVAddrDwords);
  } else {
    if (Subtarget->hasGFX90AInsts()) {
      Opcode = AMDGPU::getMIMGOpcode(IntrOpcode, AMDGPU::MIMGEncGfx90a,
                                     NumVDataDwords, NumVAddrDwords);
      if (Opcode == -1)
        report_fatal_error(
            "requested image instruction is not supported on this GPU");
    }
    if (Opcode == -1 &&
        Subtarget->getGeneration() >= AMDGPUSubtarget::VOLCANIC_ISLANDS)
      Opcode = AMDGPU::getMIMGOpcode(IntrOpcode, AMDGPU::MIMGEncGfx8,
                                     NumVDataDwords, NumVAddrDwords);
    if (Opcode == -1)
      Opcode = AMDGPU::getMIMGOpcode(IntrOpcode, AMDGPU::MIMGEncGfx6,
                                     NumVDataDwords, NumVAddrDwords);
  }
  assert(Opcode != -1);

  MachineSDNode *NewNode = DAG.getMachineNode(Opcode, DL, ResultTypes, Ops);
  if (auto MemOp = dyn_cast<MemSDNode>(Op)) {
    MachineMemOperand *MemRef = MemOp->getMemOperand();
    DAG.setNodeMemRefs(NewNode, {MemRef});
  }

  if (BaseOpcode->AtomicX2) {
    SmallVector<SDValue, 1> Elt;
    DAG.ExtractVectorElements(SDValue(NewNode, 0), Elt, 0, 1);
    return DAG.getMergeValues({Elt[0], SDValue(NewNode, 1)}, DL);
  }
  if (BaseOpcode->Store)
    return SDValue(NewNode, 0);
  return constructRetValue(DAG, NewNode,
                           OrigResultTypes, IsTexFail,
                           Subtarget->hasUnpackedD16VMem(), IsD16,
                           DMaskLanes, NumVDataDwords, DL);
}

SDValue SITargetLowering::lowerSBuffer(EVT VT, SDLoc DL, SDValue Rsrc,
                                       SDValue Offset, SDValue CachePolicy,
                                       SelectionDAG &DAG) const {
  MachineFunction &MF = DAG.getMachineFunction();

  const DataLayout &DataLayout = DAG.getDataLayout();
  Align Alignment =
      DataLayout.getABITypeAlign(VT.getTypeForEVT(*DAG.getContext()));

  MachineMemOperand *MMO = MF.getMachineMemOperand(
      MachinePointerInfo(),
      MachineMemOperand::MOLoad | MachineMemOperand::MODereferenceable |
          MachineMemOperand::MOInvariant,
      VT.getStoreSize(), Alignment);

  if (!Offset->isDivergent()) {
    SDValue Ops[] = {
        Rsrc,
        Offset, // Offset
        CachePolicy
    };

    // Widen vec3 load to vec4.
    if (VT.isVector() && VT.getVectorNumElements() == 3) {
      EVT WidenedVT =
          EVT::getVectorVT(*DAG.getContext(), VT.getVectorElementType(), 4);
      auto WidenedOp = DAG.getMemIntrinsicNode(
          AMDGPUISD::SBUFFER_LOAD, DL, DAG.getVTList(WidenedVT), Ops, WidenedVT,
          MF.getMachineMemOperand(MMO, 0, WidenedVT.getStoreSize()));
      auto Subvector = DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, VT, WidenedOp,
                                   DAG.getVectorIdxConstant(0, DL));
      return Subvector;
    }

    return DAG.getMemIntrinsicNode(AMDGPUISD::SBUFFER_LOAD, DL,
                                   DAG.getVTList(VT), Ops, VT, MMO);
  }

  // We have a divergent offset. Emit a MUBUF buffer load instead. We can
  // assume that the buffer is unswizzled.
  SmallVector<SDValue, 4> Loads;
  unsigned NumLoads = 1;
  MVT LoadVT = VT.getSimpleVT();
  unsigned NumElts = LoadVT.isVector() ? LoadVT.getVectorNumElements() : 1;
  assert((LoadVT.getScalarType() == MVT::i32 ||
          LoadVT.getScalarType() == MVT::f32));

  if (NumElts == 8 || NumElts == 16) {
    NumLoads = NumElts / 4;
    LoadVT = MVT::getVectorVT(LoadVT.getScalarType(), 4);
  }

  SDVTList VTList = DAG.getVTList({LoadVT, MVT::Glue});
  SDValue Ops[] = {
      DAG.getEntryNode(),                               // Chain
      Rsrc,                                             // rsrc
      DAG.getConstant(0, DL, MVT::i32),                 // vindex
      {},                                               // voffset
      {},                                               // soffset
      {},                                               // offset
      CachePolicy,                                      // cachepolicy
      DAG.getTargetConstant(0, DL, MVT::i1),            // idxen
  };

  // Use the alignment to ensure that the required offsets will fit into the
  // immediate offsets.
  setBufferOffsets(Offset, DAG, &Ops[3],
                   NumLoads > 1 ? Align(16 * NumLoads) : Align(4));

  uint64_t InstOffset = cast<ConstantSDNode>(Ops[5])->getZExtValue();
  for (unsigned i = 0; i < NumLoads; ++i) {
    Ops[5] = DAG.getTargetConstant(InstOffset + 16 * i, DL, MVT::i32);
    Loads.push_back(getMemIntrinsicNode(AMDGPUISD::BUFFER_LOAD, DL, VTList, Ops,
                                        LoadVT, MMO, DAG));
  }

  if (NumElts == 8 || NumElts == 16)
    return DAG.getNode(ISD::CONCAT_VECTORS, DL, VT, Loads);

  return Loads[0];
}

SDValue SITargetLowering::LowerINTRINSIC_WO_CHAIN(SDValue Op,
                                                  SelectionDAG &DAG) const {
  MachineFunction &MF = DAG.getMachineFunction();
  auto MFI = MF.getInfo<SIMachineFunctionInfo>();

  EVT VT = Op.getValueType();
  SDLoc DL(Op);
  unsigned IntrinsicID = cast<ConstantSDNode>(Op.getOperand(0))->getZExtValue();

  // TODO: Should this propagate fast-math-flags?

  switch (IntrinsicID) {
  case Intrinsic::amdgcn_implicit_buffer_ptr: {
    if (getSubtarget()->isAmdHsaOrMesa(MF.getFunction()))
      return emitNonHSAIntrinsicError(DAG, DL, VT);
    return getPreloadedValue(DAG, *MFI, VT,
                             AMDGPUFunctionArgInfo::IMPLICIT_BUFFER_PTR);
  }
  case Intrinsic::amdgcn_dispatch_ptr:
  case Intrinsic::amdgcn_queue_ptr: {
    if (!Subtarget->isAmdHsaOrMesa(MF.getFunction())) {
      DiagnosticInfoUnsupported BadIntrin(
          MF.getFunction(), "unsupported hsa intrinsic without hsa target",
          DL.getDebugLoc());
      DAG.getContext()->diagnose(BadIntrin);
      return DAG.getUNDEF(VT);
    }

    auto RegID = IntrinsicID == Intrinsic::amdgcn_dispatch_ptr ?
      AMDGPUFunctionArgInfo::DISPATCH_PTR : AMDGPUFunctionArgInfo::QUEUE_PTR;
    return getPreloadedValue(DAG, *MFI, VT, RegID);
  }
  case Intrinsic::amdgcn_implicitarg_ptr: {
    if (MFI->isEntryFunction())
      return getImplicitArgPtr(DAG, DL);
    return getPreloadedValue(DAG, *MFI, VT,
                             AMDGPUFunctionArgInfo::IMPLICIT_ARG_PTR);
  }
  case Intrinsic::amdgcn_kernarg_segment_ptr: {
    if (!AMDGPU::isKernel(MF.getFunction().getCallingConv())) {
      // This only makes sense to call in a kernel, so just lower to null.
      return DAG.getConstant(0, DL, VT);
    }

    return getPreloadedValue(DAG, *MFI, VT,
                             AMDGPUFunctionArgInfo::KERNARG_SEGMENT_PTR);
  }
  case Intrinsic::amdgcn_dispatch_id: {
    return getPreloadedValue(DAG, *MFI, VT, AMDGPUFunctionArgInfo::DISPATCH_ID);
  }
  case Intrinsic::amdgcn_rcp:
    return DAG.getNode(AMDGPUISD::RCP, DL, VT, Op.getOperand(1));
  case Intrinsic::amdgcn_rsq:
    return DAG.getNode(AMDGPUISD::RSQ, DL, VT, Op.getOperand(1));
  case Intrinsic::amdgcn_rsq_legacy:
    if (Subtarget->getGeneration() >= AMDGPUSubtarget::VOLCANIC_ISLANDS)
      return emitRemovedIntrinsicError(DAG, DL, VT);
    return SDValue();
  case Intrinsic::amdgcn_rcp_legacy:
    if (Subtarget->getGeneration() >= AMDGPUSubtarget::VOLCANIC_ISLANDS)
      return emitRemovedIntrinsicError(DAG, DL, VT);
    return DAG.getNode(AMDGPUISD::RCP_LEGACY, DL, VT, Op.getOperand(1));
  case Intrinsic::amdgcn_rsq_clamp: {
    if (Subtarget->getGeneration() < AMDGPUSubtarget::VOLCANIC_ISLANDS)
      return DAG.getNode(AMDGPUISD::RSQ_CLAMP, DL, VT, Op.getOperand(1));

    Type *Type = VT.getTypeForEVT(*DAG.getContext());
    APFloat Max = APFloat::getLargest(Type->getFltSemantics());
    APFloat Min = APFloat::getLargest(Type->getFltSemantics(), true);

    SDValue Rsq = DAG.getNode(AMDGPUISD::RSQ, DL, VT, Op.getOperand(1));
    SDValue Tmp = DAG.getNode(ISD::FMINNUM, DL, VT, Rsq,
                              DAG.getConstantFP(Max, DL, VT));
    return DAG.getNode(ISD::FMAXNUM, DL, VT, Tmp,
                       DAG.getConstantFP(Min, DL, VT));
  }
  case Intrinsic::r600_read_ngroups_x:
    if (Subtarget->isAmdHsaOS())
      return emitNonHSAIntrinsicError(DAG, DL, VT);

    return lowerKernargMemParameter(DAG, VT, VT, DL, DAG.getEntryNode(),
                                    SI::KernelInputOffsets::NGROUPS_X, Align(4),
                                    false);
  case Intrinsic::r600_read_ngroups_y:
    if (Subtarget->isAmdHsaOS())
      return emitNonHSAIntrinsicError(DAG, DL, VT);

    return lowerKernargMemParameter(DAG, VT, VT, DL, DAG.getEntryNode(),
                                    SI::KernelInputOffsets::NGROUPS_Y, Align(4),
                                    false);
  case Intrinsic::r600_read_ngroups_z:
    if (Subtarget->isAmdHsaOS())
      return emitNonHSAIntrinsicError(DAG, DL, VT);

    return lowerKernargMemParameter(DAG, VT, VT, DL, DAG.getEntryNode(),
                                    SI::KernelInputOffsets::NGROUPS_Z, Align(4),
                                    false);
  case Intrinsic::r600_read_global_size_x:
    if (Subtarget->isAmdHsaOS())
      return emitNonHSAIntrinsicError(DAG, DL, VT);

    return lowerKernargMemParameter(DAG, VT, VT, DL, DAG.getEntryNode(),
                                    SI::KernelInputOffsets::GLOBAL_SIZE_X,
                                    Align(4), false);
  case Intrinsic::r600_read_global_size_y:
    if (Subtarget->isAmdHsaOS())
      return emitNonHSAIntrinsicError(DAG, DL, VT);

    return lowerKernargMemParameter(DAG, VT, VT, DL, DAG.getEntryNode(),
                                    SI::KernelInputOffsets::GLOBAL_SIZE_Y,
                                    Align(4), false);
  case Intrinsic::r600_read_global_size_z:
    if (Subtarget->isAmdHsaOS())
      return emitNonHSAIntrinsicError(DAG, DL, VT);

    return lowerKernargMemParameter(DAG, VT, VT, DL, DAG.getEntryNode(),
                                    SI::KernelInputOffsets::GLOBAL_SIZE_Z,
                                    Align(4), false);
  case Intrinsic::r600_read_local_size_x:
    if (Subtarget->isAmdHsaOS())
      return emitNonHSAIntrinsicError(DAG, DL, VT);

    return lowerImplicitZextParam(DAG, Op, MVT::i16,
                                  SI::KernelInputOffsets::LOCAL_SIZE_X);
  case Intrinsic::r600_read_local_size_y:
    if (Subtarget->isAmdHsaOS())
      return emitNonHSAIntrinsicError(DAG, DL, VT);

    return lowerImplicitZextParam(DAG, Op, MVT::i16,
                                  SI::KernelInputOffsets::LOCAL_SIZE_Y);
  case Intrinsic::r600_read_local_size_z:
    if (Subtarget->isAmdHsaOS())
      return emitNonHSAIntrinsicError(DAG, DL, VT);

    return lowerImplicitZextParam(DAG, Op, MVT::i16,
                                  SI::KernelInputOffsets::LOCAL_SIZE_Z);
  case Intrinsic::amdgcn_workgroup_id_x:
    return getPreloadedValue(DAG, *MFI, VT,
                             AMDGPUFunctionArgInfo::WORKGROUP_ID_X);
  case Intrinsic::amdgcn_workgroup_id_y:
    return getPreloadedValue(DAG, *MFI, VT,
                             AMDGPUFunctionArgInfo::WORKGROUP_ID_Y);
  case Intrinsic::amdgcn_workgroup_id_z:
    return getPreloadedValue(DAG, *MFI, VT,
                             AMDGPUFunctionArgInfo::WORKGROUP_ID_Z);
  case Intrinsic::amdgcn_workitem_id_x:
    if (Subtarget->getMaxWorkitemID(MF.getFunction(), 0) == 0)
      return DAG.getConstant(0, DL, MVT::i32);

    return loadInputValue(DAG, &AMDGPU::VGPR_32RegClass, MVT::i32,
                          SDLoc(DAG.getEntryNode()),
                          MFI->getArgInfo().WorkItemIDX);
  case Intrinsic::amdgcn_workitem_id_y:
    if (Subtarget->getMaxWorkitemID(MF.getFunction(), 1) == 0)
      return DAG.getConstant(0, DL, MVT::i32);

    return loadInputValue(DAG, &AMDGPU::VGPR_32RegClass, MVT::i32,
                          SDLoc(DAG.getEntryNode()),
                          MFI->getArgInfo().WorkItemIDY);
  case Intrinsic::amdgcn_workitem_id_z:
    if (Subtarget->getMaxWorkitemID(MF.getFunction(), 2) == 0)
      return DAG.getConstant(0, DL, MVT::i32);

    return loadInputValue(DAG, &AMDGPU::VGPR_32RegClass, MVT::i32,
                          SDLoc(DAG.getEntryNode()),
                          MFI->getArgInfo().WorkItemIDZ);
  case Intrinsic::amdgcn_wavefrontsize:
    return DAG.getConstant(MF.getSubtarget<GCNSubtarget>().getWavefrontSize(),
                           SDLoc(Op), MVT::i32);
  case Intrinsic::amdgcn_s_buffer_load: {
    unsigned CPol = cast<ConstantSDNode>(Op.getOperand(3))->getZExtValue();
    if (CPol & ~AMDGPU::CPol::ALL)
      return Op;
    return lowerSBuffer(VT, DL, Op.getOperand(1), Op.getOperand(2), Op.getOperand(3),
                        DAG);
  }
  case Intrinsic::amdgcn_fdiv_fast:
    return lowerFDIV_FAST(Op, DAG);
  case Intrinsic::amdgcn_sin:
    return DAG.getNode(AMDGPUISD::SIN_HW, DL, VT, Op.getOperand(1));

  case Intrinsic::amdgcn_cos:
    return DAG.getNode(AMDGPUISD::COS_HW, DL, VT, Op.getOperand(1));

  case Intrinsic::amdgcn_mul_u24:
    return DAG.getNode(AMDGPUISD::MUL_U24, DL, VT, Op.getOperand(1), Op.getOperand(2));
  case Intrinsic::amdgcn_mul_i24:
    return DAG.getNode(AMDGPUISD::MUL_I24, DL, VT, Op.getOperand(1), Op.getOperand(2));

  case Intrinsic::amdgcn_log_clamp: {
    if (Subtarget->getGeneration() < AMDGPUSubtarget::VOLCANIC_ISLANDS)
      return SDValue();

    return emitRemovedIntrinsicError(DAG, DL, VT);
  }
  case Intrinsic::amdgcn_ldexp:
    return DAG.getNode(AMDGPUISD::LDEXP, DL, VT,
                       Op.getOperand(1), Op.getOperand(2));

  case Intrinsic::amdgcn_fract:
    return DAG.getNode(AMDGPUISD::FRACT, DL, VT, Op.getOperand(1));

  case Intrinsic::amdgcn_class:
    return DAG.getNode(AMDGPUISD::FP_CLASS, DL, VT,
                       Op.getOperand(1), Op.getOperand(2));
  case Intrinsic::amdgcn_div_fmas:
    return DAG.getNode(AMDGPUISD::DIV_FMAS, DL, VT,
                       Op.getOperand(1), Op.getOperand(2), Op.getOperand(3),
                       Op.getOperand(4));

  case Intrinsic::amdgcn_div_fixup:
    return DAG.getNode(AMDGPUISD::DIV_FIXUP, DL, VT,
                       Op.getOperand(1), Op.getOperand(2), Op.getOperand(3));

  case Intrinsic::amdgcn_div_scale: {
    const ConstantSDNode *Param = cast<ConstantSDNode>(Op.getOperand(3));

    // Translate to the operands expected by the machine instruction. The
    // first parameter must be the same as the first instruction.
    SDValue Numerator = Op.getOperand(1);
    SDValue Denominator = Op.getOperand(2);

    // Note this order is opposite of the machine instruction's operations,
    // which is s0.f = Quotient, s1.f = Denominator, s2.f = Numerator. The
    // intrinsic has the numerator as the first operand to match a normal
    // division operation.

    SDValue Src0 = Param->isAllOnes() ? Numerator : Denominator;

    return DAG.getNode(AMDGPUISD::DIV_SCALE, DL, Op->getVTList(), Src0,
                       Denominator, Numerator);
  }
  case Intrinsic::amdgcn_icmp: {
    // There is a Pat that handles this variant, so return it as-is.
    if (Op.getOperand(1).getValueType() == MVT::i1 &&
        Op.getConstantOperandVal(2) == 0 &&
        Op.getConstantOperandVal(3) == ICmpInst::Predicate::ICMP_NE)
      return Op;
    return lowerICMPIntrinsic(*this, Op.getNode(), DAG);
  }
  case Intrinsic::amdgcn_fcmp: {
    return lowerFCMPIntrinsic(*this, Op.getNode(), DAG);
  }
  case Intrinsic::amdgcn_ballot:
    return lowerBALLOTIntrinsic(*this, Op.getNode(), DAG);
  case Intrinsic::amdgcn_fmed3:
    return DAG.getNode(AMDGPUISD::FMED3, DL, VT,
                       Op.getOperand(1), Op.getOperand(2), Op.getOperand(3));
  case Intrinsic::amdgcn_fdot2:
    return DAG.getNode(AMDGPUISD::FDOT2, DL, VT,
                       Op.getOperand(1), Op.getOperand(2), Op.getOperand(3),
                       Op.getOperand(4));
  case Intrinsic::amdgcn_fmul_legacy:
    return DAG.getNode(AMDGPUISD::FMUL_LEGACY, DL, VT,
                       Op.getOperand(1), Op.getOperand(2));
  case Intrinsic::amdgcn_sffbh:
    return DAG.getNode(AMDGPUISD::FFBH_I32, DL, VT, Op.getOperand(1));
  case Intrinsic::amdgcn_sbfe:
    return DAG.getNode(AMDGPUISD::BFE_I32, DL, VT,
                       Op.getOperand(1), Op.getOperand(2), Op.getOperand(3));
  case Intrinsic::amdgcn_ubfe:
    return DAG.getNode(AMDGPUISD::BFE_U32, DL, VT,
                       Op.getOperand(1), Op.getOperand(2), Op.getOperand(3));
  case Intrinsic::amdgcn_cvt_pkrtz:
  case Intrinsic::amdgcn_cvt_pknorm_i16:
  case Intrinsic::amdgcn_cvt_pknorm_u16:
  case Intrinsic::amdgcn_cvt_pk_i16:
  case Intrinsic::amdgcn_cvt_pk_u16: {
    // FIXME: Stop adding cast if v2f16/v2i16 are legal.
    EVT VT = Op.getValueType();
    unsigned Opcode;

    if (IntrinsicID == Intrinsic::amdgcn_cvt_pkrtz)
      Opcode = AMDGPUISD::CVT_PKRTZ_F16_F32;
    else if (IntrinsicID == Intrinsic::amdgcn_cvt_pknorm_i16)
      Opcode = AMDGPUISD::CVT_PKNORM_I16_F32;
    else if (IntrinsicID == Intrinsic::amdgcn_cvt_pknorm_u16)
      Opcode = AMDGPUISD::CVT_PKNORM_U16_F32;
    else if (IntrinsicID == Intrinsic::amdgcn_cvt_pk_i16)
      Opcode = AMDGPUISD::CVT_PK_I16_I32;
    else
      Opcode = AMDGPUISD::CVT_PK_U16_U32;

    if (isTypeLegal(VT))
      return DAG.getNode(Opcode, DL, VT, Op.getOperand(1), Op.getOperand(2));

    SDValue Node = DAG.getNode(Opcode, DL, MVT::i32,
                               Op.getOperand(1), Op.getOperand(2));
    return DAG.getNode(ISD::BITCAST, DL, VT, Node);
  }
  case Intrinsic::amdgcn_fmad_ftz:
    return DAG.getNode(AMDGPUISD::FMAD_FTZ, DL, VT, Op.getOperand(1),
                       Op.getOperand(2), Op.getOperand(3));

  case Intrinsic::amdgcn_if_break:
    return SDValue(DAG.getMachineNode(AMDGPU::SI_IF_BREAK, DL, VT,
                                      Op->getOperand(1), Op->getOperand(2)), 0);

  case Intrinsic::amdgcn_groupstaticsize: {
    Triple::OSType OS = getTargetMachine().getTargetTriple().getOS();
    if (OS == Triple::AMDHSA || OS == Triple::AMDPAL)
      return Op;

    const Module *M = MF.getFunction().getParent();
    const GlobalValue *GV =
        M->getNamedValue(Intrinsic::getName(Intrinsic::amdgcn_groupstaticsize));
    SDValue GA = DAG.getTargetGlobalAddress(GV, DL, MVT::i32, 0,
                                            SIInstrInfo::MO_ABS32_LO);
    return {DAG.getMachineNode(AMDGPU::S_MOV_B32, DL, MVT::i32, GA), 0};
  }
  case Intrinsic::amdgcn_is_shared:
  case Intrinsic::amdgcn_is_private: {
    SDLoc SL(Op);
    unsigned AS = (IntrinsicID == Intrinsic::amdgcn_is_shared) ?
      AMDGPUAS::LOCAL_ADDRESS : AMDGPUAS::PRIVATE_ADDRESS;
    SDValue Aperture = getSegmentAperture(AS, SL, DAG);
    SDValue SrcVec = DAG.getNode(ISD::BITCAST, DL, MVT::v2i32,
                                 Op.getOperand(1));

    SDValue SrcHi = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, SL, MVT::i32, SrcVec,
                                DAG.getConstant(1, SL, MVT::i32));
    return DAG.getSetCC(SL, MVT::i1, SrcHi, Aperture, ISD::SETEQ);
  }
  case Intrinsic::amdgcn_perm:
    return DAG.getNode(AMDGPUISD::PERM, DL, MVT::i32, Op.getOperand(1),
                       Op.getOperand(2), Op.getOperand(3));
  case Intrinsic::amdgcn_reloc_constant: {
    Module *M = const_cast<Module *>(MF.getFunction().getParent());
    const MDNode *Metadata = cast<MDNodeSDNode>(Op.getOperand(1))->getMD();
    auto SymbolName = cast<MDString>(Metadata->getOperand(0))->getString();
    auto RelocSymbol = cast<GlobalVariable>(
        M->getOrInsertGlobal(SymbolName, Type::getInt32Ty(M->getContext())));
    SDValue GA = DAG.getTargetGlobalAddress(RelocSymbol, DL, MVT::i32, 0,
                                            SIInstrInfo::MO_ABS32_LO);
    return {DAG.getMachineNode(AMDGPU::S_MOV_B32, DL, MVT::i32, GA), 0};
  }
  default:
    if (const AMDGPU::ImageDimIntrinsicInfo *ImageDimIntr =
            AMDGPU::getImageDimIntrinsicInfo(IntrinsicID))
      return lowerImage(Op, ImageDimIntr, DAG, false);

    return Op;
  }
}

/// Update \p MMO based on the offset inputs to an intrinsic.
static void updateBufferMMO(MachineMemOperand *MMO, SDValue VOffset,
                            SDValue SOffset, SDValue Offset,
                            SDValue VIndex = SDValue()) {
  if (!isa<ConstantSDNode>(VOffset) || !isa<ConstantSDNode>(SOffset) ||
      !isa<ConstantSDNode>(Offset)) {
    // The combined offset is not known to be constant, so we cannot represent
    // it in the MMO. Give up.
    MMO->setValue((Value *)nullptr);
    return;
  }

  if (VIndex && (!isa<ConstantSDNode>(VIndex) ||
                 !cast<ConstantSDNode>(VIndex)->isZero())) {
    // The strided index component of the address is not known to be zero, so we
    // cannot represent it in the MMO. Give up.
    MMO->setValue((Value *)nullptr);
    return;
  }

  MMO->setOffset(cast<ConstantSDNode>(VOffset)->getSExtValue() +
                 cast<ConstantSDNode>(SOffset)->getSExtValue() +
                 cast<ConstantSDNode>(Offset)->getSExtValue());
}

SDValue SITargetLowering::lowerRawBufferAtomicIntrin(SDValue Op,
                                                     SelectionDAG &DAG,
                                                     unsigned NewOpcode) const {
  SDLoc DL(Op);

  SDValue VData = Op.getOperand(2);
  auto Offsets = splitBufferOffsets(Op.getOperand(4), DAG);
  SDValue Ops[] = {
    Op.getOperand(0), // Chain
    VData,            // vdata
    Op.getOperand(3), // rsrc
    DAG.getConstant(0, DL, MVT::i32), // vindex
    Offsets.first,    // voffset
    Op.getOperand(5), // soffset
    Offsets.second,   // offset
    Op.getOperand(6), // cachepolicy
    DAG.getTargetConstant(0, DL, MVT::i1), // idxen
  };

  auto *M = cast<MemSDNode>(Op);
  updateBufferMMO(M->getMemOperand(), Ops[4], Ops[5], Ops[6]);

  EVT MemVT = VData.getValueType();
  return DAG.getMemIntrinsicNode(NewOpcode, DL, Op->getVTList(), Ops, MemVT,
                                 M->getMemOperand());
}

// Return a value to use for the idxen operand by examining the vindex operand.
static unsigned getIdxEn(SDValue VIndex) {
  if (auto VIndexC = dyn_cast<ConstantSDNode>(VIndex))
    // No need to set idxen if vindex is known to be zero.
    return VIndexC->getZExtValue() != 0;
  return 1;
}

SDValue
SITargetLowering::lowerStructBufferAtomicIntrin(SDValue Op, SelectionDAG &DAG,
                                                unsigned NewOpcode) const {
  SDLoc DL(Op);

  SDValue VData = Op.getOperand(2);
  auto Offsets = splitBufferOffsets(Op.getOperand(5), DAG);
  SDValue Ops[] = {
    Op.getOperand(0), // Chain
    VData,            // vdata
    Op.getOperand(3), // rsrc
    Op.getOperand(4), // vindex
    Offsets.first,    // voffset
    Op.getOperand(6), // soffset
    Offsets.second,   // offset
    Op.getOperand(7), // cachepolicy
    DAG.getTargetConstant(1, DL, MVT::i1), // idxen
  };

  auto *M = cast<MemSDNode>(Op);
  updateBufferMMO(M->getMemOperand(), Ops[4], Ops[5], Ops[6], Ops[3]);

  EVT MemVT = VData.getValueType();
  return DAG.getMemIntrinsicNode(NewOpcode, DL, Op->getVTList(), Ops, MemVT,
                                 M->getMemOperand());
}

SDValue SITargetLowering::LowerINTRINSIC_W_CHAIN(SDValue Op,
                                                 SelectionDAG &DAG) const {
  unsigned IntrID = cast<ConstantSDNode>(Op.getOperand(1))->getZExtValue();
  SDLoc DL(Op);

  switch (IntrID) {
  case Intrinsic::amdgcn_ds_ordered_add:
  case Intrinsic::amdgcn_ds_ordered_swap: {
    MemSDNode *M = cast<MemSDNode>(Op);
    SDValue Chain = M->getOperand(0);
    SDValue M0 = M->getOperand(2);
    SDValue Value = M->getOperand(3);
    unsigned IndexOperand = M->getConstantOperandVal(7);
    unsigned WaveRelease = M->getConstantOperandVal(8);
    unsigned WaveDone = M->getConstantOperandVal(9);

    unsigned OrderedCountIndex = IndexOperand & 0x3f;
    IndexOperand &= ~0x3f;
    unsigned CountDw = 0;

    if (Subtarget->getGeneration() >= AMDGPUSubtarget::GFX10) {
      CountDw = (IndexOperand >> 24) & 0xf;
      IndexOperand &= ~(0xf << 24);

      if (CountDw < 1 || CountDw > 4) {
        report_fatal_error(
            "ds_ordered_count: dword count must be between 1 and 4");
      }
    }

    if (IndexOperand)
      report_fatal_error("ds_ordered_count: bad index operand");

    if (WaveDone && !WaveRelease)
      report_fatal_error("ds_ordered_count: wave_done requires wave_release");

    unsigned Instruction = IntrID == Intrinsic::amdgcn_ds_ordered_add ? 0 : 1;
    unsigned ShaderType =
        SIInstrInfo::getDSShaderTypeValue(DAG.getMachineFunction());
    unsigned Offset0 = OrderedCountIndex << 2;
    unsigned Offset1 = WaveRelease | (WaveDone << 1) | (ShaderType << 2) |
                       (Instruction << 4);

    if (Subtarget->getGeneration() >= AMDGPUSubtarget::GFX10)
      Offset1 |= (CountDw - 1) << 6;

    unsigned Offset = Offset0 | (Offset1 << 8);

    SDValue Ops[] = {
      Chain,
      Value,
      DAG.getTargetConstant(Offset, DL, MVT::i16),
      copyToM0(DAG, Chain, DL, M0).getValue(1), // Glue
    };
    return DAG.getMemIntrinsicNode(AMDGPUISD::DS_ORDERED_COUNT, DL,
                                   M->getVTList(), Ops, M->getMemoryVT(),
                                   M->getMemOperand());
  }
  case Intrinsic::amdgcn_ds_fadd: {
    MemSDNode *M = cast<MemSDNode>(Op);
    unsigned Opc;
    switch (IntrID) {
    case Intrinsic::amdgcn_ds_fadd:
      Opc = ISD::ATOMIC_LOAD_FADD;
      break;
    }

    return DAG.getAtomic(Opc, SDLoc(Op), M->getMemoryVT(),
                         M->getOperand(0), M->getOperand(2), M->getOperand(3),
                         M->getMemOperand());
  }
  case Intrinsic::amdgcn_atomic_inc:
  case Intrinsic::amdgcn_atomic_dec:
  case Intrinsic::amdgcn_ds_fmin:
  case Intrinsic::amdgcn_ds_fmax: {
    MemSDNode *M = cast<MemSDNode>(Op);
    unsigned Opc;
    switch (IntrID) {
    case Intrinsic::amdgcn_atomic_inc:
      Opc = AMDGPUISD::ATOMIC_INC;
      break;
    case Intrinsic::amdgcn_atomic_dec:
      Opc = AMDGPUISD::ATOMIC_DEC;
      break;
    case Intrinsic::amdgcn_ds_fmin:
      Opc = AMDGPUISD::ATOMIC_LOAD_FMIN;
      break;
    case Intrinsic::amdgcn_ds_fmax:
      Opc = AMDGPUISD::ATOMIC_LOAD_FMAX;
      break;
    default:
      llvm_unreachable("Unknown intrinsic!");
    }
    SDValue Ops[] = {
      M->getOperand(0), // Chain
      M->getOperand(2), // Ptr
      M->getOperand(3)  // Value
    };

    return DAG.getMemIntrinsicNode(Opc, SDLoc(Op), M->getVTList(), Ops,
                                   M->getMemoryVT(), M->getMemOperand());
  }
  case Intrinsic::amdgcn_buffer_load:
  case Intrinsic::amdgcn_buffer_load_format: {
    unsigned Glc = cast<ConstantSDNode>(Op.getOperand(5))->getZExtValue();
    unsigned Slc = cast<ConstantSDNode>(Op.getOperand(6))->getZExtValue();
    unsigned IdxEn = getIdxEn(Op.getOperand(3));
    SDValue Ops[] = {
      Op.getOperand(0), // Chain
      Op.getOperand(2), // rsrc
      Op.getOperand(3), // vindex
      SDValue(),        // voffset -- will be set by setBufferOffsets
      SDValue(),        // soffset -- will be set by setBufferOffsets
      SDValue(),        // offset -- will be set by setBufferOffsets
      DAG.getTargetConstant(Glc | (Slc << 1), DL, MVT::i32), // cachepolicy
      DAG.getTargetConstant(IdxEn, DL, MVT::i1), // idxen
    };
    setBufferOffsets(Op.getOperand(4), DAG, &Ops[3]);

    unsigned Opc = (IntrID == Intrinsic::amdgcn_buffer_load) ?
        AMDGPUISD::BUFFER_LOAD : AMDGPUISD::BUFFER_LOAD_FORMAT;

    EVT VT = Op.getValueType();
    EVT IntVT = VT.changeTypeToInteger();
    auto *M = cast<MemSDNode>(Op);
    updateBufferMMO(M->getMemOperand(), Ops[3], Ops[4], Ops[5], Ops[2]);
    EVT LoadVT = Op.getValueType();

    if (LoadVT.getScalarType() == MVT::f16)
      return adjustLoadValueType(AMDGPUISD::BUFFER_LOAD_FORMAT_D16,
                                 M, DAG, Ops);

    // Handle BUFFER_LOAD_BYTE/UBYTE/SHORT/USHORT overloaded intrinsics
    if (LoadVT.getScalarType() == MVT::i8 ||
        LoadVT.getScalarType() == MVT::i16)
      return handleByteShortBufferLoads(DAG, LoadVT, DL, Ops, M);

    return getMemIntrinsicNode(Opc, DL, Op->getVTList(), Ops, IntVT,
                               M->getMemOperand(), DAG);
  }
  case Intrinsic::amdgcn_raw_buffer_load:
  case Intrinsic::amdgcn_raw_buffer_load_format: {
    const bool IsFormat = IntrID == Intrinsic::amdgcn_raw_buffer_load_format;

    auto Offsets = splitBufferOffsets(Op.getOperand(3), DAG);
    SDValue Ops[] = {
      Op.getOperand(0), // Chain
      Op.getOperand(2), // rsrc
      DAG.getConstant(0, DL, MVT::i32), // vindex
      Offsets.first,    // voffset
      Op.getOperand(4), // soffset
      Offsets.second,   // offset
      Op.getOperand(5), // cachepolicy, swizzled buffer
      DAG.getTargetConstant(0, DL, MVT::i1), // idxen
    };

    auto *M = cast<MemSDNode>(Op);
    updateBufferMMO(M->getMemOperand(), Ops[3], Ops[4], Ops[5]);
    return lowerIntrinsicLoad(M, IsFormat, DAG, Ops);
  }
  case Intrinsic::amdgcn_struct_buffer_load:
  case Intrinsic::amdgcn_struct_buffer_load_format: {
    const bool IsFormat = IntrID == Intrinsic::amdgcn_struct_buffer_load_format;

    auto Offsets = splitBufferOffsets(Op.getOperand(4), DAG);
    SDValue Ops[] = {
      Op.getOperand(0), // Chain
      Op.getOperand(2), // rsrc
      Op.getOperand(3), // vindex
      Offsets.first,    // voffset
      Op.getOperand(5), // soffset
      Offsets.second,   // offset
      Op.getOperand(6), // cachepolicy, swizzled buffer
      DAG.getTargetConstant(1, DL, MVT::i1), // idxen
    };

    auto *M = cast<MemSDNode>(Op);
    updateBufferMMO(M->getMemOperand(), Ops[3], Ops[4], Ops[5], Ops[2]);
    return lowerIntrinsicLoad(cast<MemSDNode>(Op), IsFormat, DAG, Ops);
  }
  case Intrinsic::amdgcn_tbuffer_load: {
    MemSDNode *M = cast<MemSDNode>(Op);
    EVT LoadVT = Op.getValueType();

    unsigned Dfmt = cast<ConstantSDNode>(Op.getOperand(7))->getZExtValue();
    unsigned Nfmt = cast<ConstantSDNode>(Op.getOperand(8))->getZExtValue();
    unsigned Glc = cast<ConstantSDNode>(Op.getOperand(9))->getZExtValue();
    unsigned Slc = cast<ConstantSDNode>(Op.getOperand(10))->getZExtValue();
    unsigned IdxEn = getIdxEn(Op.getOperand(3));
    SDValue Ops[] = {
      Op.getOperand(0),  // Chain
      Op.getOperand(2),  // rsrc
      Op.getOperand(3),  // vindex
      Op.getOperand(4),  // voffset
      Op.getOperand(5),  // soffset
      Op.getOperand(6),  // offset
      DAG.getTargetConstant(Dfmt | (Nfmt << 4), DL, MVT::i32), // format
      DAG.getTargetConstant(Glc | (Slc << 1), DL, MVT::i32), // cachepolicy
      DAG.getTargetConstant(IdxEn, DL, MVT::i1) // idxen
    };

    if (LoadVT.getScalarType() == MVT::f16)
      return adjustLoadValueType(AMDGPUISD::TBUFFER_LOAD_FORMAT_D16,
                                 M, DAG, Ops);
    return getMemIntrinsicNode(AMDGPUISD::TBUFFER_LOAD_FORMAT, DL,
                               Op->getVTList(), Ops, LoadVT, M->getMemOperand(),
                               DAG);
  }
  case Intrinsic::amdgcn_raw_tbuffer_load: {
    MemSDNode *M = cast<MemSDNode>(Op);
    EVT LoadVT = Op.getValueType();
    auto Offsets = splitBufferOffsets(Op.getOperand(3), DAG);

    SDValue Ops[] = {
      Op.getOperand(0),  // Chain
      Op.getOperand(2),  // rsrc
      DAG.getConstant(0, DL, MVT::i32), // vindex
      Offsets.first,     // voffset
      Op.getOperand(4),  // soffset
      Offsets.second,    // offset
      Op.getOperand(5),  // format
      Op.getOperand(6),  // cachepolicy, swizzled buffer
      DAG.getTargetConstant(0, DL, MVT::i1), // idxen
    };

    if (LoadVT.getScalarType() == MVT::f16)
      return adjustLoadValueType(AMDGPUISD::TBUFFER_LOAD_FORMAT_D16,
                                 M, DAG, Ops);
    return getMemIntrinsicNode(AMDGPUISD::TBUFFER_LOAD_FORMAT, DL,
                               Op->getVTList(), Ops, LoadVT, M->getMemOperand(),
                               DAG);
  }
  case Intrinsic::amdgcn_struct_tbuffer_load: {
    MemSDNode *M = cast<MemSDNode>(Op);
    EVT LoadVT = Op.getValueType();
    auto Offsets = splitBufferOffsets(Op.getOperand(4), DAG);

    SDValue Ops[] = {
      Op.getOperand(0),  // Chain
      Op.getOperand(2),  // rsrc
      Op.getOperand(3),  // vindex
      Offsets.first,     // voffset
      Op.getOperand(5),  // soffset
      Offsets.second,    // offset
      Op.getOperand(6),  // format
      Op.getOperand(7),  // cachepolicy, swizzled buffer
      DAG.getTargetConstant(1, DL, MVT::i1), // idxen
    };

    if (LoadVT.getScalarType() == MVT::f16)
      return adjustLoadValueType(AMDGPUISD::TBUFFER_LOAD_FORMAT_D16,
                                 M, DAG, Ops);
    return getMemIntrinsicNode(AMDGPUISD::TBUFFER_LOAD_FORMAT, DL,
                               Op->getVTList(), Ops, LoadVT, M->getMemOperand(),
                               DAG);
  }
  case Intrinsic::amdgcn_buffer_atomic_swap:
  case Intrinsic::amdgcn_buffer_atomic_add:
  case Intrinsic::amdgcn_buffer_atomic_sub:
  case Intrinsic::amdgcn_buffer_atomic_csub:
  case Intrinsic::amdgcn_buffer_atomic_smin:
  case Intrinsic::amdgcn_buffer_atomic_umin:
  case Intrinsic::amdgcn_buffer_atomic_smax:
  case Intrinsic::amdgcn_buffer_atomic_umax:
  case Intrinsic::amdgcn_buffer_atomic_and:
  case Intrinsic::amdgcn_buffer_atomic_or:
  case Intrinsic::amdgcn_buffer_atomic_xor:
  case Intrinsic::amdgcn_buffer_atomic_fadd: {
    unsigned Slc = cast<ConstantSDNode>(Op.getOperand(6))->getZExtValue();
    unsigned IdxEn = getIdxEn(Op.getOperand(4));
    SDValue Ops[] = {
      Op.getOperand(0), // Chain
      Op.getOperand(2), // vdata
      Op.getOperand(3), // rsrc
      Op.getOperand(4), // vindex
      SDValue(),        // voffset -- will be set by setBufferOffsets
      SDValue(),        // soffset -- will be set by setBufferOffsets
      SDValue(),        // offset -- will be set by setBufferOffsets
      DAG.getTargetConstant(Slc << 1, DL, MVT::i32), // cachepolicy
      DAG.getTargetConstant(IdxEn, DL, MVT::i1), // idxen
    };
    setBufferOffsets(Op.getOperand(5), DAG, &Ops[4]);

    EVT VT = Op.getValueType();

    auto *M = cast<MemSDNode>(Op);
    updateBufferMMO(M->getMemOperand(), Ops[4], Ops[5], Ops[6], Ops[3]);
    unsigned Opcode = 0;

    switch (IntrID) {
    case Intrinsic::amdgcn_buffer_atomic_swap:
      Opcode = AMDGPUISD::BUFFER_ATOMIC_SWAP;
      break;
    case Intrinsic::amdgcn_buffer_atomic_add:
      Opcode = AMDGPUISD::BUFFER_ATOMIC_ADD;
      break;
    case Intrinsic::amdgcn_buffer_atomic_sub:
      Opcode = AMDGPUISD::BUFFER_ATOMIC_SUB;
      break;
    case Intrinsic::amdgcn_buffer_atomic_csub:
      Opcode = AMDGPUISD::BUFFER_ATOMIC_CSUB;
      break;
    case Intrinsic::amdgcn_buffer_atomic_smin:
      Opcode = AMDGPUISD::BUFFER_ATOMIC_SMIN;
      break;
    case Intrinsic::amdgcn_buffer_atomic_umin:
      Opcode = AMDGPUISD::BUFFER_ATOMIC_UMIN;
      break;
    case Intrinsic::amdgcn_buffer_atomic_smax:
      Opcode = AMDGPUISD::BUFFER_ATOMIC_SMAX;
      break;
    case Intrinsic::amdgcn_buffer_atomic_umax:
      Opcode = AMDGPUISD::BUFFER_ATOMIC_UMAX;
      break;
    case Intrinsic::amdgcn_buffer_atomic_and:
      Opcode = AMDGPUISD::BUFFER_ATOMIC_AND;
      break;
    case Intrinsic::amdgcn_buffer_atomic_or:
      Opcode = AMDGPUISD::BUFFER_ATOMIC_OR;
      break;
    case Intrinsic::amdgcn_buffer_atomic_xor:
      Opcode = AMDGPUISD::BUFFER_ATOMIC_XOR;
      break;
    case Intrinsic::amdgcn_buffer_atomic_fadd:
      if (!Op.getValue(0).use_empty() && !Subtarget->hasGFX90AInsts()) {
        DiagnosticInfoUnsupported
          NoFpRet(DAG.getMachineFunction().getFunction(),
                  "return versions of fp atomics not supported",
                  DL.getDebugLoc(), DS_Error);
        DAG.getContext()->diagnose(NoFpRet);
        return SDValue();
      }
      Opcode = AMDGPUISD::BUFFER_ATOMIC_FADD;
      break;
    default:
      llvm_unreachable("unhandled atomic opcode");
    }

    return DAG.getMemIntrinsicNode(Opcode, DL, Op->getVTList(), Ops, VT,
                                   M->getMemOperand());
  }
  case Intrinsic::amdgcn_raw_buffer_atomic_fadd:
    return lowerRawBufferAtomicIntrin(Op, DAG, AMDGPUISD::BUFFER_ATOMIC_FADD);
  case Intrinsic::amdgcn_struct_buffer_atomic_fadd:
    return lowerStructBufferAtomicIntrin(Op, DAG, AMDGPUISD::BUFFER_ATOMIC_FADD);
  case Intrinsic::amdgcn_raw_buffer_atomic_fmin:
    return lowerRawBufferAtomicIntrin(Op, DAG, AMDGPUISD::BUFFER_ATOMIC_FMIN);
  case Intrinsic::amdgcn_struct_buffer_atomic_fmin:
    return lowerStructBufferAtomicIntrin(Op, DAG, AMDGPUISD::BUFFER_ATOMIC_FMIN);
  case Intrinsic::amdgcn_raw_buffer_atomic_fmax:
    return lowerRawBufferAtomicIntrin(Op, DAG, AMDGPUISD::BUFFER_ATOMIC_FMAX);
  case Intrinsic::amdgcn_struct_buffer_atomic_fmax:
    return lowerStructBufferAtomicIntrin(Op, DAG, AMDGPUISD::BUFFER_ATOMIC_FMAX);
  case Intrinsic::amdgcn_raw_buffer_atomic_swap:
    return lowerRawBufferAtomicIntrin(Op, DAG, AMDGPUISD::BUFFER_ATOMIC_SWAP);
  case Intrinsic::amdgcn_raw_buffer_atomic_add:
    return lowerRawBufferAtomicIntrin(Op, DAG, AMDGPUISD::BUFFER_ATOMIC_ADD);
  case Intrinsic::amdgcn_raw_buffer_atomic_sub:
    return lowerRawBufferAtomicIntrin(Op, DAG, AMDGPUISD::BUFFER_ATOMIC_SUB);
  case Intrinsic::amdgcn_raw_buffer_atomic_smin:
    return lowerRawBufferAtomicIntrin(Op, DAG, AMDGPUISD::BUFFER_ATOMIC_SMIN);
  case Intrinsic::amdgcn_raw_buffer_atomic_umin:
    return lowerRawBufferAtomicIntrin(Op, DAG, AMDGPUISD::BUFFER_ATOMIC_UMIN);
  case Intrinsic::amdgcn_raw_buffer_atomic_smax:
    return lowerRawBufferAtomicIntrin(Op, DAG, AMDGPUISD::BUFFER_ATOMIC_SMAX);
  case Intrinsic::amdgcn_raw_buffer_atomic_umax:
    return lowerRawBufferAtomicIntrin(Op, DAG, AMDGPUISD::BUFFER_ATOMIC_UMAX);
  case Intrinsic::amdgcn_raw_buffer_atomic_and:
    return lowerRawBufferAtomicIntrin(Op, DAG, AMDGPUISD::BUFFER_ATOMIC_AND);
  case Intrinsic::amdgcn_raw_buffer_atomic_or:
    return lowerRawBufferAtomicIntrin(Op, DAG, AMDGPUISD::BUFFER_ATOMIC_OR);
  case Intrinsic::amdgcn_raw_buffer_atomic_xor:
    return lowerRawBufferAtomicIntrin(Op, DAG, AMDGPUISD::BUFFER_ATOMIC_XOR);
  case Intrinsic::amdgcn_raw_buffer_atomic_inc:
    return lowerRawBufferAtomicIntrin(Op, DAG, AMDGPUISD::BUFFER_ATOMIC_INC);
  case Intrinsic::amdgcn_raw_buffer_atomic_dec:
    return lowerRawBufferAtomicIntrin(Op, DAG, AMDGPUISD::BUFFER_ATOMIC_DEC);
  case Intrinsic::amdgcn_struct_buffer_atomic_swap:
    return lowerStructBufferAtomicIntrin(Op, DAG,
                                         AMDGPUISD::BUFFER_ATOMIC_SWAP);
  case Intrinsic::amdgcn_struct_buffer_atomic_add:
    return lowerStructBufferAtomicIntrin(Op, DAG, AMDGPUISD::BUFFER_ATOMIC_ADD);
  case Intrinsic::amdgcn_struct_buffer_atomic_sub:
    return lowerStructBufferAtomicIntrin(Op, DAG, AMDGPUISD::BUFFER_ATOMIC_SUB);
  case Intrinsic::amdgcn_struct_buffer_atomic_smin:
    return lowerStructBufferAtomicIntrin(Op, DAG,
                                         AMDGPUISD::BUFFER_ATOMIC_SMIN);
  case Intrinsic::amdgcn_struct_buffer_atomic_umin:
    return lowerStructBufferAtomicIntrin(Op, DAG,
                                         AMDGPUISD::BUFFER_ATOMIC_UMIN);
  case Intrinsic::amdgcn_struct_buffer_atomic_smax:
    return lowerStructBufferAtomicIntrin(Op, DAG,
                                         AMDGPUISD::BUFFER_ATOMIC_SMAX);
  case Intrinsic::amdgcn_struct_buffer_atomic_umax:
    return lowerStructBufferAtomicIntrin(Op, DAG,
                                         AMDGPUISD::BUFFER_ATOMIC_UMAX);
  case Intrinsic::amdgcn_struct_buffer_atomic_and:
    return lowerStructBufferAtomicIntrin(Op, DAG, AMDGPUISD::BUFFER_ATOMIC_AND);
  case Intrinsic::amdgcn_struct_buffer_atomic_or:
    return lowerStructBufferAtomicIntrin(Op, DAG, AMDGPUISD::BUFFER_ATOMIC_OR);
  case Intrinsic::amdgcn_struct_buffer_atomic_xor:
    return lowerStructBufferAtomicIntrin(Op, DAG, AMDGPUISD::BUFFER_ATOMIC_XOR);
  case Intrinsic::amdgcn_struct_buffer_atomic_inc:
    return lowerStructBufferAtomicIntrin(Op, DAG, AMDGPUISD::BUFFER_ATOMIC_INC);
  case Intrinsic::amdgcn_struct_buffer_atomic_dec:
    return lowerStructBufferAtomicIntrin(Op, DAG, AMDGPUISD::BUFFER_ATOMIC_DEC);

  case Intrinsic::amdgcn_buffer_atomic_cmpswap: {
    unsigned Slc = cast<ConstantSDNode>(Op.getOperand(7))->getZExtValue();
    unsigned IdxEn = getIdxEn(Op.getOperand(5));
    SDValue Ops[] = {
      Op.getOperand(0), // Chain
      Op.getOperand(2), // src
      Op.getOperand(3), // cmp
      Op.getOperand(4), // rsrc
      Op.getOperand(5), // vindex
      SDValue(),        // voffset -- will be set by setBufferOffsets
      SDValue(),        // soffset -- will be set by setBufferOffsets
      SDValue(),        // offset -- will be set by setBufferOffsets
      DAG.getTargetConstant(Slc << 1, DL, MVT::i32), // cachepolicy
      DAG.getTargetConstant(IdxEn, DL, MVT::i1), // idxen
    };
    setBufferOffsets(Op.getOperand(6), DAG, &Ops[5]);

    EVT VT = Op.getValueType();
    auto *M = cast<MemSDNode>(Op);
    updateBufferMMO(M->getMemOperand(), Ops[5], Ops[6], Ops[7], Ops[4]);

    return DAG.getMemIntrinsicNode(AMDGPUISD::BUFFER_ATOMIC_CMPSWAP, DL,
                                   Op->getVTList(), Ops, VT, M->getMemOperand());
  }
  case Intrinsic::amdgcn_raw_buffer_atomic_cmpswap: {
    auto Offsets = splitBufferOffsets(Op.getOperand(5), DAG);
    SDValue Ops[] = {
      Op.getOperand(0), // Chain
      Op.getOperand(2), // src
      Op.getOperand(3), // cmp
      Op.getOperand(4), // rsrc
      DAG.getConstant(0, DL, MVT::i32), // vindex
      Offsets.first,    // voffset
      Op.getOperand(6), // soffset
      Offsets.second,   // offset
      Op.getOperand(7), // cachepolicy
      DAG.getTargetConstant(0, DL, MVT::i1), // idxen
    };
    EVT VT = Op.getValueType();
    auto *M = cast<MemSDNode>(Op);
    updateBufferMMO(M->getMemOperand(), Ops[5], Ops[6], Ops[7]);

    return DAG.getMemIntrinsicNode(AMDGPUISD::BUFFER_ATOMIC_CMPSWAP, DL,
                                   Op->getVTList(), Ops, VT, M->getMemOperand());
  }
  case Intrinsic::amdgcn_struct_buffer_atomic_cmpswap: {
    auto Offsets = splitBufferOffsets(Op.getOperand(6), DAG);
    SDValue Ops[] = {
      Op.getOperand(0), // Chain
      Op.getOperand(2), // src
      Op.getOperand(3), // cmp
      Op.getOperand(4), // rsrc
      Op.getOperand(5), // vindex
      Offsets.first,    // voffset
      Op.getOperand(7), // soffset
      Offsets.second,   // offset
      Op.getOperand(8), // cachepolicy
      DAG.getTargetConstant(1, DL, MVT::i1), // idxen
    };
    EVT VT = Op.getValueType();
    auto *M = cast<MemSDNode>(Op);
    updateBufferMMO(M->getMemOperand(), Ops[5], Ops[6], Ops[7], Ops[4]);

    return DAG.getMemIntrinsicNode(AMDGPUISD::BUFFER_ATOMIC_CMPSWAP, DL,
                                   Op->getVTList(), Ops, VT, M->getMemOperand());
  }
  case Intrinsic::amdgcn_image_bvh_intersect_ray: {
    MemSDNode *M = cast<MemSDNode>(Op);
    SDValue NodePtr = M->getOperand(2);
    SDValue RayExtent = M->getOperand(3);
    SDValue RayOrigin = M->getOperand(4);
    SDValue RayDir = M->getOperand(5);
    SDValue RayInvDir = M->getOperand(6);
    SDValue TDescr = M->getOperand(7);

    assert(NodePtr.getValueType() == MVT::i32 ||
           NodePtr.getValueType() == MVT::i64);
    assert(RayDir.getValueType() == MVT::v3f16 ||
           RayDir.getValueType() == MVT::v3f32);

    if (!Subtarget->hasGFX10_AEncoding()) {
      emitRemovedIntrinsicError(DAG, DL, Op.getValueType());
      return SDValue();
    }

    const bool IsA16 = RayDir.getValueType().getVectorElementType() == MVT::f16;
    const bool Is64 = NodePtr.getValueType() == MVT::i64;
    const unsigned NumVDataDwords = 4;
    const unsigned NumVAddrDwords = IsA16 ? (Is64 ? 9 : 8) : (Is64 ? 12 : 11);
    const bool UseNSA = Subtarget->hasNSAEncoding() &&
                        NumVAddrDwords <= Subtarget->getNSAMaxSize();
    const unsigned BaseOpcodes[2][2] = {
        {AMDGPU::IMAGE_BVH_INTERSECT_RAY, AMDGPU::IMAGE_BVH_INTERSECT_RAY_a16},
        {AMDGPU::IMAGE_BVH64_INTERSECT_RAY,
         AMDGPU::IMAGE_BVH64_INTERSECT_RAY_a16}};
    int Opcode;
    if (UseNSA) {
      Opcode = AMDGPU::getMIMGOpcode(BaseOpcodes[Is64][IsA16],
                                     AMDGPU::MIMGEncGfx10NSA, NumVDataDwords,
                                     NumVAddrDwords);
    } else {
      Opcode = AMDGPU::getMIMGOpcode(
          BaseOpcodes[Is64][IsA16], AMDGPU::MIMGEncGfx10Default, NumVDataDwords,
          PowerOf2Ceil(NumVAddrDwords));
    }
    assert(Opcode != -1);

    SmallVector<SDValue, 16> Ops;

    auto packLanes = [&DAG, &Ops, &DL] (SDValue Op, bool IsAligned) {
      SmallVector<SDValue, 3> Lanes;
      DAG.ExtractVectorElements(Op, Lanes, 0, 3);
      if (Lanes[0].getValueSizeInBits() == 32) {
        for (unsigned I = 0; I < 3; ++I)
          Ops.push_back(DAG.getBitcast(MVT::i32, Lanes[I]));
      } else {
        if (IsAligned) {
          Ops.push_back(
            DAG.getBitcast(MVT::i32,
                           DAG.getBuildVector(MVT::v2f16, DL,
                                              { Lanes[0], Lanes[1] })));
          Ops.push_back(Lanes[2]);
        } else {
          SDValue Elt0 = Ops.pop_back_val();
          Ops.push_back(
            DAG.getBitcast(MVT::i32,
                           DAG.getBuildVector(MVT::v2f16, DL,
                                              { Elt0, Lanes[0] })));
          Ops.push_back(
            DAG.getBitcast(MVT::i32,
                           DAG.getBuildVector(MVT::v2f16, DL,
                                              { Lanes[1], Lanes[2] })));
        }
      }
    };

    if (Is64)
      DAG.ExtractVectorElements(DAG.getBitcast(MVT::v2i32, NodePtr), Ops, 0, 2);
    else
      Ops.push_back(NodePtr);

    Ops.push_back(DAG.getBitcast(MVT::i32, RayExtent));
    packLanes(RayOrigin, true);
    packLanes(RayDir, true);
    packLanes(RayInvDir, false);

    if (!UseNSA) {
      // Build a single vector containing all the operands so far prepared.
      if (NumVAddrDwords > 8) {
        SDValue Undef = DAG.getUNDEF(MVT::i32);
        Ops.append(16 - Ops.size(), Undef);
      }
      assert(Ops.size() == 8 || Ops.size() == 16);
      SDValue MergedOps = DAG.getBuildVector(
          Ops.size() == 16 ? MVT::v16i32 : MVT::v8i32, DL, Ops);
      Ops.clear();
      Ops.push_back(MergedOps);
    }

    Ops.push_back(TDescr);
    if (IsA16)
      Ops.push_back(DAG.getTargetConstant(1, DL, MVT::i1));
    Ops.push_back(M->getChain());

    auto *NewNode = DAG.getMachineNode(Opcode, DL, M->getVTList(), Ops);
    MachineMemOperand *MemRef = M->getMemOperand();
    DAG.setNodeMemRefs(NewNode, {MemRef});
    return SDValue(NewNode, 0);
  }
  case Intrinsic::amdgcn_global_atomic_fadd:
    if (!Op.getValue(0).use_empty() && !Subtarget->hasGFX90AInsts()) {
      DiagnosticInfoUnsupported
        NoFpRet(DAG.getMachineFunction().getFunction(),
                "return versions of fp atomics not supported",
                DL.getDebugLoc(), DS_Error);
      DAG.getContext()->diagnose(NoFpRet);
      return SDValue();
    }
    LLVM_FALLTHROUGH;
  case Intrinsic::amdgcn_global_atomic_fmin:
  case Intrinsic::amdgcn_global_atomic_fmax:
  case Intrinsic::amdgcn_flat_atomic_fadd:
  case Intrinsic::amdgcn_flat_atomic_fmin:
  case Intrinsic::amdgcn_flat_atomic_fmax: {
    MemSDNode *M = cast<MemSDNode>(Op);
    SDValue Ops[] = {
      M->getOperand(0), // Chain
      M->getOperand(2), // Ptr
      M->getOperand(3)  // Value
    };
    unsigned Opcode = 0;
    switch (IntrID) {
    case Intrinsic::amdgcn_global_atomic_fadd:
    case Intrinsic::amdgcn_flat_atomic_fadd: {
      EVT VT = Op.getOperand(3).getValueType();
      return DAG.getAtomic(ISD::ATOMIC_LOAD_FADD, DL, VT,
                           DAG.getVTList(VT, MVT::Other), Ops,
                           M->getMemOperand());
    }
    case Intrinsic::amdgcn_global_atomic_fmin:
    case Intrinsic::amdgcn_flat_atomic_fmin: {
      Opcode = AMDGPUISD::ATOMIC_LOAD_FMIN;
      break;
    }
    case Intrinsic::amdgcn_global_atomic_fmax:
    case Intrinsic::amdgcn_flat_atomic_fmax: {
      Opcode = AMDGPUISD::ATOMIC_LOAD_FMAX;
      break;
    }
    default:
      llvm_unreachable("unhandled atomic opcode");
    }
    return DAG.getMemIntrinsicNode(Opcode, SDLoc(Op),
                                   M->getVTList(), Ops, M->getMemoryVT(),
                                   M->getMemOperand());
  }
  default:

    if (const AMDGPU::ImageDimIntrinsicInfo *ImageDimIntr =
            AMDGPU::getImageDimIntrinsicInfo(IntrID))
      return lowerImage(Op, ImageDimIntr, DAG, true);

    return SDValue();
  }
}

// Call DAG.getMemIntrinsicNode for a load, but first widen a dwordx3 type to
// dwordx4 if on SI.
SDValue SITargetLowering::getMemIntrinsicNode(unsigned Opcode, const SDLoc &DL,
                                              SDVTList VTList,
                                              ArrayRef<SDValue> Ops, EVT MemVT,
                                              MachineMemOperand *MMO,
                                              SelectionDAG &DAG) const {
  EVT VT = VTList.VTs[0];
  EVT WidenedVT = VT;
  EVT WidenedMemVT = MemVT;
  if (!Subtarget->hasDwordx3LoadStores() &&
      (WidenedVT == MVT::v3i32 || WidenedVT == MVT::v3f32)) {
    WidenedVT = EVT::getVectorVT(*DAG.getContext(),
                                 WidenedVT.getVectorElementType(), 4);
    WidenedMemVT = EVT::getVectorVT(*DAG.getContext(),
                                    WidenedMemVT.getVectorElementType(), 4);
    MMO = DAG.getMachineFunction().getMachineMemOperand(MMO, 0, 16);
  }

  assert(VTList.NumVTs == 2);
  SDVTList WidenedVTList = DAG.getVTList(WidenedVT, VTList.VTs[1]);

  auto NewOp = DAG.getMemIntrinsicNode(Opcode, DL, WidenedVTList, Ops,
                                       WidenedMemVT, MMO);
  if (WidenedVT != VT) {
    auto Extract = DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, VT, NewOp,
                               DAG.getVectorIdxConstant(0, DL));
    NewOp = DAG.getMergeValues({ Extract, SDValue(NewOp.getNode(), 1) }, DL);
  }
  return NewOp;
}

SDValue SITargetLowering::handleD16VData(SDValue VData, SelectionDAG &DAG,
                                         bool ImageStore) const {
  EVT StoreVT = VData.getValueType();

  // No change for f16 and legal vector D16 types.
  if (!StoreVT.isVector())
    return VData;

  SDLoc DL(VData);
  unsigned NumElements = StoreVT.getVectorNumElements();

  if (Subtarget->hasUnpackedD16VMem()) {
    // We need to unpack the packed data to store.
    EVT IntStoreVT = StoreVT.changeTypeToInteger();
    SDValue IntVData = DAG.getNode(ISD::BITCAST, DL, IntStoreVT, VData);

    EVT EquivStoreVT =
        EVT::getVectorVT(*DAG.getContext(), MVT::i32, NumElements);
    SDValue ZExt = DAG.getNode(ISD::ZERO_EXTEND, DL, EquivStoreVT, IntVData);
    return DAG.UnrollVectorOp(ZExt.getNode());
  }

  // The sq block of gfx8.1 does not estimate register use correctly for d16
  // image store instructions. The data operand is computed as if it were not a
  // d16 image instruction.
  if (ImageStore && Subtarget->hasImageStoreD16Bug()) {
    // Bitcast to i16
    EVT IntStoreVT = StoreVT.changeTypeToInteger();
    SDValue IntVData = DAG.getNode(ISD::BITCAST, DL, IntStoreVT, VData);

    // Decompose into scalars
    SmallVector<SDValue, 4> Elts;
    DAG.ExtractVectorElements(IntVData, Elts);

    // Group pairs of i16 into v2i16 and bitcast to i32
    SmallVector<SDValue, 4> PackedElts;
    for (unsigned I = 0; I < Elts.size() / 2; I += 1) {
      SDValue Pair =
          DAG.getBuildVector(MVT::v2i16, DL, {Elts[I * 2], Elts[I * 2 + 1]});
      SDValue IntPair = DAG.getNode(ISD::BITCAST, DL, MVT::i32, Pair);
      PackedElts.push_back(IntPair);
    }
    if ((NumElements % 2) == 1) {
      // Handle v3i16
      unsigned I = Elts.size() / 2;
      SDValue Pair = DAG.getBuildVector(MVT::v2i16, DL,
                                        {Elts[I * 2], DAG.getUNDEF(MVT::i16)});
      SDValue IntPair = DAG.getNode(ISD::BITCAST, DL, MVT::i32, Pair);
      PackedElts.push_back(IntPair);
    }

    // Pad using UNDEF
    PackedElts.resize(Elts.size(), DAG.getUNDEF(MVT::i32));

    // Build final vector
    EVT VecVT =
        EVT::getVectorVT(*DAG.getContext(), MVT::i32, PackedElts.size());
    return DAG.getBuildVector(VecVT, DL, PackedElts);
  }

  if (NumElements == 3) {
    EVT IntStoreVT =
        EVT::getIntegerVT(*DAG.getContext(), StoreVT.getStoreSizeInBits());
    SDValue IntVData = DAG.getNode(ISD::BITCAST, DL, IntStoreVT, VData);

    EVT WidenedStoreVT = EVT::getVectorVT(
        *DAG.getContext(), StoreVT.getVectorElementType(), NumElements + 1);
    EVT WidenedIntVT = EVT::getIntegerVT(*DAG.getContext(),
                                         WidenedStoreVT.getStoreSizeInBits());
    SDValue ZExt = DAG.getNode(ISD::ZERO_EXTEND, DL, WidenedIntVT, IntVData);
    return DAG.getNode(ISD::BITCAST, DL, WidenedStoreVT, ZExt);
  }

  assert(isTypeLegal(StoreVT));
  return VData;
}

SDValue SITargetLowering::LowerINTRINSIC_VOID(SDValue Op,
                                              SelectionDAG &DAG) const {
  SDLoc DL(Op);
  SDValue Chain = Op.getOperand(0);
  unsigned IntrinsicID = cast<ConstantSDNode>(Op.getOperand(1))->getZExtValue();
  MachineFunction &MF = DAG.getMachineFunction();

  switch (IntrinsicID) {
  case Intrinsic::amdgcn_exp_compr: {
    SDValue Src0 = Op.getOperand(4);
    SDValue Src1 = Op.getOperand(5);
    // Hack around illegal type on SI by directly selecting it.
    if (isTypeLegal(Src0.getValueType()))
      return SDValue();

    const ConstantSDNode *Done = cast<ConstantSDNode>(Op.getOperand(6));
    SDValue Undef = DAG.getUNDEF(MVT::f32);
    const SDValue Ops[] = {
      Op.getOperand(2), // tgt
      DAG.getNode(ISD::BITCAST, DL, MVT::f32, Src0), // src0
      DAG.getNode(ISD::BITCAST, DL, MVT::f32, Src1), // src1
      Undef, // src2
      Undef, // src3
      Op.getOperand(7), // vm
      DAG.getTargetConstant(1, DL, MVT::i1), // compr
      Op.getOperand(3), // en
      Op.getOperand(0) // Chain
    };

    unsigned Opc = Done->isZero() ? AMDGPU::EXP : AMDGPU::EXP_DONE;
    return SDValue(DAG.getMachineNode(Opc, DL, Op->getVTList(), Ops), 0);
  }
  case Intrinsic::amdgcn_s_barrier: {
    if (getTargetMachine().getOptLevel() > CodeGenOpt::None) {
      const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
      unsigned WGSize = ST.getFlatWorkGroupSizes(MF.getFunction()).second;
      if (WGSize <= ST.getWavefrontSize())
        return SDValue(DAG.getMachineNode(AMDGPU::WAVE_BARRIER, DL, MVT::Other,
                                          Op.getOperand(0)), 0);
    }
    return SDValue();
  };
  case Intrinsic::amdgcn_tbuffer_store: {
    SDValue VData = Op.getOperand(2);
    bool IsD16 = (VData.getValueType().getScalarType() == MVT::f16);
    if (IsD16)
      VData = handleD16VData(VData, DAG);
    unsigned Dfmt = cast<ConstantSDNode>(Op.getOperand(8))->getZExtValue();
    unsigned Nfmt = cast<ConstantSDNode>(Op.getOperand(9))->getZExtValue();
    unsigned Glc = cast<ConstantSDNode>(Op.getOperand(10))->getZExtValue();
    unsigned Slc = cast<ConstantSDNode>(Op.getOperand(11))->getZExtValue();
    unsigned IdxEn = getIdxEn(Op.getOperand(4));
    SDValue Ops[] = {
      Chain,
      VData,             // vdata
      Op.getOperand(3),  // rsrc
      Op.getOperand(4),  // vindex
      Op.getOperand(5),  // voffset
      Op.getOperand(6),  // soffset
      Op.getOperand(7),  // offset
      DAG.getTargetConstant(Dfmt | (Nfmt << 4), DL, MVT::i32), // format
      DAG.getTargetConstant(Glc | (Slc << 1), DL, MVT::i32), // cachepolicy
      DAG.getTargetConstant(IdxEn, DL, MVT::i1), // idxen
    };
    unsigned Opc = IsD16 ? AMDGPUISD::TBUFFER_STORE_FORMAT_D16 :
                           AMDGPUISD::TBUFFER_STORE_FORMAT;
    MemSDNode *M = cast<MemSDNode>(Op);
    return DAG.getMemIntrinsicNode(Opc, DL, Op->getVTList(), Ops,
                                   M->getMemoryVT(), M->getMemOperand());
  }

  case Intrinsic::amdgcn_struct_tbuffer_store: {
    SDValue VData = Op.getOperand(2);
    bool IsD16 = (VData.getValueType().getScalarType() == MVT::f16);
    if (IsD16)
      VData = handleD16VData(VData, DAG);
    auto Offsets = splitBufferOffsets(Op.getOperand(5), DAG);
    SDValue Ops[] = {
      Chain,
      VData,             // vdata
      Op.getOperand(3),  // rsrc
      Op.getOperand(4),  // vindex
      Offsets.first,     // voffset
      Op.getOperand(6),  // soffset
      Offsets.second,    // offset
      Op.getOperand(7),  // format
      Op.getOperand(8),  // cachepolicy, swizzled buffer
      DAG.getTargetConstant(1, DL, MVT::i1), // idxen
    };
    unsigned Opc = IsD16 ? AMDGPUISD::TBUFFER_STORE_FORMAT_D16 :
                           AMDGPUISD::TBUFFER_STORE_FORMAT;
    MemSDNode *M = cast<MemSDNode>(Op);
    return DAG.getMemIntrinsicNode(Opc, DL, Op->getVTList(), Ops,
                                   M->getMemoryVT(), M->getMemOperand());
  }

  case Intrinsic::amdgcn_raw_tbuffer_store: {
    SDValue VData = Op.getOperand(2);
    bool IsD16 = (VData.getValueType().getScalarType() == MVT::f16);
    if (IsD16)
      VData = handleD16VData(VData, DAG);
    auto Offsets = splitBufferOffsets(Op.getOperand(4), DAG);
    SDValue Ops[] = {
      Chain,
      VData,             // vdata
      Op.getOperand(3),  // rsrc
      DAG.getConstant(0, DL, MVT::i32), // vindex
      Offsets.first,     // voffset
      Op.getOperand(5),  // soffset
      Offsets.second,    // offset
      Op.getOperand(6),  // format
      Op.getOperand(7),  // cachepolicy, swizzled buffer
      DAG.getTargetConstant(0, DL, MVT::i1), // idxen
    };
    unsigned Opc = IsD16 ? AMDGPUISD::TBUFFER_STORE_FORMAT_D16 :
                           AMDGPUISD::TBUFFER_STORE_FORMAT;
    MemSDNode *M = cast<MemSDNode>(Op);
    return DAG.getMemIntrinsicNode(Opc, DL, Op->getVTList(), Ops,
                                   M->getMemoryVT(), M->getMemOperand());
  }

  case Intrinsic::amdgcn_buffer_store:
  case Intrinsic::amdgcn_buffer_store_format: {
    SDValue VData = Op.getOperand(2);
    bool IsD16 = (VData.getValueType().getScalarType() == MVT::f16);
    if (IsD16)
      VData = handleD16VData(VData, DAG);
    unsigned Glc = cast<ConstantSDNode>(Op.getOperand(6))->getZExtValue();
    unsigned Slc = cast<ConstantSDNode>(Op.getOperand(7))->getZExtValue();
    unsigned IdxEn = getIdxEn(Op.getOperand(4));
    SDValue Ops[] = {
      Chain,
      VData,
      Op.getOperand(3), // rsrc
      Op.getOperand(4), // vindex
      SDValue(), // voffset -- will be set by setBufferOffsets
      SDValue(), // soffset -- will be set by setBufferOffsets
      SDValue(), // offset -- will be set by setBufferOffsets
      DAG.getTargetConstant(Glc | (Slc << 1), DL, MVT::i32), // cachepolicy
      DAG.getTargetConstant(IdxEn, DL, MVT::i1), // idxen
    };
    setBufferOffsets(Op.getOperand(5), DAG, &Ops[4]);

    unsigned Opc = IntrinsicID == Intrinsic::amdgcn_buffer_store ?
                   AMDGPUISD::BUFFER_STORE : AMDGPUISD::BUFFER_STORE_FORMAT;
    Opc = IsD16 ? AMDGPUISD::BUFFER_STORE_FORMAT_D16 : Opc;
    MemSDNode *M = cast<MemSDNode>(Op);
    updateBufferMMO(M->getMemOperand(), Ops[4], Ops[5], Ops[6], Ops[3]);

    // Handle BUFFER_STORE_BYTE/SHORT overloaded intrinsics
    EVT VDataType = VData.getValueType().getScalarType();
    if (VDataType == MVT::i8 || VDataType == MVT::i16)
      return handleByteShortBufferStores(DAG, VDataType, DL, Ops, M);

    return DAG.getMemIntrinsicNode(Opc, DL, Op->getVTList(), Ops,
                                   M->getMemoryVT(), M->getMemOperand());
  }

  case Intrinsic::amdgcn_raw_buffer_store:
  case Intrinsic::amdgcn_raw_buffer_store_format: {
    const bool IsFormat =
        IntrinsicID == Intrinsic::amdgcn_raw_buffer_store_format;

    SDValue VData = Op.getOperand(2);
    EVT VDataVT = VData.getValueType();
    EVT EltType = VDataVT.getScalarType();
    bool IsD16 = IsFormat && (EltType.getSizeInBits() == 16);
    if (IsD16) {
      VData = handleD16VData(VData, DAG);
      VDataVT = VData.getValueType();
    }

    if (!isTypeLegal(VDataVT)) {
      VData =
          DAG.getNode(ISD::BITCAST, DL,
                      getEquivalentMemType(*DAG.getContext(), VDataVT), VData);
    }

    auto Offsets = splitBufferOffsets(Op.getOperand(4), DAG);
    SDValue Ops[] = {
      Chain,
      VData,
      Op.getOperand(3), // rsrc
      DAG.getConstant(0, DL, MVT::i32), // vindex
      Offsets.first,    // voffset
      Op.getOperand(5), // soffset
      Offsets.second,   // offset
      Op.getOperand(6), // cachepolicy, swizzled buffer
      DAG.getTargetConstant(0, DL, MVT::i1), // idxen
    };
    unsigned Opc =
        IsFormat ? AMDGPUISD::BUFFER_STORE_FORMAT : AMDGPUISD::BUFFER_STORE;
    Opc = IsD16 ? AMDGPUISD::BUFFER_STORE_FORMAT_D16 : Opc;
    MemSDNode *M = cast<MemSDNode>(Op);
    updateBufferMMO(M->getMemOperand(), Ops[4], Ops[5], Ops[6]);

    // Handle BUFFER_STORE_BYTE/SHORT overloaded intrinsics
    if (!IsD16 && !VDataVT.isVector() && EltType.getSizeInBits() < 32)
      return handleByteShortBufferStores(DAG, VDataVT, DL, Ops, M);

    return DAG.getMemIntrinsicNode(Opc, DL, Op->getVTList(), Ops,
                                   M->getMemoryVT(), M->getMemOperand());
  }

  case Intrinsic::amdgcn_struct_buffer_store:
  case Intrinsic::amdgcn_struct_buffer_store_format: {
    const bool IsFormat =
        IntrinsicID == Intrinsic::amdgcn_struct_buffer_store_format;

    SDValue VData = Op.getOperand(2);
    EVT VDataVT = VData.getValueType();
    EVT EltType = VDataVT.getScalarType();
    bool IsD16 = IsFormat && (EltType.getSizeInBits() == 16);

    if (IsD16) {
      VData = handleD16VData(VData, DAG);
      VDataVT = VData.getValueType();
    }

    if (!isTypeLegal(VDataVT)) {
      VData =
          DAG.getNode(ISD::BITCAST, DL,
                      getEquivalentMemType(*DAG.getContext(), VDataVT), VData);
    }

    auto Offsets = splitBufferOffsets(Op.getOperand(5), DAG);
    SDValue Ops[] = {
      Chain,
      VData,
      Op.getOperand(3), // rsrc
      Op.getOperand(4), // vindex
      Offsets.first,    // voffset
      Op.getOperand(6), // soffset
      Offsets.second,   // offset
      Op.getOperand(7), // cachepolicy, swizzled buffer
      DAG.getTargetConstant(1, DL, MVT::i1), // idxen
    };
    unsigned Opc = IntrinsicID == Intrinsic::amdgcn_struct_buffer_store ?
                   AMDGPUISD::BUFFER_STORE : AMDGPUISD::BUFFER_STORE_FORMAT;
    Opc = IsD16 ? AMDGPUISD::BUFFER_STORE_FORMAT_D16 : Opc;
    MemSDNode *M = cast<MemSDNode>(Op);
    updateBufferMMO(M->getMemOperand(), Ops[4], Ops[5], Ops[6], Ops[3]);

    // Handle BUFFER_STORE_BYTE/SHORT overloaded intrinsics
    EVT VDataType = VData.getValueType().getScalarType();
    if (!IsD16 && !VDataVT.isVector() && EltType.getSizeInBits() < 32)
      return handleByteShortBufferStores(DAG, VDataType, DL, Ops, M);

    return DAG.getMemIntrinsicNode(Opc, DL, Op->getVTList(), Ops,
                                   M->getMemoryVT(), M->getMemOperand());
  }
  case Intrinsic::amdgcn_end_cf:
    return SDValue(DAG.getMachineNode(AMDGPU::SI_END_CF, DL, MVT::Other,
                                      Op->getOperand(2), Chain), 0);

  default: {
    if (const AMDGPU::ImageDimIntrinsicInfo *ImageDimIntr =
            AMDGPU::getImageDimIntrinsicInfo(IntrinsicID))
      return lowerImage(Op, ImageDimIntr, DAG, true);

    return Op;
  }
  }
}

// The raw.(t)buffer and struct.(t)buffer intrinsics have two offset args:
// offset (the offset that is included in bounds checking and swizzling, to be
// split between the instruction's voffset and immoffset fields) and soffset
// (the offset that is excluded from bounds checking and swizzling, to go in
// the instruction's soffset field).  This function takes the first kind of
// offset and figures out how to split it between voffset and immoffset.
std::pair<SDValue, SDValue> SITargetLowering::splitBufferOffsets(
    SDValue Offset, SelectionDAG &DAG) const {
  SDLoc DL(Offset);
  const unsigned MaxImm = 4095;
  SDValue N0 = Offset;
  ConstantSDNode *C1 = nullptr;

  if ((C1 = dyn_cast<ConstantSDNode>(N0)))
    N0 = SDValue();
  else if (DAG.isBaseWithConstantOffset(N0)) {
    C1 = cast<ConstantSDNode>(N0.getOperand(1));
    N0 = N0.getOperand(0);
  }

  if (C1) {
    unsigned ImmOffset = C1->getZExtValue();
    // If the immediate value is too big for the immoffset field, put the value
    // and -4096 into the immoffset field so that the value that is copied/added
    // for the voffset field is a multiple of 4096, and it stands more chance
    // of being CSEd with the copy/add for another similar load/store.
    // However, do not do that rounding down to a multiple of 4096 if that is a
    // negative number, as it appears to be illegal to have a negative offset
    // in the vgpr, even if adding the immediate offset makes it positive.
    unsigned Overflow = ImmOffset & ~MaxImm;
    ImmOffset -= Overflow;
    if ((int32_t)Overflow < 0) {
      Overflow += ImmOffset;
      ImmOffset = 0;
    }
    C1 = cast<ConstantSDNode>(DAG.getTargetConstant(ImmOffset, DL, MVT::i32));
    if (Overflow) {
      auto OverflowVal = DAG.getConstant(Overflow, DL, MVT::i32);
      if (!N0)
        N0 = OverflowVal;
      else {
        SDValue Ops[] = { N0, OverflowVal };
        N0 = DAG.getNode(ISD::ADD, DL, MVT::i32, Ops);
      }
    }
  }
  if (!N0)
    N0 = DAG.getConstant(0, DL, MVT::i32);
  if (!C1)
    C1 = cast<ConstantSDNode>(DAG.getTargetConstant(0, DL, MVT::i32));
  return {N0, SDValue(C1, 0)};
}

// Analyze a combined offset from an amdgcn_buffer_ intrinsic and store the
// three offsets (voffset, soffset and instoffset) into the SDValue[3] array
// pointed to by Offsets.
void SITargetLowering::setBufferOffsets(SDValue CombinedOffset,
                                        SelectionDAG &DAG, SDValue *Offsets,
                                        Align Alignment) const {
  SDLoc DL(CombinedOffset);
  if (auto C = dyn_cast<ConstantSDNode>(CombinedOffset)) {
    uint32_t Imm = C->getZExtValue();
    uint32_t SOffset, ImmOffset;
    if (AMDGPU::splitMUBUFOffset(Imm, SOffset, ImmOffset, Subtarget,
                                 Alignment)) {
      Offsets[0] = DAG.getConstant(0, DL, MVT::i32);
      Offsets[1] = DAG.getConstant(SOffset, DL, MVT::i32);
      Offsets[2] = DAG.getTargetConstant(ImmOffset, DL, MVT::i32);
      return;
    }
  }
  if (DAG.isBaseWithConstantOffset(CombinedOffset)) {
    SDValue N0 = CombinedOffset.getOperand(0);
    SDValue N1 = CombinedOffset.getOperand(1);
    uint32_t SOffset, ImmOffset;
    int Offset = cast<ConstantSDNode>(N1)->getSExtValue();
    if (Offset >= 0 && AMDGPU::splitMUBUFOffset(Offset, SOffset, ImmOffset,
                                                Subtarget, Alignment)) {
      Offsets[0] = N0;
      Offsets[1] = DAG.getConstant(SOffset, DL, MVT::i32);
      Offsets[2] = DAG.getTargetConstant(ImmOffset, DL, MVT::i32);
      return;
    }
  }
  Offsets[0] = CombinedOffset;
  Offsets[1] = DAG.getConstant(0, DL, MVT::i32);
  Offsets[2] = DAG.getTargetConstant(0, DL, MVT::i32);
}

// Handle 8 bit and 16 bit buffer loads
SDValue SITargetLowering::handleByteShortBufferLoads(SelectionDAG &DAG,
                                                     EVT LoadVT, SDLoc DL,
                                                     ArrayRef<SDValue> Ops,
                                                     MemSDNode *M) const {
  EVT IntVT = LoadVT.changeTypeToInteger();
  unsigned Opc = (LoadVT.getScalarType() == MVT::i8) ?
         AMDGPUISD::BUFFER_LOAD_UBYTE : AMDGPUISD::BUFFER_LOAD_USHORT;

  SDVTList ResList = DAG.getVTList(MVT::i32, MVT::Other);
  SDValue BufferLoad = DAG.getMemIntrinsicNode(Opc, DL, ResList,
                                               Ops, IntVT,
                                               M->getMemOperand());
  SDValue LoadVal = DAG.getNode(ISD::TRUNCATE, DL, IntVT, BufferLoad);
  LoadVal = DAG.getNode(ISD::BITCAST, DL, LoadVT, LoadVal);

  return DAG.getMergeValues({LoadVal, BufferLoad.getValue(1)}, DL);
}

// Handle 8 bit and 16 bit buffer stores
SDValue SITargetLowering::handleByteShortBufferStores(SelectionDAG &DAG,
                                                      EVT VDataType, SDLoc DL,
                                                      SDValue Ops[],
                                                      MemSDNode *M) const {
  if (VDataType == MVT::f16)
    Ops[1] = DAG.getNode(ISD::BITCAST, DL, MVT::i16, Ops[1]);

  SDValue BufferStoreExt = DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i32, Ops[1]);
  Ops[1] = BufferStoreExt;
  unsigned Opc = (VDataType == MVT::i8) ? AMDGPUISD::BUFFER_STORE_BYTE :
                                 AMDGPUISD::BUFFER_STORE_SHORT;
  ArrayRef<SDValue> OpsRef = makeArrayRef(&Ops[0], 9);
  return DAG.getMemIntrinsicNode(Opc, DL, M->getVTList(), OpsRef, VDataType,
                                     M->getMemOperand());
}

static SDValue getLoadExtOrTrunc(SelectionDAG &DAG,
                                 ISD::LoadExtType ExtType, SDValue Op,
                                 const SDLoc &SL, EVT VT) {
  if (VT.bitsLT(Op.getValueType()))
    return DAG.getNode(ISD::TRUNCATE, SL, VT, Op);

  switch (ExtType) {
  case ISD::SEXTLOAD:
    return DAG.getNode(ISD::SIGN_EXTEND, SL, VT, Op);
  case ISD::ZEXTLOAD:
    return DAG.getNode(ISD::ZERO_EXTEND, SL, VT, Op);
  case ISD::EXTLOAD:
    return DAG.getNode(ISD::ANY_EXTEND, SL, VT, Op);
  case ISD::NON_EXTLOAD:
    return Op;
  }

  llvm_unreachable("invalid ext type");
}

SDValue SITargetLowering::widenLoad(LoadSDNode *Ld, DAGCombinerInfo &DCI) const {
  SelectionDAG &DAG = DCI.DAG;
  if (Ld->getAlignment() < 4 || Ld->isDivergent())
    return SDValue();

  // FIXME: Constant loads should all be marked invariant.
  unsigned AS = Ld->getAddressSpace();
  if (AS != AMDGPUAS::CONSTANT_ADDRESS &&
      AS != AMDGPUAS::CONSTANT_ADDRESS_32BIT &&
      (AS != AMDGPUAS::GLOBAL_ADDRESS || !Ld->isInvariant()))
    return SDValue();

  // Don't do this early, since it may interfere with adjacent load merging for
  // illegal types. We can avoid losing alignment information for exotic types
  // pre-legalize.
  EVT MemVT = Ld->getMemoryVT();
  if ((MemVT.isSimple() && !DCI.isAfterLegalizeDAG()) ||
      MemVT.getSizeInBits() >= 32)
    return SDValue();

  SDLoc SL(Ld);

  assert((!MemVT.isVector() || Ld->getExtensionType() == ISD::NON_EXTLOAD) &&
         "unexpected vector extload");

  // TODO: Drop only high part of range.
  SDValue Ptr = Ld->getBasePtr();
  SDValue NewLoad = DAG.getLoad(ISD::UNINDEXED, ISD::NON_EXTLOAD,
                                MVT::i32, SL, Ld->getChain(), Ptr,
                                Ld->getOffset(),
                                Ld->getPointerInfo(), MVT::i32,
                                Ld->getAlignment(),
                                Ld->getMemOperand()->getFlags(),
                                Ld->getAAInfo(),
                                nullptr); // Drop ranges

  EVT TruncVT = EVT::getIntegerVT(*DAG.getContext(), MemVT.getSizeInBits());
  if (MemVT.isFloatingPoint()) {
    assert(Ld->getExtensionType() == ISD::NON_EXTLOAD &&
           "unexpected fp extload");
    TruncVT = MemVT.changeTypeToInteger();
  }

  SDValue Cvt = NewLoad;
  if (Ld->getExtensionType() == ISD::SEXTLOAD) {
    Cvt = DAG.getNode(ISD::SIGN_EXTEND_INREG, SL, MVT::i32, NewLoad,
                      DAG.getValueType(TruncVT));
  } else if (Ld->getExtensionType() == ISD::ZEXTLOAD ||
             Ld->getExtensionType() == ISD::NON_EXTLOAD) {
    Cvt = DAG.getZeroExtendInReg(NewLoad, SL, TruncVT);
  } else {
    assert(Ld->getExtensionType() == ISD::EXTLOAD);
  }

  EVT VT = Ld->getValueType(0);
  EVT IntVT = EVT::getIntegerVT(*DAG.getContext(), VT.getSizeInBits());

  DCI.AddToWorklist(Cvt.getNode());

  // We may need to handle exotic cases, such as i16->i64 extloads, so insert
  // the appropriate extension from the 32-bit load.
  Cvt = getLoadExtOrTrunc(DAG, Ld->getExtensionType(), Cvt, SL, IntVT);
  DCI.AddToWorklist(Cvt.getNode());

  // Handle conversion back to floating point if necessary.
  Cvt = DAG.getNode(ISD::BITCAST, SL, VT, Cvt);

  return DAG.getMergeValues({ Cvt, NewLoad.getValue(1) }, SL);
}

SDValue SITargetLowering::LowerLOAD(SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
  LoadSDNode *Load = cast<LoadSDNode>(Op);
  ISD::LoadExtType ExtType = Load->getExtensionType();
  EVT MemVT = Load->getMemoryVT();

  if (ExtType == ISD::NON_EXTLOAD && MemVT.getSizeInBits() < 32) {
    if (MemVT == MVT::i16 && isTypeLegal(MVT::i16))
      return SDValue();

    // FIXME: Copied from PPC
    // First, load into 32 bits, then truncate to 1 bit.

    SDValue Chain = Load->getChain();
    SDValue BasePtr = Load->getBasePtr();
    MachineMemOperand *MMO = Load->getMemOperand();

    EVT RealMemVT = (MemVT == MVT::i1) ? MVT::i8 : MVT::i16;

    SDValue NewLD = DAG.getExtLoad(ISD::EXTLOAD, DL, MVT::i32, Chain,
                                   BasePtr, RealMemVT, MMO);

    if (!MemVT.isVector()) {
      SDValue Ops[] = {
        DAG.getNode(ISD::TRUNCATE, DL, MemVT, NewLD),
        NewLD.getValue(1)
      };

      return DAG.getMergeValues(Ops, DL);
    }

    SmallVector<SDValue, 3> Elts;
    for (unsigned I = 0, N = MemVT.getVectorNumElements(); I != N; ++I) {
      SDValue Elt = DAG.getNode(ISD::SRL, DL, MVT::i32, NewLD,
                                DAG.getConstant(I, DL, MVT::i32));

      Elts.push_back(DAG.getNode(ISD::TRUNCATE, DL, MVT::i1, Elt));
    }

    SDValue Ops[] = {
      DAG.getBuildVector(MemVT, DL, Elts),
      NewLD.getValue(1)
    };

    return DAG.getMergeValues(Ops, DL);
  }

  if (!MemVT.isVector())
    return SDValue();

  assert(Op.getValueType().getVectorElementType() == MVT::i32 &&
         "Custom lowering for non-i32 vectors hasn't been implemented.");

  unsigned Alignment = Load->getAlignment();
  unsigned AS = Load->getAddressSpace();
  if (Subtarget->hasLDSMisalignedBug() &&
      AS == AMDGPUAS::FLAT_ADDRESS &&
      Alignment < MemVT.getStoreSize() && MemVT.getSizeInBits() > 32) {
    return SplitVectorLoad(Op, DAG);
  }

  MachineFunction &MF = DAG.getMachineFunction();
  SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
  // If there is a possibilty that flat instruction access scratch memory
  // then we need to use the same legalization rules we use for private.
  if (AS == AMDGPUAS::FLAT_ADDRESS &&
      !Subtarget->hasMultiDwordFlatScratchAddressing())
    AS = MFI->hasFlatScratchInit() ?
         AMDGPUAS::PRIVATE_ADDRESS : AMDGPUAS::GLOBAL_ADDRESS;

  unsigned NumElements = MemVT.getVectorNumElements();

  if (AS == AMDGPUAS::CONSTANT_ADDRESS ||
      AS == AMDGPUAS::CONSTANT_ADDRESS_32BIT) {
    if (!Op->isDivergent() && Alignment >= 4 && NumElements < 32) {
      if (MemVT.isPow2VectorType())
        return SDValue();
      return WidenOrSplitVectorLoad(Op, DAG);
    }
    // Non-uniform loads will be selected to MUBUF instructions, so they
    // have the same legalization requirements as global and private
    // loads.
    //
  }

  if (AS == AMDGPUAS::CONSTANT_ADDRESS ||
      AS == AMDGPUAS::CONSTANT_ADDRESS_32BIT ||
      AS == AMDGPUAS::GLOBAL_ADDRESS) {
    if (Subtarget->getScalarizeGlobalBehavior() && !Op->isDivergent() &&
        Load->isSimple() && isMemOpHasNoClobberedMemOperand(Load) &&
        Alignment >= 4 && NumElements < 32) {
      if (MemVT.isPow2VectorType())
        return SDValue();
      return WidenOrSplitVectorLoad(Op, DAG);
    }
    // Non-uniform loads will be selected to MUBUF instructions, so they
    // have the same legalization requirements as global and private
    // loads.
    //
  }
  if (AS == AMDGPUAS::CONSTANT_ADDRESS ||
      AS == AMDGPUAS::CONSTANT_ADDRESS_32BIT ||
      AS == AMDGPUAS::GLOBAL_ADDRESS ||
      AS == AMDGPUAS::FLAT_ADDRESS) {
    if (NumElements > 4)
      return SplitVectorLoad(Op, DAG);
    // v3 loads not supported on SI.
    if (NumElements == 3 && !Subtarget->hasDwordx3LoadStores())
      return WidenOrSplitVectorLoad(Op, DAG);

    // v3 and v4 loads are supported for private and global memory.
    return SDValue();
  }
  if (AS == AMDGPUAS::PRIVATE_ADDRESS) {
    // Depending on the setting of the private_element_size field in the
    // resource descriptor, we can only make private accesses up to a certain
    // size.
    switch (Subtarget->getMaxPrivateElementSize()) {
    case 4: {
      SDValue Ops[2];
      std::tie(Ops[0], Ops[1]) = scalarizeVectorLoad(Load, DAG);
      return DAG.getMergeValues(Ops, DL);
    }
    case 8:
      if (NumElements > 2)
        return SplitVectorLoad(Op, DAG);
      return SDValue();
    case 16:
      // Same as global/flat
      if (NumElements > 4)
        return SplitVectorLoad(Op, DAG);
      // v3 loads not supported on SI.
      if (NumElements == 3 && !Subtarget->hasDwordx3LoadStores())
        return WidenOrSplitVectorLoad(Op, DAG);

      return SDValue();
    default:
      llvm_unreachable("unsupported private_element_size");
    }
  } else if (AS == AMDGPUAS::LOCAL_ADDRESS || AS == AMDGPUAS::REGION_ADDRESS) {
    // Use ds_read_b128 or ds_read_b96 when possible.
    if (Subtarget->hasDS96AndDS128() &&
        ((Subtarget->useDS128() && MemVT.getStoreSize() == 16) ||
         MemVT.getStoreSize() == 12) &&
        allowsMisalignedMemoryAccessesImpl(MemVT.getSizeInBits(), AS,
                                           Load->getAlign()))
      return SDValue();

    if (NumElements > 2)
      return SplitVectorLoad(Op, DAG);

    // SI has a hardware bug in the LDS / GDS boounds checking: if the base
    // address is negative, then the instruction is incorrectly treated as
    // out-of-bounds even if base + offsets is in bounds. Split vectorized
    // loads here to avoid emitting ds_read2_b32. We may re-combine the
    // load later in the SILoadStoreOptimizer.
    if (Subtarget->getGeneration() == AMDGPUSubtarget::SOUTHERN_ISLANDS &&
        NumElements == 2 && MemVT.getStoreSize() == 8 &&
        Load->getAlignment() < 8) {
      return SplitVectorLoad(Op, DAG);
    }
  }

  if (!allowsMemoryAccessForAlignment(*DAG.getContext(), DAG.getDataLayout(),
                                      MemVT, *Load->getMemOperand())) {
    SDValue Ops[2];
    std::tie(Ops[0], Ops[1]) = expandUnalignedLoad(Load, DAG);
    return DAG.getMergeValues(Ops, DL);
  }

  return SDValue();
}

SDValue SITargetLowering::LowerSELECT(SDValue Op, SelectionDAG &DAG) const {
  EVT VT = Op.getValueType();
  if (VT.getSizeInBits() == 128)
    return splitTernaryVectorOp(Op, DAG);

  assert(VT.getSizeInBits() == 64);

  SDLoc DL(Op);
  SDValue Cond = Op.getOperand(0);

  SDValue Zero = DAG.getConstant(0, DL, MVT::i32);
  SDValue One = DAG.getConstant(1, DL, MVT::i32);

  SDValue LHS = DAG.getNode(ISD::BITCAST, DL, MVT::v2i32, Op.getOperand(1));
  SDValue RHS = DAG.getNode(ISD::BITCAST, DL, MVT::v2i32, Op.getOperand(2));

  SDValue Lo0 = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, MVT::i32, LHS, Zero);
  SDValue Lo1 = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, MVT::i32, RHS, Zero);

  SDValue Lo = DAG.getSelect(DL, MVT::i32, Cond, Lo0, Lo1);

  SDValue Hi0 = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, MVT::i32, LHS, One);
  SDValue Hi1 = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, MVT::i32, RHS, One);

  SDValue Hi = DAG.getSelect(DL, MVT::i32, Cond, Hi0, Hi1);

  SDValue Res = DAG.getBuildVector(MVT::v2i32, DL, {Lo, Hi});
  return DAG.getNode(ISD::BITCAST, DL, VT, Res);
}

// Catch division cases where we can use shortcuts with rcp and rsq
// instructions.
SDValue SITargetLowering::lowerFastUnsafeFDIV(SDValue Op,
                                              SelectionDAG &DAG) const {
  SDLoc SL(Op);
  SDValue LHS = Op.getOperand(0);
  SDValue RHS = Op.getOperand(1);
  EVT VT = Op.getValueType();
  const SDNodeFlags Flags = Op->getFlags();

  bool AllowInaccurateRcp = Flags.hasApproximateFuncs();

  // Without !fpmath accuracy information, we can't do more because we don't
  // know exactly whether rcp is accurate enough to meet !fpmath requirement.
  if (!AllowInaccurateRcp)
    return SDValue();

  if (const ConstantFPSDNode *CLHS = dyn_cast<ConstantFPSDNode>(LHS)) {
    if (CLHS->isExactlyValue(1.0)) {
      // v_rcp_f32 and v_rsq_f32 do not support denormals, and according to
      // the CI documentation has a worst case error of 1 ulp.
      // OpenCL requires <= 2.5 ulp for 1.0 / x, so it should always be OK to
      // use it as long as we aren't trying to use denormals.
      //
      // v_rcp_f16 and v_rsq_f16 DO support denormals.

      // 1.0 / sqrt(x) -> rsq(x)

      // XXX - Is UnsafeFPMath sufficient to do this for f64? The maximum ULP
      // error seems really high at 2^29 ULP.
      if (RHS.getOpcode() == ISD::FSQRT)
        return DAG.getNode(AMDGPUISD::RSQ, SL, VT, RHS.getOperand(0));

      // 1.0 / x -> rcp(x)
      return DAG.getNode(AMDGPUISD::RCP, SL, VT, RHS);
    }

    // Same as for 1.0, but expand the sign out of the constant.
    if (CLHS->isExactlyValue(-1.0)) {
      // -1.0 / x -> rcp (fneg x)
      SDValue FNegRHS = DAG.getNode(ISD::FNEG, SL, VT, RHS);
      return DAG.getNode(AMDGPUISD::RCP, SL, VT, FNegRHS);
    }
  }

  // Turn into multiply by the reciprocal.
  // x / y -> x * (1.0 / y)
  SDValue Recip = DAG.getNode(AMDGPUISD::RCP, SL, VT, RHS);
  return DAG.getNode(ISD::FMUL, SL, VT, LHS, Recip, Flags);
}

SDValue SITargetLowering::lowerFastUnsafeFDIV64(SDValue Op,
                                                SelectionDAG &DAG) const {
  SDLoc SL(Op);
  SDValue X = Op.getOperand(0);
  SDValue Y = Op.getOperand(1);
  EVT VT = Op.getValueType();
  const SDNodeFlags Flags = Op->getFlags();

  bool AllowInaccurateDiv = Flags.hasApproximateFuncs() ||
                            DAG.getTarget().Options.UnsafeFPMath;
  if (!AllowInaccurateDiv)
    return SDValue();

  SDValue NegY = DAG.getNode(ISD::FNEG, SL, VT, Y);
  SDValue One = DAG.getConstantFP(1.0, SL, VT);

  SDValue R = DAG.getNode(AMDGPUISD::RCP, SL, VT, Y);
  SDValue Tmp0 = DAG.getNode(ISD::FMA, SL, VT, NegY, R, One);

  R = DAG.getNode(ISD::FMA, SL, VT, Tmp0, R, R);
  SDValue Tmp1 = DAG.getNode(ISD::FMA, SL, VT, NegY, R, One);
  R = DAG.getNode(ISD::FMA, SL, VT, Tmp1, R, R);
  SDValue Ret = DAG.getNode(ISD::FMUL, SL, VT, X, R);
  SDValue Tmp2 = DAG.getNode(ISD::FMA, SL, VT, NegY, Ret, X);
  return DAG.getNode(ISD::FMA, SL, VT, Tmp2, R, Ret);
}

static SDValue getFPBinOp(SelectionDAG &DAG, unsigned Opcode, const SDLoc &SL,
                          EVT VT, SDValue A, SDValue B, SDValue GlueChain,
                          SDNodeFlags Flags) {
  if (GlueChain->getNumValues() <= 1) {
    return DAG.getNode(Opcode, SL, VT, A, B, Flags);
  }

  assert(GlueChain->getNumValues() == 3);

  SDVTList VTList = DAG.getVTList(VT, MVT::Other, MVT::Glue);
  switch (Opcode) {
  default: llvm_unreachable("no chain equivalent for opcode");
  case ISD::FMUL:
    Opcode = AMDGPUISD::FMUL_W_CHAIN;
    break;
  }

  return DAG.getNode(Opcode, SL, VTList,
                     {GlueChain.getValue(1), A, B, GlueChain.getValue(2)},
                     Flags);
}

static SDValue getFPTernOp(SelectionDAG &DAG, unsigned Opcode, const SDLoc &SL,
                           EVT VT, SDValue A, SDValue B, SDValue C,
                           SDValue GlueChain, SDNodeFlags Flags) {
  if (GlueChain->getNumValues() <= 1) {
    return DAG.getNode(Opcode, SL, VT, {A, B, C}, Flags);
  }

  assert(GlueChain->getNumValues() == 3);

  SDVTList VTList = DAG.getVTList(VT, MVT::Other, MVT::Glue);
  switch (Opcode) {
  default: llvm_unreachable("no chain equivalent for opcode");
  case ISD::FMA:
    Opcode = AMDGPUISD::FMA_W_CHAIN;
    break;
  }

  return DAG.getNode(Opcode, SL, VTList,
                     {GlueChain.getValue(1), A, B, C, GlueChain.getValue(2)},
                     Flags);
}

SDValue SITargetLowering::LowerFDIV16(SDValue Op, SelectionDAG &DAG) const {
  if (SDValue FastLowered = lowerFastUnsafeFDIV(Op, DAG))
    return FastLowered;

  SDLoc SL(Op);
  SDValue Src0 = Op.getOperand(0);
  SDValue Src1 = Op.getOperand(1);

  SDValue CvtSrc0 = DAG.getNode(ISD::FP_EXTEND, SL, MVT::f32, Src0);
  SDValue CvtSrc1 = DAG.getNode(ISD::FP_EXTEND, SL, MVT::f32, Src1);

  SDValue RcpSrc1 = DAG.getNode(AMDGPUISD::RCP, SL, MVT::f32, CvtSrc1);
  SDValue Quot = DAG.getNode(ISD::FMUL, SL, MVT::f32, CvtSrc0, RcpSrc1);

  SDValue FPRoundFlag = DAG.getTargetConstant(0, SL, MVT::i32);
  SDValue BestQuot = DAG.getNode(ISD::FP_ROUND, SL, MVT::f16, Quot, FPRoundFlag);

  return DAG.getNode(AMDGPUISD::DIV_FIXUP, SL, MVT::f16, BestQuot, Src1, Src0);
}

// Faster 2.5 ULP division that does not support denormals.
SDValue SITargetLowering::lowerFDIV_FAST(SDValue Op, SelectionDAG &DAG) const {
  SDLoc SL(Op);
  SDValue LHS = Op.getOperand(1);
  SDValue RHS = Op.getOperand(2);

  SDValue r1 = DAG.getNode(ISD::FABS, SL, MVT::f32, RHS);

  const APFloat K0Val(BitsToFloat(0x6f800000));
  const SDValue K0 = DAG.getConstantFP(K0Val, SL, MVT::f32);

  const APFloat K1Val(BitsToFloat(0x2f800000));
  const SDValue K1 = DAG.getConstantFP(K1Val, SL, MVT::f32);

  const SDValue One = DAG.getConstantFP(1.0, SL, MVT::f32);

  EVT SetCCVT =
    getSetCCResultType(DAG.getDataLayout(), *DAG.getContext(), MVT::f32);

  SDValue r2 = DAG.getSetCC(SL, SetCCVT, r1, K0, ISD::SETOGT);

  SDValue r3 = DAG.getNode(ISD::SELECT, SL, MVT::f32, r2, K1, One);

  // TODO: Should this propagate fast-math-flags?
  r1 = DAG.getNode(ISD::FMUL, SL, MVT::f32, RHS, r3);

  // rcp does not support denormals.
  SDValue r0 = DAG.getNode(AMDGPUISD::RCP, SL, MVT::f32, r1);

  SDValue Mul = DAG.getNode(ISD::FMUL, SL, MVT::f32, LHS, r0);

  return DAG.getNode(ISD::FMUL, SL, MVT::f32, r3, Mul);
}

// Returns immediate value for setting the F32 denorm mode when using the
// S_DENORM_MODE instruction.
static SDValue getSPDenormModeValue(int SPDenormMode, SelectionDAG &DAG,
                                    const SDLoc &SL, const GCNSubtarget *ST) {
  assert(ST->hasDenormModeInst() && "Requires S_DENORM_MODE");
  int DPDenormModeDefault = hasFP64FP16Denormals(DAG.getMachineFunction())
                                ? FP_DENORM_FLUSH_NONE
                                : FP_DENORM_FLUSH_IN_FLUSH_OUT;

  int Mode = SPDenormMode | (DPDenormModeDefault << 2);
  return DAG.getTargetConstant(Mode, SL, MVT::i32);
}

SDValue SITargetLowering::LowerFDIV32(SDValue Op, SelectionDAG &DAG) const {
  if (SDValue FastLowered = lowerFastUnsafeFDIV(Op, DAG))
    return FastLowered;

  // The selection matcher assumes anything with a chain selecting to a
  // mayRaiseFPException machine instruction. Since we're introducing a chain
  // here, we need to explicitly report nofpexcept for the regular fdiv
  // lowering.
  SDNodeFlags Flags = Op->getFlags();
  Flags.setNoFPExcept(true);

  SDLoc SL(Op);
  SDValue LHS = Op.getOperand(0);
  SDValue RHS = Op.getOperand(1);

  const SDValue One = DAG.getConstantFP(1.0, SL, MVT::f32);

  SDVTList ScaleVT = DAG.getVTList(MVT::f32, MVT::i1);

  SDValue DenominatorScaled = DAG.getNode(AMDGPUISD::DIV_SCALE, SL, ScaleVT,
                                          {RHS, RHS, LHS}, Flags);
  SDValue NumeratorScaled = DAG.getNode(AMDGPUISD::DIV_SCALE, SL, ScaleVT,
                                        {LHS, RHS, LHS}, Flags);

  // Denominator is scaled to not be denormal, so using rcp is ok.
  SDValue ApproxRcp = DAG.getNode(AMDGPUISD::RCP, SL, MVT::f32,
                                  DenominatorScaled, Flags);
  SDValue NegDivScale0 = DAG.getNode(ISD::FNEG, SL, MVT::f32,
                                     DenominatorScaled, Flags);

  const unsigned Denorm32Reg = AMDGPU::Hwreg::ID_MODE |
                               (4 << AMDGPU::Hwreg::OFFSET_SHIFT_) |
                               (1 << AMDGPU::Hwreg::WIDTH_M1_SHIFT_);
  const SDValue BitField = DAG.getTargetConstant(Denorm32Reg, SL, MVT::i32);

  const bool HasFP32Denormals = hasFP32Denormals(DAG.getMachineFunction());

  if (!HasFP32Denormals) {
    // Note we can't use the STRICT_FMA/STRICT_FMUL for the non-strict FDIV
    // lowering. The chain dependence is insufficient, and we need glue. We do
    // not need the glue variants in a strictfp function.

    SDVTList BindParamVTs = DAG.getVTList(MVT::Other, MVT::Glue);

    SDNode *EnableDenorm;
    if (Subtarget->hasDenormModeInst()) {
      const SDValue EnableDenormValue =
          getSPDenormModeValue(FP_DENORM_FLUSH_NONE, DAG, SL, Subtarget);

      EnableDenorm = DAG.getNode(AMDGPUISD::DENORM_MODE, SL, BindParamVTs,
                                 DAG.getEntryNode(), EnableDenormValue).getNode();
    } else {
      const SDValue EnableDenormValue = DAG.getConstant(FP_DENORM_FLUSH_NONE,
                                                        SL, MVT::i32);
      EnableDenorm =
          DAG.getMachineNode(AMDGPU::S_SETREG_B32, SL, BindParamVTs,
                             {EnableDenormValue, BitField, DAG.getEntryNode()});
    }

    SDValue Ops[3] = {
      NegDivScale0,
      SDValue(EnableDenorm, 0),
      SDValue(EnableDenorm, 1)
    };

    NegDivScale0 = DAG.getMergeValues(Ops, SL);
  }

  SDValue Fma0 = getFPTernOp(DAG, ISD::FMA, SL, MVT::f32, NegDivScale0,
                             ApproxRcp, One, NegDivScale0, Flags);

  SDValue Fma1 = getFPTernOp(DAG, ISD::FMA, SL, MVT::f32, Fma0, ApproxRcp,
                             ApproxRcp, Fma0, Flags);

  SDValue Mul = getFPBinOp(DAG, ISD::FMUL, SL, MVT::f32, NumeratorScaled,
                           Fma1, Fma1, Flags);

  SDValue Fma2 = getFPTernOp(DAG, ISD::FMA, SL, MVT::f32, NegDivScale0, Mul,
                             NumeratorScaled, Mul, Flags);

  SDValue Fma3 = getFPTernOp(DAG, ISD::FMA, SL, MVT::f32,
                             Fma2, Fma1, Mul, Fma2, Flags);

  SDValue Fma4 = getFPTernOp(DAG, ISD::FMA, SL, MVT::f32, NegDivScale0, Fma3,
                             NumeratorScaled, Fma3, Flags);

  if (!HasFP32Denormals) {
    SDNode *DisableDenorm;
    if (Subtarget->hasDenormModeInst()) {
      const SDValue DisableDenormValue =
          getSPDenormModeValue(FP_DENORM_FLUSH_IN_FLUSH_OUT, DAG, SL, Subtarget);

      DisableDenorm = DAG.getNode(AMDGPUISD::DENORM_MODE, SL, MVT::Other,
                                  Fma4.getValue(1), DisableDenormValue,
                                  Fma4.getValue(2)).getNode();
    } else {
      const SDValue DisableDenormValue =
          DAG.getConstant(FP_DENORM_FLUSH_IN_FLUSH_OUT, SL, MVT::i32);

      DisableDenorm = DAG.getMachineNode(
          AMDGPU::S_SETREG_B32, SL, MVT::Other,
          {DisableDenormValue, BitField, Fma4.getValue(1), Fma4.getValue(2)});
    }

    SDValue OutputChain = DAG.getNode(ISD::TokenFactor, SL, MVT::Other,
                                      SDValue(DisableDenorm, 0), DAG.getRoot());
    DAG.setRoot(OutputChain);
  }

  SDValue Scale = NumeratorScaled.getValue(1);
  SDValue Fmas = DAG.getNode(AMDGPUISD::DIV_FMAS, SL, MVT::f32,
                             {Fma4, Fma1, Fma3, Scale}, Flags);

  return DAG.getNode(AMDGPUISD::DIV_FIXUP, SL, MVT::f32, Fmas, RHS, LHS, Flags);
}

SDValue SITargetLowering::LowerFDIV64(SDValue Op, SelectionDAG &DAG) const {
  if (SDValue FastLowered = lowerFastUnsafeFDIV64(Op, DAG))
    return FastLowered;

  SDLoc SL(Op);
  SDValue X = Op.getOperand(0);
  SDValue Y = Op.getOperand(1);

  const SDValue One = DAG.getConstantFP(1.0, SL, MVT::f64);

  SDVTList ScaleVT = DAG.getVTList(MVT::f64, MVT::i1);

  SDValue DivScale0 = DAG.getNode(AMDGPUISD::DIV_SCALE, SL, ScaleVT, Y, Y, X);

  SDValue NegDivScale0 = DAG.getNode(ISD::FNEG, SL, MVT::f64, DivScale0);

  SDValue Rcp = DAG.getNode(AMDGPUISD::RCP, SL, MVT::f64, DivScale0);

  SDValue Fma0 = DAG.getNode(ISD::FMA, SL, MVT::f64, NegDivScale0, Rcp, One);

  SDValue Fma1 = DAG.getNode(ISD::FMA, SL, MVT::f64, Rcp, Fma0, Rcp);

  SDValue Fma2 = DAG.getNode(ISD::FMA, SL, MVT::f64, NegDivScale0, Fma1, One);

  SDValue DivScale1 = DAG.getNode(AMDGPUISD::DIV_SCALE, SL, ScaleVT, X, Y, X);

  SDValue Fma3 = DAG.getNode(ISD::FMA, SL, MVT::f64, Fma1, Fma2, Fma1);
  SDValue Mul = DAG.getNode(ISD::FMUL, SL, MVT::f64, DivScale1, Fma3);

  SDValue Fma4 = DAG.getNode(ISD::FMA, SL, MVT::f64,
                             NegDivScale0, Mul, DivScale1);

  SDValue Scale;

  if (!Subtarget->hasUsableDivScaleConditionOutput()) {
    // Workaround a hardware bug on SI where the condition output from div_scale
    // is not usable.

    const SDValue Hi = DAG.getConstant(1, SL, MVT::i32);

    // Figure out if the scale to use for div_fmas.
    SDValue NumBC = DAG.getNode(ISD::BITCAST, SL, MVT::v2i32, X);
    SDValue DenBC = DAG.getNode(ISD::BITCAST, SL, MVT::v2i32, Y);
    SDValue Scale0BC = DAG.getNode(ISD::BITCAST, SL, MVT::v2i32, DivScale0);
    SDValue Scale1BC = DAG.getNode(ISD::BITCAST, SL, MVT::v2i32, DivScale1);

    SDValue NumHi = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, SL, MVT::i32, NumBC, Hi);
    SDValue DenHi = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, SL, MVT::i32, DenBC, Hi);

    SDValue Scale0Hi
      = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, SL, MVT::i32, Scale0BC, Hi);
    SDValue Scale1Hi
      = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, SL, MVT::i32, Scale1BC, Hi);

    SDValue CmpDen = DAG.getSetCC(SL, MVT::i1, DenHi, Scale0Hi, ISD::SETEQ);
    SDValue CmpNum = DAG.getSetCC(SL, MVT::i1, NumHi, Scale1Hi, ISD::SETEQ);
    Scale = DAG.getNode(ISD::XOR, SL, MVT::i1, CmpNum, CmpDen);
  } else {
    Scale = DivScale1.getValue(1);
  }

  SDValue Fmas = DAG.getNode(AMDGPUISD::DIV_FMAS, SL, MVT::f64,
                             Fma4, Fma3, Mul, Scale);

  return DAG.getNode(AMDGPUISD::DIV_FIXUP, SL, MVT::f64, Fmas, Y, X);
}

SDValue SITargetLowering::LowerFDIV(SDValue Op, SelectionDAG &DAG) const {
  EVT VT = Op.getValueType();

  if (VT == MVT::f32)
    return LowerFDIV32(Op, DAG);

  if (VT == MVT::f64)
    return LowerFDIV64(Op, DAG);

  if (VT == MVT::f16)
    return LowerFDIV16(Op, DAG);

  llvm_unreachable("Unexpected type for fdiv");
}

SDValue SITargetLowering::LowerSTORE(SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
  StoreSDNode *Store = cast<StoreSDNode>(Op);
  EVT VT = Store->getMemoryVT();

  if (VT == MVT::i1) {
    return DAG.getTruncStore(Store->getChain(), DL,
       DAG.getSExtOrTrunc(Store->getValue(), DL, MVT::i32),
       Store->getBasePtr(), MVT::i1, Store->getMemOperand());
  }

  assert(VT.isVector() &&
         Store->getValue().getValueType().getScalarType() == MVT::i32);

  unsigned AS = Store->getAddressSpace();
  if (Subtarget->hasLDSMisalignedBug() &&
      AS == AMDGPUAS::FLAT_ADDRESS &&
      Store->getAlignment() < VT.getStoreSize() && VT.getSizeInBits() > 32) {
    return SplitVectorStore(Op, DAG);
  }

  MachineFunction &MF = DAG.getMachineFunction();
  SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
  // If there is a possibilty that flat instruction access scratch memory
  // then we need to use the same legalization rules we use for private.
  if (AS == AMDGPUAS::FLAT_ADDRESS &&
      !Subtarget->hasMultiDwordFlatScratchAddressing())
    AS = MFI->hasFlatScratchInit() ?
         AMDGPUAS::PRIVATE_ADDRESS : AMDGPUAS::GLOBAL_ADDRESS;

  unsigned NumElements = VT.getVectorNumElements();
  if (AS == AMDGPUAS::GLOBAL_ADDRESS ||
      AS == AMDGPUAS::FLAT_ADDRESS) {
    if (NumElements > 4)
      return SplitVectorStore(Op, DAG);
    // v3 stores not supported on SI.
    if (NumElements == 3 && !Subtarget->hasDwordx3LoadStores())
      return SplitVectorStore(Op, DAG);

    if (!allowsMemoryAccessForAlignment(*DAG.getContext(), DAG.getDataLayout(),
                                        VT, *Store->getMemOperand()))
      return expandUnalignedStore(Store, DAG);

    return SDValue();
  } else if (AS == AMDGPUAS::PRIVATE_ADDRESS) {
    switch (Subtarget->getMaxPrivateElementSize()) {
    case 4:
      return scalarizeVectorStore(Store, DAG);
    case 8:
      if (NumElements > 2)
        return SplitVectorStore(Op, DAG);
      return SDValue();
    case 16:
      if (NumElements > 4 ||
          (NumElements == 3 && !Subtarget->enableFlatScratch()))
        return SplitVectorStore(Op, DAG);
      return SDValue();
    default:
      llvm_unreachable("unsupported private_element_size");
    }
  } else if (AS == AMDGPUAS::LOCAL_ADDRESS || AS == AMDGPUAS::REGION_ADDRESS) {
    // Use ds_write_b128 or ds_write_b96 when possible.
    if (Subtarget->hasDS96AndDS128() &&
        ((Subtarget->useDS128() && VT.getStoreSize() == 16) ||
         (VT.getStoreSize() == 12)) &&
        allowsMisalignedMemoryAccessesImpl(VT.getSizeInBits(), AS,
                                           Store->getAlign()))
      return SDValue();

    if (NumElements > 2)
      return SplitVectorStore(Op, DAG);

    // SI has a hardware bug in the LDS / GDS boounds checking: if the base
    // address is negative, then the instruction is incorrectly treated as
    // out-of-bounds even if base + offsets is in bounds. Split vectorized
    // stores here to avoid emitting ds_write2_b32. We may re-combine the
    // store later in the SILoadStoreOptimizer.
    if (!Subtarget->hasUsableDSOffset() &&
        NumElements == 2 && VT.getStoreSize() == 8 &&
        Store->getAlignment() < 8) {
      return SplitVectorStore(Op, DAG);
    }

    if (!allowsMemoryAccessForAlignment(*DAG.getContext(), DAG.getDataLayout(),
                                        VT, *Store->getMemOperand())) {
      if (VT.isVector())
        return SplitVectorStore(Op, DAG);
      return expandUnalignedStore(Store, DAG);
    }

    return SDValue();
  } else {
    llvm_unreachable("unhandled address space");
  }
}

SDValue SITargetLowering::LowerTrig(SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT VT = Op.getValueType();
  SDValue Arg = Op.getOperand(0);
  SDValue TrigVal;

  // Propagate fast-math flags so that the multiply we introduce can be folded
  // if Arg is already the result of a multiply by constant.
  auto Flags = Op->getFlags();

  SDValue OneOver2Pi = DAG.getConstantFP(0.5 * numbers::inv_pi, DL, VT);

  if (Subtarget->hasTrigReducedRange()) {
    SDValue MulVal = DAG.getNode(ISD::FMUL, DL, VT, Arg, OneOver2Pi, Flags);
    TrigVal = DAG.getNode(AMDGPUISD::FRACT, DL, VT, MulVal, Flags);
  } else {
    TrigVal = DAG.getNode(ISD::FMUL, DL, VT, Arg, OneOver2Pi, Flags);
  }

  switch (Op.getOpcode()) {
  case ISD::FCOS:
    return DAG.getNode(AMDGPUISD::COS_HW, SDLoc(Op), VT, TrigVal, Flags);
  case ISD::FSIN:
    return DAG.getNode(AMDGPUISD::SIN_HW, SDLoc(Op), VT, TrigVal, Flags);
  default:
    llvm_unreachable("Wrong trig opcode");
  }
}

SDValue SITargetLowering::LowerATOMIC_CMP_SWAP(SDValue Op, SelectionDAG &DAG) const {
  AtomicSDNode *AtomicNode = cast<AtomicSDNode>(Op);
  assert(AtomicNode->isCompareAndSwap());
  unsigned AS = AtomicNode->getAddressSpace();

  // No custom lowering required for local address space
  if (!AMDGPU::isFlatGlobalAddrSpace(AS))
    return Op;

  // Non-local address space requires custom lowering for atomic compare
  // and swap; cmp and swap should be in a v2i32 or v2i64 in case of _X2
  SDLoc DL(Op);
  SDValue ChainIn = Op.getOperand(0);
  SDValue Addr = Op.getOperand(1);
  SDValue Old = Op.getOperand(2);
  SDValue New = Op.getOperand(3);
  EVT VT = Op.getValueType();
  MVT SimpleVT = VT.getSimpleVT();
  MVT VecType = MVT::getVectorVT(SimpleVT, 2);

  SDValue NewOld = DAG.getBuildVector(VecType, DL, {New, Old});
  SDValue Ops[] = { ChainIn, Addr, NewOld };

  return DAG.getMemIntrinsicNode(AMDGPUISD::ATOMIC_CMP_SWAP, DL, Op->getVTList(),
                                 Ops, VT, AtomicNode->getMemOperand());
}

//===----------------------------------------------------------------------===//
// Custom DAG optimizations
//===----------------------------------------------------------------------===//

SDValue SITargetLowering::performUCharToFloatCombine(SDNode *N,
                                                     DAGCombinerInfo &DCI) const {
  EVT VT = N->getValueType(0);
  EVT ScalarVT = VT.getScalarType();
  if (ScalarVT != MVT::f32 && ScalarVT != MVT::f16)
    return SDValue();

  SelectionDAG &DAG = DCI.DAG;
  SDLoc DL(N);

  SDValue Src = N->getOperand(0);
  EVT SrcVT = Src.getValueType();

  // TODO: We could try to match extracting the higher bytes, which would be
  // easier if i8 vectors weren't promoted to i32 vectors, particularly after
  // types are legalized. v4i8 -> v4f32 is probably the only case to worry
  // about in practice.
  if (DCI.isAfterLegalizeDAG() && SrcVT == MVT::i32) {
    if (DAG.MaskedValueIsZero(Src, APInt::getHighBitsSet(32, 24))) {
      SDValue Cvt = DAG.getNode(AMDGPUISD::CVT_F32_UBYTE0, DL, MVT::f32, Src);
      DCI.AddToWorklist(Cvt.getNode());

      // For the f16 case, fold to a cast to f32 and then cast back to f16.
      if (ScalarVT != MVT::f32) {
        Cvt = DAG.getNode(ISD::FP_ROUND, DL, VT, Cvt,
                          DAG.getTargetConstant(0, DL, MVT::i32));
      }
      return Cvt;
    }
  }

  return SDValue();
}

// (shl (add x, c1), c2) -> add (shl x, c2), (shl c1, c2)

// This is a variant of
// (mul (add x, c1), c2) -> add (mul x, c2), (mul c1, c2),
//
// The normal DAG combiner will do this, but only if the add has one use since
// that would increase the number of instructions.
//
// This prevents us from seeing a constant offset that can be folded into a
// memory instruction's addressing mode. If we know the resulting add offset of
// a pointer can be folded into an addressing offset, we can replace the pointer
// operand with the add of new constant offset. This eliminates one of the uses,
// and may allow the remaining use to also be simplified.
//
SDValue SITargetLowering::performSHLPtrCombine(SDNode *N,
                                               unsigned AddrSpace,
                                               EVT MemVT,
                                               DAGCombinerInfo &DCI) const {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);

  // We only do this to handle cases where it's profitable when there are
  // multiple uses of the add, so defer to the standard combine.
  if ((N0.getOpcode() != ISD::ADD && N0.getOpcode() != ISD::OR) ||
      N0->hasOneUse())
    return SDValue();

  const ConstantSDNode *CN1 = dyn_cast<ConstantSDNode>(N1);
  if (!CN1)
    return SDValue();

  const ConstantSDNode *CAdd = dyn_cast<ConstantSDNode>(N0.getOperand(1));
  if (!CAdd)
    return SDValue();

  // If the resulting offset is too large, we can't fold it into the addressing
  // mode offset.
  APInt Offset = CAdd->getAPIntValue() << CN1->getAPIntValue();
  Type *Ty = MemVT.getTypeForEVT(*DCI.DAG.getContext());

  AddrMode AM;
  AM.HasBaseReg = true;
  AM.BaseOffs = Offset.getSExtValue();
  if (!isLegalAddressingMode(DCI.DAG.getDataLayout(), AM, Ty, AddrSpace))
    return SDValue();

  SelectionDAG &DAG = DCI.DAG;
  SDLoc SL(N);
  EVT VT = N->getValueType(0);

  SDValue ShlX = DAG.getNode(ISD::SHL, SL, VT, N0.getOperand(0), N1);
  SDValue COffset = DAG.getConstant(Offset, SL, VT);

  SDNodeFlags Flags;
  Flags.setNoUnsignedWrap(N->getFlags().hasNoUnsignedWrap() &&
                          (N0.getOpcode() == ISD::OR ||
                           N0->getFlags().hasNoUnsignedWrap()));

  return DAG.getNode(ISD::ADD, SL, VT, ShlX, COffset, Flags);
}

/// MemSDNode::getBasePtr() does not work for intrinsics, which needs to offset
/// by the chain and intrinsic ID. Theoretically we would also need to check the
/// specific intrinsic, but they all place the pointer operand first.
static unsigned getBasePtrIndex(const MemSDNode *N) {
  switch (N->getOpcode()) {
  case ISD::STORE:
  case ISD::INTRINSIC_W_CHAIN:
  case ISD::INTRINSIC_VOID:
    return 2;
  default:
    return 1;
  }
}

SDValue SITargetLowering::performMemSDNodeCombine(MemSDNode *N,
                                                  DAGCombinerInfo &DCI) const {
  SelectionDAG &DAG = DCI.DAG;
  SDLoc SL(N);

  unsigned PtrIdx = getBasePtrIndex(N);
  SDValue Ptr = N->getOperand(PtrIdx);

  // TODO: We could also do this for multiplies.
  if (Ptr.getOpcode() == ISD::SHL) {
    SDValue NewPtr = performSHLPtrCombine(Ptr.getNode(),  N->getAddressSpace(),
                                          N->getMemoryVT(), DCI);
    if (NewPtr) {
      SmallVector<SDValue, 8> NewOps(N->op_begin(), N->op_end());

      NewOps[PtrIdx] = NewPtr;
      return SDValue(DAG.UpdateNodeOperands(N, NewOps), 0);
    }
  }

  return SDValue();
}

static bool bitOpWithConstantIsReducible(unsigned Opc, uint32_t Val) {
  return (Opc == ISD::AND && (Val == 0 || Val == 0xffffffff)) ||
         (Opc == ISD::OR && (Val == 0xffffffff || Val == 0)) ||
         (Opc == ISD::XOR && Val == 0);
}

// Break up 64-bit bit operation of a constant into two 32-bit and/or/xor. This
// will typically happen anyway for a VALU 64-bit and. This exposes other 32-bit
// integer combine opportunities since most 64-bit operations are decomposed
// this way.  TODO: We won't want this for SALU especially if it is an inline
// immediate.
SDValue SITargetLowering::splitBinaryBitConstantOp(
  DAGCombinerInfo &DCI,
  const SDLoc &SL,
  unsigned Opc, SDValue LHS,
  const ConstantSDNode *CRHS) const {
  uint64_t Val = CRHS->getZExtValue();
  uint32_t ValLo = Lo_32(Val);
  uint32_t ValHi = Hi_32(Val);
  const SIInstrInfo *TII = getSubtarget()->getInstrInfo();

    if ((bitOpWithConstantIsReducible(Opc, ValLo) ||
         bitOpWithConstantIsReducible(Opc, ValHi)) ||
        (CRHS->hasOneUse() && !TII->isInlineConstant(CRHS->getAPIntValue()))) {
    // If we need to materialize a 64-bit immediate, it will be split up later
    // anyway. Avoid creating the harder to understand 64-bit immediate
    // materialization.
    return splitBinaryBitConstantOpImpl(DCI, SL, Opc, LHS, ValLo, ValHi);
  }

  return SDValue();
}

// Returns true if argument is a boolean value which is not serialized into
// memory or argument and does not require v_cndmask_b32 to be deserialized.
static bool isBoolSGPR(SDValue V) {
  if (V.getValueType() != MVT::i1)
    return false;
  switch (V.getOpcode()) {
  default:
    break;
  case ISD::SETCC:
  case AMDGPUISD::FP_CLASS:
    return true;
  case ISD::AND:
  case ISD::OR:
  case ISD::XOR:
    return isBoolSGPR(V.getOperand(0)) && isBoolSGPR(V.getOperand(1));
  }
  return false;
}

// If a constant has all zeroes or all ones within each byte return it.
// Otherwise return 0.
static uint32_t getConstantPermuteMask(uint32_t C) {
  // 0xff for any zero byte in the mask
  uint32_t ZeroByteMask = 0;
  if (!(C & 0x000000ff)) ZeroByteMask |= 0x000000ff;
  if (!(C & 0x0000ff00)) ZeroByteMask |= 0x0000ff00;
  if (!(C & 0x00ff0000)) ZeroByteMask |= 0x00ff0000;
  if (!(C & 0xff000000)) ZeroByteMask |= 0xff000000;
  uint32_t NonZeroByteMask = ~ZeroByteMask; // 0xff for any non-zero byte
  if ((NonZeroByteMask & C) != NonZeroByteMask)
    return 0; // Partial bytes selected.
  return C;
}

// Check if a node selects whole bytes from its operand 0 starting at a byte
// boundary while masking the rest. Returns select mask as in the v_perm_b32
// or -1 if not succeeded.
// Note byte select encoding:
// value 0-3 selects corresponding source byte;
// value 0xc selects zero;
// value 0xff selects 0xff.
static uint32_t getPermuteMask(SelectionDAG &DAG, SDValue V) {
  assert(V.getValueSizeInBits() == 32);

  if (V.getNumOperands() != 2)
    return ~0;

  ConstantSDNode *N1 = dyn_cast<ConstantSDNode>(V.getOperand(1));
  if (!N1)
    return ~0;

  uint32_t C = N1->getZExtValue();

  switch (V.getOpcode()) {
  default:
    break;
  case ISD::AND:
    if (uint32_t ConstMask = getConstantPermuteMask(C)) {
      return (0x03020100 & ConstMask) | (0x0c0c0c0c & ~ConstMask);
    }
    break;

  case ISD::OR:
    if (uint32_t ConstMask = getConstantPermuteMask(C)) {
      return (0x03020100 & ~ConstMask) | ConstMask;
    }
    break;

  case ISD::SHL:
    if (C % 8)
      return ~0;

    return uint32_t((0x030201000c0c0c0cull << C) >> 32);

  case ISD::SRL:
    if (C % 8)
      return ~0;

    return uint32_t(0x0c0c0c0c03020100ull >> C);
  }

  return ~0;
}

SDValue SITargetLowering::performAndCombine(SDNode *N,
                                            DAGCombinerInfo &DCI) const {
  if (DCI.isBeforeLegalize())
    return SDValue();

  SelectionDAG &DAG = DCI.DAG;
  EVT VT = N->getValueType(0);
  SDValue LHS = N->getOperand(0);
  SDValue RHS = N->getOperand(1);


  const ConstantSDNode *CRHS = dyn_cast<ConstantSDNode>(RHS);
  if (VT == MVT::i64 && CRHS) {
    if (SDValue Split
        = splitBinaryBitConstantOp(DCI, SDLoc(N), ISD::AND, LHS, CRHS))
      return Split;
  }

  if (CRHS && VT == MVT::i32) {
    // and (srl x, c), mask => shl (bfe x, nb + c, mask >> nb), nb
    // nb = number of trailing zeroes in mask
    // It can be optimized out using SDWA for GFX8+ in the SDWA peephole pass,
    // given that we are selecting 8 or 16 bit fields starting at byte boundary.
    uint64_t Mask = CRHS->getZExtValue();
    unsigned Bits = countPopulation(Mask);
    if (getSubtarget()->hasSDWA() && LHS->getOpcode() == ISD::SRL &&
        (Bits == 8 || Bits == 16) && isShiftedMask_64(Mask) && !(Mask & 1)) {
      if (auto *CShift = dyn_cast<ConstantSDNode>(LHS->getOperand(1))) {
        unsigned Shift = CShift->getZExtValue();
        unsigned NB = CRHS->getAPIntValue().countTrailingZeros();
        unsigned Offset = NB + Shift;
        if ((Offset & (Bits - 1)) == 0) { // Starts at a byte or word boundary.
          SDLoc SL(N);
          SDValue BFE = DAG.getNode(AMDGPUISD::BFE_U32, SL, MVT::i32,
                                    LHS->getOperand(0),
                                    DAG.getConstant(Offset, SL, MVT::i32),
                                    DAG.getConstant(Bits, SL, MVT::i32));
          EVT NarrowVT = EVT::getIntegerVT(*DAG.getContext(), Bits);
          SDValue Ext = DAG.getNode(ISD::AssertZext, SL, VT, BFE,
                                    DAG.getValueType(NarrowVT));
          SDValue Shl = DAG.getNode(ISD::SHL, SDLoc(LHS), VT, Ext,
                                    DAG.getConstant(NB, SDLoc(CRHS), MVT::i32));
          return Shl;
        }
      }
    }

    // and (perm x, y, c1), c2 -> perm x, y, permute_mask(c1, c2)
    if (LHS.hasOneUse() && LHS.getOpcode() == AMDGPUISD::PERM &&
        isa<ConstantSDNode>(LHS.getOperand(2))) {
      uint32_t Sel = getConstantPermuteMask(Mask);
      if (!Sel)
        return SDValue();

      // Select 0xc for all zero bytes
      Sel = (LHS.getConstantOperandVal(2) & Sel) | (~Sel & 0x0c0c0c0c);
      SDLoc DL(N);
      return DAG.getNode(AMDGPUISD::PERM, DL, MVT::i32, LHS.getOperand(0),
                         LHS.getOperand(1), DAG.getConstant(Sel, DL, MVT::i32));
    }
  }

  // (and (fcmp ord x, x), (fcmp une (fabs x), inf)) ->
  // fp_class x, ~(s_nan | q_nan | n_infinity | p_infinity)
  if (LHS.getOpcode() == ISD::SETCC && RHS.getOpcode() == ISD::SETCC) {
    ISD::CondCode LCC = cast<CondCodeSDNode>(LHS.getOperand(2))->get();
    ISD::CondCode RCC = cast<CondCodeSDNode>(RHS.getOperand(2))->get();

    SDValue X = LHS.getOperand(0);
    SDValue Y = RHS.getOperand(0);
    if (Y.getOpcode() != ISD::FABS || Y.getOperand(0) != X)
      return SDValue();

    if (LCC == ISD::SETO) {
      if (X != LHS.getOperand(1))
        return SDValue();

      if (RCC == ISD::SETUNE) {
        const ConstantFPSDNode *C1 = dyn_cast<ConstantFPSDNode>(RHS.getOperand(1));
        if (!C1 || !C1->isInfinity() || C1->isNegative())
          return SDValue();

        const uint32_t Mask = SIInstrFlags::N_NORMAL |
                              SIInstrFlags::N_SUBNORMAL |
                              SIInstrFlags::N_ZERO |
                              SIInstrFlags::P_ZERO |
                              SIInstrFlags::P_SUBNORMAL |
                              SIInstrFlags::P_NORMAL;

        static_assert(((~(SIInstrFlags::S_NAN |
                          SIInstrFlags::Q_NAN |
                          SIInstrFlags::N_INFINITY |
                          SIInstrFlags::P_INFINITY)) & 0x3ff) == Mask,
                      "mask not equal");

        SDLoc DL(N);
        return DAG.getNode(AMDGPUISD::FP_CLASS, DL, MVT::i1,
                           X, DAG.getConstant(Mask, DL, MVT::i32));
      }
    }
  }

  if (RHS.getOpcode() == ISD::SETCC && LHS.getOpcode() == AMDGPUISD::FP_CLASS)
    std::swap(LHS, RHS);

  if (LHS.getOpcode() == ISD::SETCC && RHS.getOpcode() == AMDGPUISD::FP_CLASS &&
      RHS.hasOneUse()) {
    ISD::CondCode LCC = cast<CondCodeSDNode>(LHS.getOperand(2))->get();
    // and (fcmp seto), (fp_class x, mask) -> fp_class x, mask & ~(p_nan | n_nan)
    // and (fcmp setuo), (fp_class x, mask) -> fp_class x, mask & (p_nan | n_nan)
    const ConstantSDNode *Mask = dyn_cast<ConstantSDNode>(RHS.getOperand(1));
    if ((LCC == ISD::SETO || LCC == ISD::SETUO) && Mask &&
        (RHS.getOperand(0) == LHS.getOperand(0) &&
         LHS.getOperand(0) == LHS.getOperand(1))) {
      const unsigned OrdMask = SIInstrFlags::S_NAN | SIInstrFlags::Q_NAN;
      unsigned NewMask = LCC == ISD::SETO ?
        Mask->getZExtValue() & ~OrdMask :
        Mask->getZExtValue() & OrdMask;

      SDLoc DL(N);
      return DAG.getNode(AMDGPUISD::FP_CLASS, DL, MVT::i1, RHS.getOperand(0),
                         DAG.getConstant(NewMask, DL, MVT::i32));
    }
  }

  if (VT == MVT::i32 &&
      (RHS.getOpcode() == ISD::SIGN_EXTEND || LHS.getOpcode() == ISD::SIGN_EXTEND)) {
    // and x, (sext cc from i1) => select cc, x, 0
    if (RHS.getOpcode() != ISD::SIGN_EXTEND)
      std::swap(LHS, RHS);
    if (isBoolSGPR(RHS.getOperand(0)))
      return DAG.getSelect(SDLoc(N), MVT::i32, RHS.getOperand(0),
                           LHS, DAG.getConstant(0, SDLoc(N), MVT::i32));
  }

  // and (op x, c1), (op y, c2) -> perm x, y, permute_mask(c1, c2)
  const SIInstrInfo *TII = getSubtarget()->getInstrInfo();
  if (VT == MVT::i32 && LHS.hasOneUse() && RHS.hasOneUse() &&
      N->isDivergent() && TII->pseudoToMCOpcode(AMDGPU::V_PERM_B32_e64) != -1) {
    uint32_t LHSMask = getPermuteMask(DAG, LHS);
    uint32_t RHSMask = getPermuteMask(DAG, RHS);
    if (LHSMask != ~0u && RHSMask != ~0u) {
      // Canonicalize the expression in an attempt to have fewer unique masks
      // and therefore fewer registers used to hold the masks.
      if (LHSMask > RHSMask) {
        std::swap(LHSMask, RHSMask);
        std::swap(LHS, RHS);
      }

      // Select 0xc for each lane used from source operand. Zero has 0xc mask
      // set, 0xff have 0xff in the mask, actual lanes are in the 0-3 range.
      uint32_t LHSUsedLanes = ~(LHSMask & 0x0c0c0c0c) & 0x0c0c0c0c;
      uint32_t RHSUsedLanes = ~(RHSMask & 0x0c0c0c0c) & 0x0c0c0c0c;

      // Check of we need to combine values from two sources within a byte.
      if (!(LHSUsedLanes & RHSUsedLanes) &&
          // If we select high and lower word keep it for SDWA.
          // TODO: teach SDWA to work with v_perm_b32 and remove the check.
          !(LHSUsedLanes == 0x0c0c0000 && RHSUsedLanes == 0x00000c0c)) {
        // Each byte in each mask is either selector mask 0-3, or has higher
        // bits set in either of masks, which can be 0xff for 0xff or 0x0c for
        // zero. If 0x0c is in either mask it shall always be 0x0c. Otherwise
        // mask which is not 0xff wins. By anding both masks we have a correct
        // result except that 0x0c shall be corrected to give 0x0c only.
        uint32_t Mask = LHSMask & RHSMask;
        for (unsigned I = 0; I < 32; I += 8) {
          uint32_t ByteSel = 0xff << I;
          if ((LHSMask & ByteSel) == 0x0c || (RHSMask & ByteSel) == 0x0c)
            Mask &= (0x0c << I) & 0xffffffff;
        }

        // Add 4 to each active LHS lane. It will not affect any existing 0xff
        // or 0x0c.
        uint32_t Sel = Mask | (LHSUsedLanes & 0x04040404);
        SDLoc DL(N);

        return DAG.getNode(AMDGPUISD::PERM, DL, MVT::i32,
                           LHS.getOperand(0), RHS.getOperand(0),
                           DAG.getConstant(Sel, DL, MVT::i32));
      }
    }
  }

  return SDValue();
}

SDValue SITargetLowering::performOrCombine(SDNode *N,
                                           DAGCombinerInfo &DCI) const {
  SelectionDAG &DAG = DCI.DAG;
  SDValue LHS = N->getOperand(0);
  SDValue RHS = N->getOperand(1);

  EVT VT = N->getValueType(0);
  if (VT == MVT::i1) {
    // or (fp_class x, c1), (fp_class x, c2) -> fp_class x, (c1 | c2)
    if (LHS.getOpcode() == AMDGPUISD::FP_CLASS &&
        RHS.getOpcode() == AMDGPUISD::FP_CLASS) {
      SDValue Src = LHS.getOperand(0);
      if (Src != RHS.getOperand(0))
        return SDValue();

      const ConstantSDNode *CLHS = dyn_cast<ConstantSDNode>(LHS.getOperand(1));
      const ConstantSDNode *CRHS = dyn_cast<ConstantSDNode>(RHS.getOperand(1));
      if (!CLHS || !CRHS)
        return SDValue();

      // Only 10 bits are used.
      static const uint32_t MaxMask = 0x3ff;

      uint32_t NewMask = (CLHS->getZExtValue() | CRHS->getZExtValue()) & MaxMask;
      SDLoc DL(N);
      return DAG.getNode(AMDGPUISD::FP_CLASS, DL, MVT::i1,
                         Src, DAG.getConstant(NewMask, DL, MVT::i32));
    }

    return SDValue();
  }

  // or (perm x, y, c1), c2 -> perm x, y, permute_mask(c1, c2)
  if (isa<ConstantSDNode>(RHS) && LHS.hasOneUse() &&
      LHS.getOpcode() == AMDGPUISD::PERM &&
      isa<ConstantSDNode>(LHS.getOperand(2))) {
    uint32_t Sel = getConstantPermuteMask(N->getConstantOperandVal(1));
    if (!Sel)
      return SDValue();

    Sel |= LHS.getConstantOperandVal(2);
    SDLoc DL(N);
    return DAG.getNode(AMDGPUISD::PERM, DL, MVT::i32, LHS.getOperand(0),
                       LHS.getOperand(1), DAG.getConstant(Sel, DL, MVT::i32));
  }

  // or (op x, c1), (op y, c2) -> perm x, y, permute_mask(c1, c2)
  const SIInstrInfo *TII = getSubtarget()->getInstrInfo();
  if (VT == MVT::i32 && LHS.hasOneUse() && RHS.hasOneUse() &&
      N->isDivergent() && TII->pseudoToMCOpcode(AMDGPU::V_PERM_B32_e64) != -1) {
    uint32_t LHSMask = getPermuteMask(DAG, LHS);
    uint32_t RHSMask = getPermuteMask(DAG, RHS);
    if (LHSMask != ~0u && RHSMask != ~0u) {
      // Canonicalize the expression in an attempt to have fewer unique masks
      // and therefore fewer registers used to hold the masks.
      if (LHSMask > RHSMask) {
        std::swap(LHSMask, RHSMask);
        std::swap(LHS, RHS);
      }

      // Select 0xc for each lane used from source operand. Zero has 0xc mask
      // set, 0xff have 0xff in the mask, actual lanes are in the 0-3 range.
      uint32_t LHSUsedLanes = ~(LHSMask & 0x0c0c0c0c) & 0x0c0c0c0c;
      uint32_t RHSUsedLanes = ~(RHSMask & 0x0c0c0c0c) & 0x0c0c0c0c;

      // Check of we need to combine values from two sources within a byte.
      if (!(LHSUsedLanes & RHSUsedLanes) &&
          // If we select high and lower word keep it for SDWA.
          // TODO: teach SDWA to work with v_perm_b32 and remove the check.
          !(LHSUsedLanes == 0x0c0c0000 && RHSUsedLanes == 0x00000c0c)) {
        // Kill zero bytes selected by other mask. Zero value is 0xc.
        LHSMask &= ~RHSUsedLanes;
        RHSMask &= ~LHSUsedLanes;
        // Add 4 to each active LHS lane
        LHSMask |= LHSUsedLanes & 0x04040404;
        // Combine masks
        uint32_t Sel = LHSMask | RHSMask;
        SDLoc DL(N);

        return DAG.getNode(AMDGPUISD::PERM, DL, MVT::i32,
                           LHS.getOperand(0), RHS.getOperand(0),
                           DAG.getConstant(Sel, DL, MVT::i32));
      }
    }
  }

  if (VT != MVT::i64 || DCI.isBeforeLegalizeOps())
    return SDValue();

  // TODO: This could be a generic combine with a predicate for extracting the
  // high half of an integer being free.

  // (or i64:x, (zero_extend i32:y)) ->
  //   i64 (bitcast (v2i32 build_vector (or i32:y, lo_32(x)), hi_32(x)))
  if (LHS.getOpcode() == ISD::ZERO_EXTEND &&
      RHS.getOpcode() != ISD::ZERO_EXTEND)
    std::swap(LHS, RHS);

  if (RHS.getOpcode() == ISD::ZERO_EXTEND) {
    SDValue ExtSrc = RHS.getOperand(0);
    EVT SrcVT = ExtSrc.getValueType();
    if (SrcVT == MVT::i32) {
      SDLoc SL(N);
      SDValue LowLHS, HiBits;
      std::tie(LowLHS, HiBits) = split64BitValue(LHS, DAG);
      SDValue LowOr = DAG.getNode(ISD::OR, SL, MVT::i32, LowLHS, ExtSrc);

      DCI.AddToWorklist(LowOr.getNode());
      DCI.AddToWorklist(HiBits.getNode());

      SDValue Vec = DAG.getNode(ISD::BUILD_VECTOR, SL, MVT::v2i32,
                                LowOr, HiBits);
      return DAG.getNode(ISD::BITCAST, SL, MVT::i64, Vec);
    }
  }

  const ConstantSDNode *CRHS = dyn_cast<ConstantSDNode>(N->getOperand(1));
  if (CRHS) {
    if (SDValue Split
          = splitBinaryBitConstantOp(DCI, SDLoc(N), ISD::OR,
                                     N->getOperand(0), CRHS))
      return Split;
  }

  return SDValue();
}

SDValue SITargetLowering::performXorCombine(SDNode *N,
                                            DAGCombinerInfo &DCI) const {
  if (SDValue RV = reassociateScalarOps(N, DCI.DAG))
    return RV;

  EVT VT = N->getValueType(0);
  if (VT != MVT::i64)
    return SDValue();

  SDValue LHS = N->getOperand(0);
  SDValue RHS = N->getOperand(1);

  const ConstantSDNode *CRHS = dyn_cast<ConstantSDNode>(RHS);
  if (CRHS) {
    if (SDValue Split
          = splitBinaryBitConstantOp(DCI, SDLoc(N), ISD::XOR, LHS, CRHS))
      return Split;
  }

  return SDValue();
}

SDValue SITargetLowering::performZeroExtendCombine(SDNode *N,
                                                   DAGCombinerInfo &DCI) const {
  if (!Subtarget->has16BitInsts() ||
      DCI.getDAGCombineLevel() < AfterLegalizeDAG)
    return SDValue();

  EVT VT = N->getValueType(0);
  if (VT != MVT::i32)
    return SDValue();

  SDValue Src = N->getOperand(0);
  if (Src.getValueType() != MVT::i16)
    return SDValue();

  return SDValue();
}

SDValue SITargetLowering::performSignExtendInRegCombine(SDNode *N,
                                                        DAGCombinerInfo &DCI)
                                                        const {
  SDValue Src = N->getOperand(0);
  auto *VTSign = cast<VTSDNode>(N->getOperand(1));

  if (((Src.getOpcode() == AMDGPUISD::BUFFER_LOAD_UBYTE &&
      VTSign->getVT() == MVT::i8) ||
      (Src.getOpcode() == AMDGPUISD::BUFFER_LOAD_USHORT &&
      VTSign->getVT() == MVT::i16)) &&
      Src.hasOneUse()) {
    auto *M = cast<MemSDNode>(Src);
    SDValue Ops[] = {
      Src.getOperand(0), // Chain
      Src.getOperand(1), // rsrc
      Src.getOperand(2), // vindex
      Src.getOperand(3), // voffset
      Src.getOperand(4), // soffset
      Src.getOperand(5), // offset
      Src.getOperand(6),
      Src.getOperand(7)
    };
    // replace with BUFFER_LOAD_BYTE/SHORT
    SDVTList ResList = DCI.DAG.getVTList(MVT::i32,
                                         Src.getOperand(0).getValueType());
    unsigned Opc = (Src.getOpcode() == AMDGPUISD::BUFFER_LOAD_UBYTE) ?
                   AMDGPUISD::BUFFER_LOAD_BYTE : AMDGPUISD::BUFFER_LOAD_SHORT;
    SDValue BufferLoadSignExt = DCI.DAG.getMemIntrinsicNode(Opc, SDLoc(N),
                                                          ResList,
                                                          Ops, M->getMemoryVT(),
                                                          M->getMemOperand());
    return DCI.DAG.getMergeValues({BufferLoadSignExt,
                                  BufferLoadSignExt.getValue(1)}, SDLoc(N));
  }
  return SDValue();
}

SDValue SITargetLowering::performClassCombine(SDNode *N,
                                              DAGCombinerInfo &DCI) const {
  SelectionDAG &DAG = DCI.DAG;
  SDValue Mask = N->getOperand(1);

  // fp_class x, 0 -> false
  if (const ConstantSDNode *CMask = dyn_cast<ConstantSDNode>(Mask)) {
    if (CMask->isZero())
      return DAG.getConstant(0, SDLoc(N), MVT::i1);
  }

  if (N->getOperand(0).isUndef())
    return DAG.getUNDEF(MVT::i1);

  return SDValue();
}

SDValue SITargetLowering::performRcpCombine(SDNode *N,
                                            DAGCombinerInfo &DCI) const {
  EVT VT = N->getValueType(0);
  SDValue N0 = N->getOperand(0);

  if (N0.isUndef())
    return N0;

  if (VT == MVT::f32 && (N0.getOpcode() == ISD::UINT_TO_FP ||
                         N0.getOpcode() == ISD::SINT_TO_FP)) {
    return DCI.DAG.getNode(AMDGPUISD::RCP_IFLAG, SDLoc(N), VT, N0,
                           N->getFlags());
  }

  if ((VT == MVT::f32 || VT == MVT::f16) && N0.getOpcode() == ISD::FSQRT) {
    return DCI.DAG.getNode(AMDGPUISD::RSQ, SDLoc(N), VT,
                           N0.getOperand(0), N->getFlags());
  }

  return AMDGPUTargetLowering::performRcpCombine(N, DCI);
}

bool SITargetLowering::isCanonicalized(SelectionDAG &DAG, SDValue Op,
                                       unsigned MaxDepth) const {
  unsigned Opcode = Op.getOpcode();
  if (Opcode == ISD::FCANONICALIZE)
    return true;

  if (auto *CFP = dyn_cast<ConstantFPSDNode>(Op)) {
    auto F = CFP->getValueAPF();
    if (F.isNaN() && F.isSignaling())
      return false;
    return !F.isDenormal() || denormalsEnabledForType(DAG, Op.getValueType());
  }

  // If source is a result of another standard FP operation it is already in
  // canonical form.
  if (MaxDepth == 0)
    return false;

  switch (Opcode) {
  // These will flush denorms if required.
  case ISD::FADD:
  case ISD::FSUB:
  case ISD::FMUL:
  case ISD::FCEIL:
  case ISD::FFLOOR:
  case ISD::FMA:
  case ISD::FMAD:
  case ISD::FSQRT:
  case ISD::FDIV:
  case ISD::FREM:
  case ISD::FP_ROUND:
  case ISD::FP_EXTEND:
  case AMDGPUISD::FMUL_LEGACY:
  case AMDGPUISD::FMAD_FTZ:
  case AMDGPUISD::RCP:
  case AMDGPUISD::RSQ:
  case AMDGPUISD::RSQ_CLAMP:
  case AMDGPUISD::RCP_LEGACY:
  case AMDGPUISD::RCP_IFLAG:
  case AMDGPUISD::DIV_SCALE:
  case AMDGPUISD::DIV_FMAS:
  case AMDGPUISD::DIV_FIXUP:
  case AMDGPUISD::FRACT:
  case AMDGPUISD::LDEXP:
  case AMDGPUISD::CVT_PKRTZ_F16_F32:
  case AMDGPUISD::CVT_F32_UBYTE0:
  case AMDGPUISD::CVT_F32_UBYTE1:
  case AMDGPUISD::CVT_F32_UBYTE2:
  case AMDGPUISD::CVT_F32_UBYTE3:
    return true;

  // It can/will be lowered or combined as a bit operation.
  // Need to check their input recursively to handle.
  case ISD::FNEG:
  case ISD::FABS:
  case ISD::FCOPYSIGN:
    return isCanonicalized(DAG, Op.getOperand(0), MaxDepth - 1);

  case ISD::FSIN:
  case ISD::FCOS:
  case ISD::FSINCOS:
    return Op.getValueType().getScalarType() != MVT::f16;

  case ISD::FMINNUM:
  case ISD::FMAXNUM:
  case ISD::FMINNUM_IEEE:
  case ISD::FMAXNUM_IEEE:
  case AMDGPUISD::CLAMP:
  case AMDGPUISD::FMED3:
  case AMDGPUISD::FMAX3:
  case AMDGPUISD::FMIN3: {
    // FIXME: Shouldn't treat the generic operations different based these.
    // However, we aren't really required to flush the result from
    // minnum/maxnum..

    // snans will be quieted, so we only need to worry about denormals.
    if (Subtarget->supportsMinMaxDenormModes() ||
        denormalsEnabledForType(DAG, Op.getValueType()))
      return true;

    // Flushing may be required.
    // In pre-GFX9 targets V_MIN_F32 and others do not flush denorms. For such
    // targets need to check their input recursively.

    // FIXME: Does this apply with clamp? It's implemented with max.
    for (unsigned I = 0, E = Op.getNumOperands(); I != E; ++I) {
      if (!isCanonicalized(DAG, Op.getOperand(I), MaxDepth - 1))
        return false;
    }

    return true;
  }
  case ISD::SELECT: {
    return isCanonicalized(DAG, Op.getOperand(1), MaxDepth - 1) &&
           isCanonicalized(DAG, Op.getOperand(2), MaxDepth - 1);
  }
  case ISD::BUILD_VECTOR: {
    for (unsigned i = 0, e = Op.getNumOperands(); i != e; ++i) {
      SDValue SrcOp = Op.getOperand(i);
      if (!isCanonicalized(DAG, SrcOp, MaxDepth - 1))
        return false;
    }

    return true;
  }
  case ISD::EXTRACT_VECTOR_ELT:
  case ISD::EXTRACT_SUBVECTOR: {
    return isCanonicalized(DAG, Op.getOperand(0), MaxDepth - 1);
  }
  case ISD::INSERT_VECTOR_ELT: {
    return isCanonicalized(DAG, Op.getOperand(0), MaxDepth - 1) &&
           isCanonicalized(DAG, Op.getOperand(1), MaxDepth - 1);
  }
  case ISD::UNDEF:
    // Could be anything.
    return false;

  case ISD::BITCAST:
    return isCanonicalized(DAG, Op.getOperand(0), MaxDepth - 1);
  case ISD::TRUNCATE: {
    // Hack round the mess we make when legalizing extract_vector_elt
    if (Op.getValueType() == MVT::i16) {
      SDValue TruncSrc = Op.getOperand(0);
      if (TruncSrc.getValueType() == MVT::i32 &&
          TruncSrc.getOpcode() == ISD::BITCAST &&
          TruncSrc.getOperand(0).getValueType() == MVT::v2f16) {
        return isCanonicalized(DAG, TruncSrc.getOperand(0), MaxDepth - 1);
      }
    }
    return false;
  }
  case ISD::INTRINSIC_WO_CHAIN: {
    unsigned IntrinsicID
      = cast<ConstantSDNode>(Op.getOperand(0))->getZExtValue();
    // TODO: Handle more intrinsics
    switch (IntrinsicID) {
    case Intrinsic::amdgcn_cvt_pkrtz:
    case Intrinsic::amdgcn_cubeid:
    case Intrinsic::amdgcn_frexp_mant:
    case Intrinsic::amdgcn_fdot2:
    case Intrinsic::amdgcn_rcp:
    case Intrinsic::amdgcn_rsq:
    case Intrinsic::amdgcn_rsq_clamp:
    case Intrinsic::amdgcn_rcp_legacy:
    case Intrinsic::amdgcn_rsq_legacy:
    case Intrinsic::amdgcn_trig_preop:
      return true;
    default:
      break;
    }

    LLVM_FALLTHROUGH;
  }
  default:
    return denormalsEnabledForType(DAG, Op.getValueType()) &&
           DAG.isKnownNeverSNaN(Op);
  }

  llvm_unreachable("invalid operation");
}

bool SITargetLowering::isCanonicalized(Register Reg, MachineFunction &MF,
                                       unsigned MaxDepth) const {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  MachineInstr *MI = MRI.getVRegDef(Reg);
  unsigned Opcode = MI->getOpcode();

  if (Opcode == AMDGPU::G_FCANONICALIZE)
    return true;

  Optional<FPValueAndVReg> FCR;
  // Constant splat (can be padded with undef) or scalar constant.
  if (mi_match(Reg, MRI, MIPatternMatch::m_GFCstOrSplat(FCR))) {
    if (FCR->Value.isSignaling())
      return false;
    return !FCR->Value.isDenormal() ||
           denormalsEnabledForType(MRI.getType(FCR->VReg), MF);
  }

  if (MaxDepth == 0)
    return false;

  switch (Opcode) {
  case AMDGPU::G_FMINNUM_IEEE:
  case AMDGPU::G_FMAXNUM_IEEE: {
    if (Subtarget->supportsMinMaxDenormModes() ||
        denormalsEnabledForType(MRI.getType(Reg), MF))
      return true;
    for (const MachineOperand &MO : llvm::drop_begin(MI->operands()))
      if (!isCanonicalized(MO.getReg(), MF, MaxDepth - 1))
        return false;
    return true;
  }
  default:
    return denormalsEnabledForType(MRI.getType(Reg), MF) &&
           isKnownNeverSNaN(Reg, MRI);
  }

  llvm_unreachable("invalid operation");
}

// Constant fold canonicalize.
SDValue SITargetLowering::getCanonicalConstantFP(
  SelectionDAG &DAG, const SDLoc &SL, EVT VT, const APFloat &C) const {
  // Flush denormals to 0 if not enabled.
  if (C.isDenormal() && !denormalsEnabledForType(DAG, VT))
    return DAG.getConstantFP(0.0, SL, VT);

  if (C.isNaN()) {
    APFloat CanonicalQNaN = APFloat::getQNaN(C.getSemantics());
    if (C.isSignaling()) {
      // Quiet a signaling NaN.
      // FIXME: Is this supposed to preserve payload bits?
      return DAG.getConstantFP(CanonicalQNaN, SL, VT);
    }

    // Make sure it is the canonical NaN bitpattern.
    //
    // TODO: Can we use -1 as the canonical NaN value since it's an inline
    // immediate?
    if (C.bitcastToAPInt() != CanonicalQNaN.bitcastToAPInt())
      return DAG.getConstantFP(CanonicalQNaN, SL, VT);
  }

  // Already canonical.
  return DAG.getConstantFP(C, SL, VT);
}

static bool vectorEltWillFoldAway(SDValue Op) {
  return Op.isUndef() || isa<ConstantFPSDNode>(Op);
}

SDValue SITargetLowering::performFCanonicalizeCombine(
  SDNode *N,
  DAGCombinerInfo &DCI) const {
  SelectionDAG &DAG = DCI.DAG;
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);

  // fcanonicalize undef -> qnan
  if (N0.isUndef()) {
    APFloat QNaN = APFloat::getQNaN(SelectionDAG::EVTToAPFloatSemantics(VT));
    return DAG.getConstantFP(QNaN, SDLoc(N), VT);
  }

  if (ConstantFPSDNode *CFP = isConstOrConstSplatFP(N0)) {
    EVT VT = N->getValueType(0);
    return getCanonicalConstantFP(DAG, SDLoc(N), VT, CFP->getValueAPF());
  }

  // fcanonicalize (build_vector x, k) -> build_vector (fcanonicalize x),
  //                                                   (fcanonicalize k)
  //
  // fcanonicalize (build_vector x, undef) -> build_vector (fcanonicalize x), 0

  // TODO: This could be better with wider vectors that will be split to v2f16,
  // and to consider uses since there aren't that many packed operations.
  if (N0.getOpcode() == ISD::BUILD_VECTOR && VT == MVT::v2f16 &&
      isTypeLegal(MVT::v2f16)) {
    SDLoc SL(N);
    SDValue NewElts[2];
    SDValue Lo = N0.getOperand(0);
    SDValue Hi = N0.getOperand(1);
    EVT EltVT = Lo.getValueType();

    if (vectorEltWillFoldAway(Lo) || vectorEltWillFoldAway(Hi)) {
      for (unsigned I = 0; I != 2; ++I) {
        SDValue Op = N0.getOperand(I);
        if (ConstantFPSDNode *CFP = dyn_cast<ConstantFPSDNode>(Op)) {
          NewElts[I] = getCanonicalConstantFP(DAG, SL, EltVT,
                                              CFP->getValueAPF());
        } else if (Op.isUndef()) {
          // Handled below based on what the other operand is.
          NewElts[I] = Op;
        } else {
          NewElts[I] = DAG.getNode(ISD::FCANONICALIZE, SL, EltVT, Op);
        }
      }

      // If one half is undef, and one is constant, perfer a splat vector rather
      // than the normal qNaN. If it's a register, prefer 0.0 since that's
      // cheaper to use and may be free with a packed operation.
      if (NewElts[0].isUndef()) {
        if (isa<ConstantFPSDNode>(NewElts[1]))
          NewElts[0] = isa<ConstantFPSDNode>(NewElts[1]) ?
            NewElts[1]: DAG.getConstantFP(0.0f, SL, EltVT);
      }

      if (NewElts[1].isUndef()) {
        NewElts[1] = isa<ConstantFPSDNode>(NewElts[0]) ?
          NewElts[0] : DAG.getConstantFP(0.0f, SL, EltVT);
      }

      return DAG.getBuildVector(VT, SL, NewElts);
    }
  }

  unsigned SrcOpc = N0.getOpcode();

  // If it's free to do so, push canonicalizes further up the source, which may
  // find a canonical source.
  //
  // TODO: More opcodes. Note this is unsafe for the the _ieee minnum/maxnum for
  // sNaNs.
  if (SrcOpc == ISD::FMINNUM || SrcOpc == ISD::FMAXNUM) {
    auto *CRHS = dyn_cast<ConstantFPSDNode>(N0.getOperand(1));
    if (CRHS && N0.hasOneUse()) {
      SDLoc SL(N);
      SDValue Canon0 = DAG.getNode(ISD::FCANONICALIZE, SL, VT,
                                   N0.getOperand(0));
      SDValue Canon1 = getCanonicalConstantFP(DAG, SL, VT, CRHS->getValueAPF());
      DCI.AddToWorklist(Canon0.getNode());

      return DAG.getNode(N0.getOpcode(), SL, VT, Canon0, Canon1);
    }
  }

  return isCanonicalized(DAG, N0) ? N0 : SDValue();
}

static unsigned minMaxOpcToMin3Max3Opc(unsigned Opc) {
  switch (Opc) {
  case ISD::FMAXNUM:
  case ISD::FMAXNUM_IEEE:
    return AMDGPUISD::FMAX3;
  case ISD::SMAX:
    return AMDGPUISD::SMAX3;
  case ISD::UMAX:
    return AMDGPUISD::UMAX3;
  case ISD::FMINNUM:
  case ISD::FMINNUM_IEEE:
    return AMDGPUISD::FMIN3;
  case ISD::SMIN:
    return AMDGPUISD::SMIN3;
  case ISD::UMIN:
    return AMDGPUISD::UMIN3;
  default:
    llvm_unreachable("Not a min/max opcode");
  }
}

SDValue SITargetLowering::performIntMed3ImmCombine(
  SelectionDAG &DAG, const SDLoc &SL,
  SDValue Op0, SDValue Op1, bool Signed) const {
  ConstantSDNode *K1 = dyn_cast<ConstantSDNode>(Op1);
  if (!K1)
    return SDValue();

  ConstantSDNode *K0 = dyn_cast<ConstantSDNode>(Op0.getOperand(1));
  if (!K0)
    return SDValue();

  if (Signed) {
    if (K0->getAPIntValue().sge(K1->getAPIntValue()))
      return SDValue();
  } else {
    if (K0->getAPIntValue().uge(K1->getAPIntValue()))
      return SDValue();
  }

  EVT VT = K0->getValueType(0);
  unsigned Med3Opc = Signed ? AMDGPUISD::SMED3 : AMDGPUISD::UMED3;
  if (VT == MVT::i32 || (VT == MVT::i16 && Subtarget->hasMed3_16())) {
    return DAG.getNode(Med3Opc, SL, VT,
                       Op0.getOperand(0), SDValue(K0, 0), SDValue(K1, 0));
  }

  // If there isn't a 16-bit med3 operation, convert to 32-bit.
  if (VT == MVT::i16) {
    MVT NVT = MVT::i32;
    unsigned ExtOp = Signed ? ISD::SIGN_EXTEND : ISD::ZERO_EXTEND;

    SDValue Tmp1 = DAG.getNode(ExtOp, SL, NVT, Op0->getOperand(0));
    SDValue Tmp2 = DAG.getNode(ExtOp, SL, NVT, Op0->getOperand(1));
    SDValue Tmp3 = DAG.getNode(ExtOp, SL, NVT, Op1);

    SDValue Med3 = DAG.getNode(Med3Opc, SL, NVT, Tmp1, Tmp2, Tmp3);
    return DAG.getNode(ISD::TRUNCATE, SL, VT, Med3);
  }

  return SDValue();
}

static ConstantFPSDNode *getSplatConstantFP(SDValue Op) {
  if (ConstantFPSDNode *C = dyn_cast<ConstantFPSDNode>(Op))
    return C;

  if (BuildVectorSDNode *BV = dyn_cast<BuildVectorSDNode>(Op)) {
    if (ConstantFPSDNode *C = BV->getConstantFPSplatNode())
      return C;
  }

  return nullptr;
}

SDValue SITargetLowering::performFPMed3ImmCombine(SelectionDAG &DAG,
                                                  const SDLoc &SL,
                                                  SDValue Op0,
                                                  SDValue Op1) const {
  ConstantFPSDNode *K1 = getSplatConstantFP(Op1);
  if (!K1)
    return SDValue();

  ConstantFPSDNode *K0 = getSplatConstantFP(Op0.getOperand(1));
  if (!K0)
    return SDValue();

  // Ordered >= (although NaN inputs should have folded away by now).
  if (K0->getValueAPF() > K1->getValueAPF())
    return SDValue();

  const MachineFunction &MF = DAG.getMachineFunction();
  const SIMachineFunctionInfo *Info = MF.getInfo<SIMachineFunctionInfo>();

  // TODO: Check IEEE bit enabled?
  EVT VT = Op0.getValueType();
  if (Info->getMode().DX10Clamp) {
    // If dx10_clamp is enabled, NaNs clamp to 0.0. This is the same as the
    // hardware fmed3 behavior converting to a min.
    // FIXME: Should this be allowing -0.0?
    if (K1->isExactlyValue(1.0) && K0->isExactlyValue(0.0))
      return DAG.getNode(AMDGPUISD::CLAMP, SL, VT, Op0.getOperand(0));
  }

  // med3 for f16 is only available on gfx9+, and not available for v2f16.
  if (VT == MVT::f32 || (VT == MVT::f16 && Subtarget->hasMed3_16())) {
    // This isn't safe with signaling NaNs because in IEEE mode, min/max on a
    // signaling NaN gives a quiet NaN. The quiet NaN input to the min would
    // then give the other result, which is different from med3 with a NaN
    // input.
    SDValue Var = Op0.getOperand(0);
    if (!DAG.isKnownNeverSNaN(Var))
      return SDValue();

    const SIInstrInfo *TII = getSubtarget()->getInstrInfo();

    if ((!K0->hasOneUse() ||
         TII->isInlineConstant(K0->getValueAPF().bitcastToAPInt())) &&
        (!K1->hasOneUse() ||
         TII->isInlineConstant(K1->getValueAPF().bitcastToAPInt()))) {
      return DAG.getNode(AMDGPUISD::FMED3, SL, K0->getValueType(0),
                         Var, SDValue(K0, 0), SDValue(K1, 0));
    }
  }

  return SDValue();
}

SDValue SITargetLowering::performMinMaxCombine(SDNode *N,
                                               DAGCombinerInfo &DCI) const {
  SelectionDAG &DAG = DCI.DAG;

  EVT VT = N->getValueType(0);
  unsigned Opc = N->getOpcode();
  SDValue Op0 = N->getOperand(0);
  SDValue Op1 = N->getOperand(1);

  // Only do this if the inner op has one use since this will just increases
  // register pressure for no benefit.

  if (Opc != AMDGPUISD::FMIN_LEGACY && Opc != AMDGPUISD::FMAX_LEGACY &&
      !VT.isVector() &&
      (VT == MVT::i32 || VT == MVT::f32 ||
       ((VT == MVT::f16 || VT == MVT::i16) && Subtarget->hasMin3Max3_16()))) {
    // max(max(a, b), c) -> max3(a, b, c)
    // min(min(a, b), c) -> min3(a, b, c)
    if (Op0.getOpcode() == Opc && Op0.hasOneUse()) {
      SDLoc DL(N);
      return DAG.getNode(minMaxOpcToMin3Max3Opc(Opc),
                         DL,
                         N->getValueType(0),
                         Op0.getOperand(0),
                         Op0.getOperand(1),
                         Op1);
    }

    // Try commuted.
    // max(a, max(b, c)) -> max3(a, b, c)
    // min(a, min(b, c)) -> min3(a, b, c)
    if (Op1.getOpcode() == Opc && Op1.hasOneUse()) {
      SDLoc DL(N);
      return DAG.getNode(minMaxOpcToMin3Max3Opc(Opc),
                         DL,
                         N->getValueType(0),
                         Op0,
                         Op1.getOperand(0),
                         Op1.getOperand(1));
    }
  }

  // min(max(x, K0), K1), K0 < K1 -> med3(x, K0, K1)
  if (Opc == ISD::SMIN && Op0.getOpcode() == ISD::SMAX && Op0.hasOneUse()) {
    if (SDValue Med3 = performIntMed3ImmCombine(DAG, SDLoc(N), Op0, Op1, true))
      return Med3;
  }

  if (Opc == ISD::UMIN && Op0.getOpcode() == ISD::UMAX && Op0.hasOneUse()) {
    if (SDValue Med3 = performIntMed3ImmCombine(DAG, SDLoc(N), Op0, Op1, false))
      return Med3;
  }

  // fminnum(fmaxnum(x, K0), K1), K0 < K1 && !is_snan(x) -> fmed3(x, K0, K1)
  if (((Opc == ISD::FMINNUM && Op0.getOpcode() == ISD::FMAXNUM) ||
       (Opc == ISD::FMINNUM_IEEE && Op0.getOpcode() == ISD::FMAXNUM_IEEE) ||
       (Opc == AMDGPUISD::FMIN_LEGACY &&
        Op0.getOpcode() == AMDGPUISD::FMAX_LEGACY)) &&
      (VT == MVT::f32 || VT == MVT::f64 ||
       (VT == MVT::f16 && Subtarget->has16BitInsts()) ||
       (VT == MVT::v2f16 && Subtarget->hasVOP3PInsts())) &&
      Op0.hasOneUse()) {
    if (SDValue Res = performFPMed3ImmCombine(DAG, SDLoc(N), Op0, Op1))
      return Res;
  }

  return SDValue();
}

static bool isClampZeroToOne(SDValue A, SDValue B) {
  if (ConstantFPSDNode *CA = dyn_cast<ConstantFPSDNode>(A)) {
    if (ConstantFPSDNode *CB = dyn_cast<ConstantFPSDNode>(B)) {
      // FIXME: Should this be allowing -0.0?
      return (CA->isExactlyValue(0.0) && CB->isExactlyValue(1.0)) ||
             (CA->isExactlyValue(1.0) && CB->isExactlyValue(0.0));
    }
  }

  return false;
}

// FIXME: Should only worry about snans for version with chain.
SDValue SITargetLowering::performFMed3Combine(SDNode *N,
                                              DAGCombinerInfo &DCI) const {
  EVT VT = N->getValueType(0);
  // v_med3_f32 and v_max_f32 behave identically wrt denorms, exceptions and
  // NaNs. With a NaN input, the order of the operands may change the result.

  SelectionDAG &DAG = DCI.DAG;
  SDLoc SL(N);

  SDValue Src0 = N->getOperand(0);
  SDValue Src1 = N->getOperand(1);
  SDValue Src2 = N->getOperand(2);

  if (isClampZeroToOne(Src0, Src1)) {
    // const_a, const_b, x -> clamp is safe in all cases including signaling
    // nans.
    // FIXME: Should this be allowing -0.0?
    return DAG.getNode(AMDGPUISD::CLAMP, SL, VT, Src2);
  }

  const MachineFunction &MF = DAG.getMachineFunction();
  const SIMachineFunctionInfo *Info = MF.getInfo<SIMachineFunctionInfo>();

  // FIXME: dx10_clamp behavior assumed in instcombine. Should we really bother
  // handling no dx10-clamp?
  if (Info->getMode().DX10Clamp) {
    // If NaNs is clamped to 0, we are free to reorder the inputs.

    if (isa<ConstantFPSDNode>(Src0) && !isa<ConstantFPSDNode>(Src1))
      std::swap(Src0, Src1);

    if (isa<ConstantFPSDNode>(Src1) && !isa<ConstantFPSDNode>(Src2))
      std::swap(Src1, Src2);

    if (isa<ConstantFPSDNode>(Src0) && !isa<ConstantFPSDNode>(Src1))
      std::swap(Src0, Src1);

    if (isClampZeroToOne(Src1, Src2))
      return DAG.getNode(AMDGPUISD::CLAMP, SL, VT, Src0);
  }

  return SDValue();
}

SDValue SITargetLowering::performCvtPkRTZCombine(SDNode *N,
                                                 DAGCombinerInfo &DCI) const {
  SDValue Src0 = N->getOperand(0);
  SDValue Src1 = N->getOperand(1);
  if (Src0.isUndef() && Src1.isUndef())
    return DCI.DAG.getUNDEF(N->getValueType(0));
  return SDValue();
}

// Check if EXTRACT_VECTOR_ELT/INSERT_VECTOR_ELT (<n x e>, var-idx) should be
// expanded into a set of cmp/select instructions.
bool SITargetLowering::shouldExpandVectorDynExt(unsigned EltSize,
                                                unsigned NumElem,
                                                bool IsDivergentIdx) {
  if (UseDivergentRegisterIndexing)
    return false;

  unsigned VecSize = EltSize * NumElem;

  // Sub-dword vectors of size 2 dword or less have better implementation.
  if (VecSize <= 64 && EltSize < 32)
    return false;

  // Always expand the rest of sub-dword instructions, otherwise it will be
  // lowered via memory.
  if (EltSize < 32)
    return true;

  // Always do this if var-idx is divergent, otherwise it will become a loop.
  if (IsDivergentIdx)
    return true;

  // Large vectors would yield too many compares and v_cndmask_b32 instructions.
  unsigned NumInsts = NumElem /* Number of compares */ +
                      ((EltSize + 31) / 32) * NumElem /* Number of cndmasks */;
  return NumInsts <= 16;
}

static bool shouldExpandVectorDynExt(SDNode *N) {
  SDValue Idx = N->getOperand(N->getNumOperands() - 1);
  if (isa<ConstantSDNode>(Idx))
    return false;

  SDValue Vec = N->getOperand(0);
  EVT VecVT = Vec.getValueType();
  EVT EltVT = VecVT.getVectorElementType();
  unsigned EltSize = EltVT.getSizeInBits();
  unsigned NumElem = VecVT.getVectorNumElements();

  return SITargetLowering::shouldExpandVectorDynExt(EltSize, NumElem,
                                                    Idx->isDivergent());
}

SDValue SITargetLowering::performExtractVectorEltCombine(
  SDNode *N, DAGCombinerInfo &DCI) const {
  SDValue Vec = N->getOperand(0);
  SelectionDAG &DAG = DCI.DAG;

  EVT VecVT = Vec.getValueType();
  EVT EltVT = VecVT.getVectorElementType();

  if ((Vec.getOpcode() == ISD::FNEG ||
       Vec.getOpcode() == ISD::FABS) && allUsesHaveSourceMods(N)) {
    SDLoc SL(N);
    EVT EltVT = N->getValueType(0);
    SDValue Idx = N->getOperand(1);
    SDValue Elt = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, SL, EltVT,
                              Vec.getOperand(0), Idx);
    return DAG.getNode(Vec.getOpcode(), SL, EltVT, Elt);
  }

  // ScalarRes = EXTRACT_VECTOR_ELT ((vector-BINOP Vec1, Vec2), Idx)
  //    =>
  // Vec1Elt = EXTRACT_VECTOR_ELT(Vec1, Idx)
  // Vec2Elt = EXTRACT_VECTOR_ELT(Vec2, Idx)
  // ScalarRes = scalar-BINOP Vec1Elt, Vec2Elt
  if (Vec.hasOneUse() && DCI.isBeforeLegalize()) {
    SDLoc SL(N);
    EVT EltVT = N->getValueType(0);
    SDValue Idx = N->getOperand(1);
    unsigned Opc = Vec.getOpcode();

    switch(Opc) {
    default:
      break;
      // TODO: Support other binary operations.
    case ISD::FADD:
    case ISD::FSUB:
    case ISD::FMUL:
    case ISD::ADD:
    case ISD::UMIN:
    case ISD::UMAX:
    case ISD::SMIN:
    case ISD::SMAX:
    case ISD::FMAXNUM:
    case ISD::FMINNUM:
    case ISD::FMAXNUM_IEEE:
    case ISD::FMINNUM_IEEE: {
      SDValue Elt0 = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, SL, EltVT,
                                 Vec.getOperand(0), Idx);
      SDValue Elt1 = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, SL, EltVT,
                                 Vec.getOperand(1), Idx);

      DCI.AddToWorklist(Elt0.getNode());
      DCI.AddToWorklist(Elt1.getNode());
      return DAG.getNode(Opc, SL, EltVT, Elt0, Elt1, Vec->getFlags());
    }
    }
  }

  unsigned VecSize = VecVT.getSizeInBits();
  unsigned EltSize = EltVT.getSizeInBits();

  // EXTRACT_VECTOR_ELT (<n x e>, var-idx) => n x select (e, const-idx)
  if (::shouldExpandVectorDynExt(N)) {
    SDLoc SL(N);
    SDValue Idx = N->getOperand(1);
    SDValue V;
    for (unsigned I = 0, E = VecVT.getVectorNumElements(); I < E; ++I) {
      SDValue IC = DAG.getVectorIdxConstant(I, SL);
      SDValue Elt = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, SL, EltVT, Vec, IC);
      if (I == 0)
        V = Elt;
      else
        V = DAG.getSelectCC(SL, Idx, IC, Elt, V, ISD::SETEQ);
    }
    return V;
  }

  if (!DCI.isBeforeLegalize())
    return SDValue();

  // Try to turn sub-dword accesses of vectors into accesses of the same 32-bit
  // elements. This exposes more load reduction opportunities by replacing
  // multiple small extract_vector_elements with a single 32-bit extract.
  auto *Idx = dyn_cast<ConstantSDNode>(N->getOperand(1));
  if (isa<MemSDNode>(Vec) &&
      EltSize <= 16 &&
      EltVT.isByteSized() &&
      VecSize > 32 &&
      VecSize % 32 == 0 &&
      Idx) {
    EVT NewVT = getEquivalentMemType(*DAG.getContext(), VecVT);

    unsigned BitIndex = Idx->getZExtValue() * EltSize;
    unsigned EltIdx = BitIndex / 32;
    unsigned LeftoverBitIdx = BitIndex % 32;
    SDLoc SL(N);

    SDValue Cast = DAG.getNode(ISD::BITCAST, SL, NewVT, Vec);
    DCI.AddToWorklist(Cast.getNode());

    SDValue Elt = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, SL, MVT::i32, Cast,
                              DAG.getConstant(EltIdx, SL, MVT::i32));
    DCI.AddToWorklist(Elt.getNode());
    SDValue Srl = DAG.getNode(ISD::SRL, SL, MVT::i32, Elt,
                              DAG.getConstant(LeftoverBitIdx, SL, MVT::i32));
    DCI.AddToWorklist(Srl.getNode());

    SDValue Trunc = DAG.getNode(ISD::TRUNCATE, SL, EltVT.changeTypeToInteger(), Srl);
    DCI.AddToWorklist(Trunc.getNode());
    return DAG.getNode(ISD::BITCAST, SL, EltVT, Trunc);
  }

  return SDValue();
}

SDValue
SITargetLowering::performInsertVectorEltCombine(SDNode *N,
                                                DAGCombinerInfo &DCI) const {
  SDValue Vec = N->getOperand(0);
  SDValue Idx = N->getOperand(2);
  EVT VecVT = Vec.getValueType();
  EVT EltVT = VecVT.getVectorElementType();

  // INSERT_VECTOR_ELT (<n x e>, var-idx)
  // => BUILD_VECTOR n x select (e, const-idx)
  if (!::shouldExpandVectorDynExt(N))
    return SDValue();

  SelectionDAG &DAG = DCI.DAG;
  SDLoc SL(N);
  SDValue Ins = N->getOperand(1);
  EVT IdxVT = Idx.getValueType();

  SmallVector<SDValue, 16> Ops;
  for (unsigned I = 0, E = VecVT.getVectorNumElements(); I < E; ++I) {
    SDValue IC = DAG.getConstant(I, SL, IdxVT);
    SDValue Elt = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, SL, EltVT, Vec, IC);
    SDValue V = DAG.getSelectCC(SL, Idx, IC, Ins, Elt, ISD::SETEQ);
    Ops.push_back(V);
  }

  return DAG.getBuildVector(VecVT, SL, Ops);
}

unsigned SITargetLowering::getFusedOpcode(const SelectionDAG &DAG,
                                          const SDNode *N0,
                                          const SDNode *N1) const {
  EVT VT = N0->getValueType(0);

  // Only do this if we are not trying to support denormals. v_mad_f32 does not
  // support denormals ever.
  if (((VT == MVT::f32 && !hasFP32Denormals(DAG.getMachineFunction())) ||
       (VT == MVT::f16 && !hasFP64FP16Denormals(DAG.getMachineFunction()) &&
        getSubtarget()->hasMadF16())) &&
       isOperationLegal(ISD::FMAD, VT))
    return ISD::FMAD;

  const TargetOptions &Options = DAG.getTarget().Options;
  if ((Options.AllowFPOpFusion == FPOpFusion::Fast || Options.UnsafeFPMath ||
       (N0->getFlags().hasAllowContract() &&
        N1->getFlags().hasAllowContract())) &&
      isFMAFasterThanFMulAndFAdd(DAG.getMachineFunction(), VT)) {
    return ISD::FMA;
  }

  return 0;
}

// For a reassociatable opcode perform:
// op x, (op y, z) -> op (op x, z), y, if x and z are uniform
SDValue SITargetLowering::reassociateScalarOps(SDNode *N,
                                               SelectionDAG &DAG) const {
  EVT VT = N->getValueType(0);
  if (VT != MVT::i32 && VT != MVT::i64)
    return SDValue();

  if (DAG.isBaseWithConstantOffset(SDValue(N, 0)))
    return SDValue();

  unsigned Opc = N->getOpcode();
  SDValue Op0 = N->getOperand(0);
  SDValue Op1 = N->getOperand(1);

  if (!(Op0->isDivergent() ^ Op1->isDivergent()))
    return SDValue();

  if (Op0->isDivergent())
    std::swap(Op0, Op1);

  if (Op1.getOpcode() != Opc || !Op1.hasOneUse())
    return SDValue();

  SDValue Op2 = Op1.getOperand(1);
  Op1 = Op1.getOperand(0);
  if (!(Op1->isDivergent() ^ Op2->isDivergent()))
    return SDValue();

  if (Op1->isDivergent())
    std::swap(Op1, Op2);

  SDLoc SL(N);
  SDValue Add1 = DAG.getNode(Opc, SL, VT, Op0, Op1);
  return DAG.getNode(Opc, SL, VT, Add1, Op2);
}

static SDValue getMad64_32(SelectionDAG &DAG, const SDLoc &SL,
                           EVT VT,
                           SDValue N0, SDValue N1, SDValue N2,
                           bool Signed) {
  unsigned MadOpc = Signed ? AMDGPUISD::MAD_I64_I32 : AMDGPUISD::MAD_U64_U32;
  SDVTList VTs = DAG.getVTList(MVT::i64, MVT::i1);
  SDValue Mad = DAG.getNode(MadOpc, SL, VTs, N0, N1, N2);
  return DAG.getNode(ISD::TRUNCATE, SL, VT, Mad);
}

SDValue SITargetLowering::performAddCombine(SDNode *N,
                                            DAGCombinerInfo &DCI) const {
  SelectionDAG &DAG = DCI.DAG;
  EVT VT = N->getValueType(0);
  SDLoc SL(N);
  SDValue LHS = N->getOperand(0);
  SDValue RHS = N->getOperand(1);

  if ((LHS.getOpcode() == ISD::MUL || RHS.getOpcode() == ISD::MUL)
      && Subtarget->hasMad64_32() &&
      !VT.isVector() && VT.getScalarSizeInBits() > 32 &&
      VT.getScalarSizeInBits() <= 64) {
    if (LHS.getOpcode() != ISD::MUL)
      std::swap(LHS, RHS);

    SDValue MulLHS = LHS.getOperand(0);
    SDValue MulRHS = LHS.getOperand(1);
    SDValue AddRHS = RHS;

    // TODO: Maybe restrict if SGPR inputs.
    if (numBitsUnsigned(MulLHS, DAG) <= 32 &&
        numBitsUnsigned(MulRHS, DAG) <= 32) {
      MulLHS = DAG.getZExtOrTrunc(MulLHS, SL, MVT::i32);
      MulRHS = DAG.getZExtOrTrunc(MulRHS, SL, MVT::i32);
      AddRHS = DAG.getZExtOrTrunc(AddRHS, SL, MVT::i64);
      return getMad64_32(DAG, SL, VT, MulLHS, MulRHS, AddRHS, false);
    }

    if (numBitsSigned(MulLHS, DAG) <= 32 && numBitsSigned(MulRHS, DAG) <= 32) {
      MulLHS = DAG.getSExtOrTrunc(MulLHS, SL, MVT::i32);
      MulRHS = DAG.getSExtOrTrunc(MulRHS, SL, MVT::i32);
      AddRHS = DAG.getSExtOrTrunc(AddRHS, SL, MVT::i64);
      return getMad64_32(DAG, SL, VT, MulLHS, MulRHS, AddRHS, true);
    }

    return SDValue();
  }

  if (SDValue V = reassociateScalarOps(N, DAG)) {
    return V;
  }

  if (VT != MVT::i32 || !DCI.isAfterLegalizeDAG())
    return SDValue();

  // add x, zext (setcc) => addcarry x, 0, setcc
  // add x, sext (setcc) => subcarry x, 0, setcc
  unsigned Opc = LHS.getOpcode();
  if (Opc == ISD::ZERO_EXTEND || Opc == ISD::SIGN_EXTEND ||
      Opc == ISD::ANY_EXTEND || Opc == ISD::ADDCARRY)
    std::swap(RHS, LHS);

  Opc = RHS.getOpcode();
  switch (Opc) {
  default: break;
  case ISD::ZERO_EXTEND:
  case ISD::SIGN_EXTEND:
  case ISD::ANY_EXTEND: {
    auto Cond = RHS.getOperand(0);
    // If this won't be a real VOPC output, we would still need to insert an
    // extra instruction anyway.
    if (!isBoolSGPR(Cond))
      break;
    SDVTList VTList = DAG.getVTList(MVT::i32, MVT::i1);
    SDValue Args[] = { LHS, DAG.getConstant(0, SL, MVT::i32), Cond };
    Opc = (Opc == ISD::SIGN_EXTEND) ? ISD::SUBCARRY : ISD::ADDCARRY;
    return DAG.getNode(Opc, SL, VTList, Args);
  }
  case ISD::ADDCARRY: {
    // add x, (addcarry y, 0, cc) => addcarry x, y, cc
    auto C = dyn_cast<ConstantSDNode>(RHS.getOperand(1));
    if (!C || C->getZExtValue() != 0) break;
    SDValue Args[] = { LHS, RHS.getOperand(0), RHS.getOperand(2) };
    return DAG.getNode(ISD::ADDCARRY, SDLoc(N), RHS->getVTList(), Args);
  }
  }
  return SDValue();
}

SDValue SITargetLowering::performSubCombine(SDNode *N,
                                            DAGCombinerInfo &DCI) const {
  SelectionDAG &DAG = DCI.DAG;
  EVT VT = N->getValueType(0);

  if (VT != MVT::i32)
    return SDValue();

  SDLoc SL(N);
  SDValue LHS = N->getOperand(0);
  SDValue RHS = N->getOperand(1);

  // sub x, zext (setcc) => subcarry x, 0, setcc
  // sub x, sext (setcc) => addcarry x, 0, setcc
  unsigned Opc = RHS.getOpcode();
  switch (Opc) {
  default: break;
  case ISD::ZERO_EXTEND:
  case ISD::SIGN_EXTEND:
  case ISD::ANY_EXTEND: {
    auto Cond = RHS.getOperand(0);
    // If this won't be a real VOPC output, we would still need to insert an
    // extra instruction anyway.
    if (!isBoolSGPR(Cond))
      break;
    SDVTList VTList = DAG.getVTList(MVT::i32, MVT::i1);
    SDValue Args[] = { LHS, DAG.getConstant(0, SL, MVT::i32), Cond };
    Opc = (Opc == ISD::SIGN_EXTEND) ? ISD::ADDCARRY : ISD::SUBCARRY;
    return DAG.getNode(Opc, SL, VTList, Args);
  }
  }

  if (LHS.getOpcode() == ISD::SUBCARRY) {
    // sub (subcarry x, 0, cc), y => subcarry x, y, cc
    auto C = dyn_cast<ConstantSDNode>(LHS.getOperand(1));
    if (!C || !C->isZero())
      return SDValue();
    SDValue Args[] = { LHS.getOperand(0), RHS, LHS.getOperand(2) };
    return DAG.getNode(ISD::SUBCARRY, SDLoc(N), LHS->getVTList(), Args);
  }
  return SDValue();
}

SDValue SITargetLowering::performAddCarrySubCarryCombine(SDNode *N,
  DAGCombinerInfo &DCI) const {

  if (N->getValueType(0) != MVT::i32)
    return SDValue();

  auto C = dyn_cast<ConstantSDNode>(N->getOperand(1));
  if (!C || C->getZExtValue() != 0)
    return SDValue();

  SelectionDAG &DAG = DCI.DAG;
  SDValue LHS = N->getOperand(0);

  // addcarry (add x, y), 0, cc => addcarry x, y, cc
  // subcarry (sub x, y), 0, cc => subcarry x, y, cc
  unsigned LHSOpc = LHS.getOpcode();
  unsigned Opc = N->getOpcode();
  if ((LHSOpc == ISD::ADD && Opc == ISD::ADDCARRY) ||
      (LHSOpc == ISD::SUB && Opc == ISD::SUBCARRY)) {
    SDValue Args[] = { LHS.getOperand(0), LHS.getOperand(1), N->getOperand(2) };
    return DAG.getNode(Opc, SDLoc(N), N->getVTList(), Args);
  }
  return SDValue();
}

SDValue SITargetLowering::performFAddCombine(SDNode *N,
                                             DAGCombinerInfo &DCI) const {
  if (DCI.getDAGCombineLevel() < AfterLegalizeDAG)
    return SDValue();

  SelectionDAG &DAG = DCI.DAG;
  EVT VT = N->getValueType(0);

  SDLoc SL(N);
  SDValue LHS = N->getOperand(0);
  SDValue RHS = N->getOperand(1);

  // These should really be instruction patterns, but writing patterns with
  // source modiifiers is a pain.

  // fadd (fadd (a, a), b) -> mad 2.0, a, b
  if (LHS.getOpcode() == ISD::FADD) {
    SDValue A = LHS.getOperand(0);
    if (A == LHS.getOperand(1)) {
      unsigned FusedOp = getFusedOpcode(DAG, N, LHS.getNode());
      if (FusedOp != 0) {
        const SDValue Two = DAG.getConstantFP(2.0, SL, VT);
        return DAG.getNode(FusedOp, SL, VT, A, Two, RHS);
      }
    }
  }

  // fadd (b, fadd (a, a)) -> mad 2.0, a, b
  if (RHS.getOpcode() == ISD::FADD) {
    SDValue A = RHS.getOperand(0);
    if (A == RHS.getOperand(1)) {
      unsigned FusedOp = getFusedOpcode(DAG, N, RHS.getNode());
      if (FusedOp != 0) {
        const SDValue Two = DAG.getConstantFP(2.0, SL, VT);
        return DAG.getNode(FusedOp, SL, VT, A, Two, LHS);
      }
    }
  }

  return SDValue();
}

SDValue SITargetLowering::performFSubCombine(SDNode *N,
                                             DAGCombinerInfo &DCI) const {
  if (DCI.getDAGCombineLevel() < AfterLegalizeDAG)
    return SDValue();

  SelectionDAG &DAG = DCI.DAG;
  SDLoc SL(N);
  EVT VT = N->getValueType(0);
  assert(!VT.isVector());

  // Try to get the fneg to fold into the source modifier. This undoes generic
  // DAG combines and folds them into the mad.
  //
  // Only do this if we are not trying to support denormals. v_mad_f32 does
  // not support denormals ever.
  SDValue LHS = N->getOperand(0);
  SDValue RHS = N->getOperand(1);
  if (LHS.getOpcode() == ISD::FADD) {
    // (fsub (fadd a, a), c) -> mad 2.0, a, (fneg c)
    SDValue A = LHS.getOperand(0);
    if (A == LHS.getOperand(1)) {
      unsigned FusedOp = getFusedOpcode(DAG, N, LHS.getNode());
      if (FusedOp != 0){
        const SDValue Two = DAG.getConstantFP(2.0, SL, VT);
        SDValue NegRHS = DAG.getNode(ISD::FNEG, SL, VT, RHS);

        return DAG.getNode(FusedOp, SL, VT, A, Two, NegRHS);
      }
    }
  }

  if (RHS.getOpcode() == ISD::FADD) {
    // (fsub c, (fadd a, a)) -> mad -2.0, a, c

    SDValue A = RHS.getOperand(0);
    if (A == RHS.getOperand(1)) {
      unsigned FusedOp = getFusedOpcode(DAG, N, RHS.getNode());
      if (FusedOp != 0){
        const SDValue NegTwo = DAG.getConstantFP(-2.0, SL, VT);
        return DAG.getNode(FusedOp, SL, VT, A, NegTwo, LHS);
      }
    }
  }

  return SDValue();
}

SDValue SITargetLowering::performFMACombine(SDNode *N,
                                            DAGCombinerInfo &DCI) const {
  SelectionDAG &DAG = DCI.DAG;
  EVT VT = N->getValueType(0);
  SDLoc SL(N);

  if (!Subtarget->hasDot7Insts() || VT != MVT::f32)
    return SDValue();

  // FMA((F32)S0.x, (F32)S1. x, FMA((F32)S0.y, (F32)S1.y, (F32)z)) ->
  //   FDOT2((V2F16)S0, (V2F16)S1, (F32)z))
  SDValue Op1 = N->getOperand(0);
  SDValue Op2 = N->getOperand(1);
  SDValue FMA = N->getOperand(2);

  if (FMA.getOpcode() != ISD::FMA ||
      Op1.getOpcode() != ISD::FP_EXTEND ||
      Op2.getOpcode() != ISD::FP_EXTEND)
    return SDValue();

  // fdot2_f32_f16 always flushes fp32 denormal operand and output to zero,
  // regardless of the denorm mode setting. Therefore, unsafe-fp-math/fp-contract
  // is sufficient to allow generaing fdot2.
  const TargetOptions &Options = DAG.getTarget().Options;
  if (Options.AllowFPOpFusion == FPOpFusion::Fast || Options.UnsafeFPMath ||
      (N->getFlags().hasAllowContract() &&
       FMA->getFlags().hasAllowContract())) {
    Op1 = Op1.getOperand(0);
    Op2 = Op2.getOperand(0);
    if (Op1.getOpcode() != ISD::EXTRACT_VECTOR_ELT ||
        Op2.getOpcode() != ISD::EXTRACT_VECTOR_ELT)
      return SDValue();

    SDValue Vec1 = Op1.getOperand(0);
    SDValue Idx1 = Op1.getOperand(1);
    SDValue Vec2 = Op2.getOperand(0);

    SDValue FMAOp1 = FMA.getOperand(0);
    SDValue FMAOp2 = FMA.getOperand(1);
    SDValue FMAAcc = FMA.getOperand(2);

    if (FMAOp1.getOpcode() != ISD::FP_EXTEND ||
        FMAOp2.getOpcode() != ISD::FP_EXTEND)
      return SDValue();

    FMAOp1 = FMAOp1.getOperand(0);
    FMAOp2 = FMAOp2.getOperand(0);
    if (FMAOp1.getOpcode() != ISD::EXTRACT_VECTOR_ELT ||
        FMAOp2.getOpcode() != ISD::EXTRACT_VECTOR_ELT)
      return SDValue();

    SDValue Vec3 = FMAOp1.getOperand(0);
    SDValue Vec4 = FMAOp2.getOperand(0);
    SDValue Idx2 = FMAOp1.getOperand(1);

    if (Idx1 != Op2.getOperand(1) || Idx2 != FMAOp2.getOperand(1) ||
        // Idx1 and Idx2 cannot be the same.
        Idx1 == Idx2)
      return SDValue();

    if (Vec1 == Vec2 || Vec3 == Vec4)
      return SDValue();

    if (Vec1.getValueType() != MVT::v2f16 || Vec2.getValueType() != MVT::v2f16)
      return SDValue();

    if ((Vec1 == Vec3 && Vec2 == Vec4) ||
        (Vec1 == Vec4 && Vec2 == Vec3)) {
      return DAG.getNode(AMDGPUISD::FDOT2, SL, MVT::f32, Vec1, Vec2, FMAAcc,
                         DAG.getTargetConstant(0, SL, MVT::i1));
    }
  }
  return SDValue();
}

SDValue SITargetLowering::performSetCCCombine(SDNode *N,
                                              DAGCombinerInfo &DCI) const {
  SelectionDAG &DAG = DCI.DAG;
  SDLoc SL(N);

  SDValue LHS = N->getOperand(0);
  SDValue RHS = N->getOperand(1);
  EVT VT = LHS.getValueType();
  ISD::CondCode CC = cast<CondCodeSDNode>(N->getOperand(2))->get();

  auto CRHS = dyn_cast<ConstantSDNode>(RHS);
  if (!CRHS) {
    CRHS = dyn_cast<ConstantSDNode>(LHS);
    if (CRHS) {
      std::swap(LHS, RHS);
      CC = getSetCCSwappedOperands(CC);
    }
  }

  if (CRHS) {
    if (VT == MVT::i32 && LHS.getOpcode() == ISD::SIGN_EXTEND &&
        isBoolSGPR(LHS.getOperand(0))) {
      // setcc (sext from i1 cc), -1, ne|sgt|ult) => not cc => xor cc, -1
      // setcc (sext from i1 cc), -1, eq|sle|uge) => cc
      // setcc (sext from i1 cc),  0, eq|sge|ule) => not cc => xor cc, -1
      // setcc (sext from i1 cc),  0, ne|ugt|slt) => cc
      if ((CRHS->isAllOnes() &&
           (CC == ISD::SETNE || CC == ISD::SETGT || CC == ISD::SETULT)) ||
          (CRHS->isZero() &&
           (CC == ISD::SETEQ || CC == ISD::SETGE || CC == ISD::SETULE)))
        return DAG.getNode(ISD::XOR, SL, MVT::i1, LHS.getOperand(0),
                           DAG.getConstant(-1, SL, MVT::i1));
      if ((CRHS->isAllOnes() &&
           (CC == ISD::SETEQ || CC == ISD::SETLE || CC == ISD::SETUGE)) ||
          (CRHS->isZero() &&
           (CC == ISD::SETNE || CC == ISD::SETUGT || CC == ISD::SETLT)))
        return LHS.getOperand(0);
    }

    const APInt &CRHSVal = CRHS->getAPIntValue();
    if ((CC == ISD::SETEQ || CC == ISD::SETNE) &&
        LHS.getOpcode() == ISD::SELECT &&
        isa<ConstantSDNode>(LHS.getOperand(1)) &&
        isa<ConstantSDNode>(LHS.getOperand(2)) &&
        LHS.getConstantOperandVal(1) != LHS.getConstantOperandVal(2) &&
        isBoolSGPR(LHS.getOperand(0))) {
      // Given CT != FT:
      // setcc (select cc, CT, CF), CF, eq => xor cc, -1
      // setcc (select cc, CT, CF), CF, ne => cc
      // setcc (select cc, CT, CF), CT, ne => xor cc, -1
      // setcc (select cc, CT, CF), CT, eq => cc
      const APInt &CT = LHS.getConstantOperandAPInt(1);
      const APInt &CF = LHS.getConstantOperandAPInt(2);

      if ((CF == CRHSVal && CC == ISD::SETEQ) ||
          (CT == CRHSVal && CC == ISD::SETNE))
        return DAG.getNode(ISD::XOR, SL, MVT::i1, LHS.getOperand(0),
                           DAG.getConstant(-1, SL, MVT::i1));
      if ((CF == CRHSVal && CC == ISD::SETNE) ||
          (CT == CRHSVal && CC == ISD::SETEQ))
        return LHS.getOperand(0);
    }
  }

  if (VT != MVT::f32 && VT != MVT::f64 && (Subtarget->has16BitInsts() &&
                                           VT != MVT::f16))
    return SDValue();

  // Match isinf/isfinite pattern
  // (fcmp oeq (fabs x), inf) -> (fp_class x, (p_infinity | n_infinity))
  // (fcmp one (fabs x), inf) -> (fp_class x,
  // (p_normal | n_normal | p_subnormal | n_subnormal | p_zero | n_zero)
  if ((CC == ISD::SETOEQ || CC == ISD::SETONE) && LHS.getOpcode() == ISD::FABS) {
    const ConstantFPSDNode *CRHS = dyn_cast<ConstantFPSDNode>(RHS);
    if (!CRHS)
      return SDValue();

    const APFloat &APF = CRHS->getValueAPF();
    if (APF.isInfinity() && !APF.isNegative()) {
      const unsigned IsInfMask = SIInstrFlags::P_INFINITY |
                                 SIInstrFlags::N_INFINITY;
      const unsigned IsFiniteMask = SIInstrFlags::N_ZERO |
                                    SIInstrFlags::P_ZERO |
                                    SIInstrFlags::N_NORMAL |
                                    SIInstrFlags::P_NORMAL |
                                    SIInstrFlags::N_SUBNORMAL |
                                    SIInstrFlags::P_SUBNORMAL;
      unsigned Mask = CC == ISD::SETOEQ ? IsInfMask : IsFiniteMask;
      return DAG.getNode(AMDGPUISD::FP_CLASS, SL, MVT::i1, LHS.getOperand(0),
                         DAG.getConstant(Mask, SL, MVT::i32));
    }
  }

  return SDValue();
}

SDValue SITargetLowering::performCvtF32UByteNCombine(SDNode *N,
                                                     DAGCombinerInfo &DCI) const {
  SelectionDAG &DAG = DCI.DAG;
  SDLoc SL(N);
  unsigned Offset = N->getOpcode() - AMDGPUISD::CVT_F32_UBYTE0;

  SDValue Src = N->getOperand(0);
  SDValue Shift = N->getOperand(0);

  // TODO: Extend type shouldn't matter (assuming legal types).
  if (Shift.getOpcode() == ISD::ZERO_EXTEND)
    Shift = Shift.getOperand(0);

  if (Shift.getOpcode() == ISD::SRL || Shift.getOpcode() == ISD::SHL) {
    // cvt_f32_ubyte1 (shl x,  8) -> cvt_f32_ubyte0 x
    // cvt_f32_ubyte3 (shl x, 16) -> cvt_f32_ubyte1 x
    // cvt_f32_ubyte0 (srl x, 16) -> cvt_f32_ubyte2 x
    // cvt_f32_ubyte1 (srl x, 16) -> cvt_f32_ubyte3 x
    // cvt_f32_ubyte0 (srl x,  8) -> cvt_f32_ubyte1 x
    if (auto *C = dyn_cast<ConstantSDNode>(Shift.getOperand(1))) {
      SDValue Shifted = DAG.getZExtOrTrunc(Shift.getOperand(0),
                                 SDLoc(Shift.getOperand(0)), MVT::i32);

      unsigned ShiftOffset = 8 * Offset;
      if (Shift.getOpcode() == ISD::SHL)
        ShiftOffset -= C->getZExtValue();
      else
        ShiftOffset += C->getZExtValue();

      if (ShiftOffset < 32 && (ShiftOffset % 8) == 0) {
        return DAG.getNode(AMDGPUISD::CVT_F32_UBYTE0 + ShiftOffset / 8, SL,
                           MVT::f32, Shifted);
      }
    }
  }

  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  APInt DemandedBits = APInt::getBitsSet(32, 8 * Offset, 8 * Offset + 8);
  if (TLI.SimplifyDemandedBits(Src, DemandedBits, DCI)) {
    // We simplified Src. If this node is not dead, visit it again so it is
    // folded properly.
    if (N->getOpcode() != ISD::DELETED_NODE)
      DCI.AddToWorklist(N);
    return SDValue(N, 0);
  }

  // Handle (or x, (srl y, 8)) pattern when known bits are zero.
  if (SDValue DemandedSrc =
          TLI.SimplifyMultipleUseDemandedBits(Src, DemandedBits, DAG))
    return DAG.getNode(N->getOpcode(), SL, MVT::f32, DemandedSrc);

  return SDValue();
}

SDValue SITargetLowering::performClampCombine(SDNode *N,
                                              DAGCombinerInfo &DCI) const {
  ConstantFPSDNode *CSrc = dyn_cast<ConstantFPSDNode>(N->getOperand(0));
  if (!CSrc)
    return SDValue();

  const MachineFunction &MF = DCI.DAG.getMachineFunction();
  const APFloat &F = CSrc->getValueAPF();
  APFloat Zero = APFloat::getZero(F.getSemantics());
  if (F < Zero ||
      (F.isNaN() && MF.getInfo<SIMachineFunctionInfo>()->getMode().DX10Clamp)) {
    return DCI.DAG.getConstantFP(Zero, SDLoc(N), N->getValueType(0));
  }

  APFloat One(F.getSemantics(), "1.0");
  if (F > One)
    return DCI.DAG.getConstantFP(One, SDLoc(N), N->getValueType(0));

  return SDValue(CSrc, 0);
}


SDValue SITargetLowering::PerformDAGCombine(SDNode *N,
                                            DAGCombinerInfo &DCI) const {
  if (getTargetMachine().getOptLevel() == CodeGenOpt::None)
    return SDValue();
  switch (N->getOpcode()) {
  case ISD::ADD:
    return performAddCombine(N, DCI);
  case ISD::SUB:
    return performSubCombine(N, DCI);
  case ISD::ADDCARRY:
  case ISD::SUBCARRY:
    return performAddCarrySubCarryCombine(N, DCI);
  case ISD::FADD:
    return performFAddCombine(N, DCI);
  case ISD::FSUB:
    return performFSubCombine(N, DCI);
  case ISD::SETCC:
    return performSetCCCombine(N, DCI);
  case ISD::FMAXNUM:
  case ISD::FMINNUM:
  case ISD::FMAXNUM_IEEE:
  case ISD::FMINNUM_IEEE:
  case ISD::SMAX:
  case ISD::SMIN:
  case ISD::UMAX:
  case ISD::UMIN:
  case AMDGPUISD::FMIN_LEGACY:
  case AMDGPUISD::FMAX_LEGACY:
    return performMinMaxCombine(N, DCI);
  case ISD::FMA:
    return performFMACombine(N, DCI);
  case ISD::AND:
    return performAndCombine(N, DCI);
  case ISD::OR:
    return performOrCombine(N, DCI);
  case ISD::XOR:
    return performXorCombine(N, DCI);
  case ISD::ZERO_EXTEND:
    return performZeroExtendCombine(N, DCI);
  case ISD::SIGN_EXTEND_INREG:
    return performSignExtendInRegCombine(N , DCI);
  case AMDGPUISD::FP_CLASS:
    return performClassCombine(N, DCI);
  case ISD::FCANONICALIZE:
    return performFCanonicalizeCombine(N, DCI);
  case AMDGPUISD::RCP:
    return performRcpCombine(N, DCI);
  case AMDGPUISD::FRACT:
  case AMDGPUISD::RSQ:
  case AMDGPUISD::RCP_LEGACY:
  case AMDGPUISD::RCP_IFLAG:
  case AMDGPUISD::RSQ_CLAMP:
  case AMDGPUISD::LDEXP: {
    // FIXME: This is probably wrong. If src is an sNaN, it won't be quieted
    SDValue Src = N->getOperand(0);
    if (Src.isUndef())
      return Src;
    break;
  }
  case ISD::SINT_TO_FP:
  case ISD::UINT_TO_FP:
    return performUCharToFloatCombine(N, DCI);
  case AMDGPUISD::CVT_F32_UBYTE0:
  case AMDGPUISD::CVT_F32_UBYTE1:
  case AMDGPUISD::CVT_F32_UBYTE2:
  case AMDGPUISD::CVT_F32_UBYTE3:
    return performCvtF32UByteNCombine(N, DCI);
  case AMDGPUISD::FMED3:
    return performFMed3Combine(N, DCI);
  case AMDGPUISD::CVT_PKRTZ_F16_F32:
    return performCvtPkRTZCombine(N, DCI);
  case AMDGPUISD::CLAMP:
    return performClampCombine(N, DCI);
  case ISD::SCALAR_TO_VECTOR: {
    SelectionDAG &DAG = DCI.DAG;
    EVT VT = N->getValueType(0);

    // v2i16 (scalar_to_vector i16:x) -> v2i16 (bitcast (any_extend i16:x))
    if (VT == MVT::v2i16 || VT == MVT::v2f16) {
      SDLoc SL(N);
      SDValue Src = N->getOperand(0);
      EVT EltVT = Src.getValueType();
      if (EltVT == MVT::f16)
        Src = DAG.getNode(ISD::BITCAST, SL, MVT::i16, Src);

      SDValue Ext = DAG.getNode(ISD::ANY_EXTEND, SL, MVT::i32, Src);
      return DAG.getNode(ISD::BITCAST, SL, VT, Ext);
    }

    break;
  }
  case ISD::EXTRACT_VECTOR_ELT:
    return performExtractVectorEltCombine(N, DCI);
  case ISD::INSERT_VECTOR_ELT:
    return performInsertVectorEltCombine(N, DCI);
  case ISD::LOAD: {
    if (SDValue Widended = widenLoad(cast<LoadSDNode>(N), DCI))
      return Widended;
    LLVM_FALLTHROUGH;
  }
  default: {
    if (!DCI.isBeforeLegalize()) {
      if (MemSDNode *MemNode = dyn_cast<MemSDNode>(N))
        return performMemSDNodeCombine(MemNode, DCI);
    }

    break;
  }
  }

  return AMDGPUTargetLowering::PerformDAGCombine(N, DCI);
}

/// Helper function for adjustWritemask
static unsigned SubIdx2Lane(unsigned Idx) {
  switch (Idx) {
  default: return ~0u;
  case AMDGPU::sub0: return 0;
  case AMDGPU::sub1: return 1;
  case AMDGPU::sub2: return 2;
  case AMDGPU::sub3: return 3;
  case AMDGPU::sub4: return 4; // Possible with TFE/LWE
  }
}

/// Adjust the writemask of MIMG instructions
SDNode *SITargetLowering::adjustWritemask(MachineSDNode *&Node,
                                          SelectionDAG &DAG) const {
  unsigned Opcode = Node->getMachineOpcode();

  // Subtract 1 because the vdata output is not a MachineSDNode operand.
  int D16Idx = AMDGPU::getNamedOperandIdx(Opcode, AMDGPU::OpName::d16) - 1;
  if (D16Idx >= 0 && Node->getConstantOperandVal(D16Idx))
    return Node; // not implemented for D16

  SDNode *Users[5] = { nullptr };
  unsigned Lane = 0;
  unsigned DmaskIdx = AMDGPU::getNamedOperandIdx(Opcode, AMDGPU::OpName::dmask) - 1;
  unsigned OldDmask = Node->getConstantOperandVal(DmaskIdx);
  unsigned NewDmask = 0;
  unsigned TFEIdx = AMDGPU::getNamedOperandIdx(Opcode, AMDGPU::OpName::tfe) - 1;
  unsigned LWEIdx = AMDGPU::getNamedOperandIdx(Opcode, AMDGPU::OpName::lwe) - 1;
  bool UsesTFC = ((int(TFEIdx) >= 0 && Node->getConstantOperandVal(TFEIdx)) ||
                  Node->getConstantOperandVal(LWEIdx))
                     ? true
                     : false;
  unsigned TFCLane = 0;
  bool HasChain = Node->getNumValues() > 1;

  if (OldDmask == 0) {
    // These are folded out, but on the chance it happens don't assert.
    return Node;
  }

  unsigned OldBitsSet = countPopulation(OldDmask);
  // Work out which is the TFE/LWE lane if that is enabled.
  if (UsesTFC) {
    TFCLane = OldBitsSet;
  }

  // Try to figure out the used register components
  for (SDNode::use_iterator I = Node->use_begin(), E = Node->use_end();
       I != E; ++I) {

    // Don't look at users of the chain.
    if (I.getUse().getResNo() != 0)
      continue;

    // Abort if we can't understand the usage
    if (!I->isMachineOpcode() ||
        I->getMachineOpcode() != TargetOpcode::EXTRACT_SUBREG)
      return Node;

    // Lane means which subreg of %vgpra_vgprb_vgprc_vgprd is used.
    // Note that subregs are packed, i.e. Lane==0 is the first bit set
    // in OldDmask, so it can be any of X,Y,Z,W; Lane==1 is the second bit
    // set, etc.
    Lane = SubIdx2Lane(I->getConstantOperandVal(1));
    if (Lane == ~0u)
      return Node;

    // Check if the use is for the TFE/LWE generated result at VGPRn+1.
    if (UsesTFC && Lane == TFCLane) {
      Users[Lane] = *I;
    } else {
      // Set which texture component corresponds to the lane.
      unsigned Comp;
      for (unsigned i = 0, Dmask = OldDmask; (i <= Lane) && (Dmask != 0); i++) {
        Comp = countTrailingZeros(Dmask);
        Dmask &= ~(1 << Comp);
      }

      // Abort if we have more than one user per component.
      if (Users[Lane])
        return Node;

      Users[Lane] = *I;
      NewDmask |= 1 << Comp;
    }
  }

  // Don't allow 0 dmask, as hardware assumes one channel enabled.
  bool NoChannels = !NewDmask;
  if (NoChannels) {
    if (!UsesTFC) {
      // No uses of the result and not using TFC. Then do nothing.
      return Node;
    }
    // If the original dmask has one channel - then nothing to do
    if (OldBitsSet == 1)
      return Node;
    // Use an arbitrary dmask - required for the instruction to work
    NewDmask = 1;
  }
  // Abort if there's no change
  if (NewDmask == OldDmask)
    return Node;

  unsigned BitsSet = countPopulation(NewDmask);

  // Check for TFE or LWE - increase the number of channels by one to account
  // for the extra return value
  // This will need adjustment for D16 if this is also included in
  // adjustWriteMask (this function) but at present D16 are excluded.
  unsigned NewChannels = BitsSet + UsesTFC;

  int NewOpcode =
      AMDGPU::getMaskedMIMGOp(Node->getMachineOpcode(), NewChannels);
  assert(NewOpcode != -1 &&
         NewOpcode != static_cast<int>(Node->getMachineOpcode()) &&
         "failed to find equivalent MIMG op");

  // Adjust the writemask in the node
  SmallVector<SDValue, 12> Ops;
  Ops.insert(Ops.end(), Node->op_begin(), Node->op_begin() + DmaskIdx);
  Ops.push_back(DAG.getTargetConstant(NewDmask, SDLoc(Node), MVT::i32));
  Ops.insert(Ops.end(), Node->op_begin() + DmaskIdx + 1, Node->op_end());

  MVT SVT = Node->getValueType(0).getVectorElementType().getSimpleVT();

  MVT ResultVT = NewChannels == 1 ?
    SVT : MVT::getVectorVT(SVT, NewChannels == 3 ? 4 :
                           NewChannels == 5 ? 8 : NewChannels);
  SDVTList NewVTList = HasChain ?
    DAG.getVTList(ResultVT, MVT::Other) : DAG.getVTList(ResultVT);


  MachineSDNode *NewNode = DAG.getMachineNode(NewOpcode, SDLoc(Node),
                                              NewVTList, Ops);

  if (HasChain) {
    // Update chain.
    DAG.setNodeMemRefs(NewNode, Node->memoperands());
    DAG.ReplaceAllUsesOfValueWith(SDValue(Node, 1), SDValue(NewNode, 1));
  }

  if (NewChannels == 1) {
    assert(Node->hasNUsesOfValue(1, 0));
    SDNode *Copy = DAG.getMachineNode(TargetOpcode::COPY,
                                      SDLoc(Node), Users[Lane]->getValueType(0),
                                      SDValue(NewNode, 0));
    DAG.ReplaceAllUsesWith(Users[Lane], Copy);
    return nullptr;
  }

  // Update the users of the node with the new indices
  for (unsigned i = 0, Idx = AMDGPU::sub0; i < 5; ++i) {
    SDNode *User = Users[i];
    if (!User) {
      // Handle the special case of NoChannels. We set NewDmask to 1 above, but
      // Users[0] is still nullptr because channel 0 doesn't really have a use.
      if (i || !NoChannels)
        continue;
    } else {
      SDValue Op = DAG.getTargetConstant(Idx, SDLoc(User), MVT::i32);
      DAG.UpdateNodeOperands(User, SDValue(NewNode, 0), Op);
    }

    switch (Idx) {
    default: break;
    case AMDGPU::sub0: Idx = AMDGPU::sub1; break;
    case AMDGPU::sub1: Idx = AMDGPU::sub2; break;
    case AMDGPU::sub2: Idx = AMDGPU::sub3; break;
    case AMDGPU::sub3: Idx = AMDGPU::sub4; break;
    }
  }

  DAG.RemoveDeadNode(Node);
  return nullptr;
}

static bool isFrameIndexOp(SDValue Op) {
  if (Op.getOpcode() == ISD::AssertZext)
    Op = Op.getOperand(0);

  return isa<FrameIndexSDNode>(Op);
}

/// Legalize target independent instructions (e.g. INSERT_SUBREG)
/// with frame index operands.
/// LLVM assumes that inputs are to these instructions are registers.
SDNode *SITargetLowering::legalizeTargetIndependentNode(SDNode *Node,
                                                        SelectionDAG &DAG) const {
  if (Node->getOpcode() == ISD::CopyToReg) {
    RegisterSDNode *DestReg = cast<RegisterSDNode>(Node->getOperand(1));
    SDValue SrcVal = Node->getOperand(2);

    // Insert a copy to a VReg_1 virtual register so LowerI1Copies doesn't have
    // to try understanding copies to physical registers.
    if (SrcVal.getValueType() == MVT::i1 && DestReg->getReg().isPhysical()) {
      SDLoc SL(Node);
      MachineRegisterInfo &MRI = DAG.getMachineFunction().getRegInfo();
      SDValue VReg = DAG.getRegister(
        MRI.createVirtualRegister(&AMDGPU::VReg_1RegClass), MVT::i1);

      SDNode *Glued = Node->getGluedNode();
      SDValue ToVReg
        = DAG.getCopyToReg(Node->getOperand(0), SL, VReg, SrcVal,
                         SDValue(Glued, Glued ? Glued->getNumValues() - 1 : 0));
      SDValue ToResultReg
        = DAG.getCopyToReg(ToVReg, SL, SDValue(DestReg, 0),
                           VReg, ToVReg.getValue(1));
      DAG.ReplaceAllUsesWith(Node, ToResultReg.getNode());
      DAG.RemoveDeadNode(Node);
      return ToResultReg.getNode();
    }
  }

  SmallVector<SDValue, 8> Ops;
  for (unsigned i = 0; i < Node->getNumOperands(); ++i) {
    if (!isFrameIndexOp(Node->getOperand(i))) {
      Ops.push_back(Node->getOperand(i));
      continue;
    }

    SDLoc DL(Node);
    Ops.push_back(SDValue(DAG.getMachineNode(AMDGPU::S_MOV_B32, DL,
                                     Node->getOperand(i).getValueType(),
                                     Node->getOperand(i)), 0));
  }

  return DAG.UpdateNodeOperands(Node, Ops);
}

/// Fold the instructions after selecting them.
/// Returns null if users were already updated.
SDNode *SITargetLowering::PostISelFolding(MachineSDNode *Node,
                                          SelectionDAG &DAG) const {
  const SIInstrInfo *TII = getSubtarget()->getInstrInfo();
  unsigned Opcode = Node->getMachineOpcode();

  if (TII->isMIMG(Opcode) && !TII->get(Opcode).mayStore() &&
      !TII->isGather4(Opcode) &&
      AMDGPU::getNamedOperandIdx(Opcode, AMDGPU::OpName::dmask) != -1) {
    return adjustWritemask(Node, DAG);
  }

  if (Opcode == AMDGPU::INSERT_SUBREG ||
      Opcode == AMDGPU::REG_SEQUENCE) {
    legalizeTargetIndependentNode(Node, DAG);
    return Node;
  }

  switch (Opcode) {
  case AMDGPU::V_DIV_SCALE_F32_e64:
  case AMDGPU::V_DIV_SCALE_F64_e64: {
    // Satisfy the operand register constraint when one of the inputs is
    // undefined. Ordinarily each undef value will have its own implicit_def of
    // a vreg, so force these to use a single register.
    SDValue Src0 = Node->getOperand(1);
    SDValue Src1 = Node->getOperand(3);
    SDValue Src2 = Node->getOperand(5);

    if ((Src0.isMachineOpcode() &&
         Src0.getMachineOpcode() != AMDGPU::IMPLICIT_DEF) &&
        (Src0 == Src1 || Src0 == Src2))
      break;

    MVT VT = Src0.getValueType().getSimpleVT();
    const TargetRegisterClass *RC =
        getRegClassFor(VT, Src0.getNode()->isDivergent());

    MachineRegisterInfo &MRI = DAG.getMachineFunction().getRegInfo();
    SDValue UndefReg = DAG.getRegister(MRI.createVirtualRegister(RC), VT);

    SDValue ImpDef = DAG.getCopyToReg(DAG.getEntryNode(), SDLoc(Node),
                                      UndefReg, Src0, SDValue());

    // src0 must be the same register as src1 or src2, even if the value is
    // undefined, so make sure we don't violate this constraint.
    if (Src0.isMachineOpcode() &&
        Src0.getMachineOpcode() == AMDGPU::IMPLICIT_DEF) {
      if (Src1.isMachineOpcode() &&
          Src1.getMachineOpcode() != AMDGPU::IMPLICIT_DEF)
        Src0 = Src1;
      else if (Src2.isMachineOpcode() &&
               Src2.getMachineOpcode() != AMDGPU::IMPLICIT_DEF)
        Src0 = Src2;
      else {
        assert(Src1.getMachineOpcode() == AMDGPU::IMPLICIT_DEF);
        Src0 = UndefReg;
        Src1 = UndefReg;
      }
    } else
      break;

    SmallVector<SDValue, 9> Ops(Node->op_begin(), Node->op_end());
    Ops[1] = Src0;
    Ops[3] = Src1;
    Ops[5] = Src2;
    Ops.push_back(ImpDef.getValue(1));
    return DAG.getMachineNode(Opcode, SDLoc(Node), Node->getVTList(), Ops);
  }
  default:
    break;
  }

  return Node;
}

// Any MIMG instructions that use tfe or lwe require an initialization of the
// result register that will be written in the case of a memory access failure.
// The required code is also added to tie this init code to the result of the
// img instruction.
void SITargetLowering::AddIMGInit(MachineInstr &MI) const {
  const SIInstrInfo *TII = getSubtarget()->getInstrInfo();
  const SIRegisterInfo &TRI = TII->getRegisterInfo();
  MachineRegisterInfo &MRI = MI.getMF()->getRegInfo();
  MachineBasicBlock &MBB = *MI.getParent();

  MachineOperand *TFE = TII->getNamedOperand(MI, AMDGPU::OpName::tfe);
  MachineOperand *LWE = TII->getNamedOperand(MI, AMDGPU::OpName::lwe);
  MachineOperand *D16 = TII->getNamedOperand(MI, AMDGPU::OpName::d16);

  if (!TFE && !LWE) // intersect_ray
    return;

  unsigned TFEVal = TFE ? TFE->getImm() : 0;
  unsigned LWEVal = LWE->getImm();
  unsigned D16Val = D16 ? D16->getImm() : 0;

  if (!TFEVal && !LWEVal)
    return;

  // At least one of TFE or LWE are non-zero
  // We have to insert a suitable initialization of the result value and
  // tie this to the dest of the image instruction.

  const DebugLoc &DL = MI.getDebugLoc();

  int DstIdx =
      AMDGPU::getNamedOperandIdx(MI.getOpcode(), AMDGPU::OpName::vdata);

  // Calculate which dword we have to initialize to 0.
  MachineOperand *MO_Dmask = TII->getNamedOperand(MI, AMDGPU::OpName::dmask);

  // check that dmask operand is found.
  assert(MO_Dmask && "Expected dmask operand in instruction");

  unsigned dmask = MO_Dmask->getImm();
  // Determine the number of active lanes taking into account the
  // Gather4 special case
  unsigned ActiveLanes = TII->isGather4(MI) ? 4 : countPopulation(dmask);

  bool Packed = !Subtarget->hasUnpackedD16VMem();

  unsigned InitIdx =
      D16Val && Packed ? ((ActiveLanes + 1) >> 1) + 1 : ActiveLanes + 1;

  // Abandon attempt if the dst size isn't large enough
  // - this is in fact an error but this is picked up elsewhere and
  // reported correctly.
  uint32_t DstSize = TRI.getRegSizeInBits(*TII->getOpRegClass(MI, DstIdx)) / 32;
  if (DstSize < InitIdx)
    return;

  // Create a register for the intialization value.
  Register PrevDst = MRI.createVirtualRegister(TII->getOpRegClass(MI, DstIdx));
  unsigned NewDst = 0; // Final initialized value will be in here

  // If PRTStrictNull feature is enabled (the default) then initialize
  // all the result registers to 0, otherwise just the error indication
  // register (VGPRn+1)
  unsigned SizeLeft = Subtarget->usePRTStrictNull() ? InitIdx : 1;
  unsigned CurrIdx = Subtarget->usePRTStrictNull() ? 0 : (InitIdx - 1);

  BuildMI(MBB, MI, DL, TII->get(AMDGPU::IMPLICIT_DEF), PrevDst);
  for (; SizeLeft; SizeLeft--, CurrIdx++) {
    NewDst = MRI.createVirtualRegister(TII->getOpRegClass(MI, DstIdx));
    // Initialize dword
    Register SubReg = MRI.createVirtualRegister(&AMDGPU::VGPR_32RegClass);
    BuildMI(MBB, MI, DL, TII->get(AMDGPU::V_MOV_B32_e32), SubReg)
      .addImm(0);
    // Insert into the super-reg
    BuildMI(MBB, MI, DL, TII->get(TargetOpcode::INSERT_SUBREG), NewDst)
      .addReg(PrevDst)
      .addReg(SubReg)
      .addImm(SIRegisterInfo::getSubRegFromChannel(CurrIdx));

    PrevDst = NewDst;
  }

  // Add as an implicit operand
  MI.addOperand(MachineOperand::CreateReg(NewDst, false, true));

  // Tie the just added implicit operand to the dst
  MI.tieOperands(DstIdx, MI.getNumOperands() - 1);
}

/// Assign the register class depending on the number of
/// bits set in the writemask
void SITargetLowering::AdjustInstrPostInstrSelection(MachineInstr &MI,
                                                     SDNode *Node) const {
  const SIInstrInfo *TII = getSubtarget()->getInstrInfo();

  MachineRegisterInfo &MRI = MI.getParent()->getParent()->getRegInfo();

  if (TII->isVOP3(MI.getOpcode())) {
    // Make sure constant bus requirements are respected.
    TII->legalizeOperandsVOP3(MRI, MI);

    // Prefer VGPRs over AGPRs in mAI instructions where possible.
    // This saves a chain-copy of registers and better ballance register
    // use between vgpr and agpr as agpr tuples tend to be big.
    if (MI.getDesc().OpInfo) {
      unsigned Opc = MI.getOpcode();
      const SIRegisterInfo *TRI = Subtarget->getRegisterInfo();
      for (auto I : { AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::src0),
                      AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::src1) }) {
        if (I == -1)
          break;
        MachineOperand &Op = MI.getOperand(I);
        if (!Op.isReg() || !Op.getReg().isVirtual())
          continue;
        auto *RC = TRI->getRegClassForReg(MRI, Op.getReg());
        if (!TRI->hasAGPRs(RC))
          continue;
        auto *Src = MRI.getUniqueVRegDef(Op.getReg());
        if (!Src || !Src->isCopy() ||
            !TRI->isSGPRReg(MRI, Src->getOperand(1).getReg()))
          continue;
        auto *NewRC = TRI->getEquivalentVGPRClass(RC);
        // All uses of agpr64 and agpr32 can also accept vgpr except for
        // v_accvgpr_read, but we do not produce agpr reads during selection,
        // so no use checks are needed.
        MRI.setRegClass(Op.getReg(), NewRC);
      }
    }

    return;
  }

  // Replace unused atomics with the no return version.
  int NoRetAtomicOp = AMDGPU::getAtomicNoRetOp(MI.getOpcode());
  if (NoRetAtomicOp != -1) {
    if (!Node->hasAnyUseOfValue(0)) {
      int CPolIdx = AMDGPU::getNamedOperandIdx(MI.getOpcode(),
                                               AMDGPU::OpName::cpol);
      if (CPolIdx != -1) {
        MachineOperand &CPol = MI.getOperand(CPolIdx);
        CPol.setImm(CPol.getImm() & ~AMDGPU::CPol::GLC);
      }
      MI.RemoveOperand(0);
      MI.setDesc(TII->get(NoRetAtomicOp));
      return;
    }

    // For mubuf_atomic_cmpswap, we need to have tablegen use an extract_subreg
    // instruction, because the return type of these instructions is a vec2 of
    // the memory type, so it can be tied to the input operand.
    // This means these instructions always have a use, so we need to add a
    // special case to check if the atomic has only one extract_subreg use,
    // which itself has no uses.
    if ((Node->hasNUsesOfValue(1, 0) &&
         Node->use_begin()->isMachineOpcode() &&
         Node->use_begin()->getMachineOpcode() == AMDGPU::EXTRACT_SUBREG &&
         !Node->use_begin()->hasAnyUseOfValue(0))) {
      Register Def = MI.getOperand(0).getReg();

      // Change this into a noret atomic.
      MI.setDesc(TII->get(NoRetAtomicOp));
      MI.RemoveOperand(0);

      // If we only remove the def operand from the atomic instruction, the
      // extract_subreg will be left with a use of a vreg without a def.
      // So we need to insert an implicit_def to avoid machine verifier
      // errors.
      BuildMI(*MI.getParent(), MI, MI.getDebugLoc(),
              TII->get(AMDGPU::IMPLICIT_DEF), Def);
    }
    return;
  }

  if (TII->isMIMG(MI) && !MI.mayStore())
    AddIMGInit(MI);
}

static SDValue buildSMovImm32(SelectionDAG &DAG, const SDLoc &DL,
                              uint64_t Val) {
  SDValue K = DAG.getTargetConstant(Val, DL, MVT::i32);
  return SDValue(DAG.getMachineNode(AMDGPU::S_MOV_B32, DL, MVT::i32, K), 0);
}

MachineSDNode *SITargetLowering::wrapAddr64Rsrc(SelectionDAG &DAG,
                                                const SDLoc &DL,
                                                SDValue Ptr) const {
  const SIInstrInfo *TII = getSubtarget()->getInstrInfo();

  // Build the half of the subregister with the constants before building the
  // full 128-bit register. If we are building multiple resource descriptors,
  // this will allow CSEing of the 2-component register.
  const SDValue Ops0[] = {
    DAG.getTargetConstant(AMDGPU::SGPR_64RegClassID, DL, MVT::i32),
    buildSMovImm32(DAG, DL, 0),
    DAG.getTargetConstant(AMDGPU::sub0, DL, MVT::i32),
    buildSMovImm32(DAG, DL, TII->getDefaultRsrcDataFormat() >> 32),
    DAG.getTargetConstant(AMDGPU::sub1, DL, MVT::i32)
  };

  SDValue SubRegHi = SDValue(DAG.getMachineNode(AMDGPU::REG_SEQUENCE, DL,
                                                MVT::v2i32, Ops0), 0);

  // Combine the constants and the pointer.
  const SDValue Ops1[] = {
    DAG.getTargetConstant(AMDGPU::SGPR_128RegClassID, DL, MVT::i32),
    Ptr,
    DAG.getTargetConstant(AMDGPU::sub0_sub1, DL, MVT::i32),
    SubRegHi,
    DAG.getTargetConstant(AMDGPU::sub2_sub3, DL, MVT::i32)
  };

  return DAG.getMachineNode(AMDGPU::REG_SEQUENCE, DL, MVT::v4i32, Ops1);
}

/// Return a resource descriptor with the 'Add TID' bit enabled
///        The TID (Thread ID) is multiplied by the stride value (bits [61:48]
///        of the resource descriptor) to create an offset, which is added to
///        the resource pointer.
MachineSDNode *SITargetLowering::buildRSRC(SelectionDAG &DAG, const SDLoc &DL,
                                           SDValue Ptr, uint32_t RsrcDword1,
                                           uint64_t RsrcDword2And3) const {
  SDValue PtrLo = DAG.getTargetExtractSubreg(AMDGPU::sub0, DL, MVT::i32, Ptr);
  SDValue PtrHi = DAG.getTargetExtractSubreg(AMDGPU::sub1, DL, MVT::i32, Ptr);
  if (RsrcDword1) {
    PtrHi = SDValue(DAG.getMachineNode(AMDGPU::S_OR_B32, DL, MVT::i32, PtrHi,
                                     DAG.getConstant(RsrcDword1, DL, MVT::i32)),
                    0);
  }

  SDValue DataLo = buildSMovImm32(DAG, DL,
                                  RsrcDword2And3 & UINT64_C(0xFFFFFFFF));
  SDValue DataHi = buildSMovImm32(DAG, DL, RsrcDword2And3 >> 32);

  const SDValue Ops[] = {
    DAG.getTargetConstant(AMDGPU::SGPR_128RegClassID, DL, MVT::i32),
    PtrLo,
    DAG.getTargetConstant(AMDGPU::sub0, DL, MVT::i32),
    PtrHi,
    DAG.getTargetConstant(AMDGPU::sub1, DL, MVT::i32),
    DataLo,
    DAG.getTargetConstant(AMDGPU::sub2, DL, MVT::i32),
    DataHi,
    DAG.getTargetConstant(AMDGPU::sub3, DL, MVT::i32)
  };

  return DAG.getMachineNode(AMDGPU::REG_SEQUENCE, DL, MVT::v4i32, Ops);
}

//===----------------------------------------------------------------------===//
//                         SI Inline Assembly Support
//===----------------------------------------------------------------------===//

std::pair<unsigned, const TargetRegisterClass *>
SITargetLowering::getRegForInlineAsmConstraint(const TargetRegisterInfo *TRI_,
                                               StringRef Constraint,
                                               MVT VT) const {
  const SIRegisterInfo *TRI = static_cast<const SIRegisterInfo *>(TRI_);

  const TargetRegisterClass *RC = nullptr;
  if (Constraint.size() == 1) {
    const unsigned BitWidth = VT.getSizeInBits();
    switch (Constraint[0]) {
    default:
      return TargetLowering::getRegForInlineAsmConstraint(TRI, Constraint, VT);
    case 's':
    case 'r':
      switch (BitWidth) {
      case 16:
        RC = &AMDGPU::SReg_32RegClass;
        break;
      case 64:
        RC = &AMDGPU::SGPR_64RegClass;
        break;
      default:
        RC = SIRegisterInfo::getSGPRClassForBitWidth(BitWidth);
        if (!RC)
          return std::make_pair(0U, nullptr);
        break;
      }
      break;
    case 'v':
      switch (BitWidth) {
      case 16:
        RC = &AMDGPU::VGPR_32RegClass;
        break;
      default:
        RC = TRI->getVGPRClassForBitWidth(BitWidth);
        if (!RC)
          return std::make_pair(0U, nullptr);
        break;
      }
      break;
    case 'a':
      if (!Subtarget->hasMAIInsts())
        break;
      switch (BitWidth) {
      case 16:
        RC = &AMDGPU::AGPR_32RegClass;
        break;
      default:
        RC = TRI->getAGPRClassForBitWidth(BitWidth);
        if (!RC)
          return std::make_pair(0U, nullptr);
        break;
      }
      break;
    }
    // We actually support i128, i16 and f16 as inline parameters
    // even if they are not reported as legal
    if (RC && (isTypeLegal(VT) || VT.SimpleTy == MVT::i128 ||
               VT.SimpleTy == MVT::i16 || VT.SimpleTy == MVT::f16))
      return std::make_pair(0U, RC);
  }

  if (Constraint.startswith("{") && Constraint.endswith("}")) {
    StringRef RegName(Constraint.data() + 1, Constraint.size() - 2);
    if (RegName.consume_front("v")) {
      RC = &AMDGPU::VGPR_32RegClass;
    } else if (RegName.consume_front("s")) {
      RC = &AMDGPU::SGPR_32RegClass;
    } else if (RegName.consume_front("a")) {
      RC = &AMDGPU::AGPR_32RegClass;
    }

    if (RC) {
      uint32_t Idx;
      if (RegName.consume_front("[")) {
        uint32_t End;
        bool Failed = RegName.consumeInteger(10, Idx);
        Failed |= !RegName.consume_front(":");
        Failed |= RegName.consumeInteger(10, End);
        Failed |= !RegName.consume_back("]");
        if (!Failed) {
          uint32_t Width = (End - Idx + 1) * 32;
          MCRegister Reg = RC->getRegister(Idx);
          if (SIRegisterInfo::isVGPRClass(RC))
            RC = TRI->getVGPRClassForBitWidth(Width);
          else if (SIRegisterInfo::isSGPRClass(RC))
            RC = TRI->getSGPRClassForBitWidth(Width);
          else if (SIRegisterInfo::isAGPRClass(RC))
            RC = TRI->getAGPRClassForBitWidth(Width);
          if (RC) {
            Reg = TRI->getMatchingSuperReg(Reg, AMDGPU::sub0, RC);
            return std::make_pair(Reg, RC);
          }
        }
      } else {
        bool Failed = RegName.getAsInteger(10, Idx);
        if (!Failed && Idx < RC->getNumRegs())
          return std::make_pair(RC->getRegister(Idx), RC);
      }
    }
  }

  auto Ret = TargetLowering::getRegForInlineAsmConstraint(TRI, Constraint, VT);
  if (Ret.first)
    Ret.second = TRI->getPhysRegClass(Ret.first);

  return Ret;
}

static bool isImmConstraint(StringRef Constraint) {
  if (Constraint.size() == 1) {
    switch (Constraint[0]) {
    default: break;
    case 'I':
    case 'J':
    case 'A':
    case 'B':
    case 'C':
      return true;
    }
  } else if (Constraint == "DA" ||
             Constraint == "DB") {
    return true;
  }
  return false;
}

SITargetLowering::ConstraintType
SITargetLowering::getConstraintType(StringRef Constraint) const {
  if (Constraint.size() == 1) {
    switch (Constraint[0]) {
    default: break;
    case 's':
    case 'v':
    case 'a':
      return C_RegisterClass;
    }
  }
  if (isImmConstraint(Constraint)) {
    return C_Other;
  }
  return TargetLowering::getConstraintType(Constraint);
}

static uint64_t clearUnusedBits(uint64_t Val, unsigned Size) {
  if (!AMDGPU::isInlinableIntLiteral(Val)) {
    Val = Val & maskTrailingOnes<uint64_t>(Size);
  }
  return Val;
}

void SITargetLowering::LowerAsmOperandForConstraint(SDValue Op,
                                                    std::string &Constraint,
                                                    std::vector<SDValue> &Ops,
                                                    SelectionDAG &DAG) const {
  if (isImmConstraint(Constraint)) {
    uint64_t Val;
    if (getAsmOperandConstVal(Op, Val) &&
        checkAsmConstraintVal(Op, Constraint, Val)) {
      Val = clearUnusedBits(Val, Op.getScalarValueSizeInBits());
      Ops.push_back(DAG.getTargetConstant(Val, SDLoc(Op), MVT::i64));
    }
  } else {
    TargetLowering::LowerAsmOperandForConstraint(Op, Constraint, Ops, DAG);
  }
}

bool SITargetLowering::getAsmOperandConstVal(SDValue Op, uint64_t &Val) const {
  unsigned Size = Op.getScalarValueSizeInBits();
  if (Size > 64)
    return false;

  if (Size == 16 && !Subtarget->has16BitInsts())
    return false;

  if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op)) {
    Val = C->getSExtValue();
    return true;
  }
  if (ConstantFPSDNode *C = dyn_cast<ConstantFPSDNode>(Op)) {
    Val = C->getValueAPF().bitcastToAPInt().getSExtValue();
    return true;
  }
  if (BuildVectorSDNode *V = dyn_cast<BuildVectorSDNode>(Op)) {
    if (Size != 16 || Op.getNumOperands() != 2)
      return false;
    if (Op.getOperand(0).isUndef() || Op.getOperand(1).isUndef())
      return false;
    if (ConstantSDNode *C = V->getConstantSplatNode()) {
      Val = C->getSExtValue();
      return true;
    }
    if (ConstantFPSDNode *C = V->getConstantFPSplatNode()) {
      Val = C->getValueAPF().bitcastToAPInt().getSExtValue();
      return true;
    }
  }

  return false;
}

bool SITargetLowering::checkAsmConstraintVal(SDValue Op,
                                             const std::string &Constraint,
                                             uint64_t Val) const {
  if (Constraint.size() == 1) {
    switch (Constraint[0]) {
    case 'I':
      return AMDGPU::isInlinableIntLiteral(Val);
    case 'J':
      return isInt<16>(Val);
    case 'A':
      return checkAsmConstraintValA(Op, Val);
    case 'B':
      return isInt<32>(Val);
    case 'C':
      return isUInt<32>(clearUnusedBits(Val, Op.getScalarValueSizeInBits())) ||
             AMDGPU::isInlinableIntLiteral(Val);
    default:
      break;
    }
  } else if (Constraint.size() == 2) {
    if (Constraint == "DA") {
      int64_t HiBits = static_cast<int32_t>(Val >> 32);
      int64_t LoBits = static_cast<int32_t>(Val);
      return checkAsmConstraintValA(Op, HiBits, 32) &&
             checkAsmConstraintValA(Op, LoBits, 32);
    }
    if (Constraint == "DB") {
      return true;
    }
  }
  llvm_unreachable("Invalid asm constraint");
}

bool SITargetLowering::checkAsmConstraintValA(SDValue Op,
                                              uint64_t Val,
                                              unsigned MaxSize) const {
  unsigned Size = std::min<unsigned>(Op.getScalarValueSizeInBits(), MaxSize);
  bool HasInv2Pi = Subtarget->hasInv2PiInlineImm();
  if ((Size == 16 && AMDGPU::isInlinableLiteral16(Val, HasInv2Pi)) ||
      (Size == 32 && AMDGPU::isInlinableLiteral32(Val, HasInv2Pi)) ||
      (Size == 64 && AMDGPU::isInlinableLiteral64(Val, HasInv2Pi))) {
    return true;
  }
  return false;
}

static int getAlignedAGPRClassID(unsigned UnalignedClassID) {
  switch (UnalignedClassID) {
  case AMDGPU::VReg_64RegClassID:
    return AMDGPU::VReg_64_Align2RegClassID;
  case AMDGPU::VReg_96RegClassID:
    return AMDGPU::VReg_96_Align2RegClassID;
  case AMDGPU::VReg_128RegClassID:
    return AMDGPU::VReg_128_Align2RegClassID;
  case AMDGPU::VReg_160RegClassID:
    return AMDGPU::VReg_160_Align2RegClassID;
  case AMDGPU::VReg_192RegClassID:
    return AMDGPU::VReg_192_Align2RegClassID;
  case AMDGPU::VReg_224RegClassID:
    return AMDGPU::VReg_224_Align2RegClassID;
  case AMDGPU::VReg_256RegClassID:
    return AMDGPU::VReg_256_Align2RegClassID;
  case AMDGPU::VReg_512RegClassID:
    return AMDGPU::VReg_512_Align2RegClassID;
  case AMDGPU::VReg_1024RegClassID:
    return AMDGPU::VReg_1024_Align2RegClassID;
  case AMDGPU::AReg_64RegClassID:
    return AMDGPU::AReg_64_Align2RegClassID;
  case AMDGPU::AReg_96RegClassID:
    return AMDGPU::AReg_96_Align2RegClassID;
  case AMDGPU::AReg_128RegClassID:
    return AMDGPU::AReg_128_Align2RegClassID;
  case AMDGPU::AReg_160RegClassID:
    return AMDGPU::AReg_160_Align2RegClassID;
  case AMDGPU::AReg_192RegClassID:
    return AMDGPU::AReg_192_Align2RegClassID;
  case AMDGPU::AReg_256RegClassID:
    return AMDGPU::AReg_256_Align2RegClassID;
  case AMDGPU::AReg_512RegClassID:
    return AMDGPU::AReg_512_Align2RegClassID;
  case AMDGPU::AReg_1024RegClassID:
    return AMDGPU::AReg_1024_Align2RegClassID;
  default:
    return -1;
  }
}

// Figure out which registers should be reserved for stack access. Only after
// the function is legalized do we know all of the non-spill stack objects or if
// calls are present.
void SITargetLowering::finalizeLowering(MachineFunction &MF) const {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  SIMachineFunctionInfo *Info = MF.getInfo<SIMachineFunctionInfo>();
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  const SIRegisterInfo *TRI = Subtarget->getRegisterInfo();
  const SIInstrInfo *TII = ST.getInstrInfo();

  if (Info->isEntryFunction()) {
    // Callable functions have fixed registers used for stack access.
    reservePrivateMemoryRegs(getTargetMachine(), MF, *TRI, *Info);
  }

  assert(!TRI->isSubRegister(Info->getScratchRSrcReg(),
                             Info->getStackPtrOffsetReg()));
  if (Info->getStackPtrOffsetReg() != AMDGPU::SP_REG)
    MRI.replaceRegWith(AMDGPU::SP_REG, Info->getStackPtrOffsetReg());

  // We need to worry about replacing the default register with itself in case
  // of MIR testcases missing the MFI.
  if (Info->getScratchRSrcReg() != AMDGPU::PRIVATE_RSRC_REG)
    MRI.replaceRegWith(AMDGPU::PRIVATE_RSRC_REG, Info->getScratchRSrcReg());

  if (Info->getFrameOffsetReg() != AMDGPU::FP_REG)
    MRI.replaceRegWith(AMDGPU::FP_REG, Info->getFrameOffsetReg());

  Info->limitOccupancy(MF);

  if (ST.isWave32() && !MF.empty()) {
    for (auto &MBB : MF) {
      for (auto &MI : MBB) {
        TII->fixImplicitOperands(MI);
      }
    }
  }

  // FIXME: This is a hack to fixup AGPR classes to use the properly aligned
  // classes if required. Ideally the register class constraints would differ
  // per-subtarget, but there's no easy way to achieve that right now. This is
  // not a problem for VGPRs because the correctly aligned VGPR class is implied
  // from using them as the register class for legal types.
  if (ST.needsAlignedVGPRs()) {
    for (unsigned I = 0, E = MRI.getNumVirtRegs(); I != E; ++I) {
      const Register Reg = Register::index2VirtReg(I);
      const TargetRegisterClass *RC = MRI.getRegClassOrNull(Reg);
      if (!RC)
        continue;
      int NewClassID = getAlignedAGPRClassID(RC->getID());
      if (NewClassID != -1)
        MRI.setRegClass(Reg, TRI->getRegClass(NewClassID));
    }
  }

  TargetLoweringBase::finalizeLowering(MF);
}

void SITargetLowering::computeKnownBitsForFrameIndex(
  const int FI, KnownBits &Known, const MachineFunction &MF) const {
  TargetLowering::computeKnownBitsForFrameIndex(FI, Known, MF);

  // Set the high bits to zero based on the maximum allowed scratch size per
  // wave. We can't use vaddr in MUBUF instructions if we don't know the address
  // calculation won't overflow, so assume the sign bit is never set.
  Known.Zero.setHighBits(getSubtarget()->getKnownHighZeroBitsForFrameIndex());
}

static void knownBitsForWorkitemID(const GCNSubtarget &ST, GISelKnownBits &KB,
                                   KnownBits &Known, unsigned Dim) {
  unsigned MaxValue =
      ST.getMaxWorkitemID(KB.getMachineFunction().getFunction(), Dim);
  Known.Zero.setHighBits(countLeadingZeros(MaxValue));
}

void SITargetLowering::computeKnownBitsForTargetInstr(
    GISelKnownBits &KB, Register R, KnownBits &Known, const APInt &DemandedElts,
    const MachineRegisterInfo &MRI, unsigned Depth) const {
  const MachineInstr *MI = MRI.getVRegDef(R);
  switch (MI->getOpcode()) {
  case AMDGPU::G_INTRINSIC: {
    switch (MI->getIntrinsicID()) {
    case Intrinsic::amdgcn_workitem_id_x:
      knownBitsForWorkitemID(*getSubtarget(), KB, Known, 0);
      break;
    case Intrinsic::amdgcn_workitem_id_y:
      knownBitsForWorkitemID(*getSubtarget(), KB, Known, 1);
      break;
    case Intrinsic::amdgcn_workitem_id_z:
      knownBitsForWorkitemID(*getSubtarget(), KB, Known, 2);
      break;
    case Intrinsic::amdgcn_mbcnt_lo:
    case Intrinsic::amdgcn_mbcnt_hi: {
      // These return at most the wavefront size - 1.
      unsigned Size = MRI.getType(R).getSizeInBits();
      Known.Zero.setHighBits(Size - getSubtarget()->getWavefrontSizeLog2());
      break;
    }
    case Intrinsic::amdgcn_groupstaticsize: {
      // We can report everything over the maximum size as 0. We can't report
      // based on the actual size because we don't know if it's accurate or not
      // at any given point.
      Known.Zero.setHighBits(countLeadingZeros(getSubtarget()->getLocalMemorySize()));
      break;
    }
    }
    break;
  }
  case AMDGPU::G_AMDGPU_BUFFER_LOAD_UBYTE:
    Known.Zero.setHighBits(24);
    break;
  case AMDGPU::G_AMDGPU_BUFFER_LOAD_USHORT:
    Known.Zero.setHighBits(16);
    break;
  }
}

Align SITargetLowering::computeKnownAlignForTargetInstr(
  GISelKnownBits &KB, Register R, const MachineRegisterInfo &MRI,
  unsigned Depth) const {
  const MachineInstr *MI = MRI.getVRegDef(R);
  switch (MI->getOpcode()) {
  case AMDGPU::G_INTRINSIC:
  case AMDGPU::G_INTRINSIC_W_SIDE_EFFECTS: {
    // FIXME: Can this move to generic code? What about the case where the call
    // site specifies a lower alignment?
    Intrinsic::ID IID = MI->getIntrinsicID();
    LLVMContext &Ctx = KB.getMachineFunction().getFunction().getContext();
    AttributeList Attrs = Intrinsic::getAttributes(Ctx, IID);
    if (MaybeAlign RetAlign = Attrs.getRetAlignment())
      return *RetAlign;
    return Align(1);
  }
  default:
    return Align(1);
  }
}

Align SITargetLowering::getPrefLoopAlignment(MachineLoop *ML) const {
  const Align PrefAlign = TargetLowering::getPrefLoopAlignment(ML);
  const Align CacheLineAlign = Align(64);

  // Pre-GFX10 target did not benefit from loop alignment
  if (!ML || DisableLoopAlignment ||
      (getSubtarget()->getGeneration() < AMDGPUSubtarget::GFX10) ||
      getSubtarget()->hasInstFwdPrefetchBug())
    return PrefAlign;

  // On GFX10 I$ is 4 x 64 bytes cache lines.
  // By default prefetcher keeps one cache line behind and reads two ahead.
  // We can modify it with S_INST_PREFETCH for larger loops to have two lines
  // behind and one ahead.
  // Therefor we can benefit from aligning loop headers if loop fits 192 bytes.
  // If loop fits 64 bytes it always spans no more than two cache lines and
  // does not need an alignment.
  // Else if loop is less or equal 128 bytes we do not need to modify prefetch,
  // Else if loop is less or equal 192 bytes we need two lines behind.

  const SIInstrInfo *TII = getSubtarget()->getInstrInfo();
  const MachineBasicBlock *Header = ML->getHeader();
  if (Header->getAlignment() != PrefAlign)
    return Header->getAlignment(); // Already processed.

  unsigned LoopSize = 0;
  for (const MachineBasicBlock *MBB : ML->blocks()) {
    // If inner loop block is aligned assume in average half of the alignment
    // size to be added as nops.
    if (MBB != Header)
      LoopSize += MBB->getAlignment().value() / 2;

    for (const MachineInstr &MI : *MBB) {
      LoopSize += TII->getInstSizeInBytes(MI);
      if (LoopSize > 192)
        return PrefAlign;
    }
  }

  if (LoopSize <= 64)
    return PrefAlign;

  if (LoopSize <= 128)
    return CacheLineAlign;

  // If any of parent loops is surrounded by prefetch instructions do not
  // insert new for inner loop, which would reset parent's settings.
  for (MachineLoop *P = ML->getParentLoop(); P; P = P->getParentLoop()) {
    if (MachineBasicBlock *Exit = P->getExitBlock()) {
      auto I = Exit->getFirstNonDebugInstr();
      if (I != Exit->end() && I->getOpcode() == AMDGPU::S_INST_PREFETCH)
        return CacheLineAlign;
    }
  }

  MachineBasicBlock *Pre = ML->getLoopPreheader();
  MachineBasicBlock *Exit = ML->getExitBlock();

  if (Pre && Exit) {
    BuildMI(*Pre, Pre->getFirstTerminator(), DebugLoc(),
            TII->get(AMDGPU::S_INST_PREFETCH))
      .addImm(1); // prefetch 2 lines behind PC

    BuildMI(*Exit, Exit->getFirstNonDebugInstr(), DebugLoc(),
            TII->get(AMDGPU::S_INST_PREFETCH))
      .addImm(2); // prefetch 1 line behind PC
  }

  return CacheLineAlign;
}

LLVM_ATTRIBUTE_UNUSED
static bool isCopyFromRegOfInlineAsm(const SDNode *N) {
  assert(N->getOpcode() == ISD::CopyFromReg);
  do {
    // Follow the chain until we find an INLINEASM node.
    N = N->getOperand(0).getNode();
    if (N->getOpcode() == ISD::INLINEASM ||
        N->getOpcode() == ISD::INLINEASM_BR)
      return true;
  } while (N->getOpcode() == ISD::CopyFromReg);
  return false;
}

bool SITargetLowering::isSDNodeSourceOfDivergence(
    const SDNode *N, FunctionLoweringInfo *FLI,
    LegacyDivergenceAnalysis *KDA) const {
  switch (N->getOpcode()) {
  case ISD::CopyFromReg: {
    const RegisterSDNode *R = cast<RegisterSDNode>(N->getOperand(1));
    const MachineRegisterInfo &MRI = FLI->MF->getRegInfo();
    const SIRegisterInfo *TRI = Subtarget->getRegisterInfo();
    Register Reg = R->getReg();

    // FIXME: Why does this need to consider isLiveIn?
    if (Reg.isPhysical() || MRI.isLiveIn(Reg))
      return !TRI->isSGPRReg(MRI, Reg);

    if (const Value *V = FLI->getValueFromVirtualReg(R->getReg()))
      return KDA->isDivergent(V);

    assert(Reg == FLI->DemoteRegister || isCopyFromRegOfInlineAsm(N));
    return !TRI->isSGPRReg(MRI, Reg);
  }
  case ISD::LOAD: {
    const LoadSDNode *L = cast<LoadSDNode>(N);
    unsigned AS = L->getAddressSpace();
    // A flat load may access private memory.
    return AS == AMDGPUAS::PRIVATE_ADDRESS || AS == AMDGPUAS::FLAT_ADDRESS;
  }
  case ISD::CALLSEQ_END:
    return true;
  case ISD::INTRINSIC_WO_CHAIN:
    return AMDGPU::isIntrinsicSourceOfDivergence(
        cast<ConstantSDNode>(N->getOperand(0))->getZExtValue());
  case ISD::INTRINSIC_W_CHAIN:
    return AMDGPU::isIntrinsicSourceOfDivergence(
        cast<ConstantSDNode>(N->getOperand(1))->getZExtValue());
  case AMDGPUISD::ATOMIC_CMP_SWAP:
  case AMDGPUISD::ATOMIC_INC:
  case AMDGPUISD::ATOMIC_DEC:
  case AMDGPUISD::ATOMIC_LOAD_FMIN:
  case AMDGPUISD::ATOMIC_LOAD_FMAX:
  case AMDGPUISD::BUFFER_ATOMIC_SWAP:
  case AMDGPUISD::BUFFER_ATOMIC_ADD:
  case AMDGPUISD::BUFFER_ATOMIC_SUB:
  case AMDGPUISD::BUFFER_ATOMIC_SMIN:
  case AMDGPUISD::BUFFER_ATOMIC_UMIN:
  case AMDGPUISD::BUFFER_ATOMIC_SMAX:
  case AMDGPUISD::BUFFER_ATOMIC_UMAX:
  case AMDGPUISD::BUFFER_ATOMIC_AND:
  case AMDGPUISD::BUFFER_ATOMIC_OR:
  case AMDGPUISD::BUFFER_ATOMIC_XOR:
  case AMDGPUISD::BUFFER_ATOMIC_INC:
  case AMDGPUISD::BUFFER_ATOMIC_DEC:
  case AMDGPUISD::BUFFER_ATOMIC_CMPSWAP:
  case AMDGPUISD::BUFFER_ATOMIC_CSUB:
  case AMDGPUISD::BUFFER_ATOMIC_FADD:
  case AMDGPUISD::BUFFER_ATOMIC_FMIN:
  case AMDGPUISD::BUFFER_ATOMIC_FMAX:
    // Target-specific read-modify-write atomics are sources of divergence.
    return true;
  default:
    if (auto *A = dyn_cast<AtomicSDNode>(N)) {
      // Generic read-modify-write atomics are sources of divergence.
      return A->readMem() && A->writeMem();
    }
    return false;
  }
}

bool SITargetLowering::denormalsEnabledForType(const SelectionDAG &DAG,
                                               EVT VT) const {
  switch (VT.getScalarType().getSimpleVT().SimpleTy) {
  case MVT::f32:
    return hasFP32Denormals(DAG.getMachineFunction());
  case MVT::f64:
  case MVT::f16:
    return hasFP64FP16Denormals(DAG.getMachineFunction());
  default:
    return false;
  }
}

bool SITargetLowering::denormalsEnabledForType(LLT Ty,
                                               MachineFunction &MF) const {
  switch (Ty.getScalarSizeInBits()) {
  case 32:
    return hasFP32Denormals(MF);
  case 64:
  case 16:
    return hasFP64FP16Denormals(MF);
  default:
    return false;
  }
}

bool SITargetLowering::isKnownNeverNaNForTargetNode(SDValue Op,
                                                    const SelectionDAG &DAG,
                                                    bool SNaN,
                                                    unsigned Depth) const {
  if (Op.getOpcode() == AMDGPUISD::CLAMP) {
    const MachineFunction &MF = DAG.getMachineFunction();
    const SIMachineFunctionInfo *Info = MF.getInfo<SIMachineFunctionInfo>();

    if (Info->getMode().DX10Clamp)
      return true; // Clamped to 0.
    return DAG.isKnownNeverNaN(Op.getOperand(0), SNaN, Depth + 1);
  }

  return AMDGPUTargetLowering::isKnownNeverNaNForTargetNode(Op, DAG,
                                                            SNaN, Depth);
}

// Global FP atomic instructions have a hardcoded FP mode and do not support
// FP32 denormals, and only support v2f16 denormals.
static bool fpModeMatchesGlobalFPAtomicMode(const AtomicRMWInst *RMW) {
  const fltSemantics &Flt = RMW->getType()->getScalarType()->getFltSemantics();
  auto DenormMode = RMW->getParent()->getParent()->getDenormalMode(Flt);
  if (&Flt == &APFloat::IEEEsingle())
    return DenormMode == DenormalMode::getPreserveSign();
  return DenormMode == DenormalMode::getIEEE();
}

TargetLowering::AtomicExpansionKind
SITargetLowering::shouldExpandAtomicRMWInIR(AtomicRMWInst *RMW) const {

  auto ReportUnsafeHWInst = [&](TargetLowering::AtomicExpansionKind Kind) {
    OptimizationRemarkEmitter ORE(RMW->getFunction());
    LLVMContext &Ctx = RMW->getFunction()->getContext();
    SmallVector<StringRef> SSNs;
    Ctx.getSyncScopeNames(SSNs);
    auto MemScope = SSNs[RMW->getSyncScopeID()].empty()
                        ? "system"
                        : SSNs[RMW->getSyncScopeID()];
    ORE.emit([&]() {
      return OptimizationRemark(DEBUG_TYPE, "Passed", RMW)
             << "Hardware instruction generated for atomic "
             << RMW->getOperationName(RMW->getOperation())
             << " operation at memory scope " << MemScope
             << " due to an unsafe request.";
    });
    return Kind;
  };

  switch (RMW->getOperation()) {
  case AtomicRMWInst::FAdd: {
    Type *Ty = RMW->getType();

    // We don't have a way to support 16-bit atomics now, so just leave them
    // as-is.
    if (Ty->isHalfTy())
      return AtomicExpansionKind::None;

    if (!Ty->isFloatTy() && (!Subtarget->hasGFX90AInsts() || !Ty->isDoubleTy()))
      return AtomicExpansionKind::CmpXChg;

    unsigned AS = RMW->getPointerAddressSpace();

    if ((AS == AMDGPUAS::GLOBAL_ADDRESS || AS == AMDGPUAS::FLAT_ADDRESS) &&
         Subtarget->hasAtomicFaddInsts()) {
      // The amdgpu-unsafe-fp-atomics attribute enables generation of unsafe
      // floating point atomic instructions. May generate more efficient code,
      // but may not respect rounding and denormal modes, and may give incorrect
      // results for certain memory destinations.
      if (RMW->getFunction()
              ->getFnAttribute("amdgpu-unsafe-fp-atomics")
              .getValueAsString() != "true")
        return AtomicExpansionKind::CmpXChg;

      if (Subtarget->hasGFX90AInsts()) {
        if (Ty->isFloatTy() && AS == AMDGPUAS::FLAT_ADDRESS)
          return AtomicExpansionKind::CmpXChg;

        auto SSID = RMW->getSyncScopeID();
        if (SSID == SyncScope::System ||
            SSID == RMW->getContext().getOrInsertSyncScopeID("one-as"))
          return AtomicExpansionKind::CmpXChg;

        return ReportUnsafeHWInst(AtomicExpansionKind::None);
      }

      if (AS == AMDGPUAS::FLAT_ADDRESS)
        return AtomicExpansionKind::CmpXChg;

      return RMW->use_empty() ? ReportUnsafeHWInst(AtomicExpansionKind::None)
                              : AtomicExpansionKind::CmpXChg;
    }

    // DS FP atomics do repect the denormal mode, but the rounding mode is fixed
    // to round-to-nearest-even.
    // The only exception is DS_ADD_F64 which never flushes regardless of mode.
    if (AS == AMDGPUAS::LOCAL_ADDRESS && Subtarget->hasLDSFPAtomicAdd()) {
      if (!Ty->isDoubleTy())
        return AtomicExpansionKind::None;

      if (fpModeMatchesGlobalFPAtomicMode(RMW))
        return AtomicExpansionKind::None;

      return RMW->getFunction()
                         ->getFnAttribute("amdgpu-unsafe-fp-atomics")
                         .getValueAsString() == "true"
                 ? ReportUnsafeHWInst(AtomicExpansionKind::None)
                 : AtomicExpansionKind::CmpXChg;
    }

    return AtomicExpansionKind::CmpXChg;
  }
  default:
    break;
  }

  return AMDGPUTargetLowering::shouldExpandAtomicRMWInIR(RMW);
}

const TargetRegisterClass *
SITargetLowering::getRegClassFor(MVT VT, bool isDivergent) const {
  const TargetRegisterClass *RC = TargetLoweringBase::getRegClassFor(VT, false);
  const SIRegisterInfo *TRI = Subtarget->getRegisterInfo();
  if (RC == &AMDGPU::VReg_1RegClass && !isDivergent)
    return Subtarget->getWavefrontSize() == 64 ? &AMDGPU::SReg_64RegClass
                                               : &AMDGPU::SReg_32RegClass;
  if (!TRI->isSGPRClass(RC) && !isDivergent)
    return TRI->getEquivalentSGPRClass(RC);
  else if (TRI->isSGPRClass(RC) && isDivergent)
    return TRI->getEquivalentVGPRClass(RC);

  return RC;
}

// FIXME: This is a workaround for DivergenceAnalysis not understanding always
// uniform values (as produced by the mask results of control flow intrinsics)
// used outside of divergent blocks. The phi users need to also be treated as
// always uniform.
static bool hasCFUser(const Value *V, SmallPtrSet<const Value *, 16> &Visited,
                      unsigned WaveSize) {
  // FIXME: We asssume we never cast the mask results of a control flow
  // intrinsic.
  // Early exit if the type won't be consistent as a compile time hack.
  IntegerType *IT = dyn_cast<IntegerType>(V->getType());
  if (!IT || IT->getBitWidth() != WaveSize)
    return false;

  if (!isa<Instruction>(V))
    return false;
  if (!Visited.insert(V).second)
    return false;
  bool Result = false;
  for (auto U : V->users()) {
    if (const IntrinsicInst *Intrinsic = dyn_cast<IntrinsicInst>(U)) {
      if (V == U->getOperand(1)) {
        switch (Intrinsic->getIntrinsicID()) {
        default:
          Result = false;
          break;
        case Intrinsic::amdgcn_if_break:
        case Intrinsic::amdgcn_if:
        case Intrinsic::amdgcn_else:
          Result = true;
          break;
        }
      }
      if (V == U->getOperand(0)) {
        switch (Intrinsic->getIntrinsicID()) {
        default:
          Result = false;
          break;
        case Intrinsic::amdgcn_end_cf:
        case Intrinsic::amdgcn_loop:
          Result = true;
          break;
        }
      }
    } else {
      Result = hasCFUser(U, Visited, WaveSize);
    }
    if (Result)
      break;
  }
  return Result;
}

bool SITargetLowering::requiresUniformRegister(MachineFunction &MF,
                                               const Value *V) const {
  if (const CallInst *CI = dyn_cast<CallInst>(V)) {
    if (CI->isInlineAsm()) {
      // FIXME: This cannot give a correct answer. This should only trigger in
      // the case where inline asm returns mixed SGPR and VGPR results, used
      // outside the defining block. We don't have a specific result to
      // consider, so this assumes if any value is SGPR, the overall register
      // also needs to be SGPR.
      const SIRegisterInfo *SIRI = Subtarget->getRegisterInfo();
      TargetLowering::AsmOperandInfoVector TargetConstraints = ParseConstraints(
          MF.getDataLayout(), Subtarget->getRegisterInfo(), *CI);
      for (auto &TC : TargetConstraints) {
        if (TC.Type == InlineAsm::isOutput) {
          ComputeConstraintToUse(TC, SDValue());
          const TargetRegisterClass *RC = getRegForInlineAsmConstraint(
              SIRI, TC.ConstraintCode, TC.ConstraintVT).second;
          if (RC && SIRI->isSGPRClass(RC))
            return true;
        }
      }
    }
  }
  SmallPtrSet<const Value *, 16> Visited;
  return hasCFUser(V, Visited, Subtarget->getWavefrontSize());
}

std::pair<InstructionCost, MVT>
SITargetLowering::getTypeLegalizationCost(const DataLayout &DL,
                                          Type *Ty) const {
  std::pair<InstructionCost, MVT> Cost =
      TargetLoweringBase::getTypeLegalizationCost(DL, Ty);
  auto Size = DL.getTypeSizeInBits(Ty);
  // Maximum load or store can handle 8 dwords for scalar and 4 for
  // vector ALU. Let's assume anything above 8 dwords is expensive
  // even if legal.
  if (Size <= 256)
    return Cost;

  Cost.first += (Size + 255) / 256;
  return Cost;
}

bool SITargetLowering::hasMemSDNodeUser(SDNode *N) const {
  SDNode::use_iterator I = N->use_begin(), E = N->use_end();
  for (; I != E; ++I) {
    if (MemSDNode *M = dyn_cast<MemSDNode>(*I)) {
      if (getBasePtrIndex(M) == I.getOperandNo())
        return true;
    }
  }
  return false;
}

bool SITargetLowering::isReassocProfitable(SelectionDAG &DAG, SDValue N0,
                                           SDValue N1) const {
  if (!N0.hasOneUse())
    return false;
  // Take care of the oportunity to keep N0 uniform
  if (N0->isDivergent() || !N1->isDivergent())
    return true;
  // Check if we have a good chance to form the memory access pattern with the
  // base and offset
  return (DAG.isBaseWithConstantOffset(N0) &&
          hasMemSDNodeUser(*N0->use_begin()));
}
