//===-- AMDILVersion.td - Barrier Instruction/Intrinsic definitions------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===--------------------------------------------------------------------===//
// Intrinsic operation support
//===--------------------------------------------------------------------===//
let TargetPrefix = "AMDIL", isTarget = 1 in {
def int_AMDIL_barrier   : GCCBuiltin<"barrier">,
        BinaryIntNoRetInt;
def int_AMDIL_barrier_global   : GCCBuiltin<"barrierGlobal">,
      BinaryIntNoRetInt;
def int_AMDIL_barrier_local   : GCCBuiltin<"barrierLocal">,
      BinaryIntNoRetInt;
def int_AMDIL_barrier_region   : GCCBuiltin<"barrierRegion">,
      BinaryIntNoRetInt;
def int_AMDIL_get_region_id : GCCBuiltin<"__amdil_get_region_id_int">,
    Intrinsic<[llvm_v4i32_ty], [], []>;
def int_AMDIL_get_region_local_id : GCCBuiltin<"__amdil_get_region_local_id_int">,
    Intrinsic<[llvm_v4i32_ty], [], []>;
def int_AMDIL_get_num_regions : GCCBuiltin<"__amdil_get_num_regions_int">,
    Intrinsic<[llvm_v4i32_ty], [], []>;
def int_AMDIL_get_region_size : GCCBuiltin<"__amdil_get_region_size_int">,
    Intrinsic<[llvm_v4i32_ty], [], []>;
}

let isCall=1, isNotDuplicable=1 in {
  let Predicates=[hasRegionAS] in {
def BARRIER_EGNI : BinaryOpNoRet<IL_OP_BARRIER, (outs),
      (ins GPRI32:$flag, GPRI32:$id),
      "fence_threads_memory_lds_gds_gws",
      [(int_AMDIL_barrier GPRI32:$flag, GPRI32:$id)]>;
}
let Predicates=[noRegionAS] in {
def BARRIER_7XX : BinaryOpNoRet<IL_OP_BARRIER, (outs),
      (ins GPRI32:$flag, GPRI32:$id),
      "fence_threads_memory_lds",
      [(int_AMDIL_barrier GPRI32:$flag, GPRI32:$id)]>;
}

def BARRIER_LOCAL : BinaryOpNoRet<IL_OP_BARRIER_LOCAL, (outs),
      (ins GPRI32:$flag, GPRI32:$id),
      "fence_threads_lds",
      [(int_AMDIL_barrier_local GPRI32:$flag, GPRI32:$id)]>;

def BARRIER_GLOBAL : BinaryOpNoRet<IL_OP_BARRIER_GLOBAL, (outs),
      (ins GPRI32:$flag, GPRI32:$id),
      "fence_threads_memory",
      [(int_AMDIL_barrier_global GPRI32:$flag, GPRI32:$id)]>;

def BARRIER_REGION : BinaryOpNoRet<IL_OP_BARRIER_REGION, (outs),
    (ins GPRI32:$flag, GPRI32:$id),
    "fence_threads_gds",
    [(int_AMDIL_barrier_region GPRI32:$flag, GPRI32:$id)]>;
}
