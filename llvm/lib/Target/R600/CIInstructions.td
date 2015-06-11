//===-- CIInstructions.td - CI Instruction Defintions ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Instruction definitions for CI and newer.
//===----------------------------------------------------------------------===//


def isCIVI : Predicate <
  "Subtarget->getGeneration() == AMDGPUSubtarget::SEA_ISLANDS || "
  "Subtarget->getGeneration() == AMDGPUSubtarget::VOLCANIC_ISLANDS"
>, AssemblerPredicate<"FeatureCIInsts">;

def HasFlatAddressSpace : Predicate<"Subtarget->hasFlatAddressSpace()">;

//===----------------------------------------------------------------------===//
// VOP1 Instructions
//===----------------------------------------------------------------------===//

let SubtargetPredicate = isCIVI in {

defm V_TRUNC_F64 : VOP1Inst <vop1<0x17>, "v_trunc_f64",
  VOP_F64_F64, ftrunc
>;
defm V_CEIL_F64 : VOP1Inst <vop1<0x18>, "v_ceil_f64",
  VOP_F64_F64, fceil
>;
defm V_FLOOR_F64 : VOP1Inst <vop1<0x1A>, "v_floor_f64",
  VOP_F64_F64, ffloor
>;
defm V_RNDNE_F64 : VOP1Inst <vop1<0x19>, "v_rndne_f64",
  VOP_F64_F64, frint
>;
defm V_LOG_LEGACY_F32 : VOP1Inst <vop1<0x45, 0x4c>, "v_log_legacy_f32",
  VOP_F32_F32
>;
defm V_EXP_LEGACY_F32 : VOP1Inst <vop1<0x46, 0x4b>, "v_exp_legacy_f32",
  VOP_F32_F32
>;
} // End SubtargetPredicate = isCIVI

//===----------------------------------------------------------------------===//
// Flat Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasFlatAddressSpace] in {
def FLAT_LOAD_UBYTE : FLAT_Load_Helper <0x00000008, "flat_load_ubyte", VGPR_32>;
def FLAT_LOAD_SBYTE : FLAT_Load_Helper <0x00000009, "flat_load_sbyte", VGPR_32>;
def FLAT_LOAD_USHORT : FLAT_Load_Helper <0x0000000a, "flat_load_ushort", VGPR_32>;
def FLAT_LOAD_SSHORT : FLAT_Load_Helper <0x0000000b, "flat_load_sshort", VGPR_32>;
def FLAT_LOAD_DWORD : FLAT_Load_Helper <0x0000000c, "flat_load_dword", VGPR_32>;
def FLAT_LOAD_DWORDX2 : FLAT_Load_Helper <0x0000000d, "flat_load_dwordx2", VReg_64>;
def FLAT_LOAD_DWORDX4 : FLAT_Load_Helper <0x0000000e, "flat_load_dwordx4", VReg_128>;
def FLAT_LOAD_DWORDX3 : FLAT_Load_Helper <0x00000010, "flat_load_dwordx3", VReg_96>;

def FLAT_STORE_BYTE : FLAT_Store_Helper <
  0x00000018, "flat_store_byte", VGPR_32
>;

def FLAT_STORE_SHORT : FLAT_Store_Helper <
  0x0000001a, "flat_store_short", VGPR_32
>;

def FLAT_STORE_DWORD : FLAT_Store_Helper <
  0x0000001c, "flat_store_dword", VGPR_32
>;

def FLAT_STORE_DWORDX2 : FLAT_Store_Helper <
  0x0000001d, "flat_store_dwordx2", VReg_64
>;

def FLAT_STORE_DWORDX4 : FLAT_Store_Helper <
  0x0000001e, "flat_store_dwordx4", VReg_128
>;

def FLAT_STORE_DWORDX3 : FLAT_Store_Helper <
  0x0000001e, "flat_store_dwordx3", VReg_96
>;

//def FLAT_ATOMIC_SWAP : FLAT_ <0x00000030, "flat_atomic_swap", []>;
//def FLAT_ATOMIC_CMPSWAP : FLAT_ <0x00000031, "flat_atomic_cmpswap", []>;
//def FLAT_ATOMIC_ADD : FLAT_ <0x00000032, "flat_atomic_add", []>;
//def FLAT_ATOMIC_SUB : FLAT_ <0x00000033, "flat_atomic_sub", []>;
//def FLAT_ATOMIC_RSUB : FLAT_ <0x00000034, "flat_atomic_rsub", []>;
//def FLAT_ATOMIC_SMIN : FLAT_ <0x00000035, "flat_atomic_smin", []>;
//def FLAT_ATOMIC_UMIN : FLAT_ <0x00000036, "flat_atomic_umin", []>;
//def FLAT_ATOMIC_SMAX : FLAT_ <0x00000037, "flat_atomic_smax", []>;
//def FLAT_ATOMIC_UMAX : FLAT_ <0x00000038, "flat_atomic_umax", []>;
//def FLAT_ATOMIC_AND : FLAT_ <0x00000039, "flat_atomic_and", []>;
//def FLAT_ATOMIC_OR : FLAT_ <0x0000003a, "flat_atomic_or", []>;
//def FLAT_ATOMIC_XOR : FLAT_ <0x0000003b, "flat_atomic_xor", []>;
//def FLAT_ATOMIC_INC : FLAT_ <0x0000003c, "flat_atomic_inc", []>;
//def FLAT_ATOMIC_DEC : FLAT_ <0x0000003d, "flat_atomic_dec", []>;
//def FLAT_ATOMIC_FCMPSWAP : FLAT_ <0x0000003e, "flat_atomic_fcmpswap", []>;
//def FLAT_ATOMIC_FMIN : FLAT_ <0x0000003f, "flat_atomic_fmin", []>;
//def FLAT_ATOMIC_FMAX : FLAT_ <0x00000040, "flat_atomic_fmax", []>;
//def FLAT_ATOMIC_SWAP_X2 : FLAT_X2 <0x00000050, "flat_atomic_swap_x2", []>;
//def FLAT_ATOMIC_CMPSWAP_X2 : FLAT_X2 <0x00000051, "flat_atomic_cmpswap_x2", []>;
//def FLAT_ATOMIC_ADD_X2 : FLAT_X2 <0x00000052, "flat_atomic_add_x2", []>;
//def FLAT_ATOMIC_SUB_X2 : FLAT_X2 <0x00000053, "flat_atomic_sub_x2", []>;
//def FLAT_ATOMIC_RSUB_X2 : FLAT_X2 <0x00000054, "flat_atomic_rsub_x2", []>;
//def FLAT_ATOMIC_SMIN_X2 : FLAT_X2 <0x00000055, "flat_atomic_smin_x2", []>;
//def FLAT_ATOMIC_UMIN_X2 : FLAT_X2 <0x00000056, "flat_atomic_umin_x2", []>;
//def FLAT_ATOMIC_SMAX_X2 : FLAT_X2 <0x00000057, "flat_atomic_smax_x2", []>;
//def FLAT_ATOMIC_UMAX_X2 : FLAT_X2 <0x00000058, "flat_atomic_umax_x2", []>;
//def FLAT_ATOMIC_AND_X2 : FLAT_X2 <0x00000059, "flat_atomic_and_x2", []>;
//def FLAT_ATOMIC_OR_X2 : FLAT_X2 <0x0000005a, "flat_atomic_or_x2", []>;
//def FLAT_ATOMIC_XOR_X2 : FLAT_X2 <0x0000005b, "flat_atomic_xor_x2", []>;
//def FLAT_ATOMIC_INC_X2 : FLAT_X2 <0x0000005c, "flat_atomic_inc_x2", []>;
//def FLAT_ATOMIC_DEC_X2 : FLAT_X2 <0x0000005d, "flat_atomic_dec_x2", []>;
//def FLAT_ATOMIC_FCMPSWAP_X2 : FLAT_X2 <0x0000005e, "flat_atomic_fcmpswap_x2", []>;
//def FLAT_ATOMIC_FMIN_X2 : FLAT_X2 <0x0000005f, "flat_atomic_fmin_x2", []>;
//def FLAT_ATOMIC_FMAX_X2 : FLAT_X2 <0x00000060, "flat_atomic_fmax_x2", []>;

//===----------------------------------------------------------------------===//
// Flat Patterns
//===----------------------------------------------------------------------===//

class FLATLoad_Pattern <FLAT Instr_ADDR64, ValueType vt,
                             PatFrag flat_ld> :
  Pat <(vt (flat_ld i64:$ptr)),
       (Instr_ADDR64 $ptr)
>;

def : FLATLoad_Pattern <FLAT_LOAD_SBYTE, i32, sextloadi8_flat>;
def : FLATLoad_Pattern <FLAT_LOAD_UBYTE, i32, az_extloadi8_flat>;
def : FLATLoad_Pattern <FLAT_LOAD_SSHORT, i32, sextloadi16_flat>;
def : FLATLoad_Pattern <FLAT_LOAD_USHORT, i32, az_extloadi16_flat>;
def : FLATLoad_Pattern <FLAT_LOAD_DWORD, i32, flat_load>;
def : FLATLoad_Pattern <FLAT_LOAD_DWORDX2, i64, flat_load>;
def : FLATLoad_Pattern <FLAT_LOAD_DWORDX2, i64, az_extloadi32_flat>;
def : FLATLoad_Pattern <FLAT_LOAD_DWORDX2, v2i32, flat_load>;
def : FLATLoad_Pattern <FLAT_LOAD_DWORDX4, v4i32, flat_load>;

class FLATStore_Pattern <FLAT Instr, ValueType vt, PatFrag st> :
  Pat <(st vt:$value, i64:$ptr),
        (Instr $value, $ptr)
  >;

def : FLATStore_Pattern <FLAT_STORE_BYTE, i32, truncstorei8_flat>;
def : FLATStore_Pattern <FLAT_STORE_SHORT, i32, truncstorei16_flat>;
def : FLATStore_Pattern <FLAT_STORE_DWORD, i32, flat_store>;
def : FLATStore_Pattern <FLAT_STORE_DWORDX2, i64, flat_store>;
def : FLATStore_Pattern <FLAT_STORE_DWORDX2, v2i32, flat_store>;
def : FLATStore_Pattern <FLAT_STORE_DWORDX4, v4i32, flat_store>;

} // End HasFlatAddressSpace predicate

