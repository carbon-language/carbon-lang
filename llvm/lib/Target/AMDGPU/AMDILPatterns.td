//===- AMDILPatterns.td - AMDIL Target Patterns------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//==-----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Store pattern fragments
//===----------------------------------------------------------------------===//
def truncstorei64 : PatFrag<(ops node:$val, node:$ptr),
                           (truncstore node:$val, node:$ptr), [{
  return cast<StoreSDNode>(N)->getMemoryVT() == MVT::i64;
}]>;
def truncstorev2i8 : PatFrag<(ops node:$val, node:$ptr),
                           (truncstore node:$val, node:$ptr), [{
  return cast<StoreSDNode>(N)->getMemoryVT() == MVT::v2i8;
}]>;
def truncstorev2i16 : PatFrag<(ops node:$val, node:$ptr),
                            (truncstore node:$val, node:$ptr), [{
  return cast<StoreSDNode>(N)->getMemoryVT() == MVT::v2i16;
}]>;
def truncstorev2i32 : PatFrag<(ops node:$val, node:$ptr),
                            (truncstore node:$val, node:$ptr), [{
  return cast<StoreSDNode>(N)->getMemoryVT() == MVT::v2i32;
}]>;
def truncstorev2i64 : PatFrag<(ops node:$val, node:$ptr),
                            (truncstore node:$val, node:$ptr), [{
  return cast<StoreSDNode>(N)->getMemoryVT() == MVT::v2i64;
}]>;
def truncstorev2f32 : PatFrag<(ops node:$val, node:$ptr),
                            (truncstore node:$val, node:$ptr), [{
  return cast<StoreSDNode>(N)->getMemoryVT() == MVT::v2f32;
}]>;
def truncstorev2f64 : PatFrag<(ops node:$val, node:$ptr),
                            (truncstore node:$val, node:$ptr), [{
  return cast<StoreSDNode>(N)->getMemoryVT() == MVT::v2f64;
}]>;
def truncstorev4i8 : PatFrag<(ops node:$val, node:$ptr),
                           (truncstore node:$val, node:$ptr), [{
  return cast<StoreSDNode>(N)->getMemoryVT() == MVT::v4i8;
}]>;
def truncstorev4i16 : PatFrag<(ops node:$val, node:$ptr),
                            (truncstore node:$val, node:$ptr), [{
  return cast<StoreSDNode>(N)->getMemoryVT() == MVT::v4i16;
}]>;
def truncstorev4i32 : PatFrag<(ops node:$val, node:$ptr),
                            (truncstore node:$val, node:$ptr), [{
  return cast<StoreSDNode>(N)->getMemoryVT() == MVT::v4i32;
}]>;
def truncstorev4f32 : PatFrag<(ops node:$val, node:$ptr),
                            (truncstore node:$val, node:$ptr), [{
  return cast<StoreSDNode>(N)->getMemoryVT() == MVT::v4f32;
}]>;

def global_store : PatFrag<(ops node:$val, node:$ptr),
    (store node:$val, node:$ptr), [{
        return isGlobalStore(dyn_cast<StoreSDNode>(N));
}]>;
def private_store : PatFrag<(ops node:$val, node:$ptr),
    (store node:$val, node:$ptr), [{
        return isPrivateStore(dyn_cast<StoreSDNode>(N));
}]>;
def local_store : PatFrag<(ops node:$val, node:$ptr),
    (store node:$val, node:$ptr), [{
        return isLocalStore(dyn_cast<StoreSDNode>(N));
}]>;
def region_store : PatFrag<(ops node:$val, node:$ptr),
    (store node:$val, node:$ptr), [{
        return isRegionStore(dyn_cast<StoreSDNode>(N));
}]>;
def global_i8trunc_store : PatFrag<(ops node:$val, node:$ptr),
    (truncstorei8 node:$val, node:$ptr), [{
        return isGlobalStore(dyn_cast<StoreSDNode>(N));
}]>;
def global_i16trunc_store : PatFrag<(ops node:$val, node:$ptr),
    (truncstorei16 node:$val, node:$ptr), [{
        return isGlobalStore(dyn_cast<StoreSDNode>(N));
}]>;
def global_i32trunc_store : PatFrag<(ops node:$val, node:$ptr),
    (truncstorei32 node:$val, node:$ptr), [{
        return isGlobalStore(dyn_cast<StoreSDNode>(N));
}]>;
def global_i64trunc_store : PatFrag<(ops node:$val, node:$ptr),
    (truncstorei64 node:$val, node:$ptr), [{
        return isGlobalStore(dyn_cast<StoreSDNode>(N));
}]>;
def global_f32trunc_store : PatFrag<(ops node:$val, node:$ptr),
    (truncstoref32 node:$val, node:$ptr), [{
        return isGlobalStore(dyn_cast<StoreSDNode>(N));
}]>;
def global_f64trunc_store : PatFrag<(ops node:$val, node:$ptr),
    (truncstoref64 node:$val, node:$ptr), [{
        return isGlobalStore(dyn_cast<StoreSDNode>(N));
}]>;
def global_v2i8trunc_store : PatFrag<(ops node:$val, node:$ptr),
    (truncstorev2i8 node:$val, node:$ptr), [{
        return isGlobalStore(dyn_cast<StoreSDNode>(N));
}]>;
def global_v2i16trunc_store : PatFrag<(ops node:$val, node:$ptr),
    (truncstorev2i16 node:$val, node:$ptr), [{
        return isGlobalStore(dyn_cast<StoreSDNode>(N));
}]>;
def global_v2i32trunc_store : PatFrag<(ops node:$val, node:$ptr),
    (truncstorev2i32 node:$val, node:$ptr), [{
        return isGlobalStore(dyn_cast<StoreSDNode>(N));
}]>;
def global_v2i64trunc_store : PatFrag<(ops node:$val, node:$ptr),
    (truncstorev2i64 node:$val, node:$ptr), [{
        return isGlobalStore(dyn_cast<StoreSDNode>(N));
}]>;
def global_v2f32trunc_store : PatFrag<(ops node:$val, node:$ptr),
    (truncstorev2f32 node:$val, node:$ptr), [{
        return isGlobalStore(dyn_cast<StoreSDNode>(N));
}]>;
def global_v2f64trunc_store : PatFrag<(ops node:$val, node:$ptr),
    (truncstorev2f64 node:$val, node:$ptr), [{
        return isGlobalStore(dyn_cast<StoreSDNode>(N));
}]>;
def global_v4i8trunc_store : PatFrag<(ops node:$val, node:$ptr),
    (truncstorev4i8 node:$val, node:$ptr), [{
        return isGlobalStore(dyn_cast<StoreSDNode>(N));
}]>;
def global_v4i16trunc_store : PatFrag<(ops node:$val, node:$ptr),
    (truncstorev4i16 node:$val, node:$ptr), [{
        return isGlobalStore(dyn_cast<StoreSDNode>(N));
}]>;
def global_v4i32trunc_store : PatFrag<(ops node:$val, node:$ptr),
    (truncstorev4i32 node:$val, node:$ptr), [{
        return isGlobalStore(dyn_cast<StoreSDNode>(N));
}]>;
def global_v4f32trunc_store : PatFrag<(ops node:$val, node:$ptr),
    (truncstorev4f32 node:$val, node:$ptr), [{
        return isGlobalStore(dyn_cast<StoreSDNode>(N));
}]>;
def private_trunc_store : PatFrag<(ops node:$val, node:$ptr),
    (truncstore node:$val, node:$ptr), [{
        return isPrivateStore(dyn_cast<StoreSDNode>(N));
}]>;
def private_i8trunc_store : PatFrag<(ops node:$val, node:$ptr),
    (truncstorei8 node:$val, node:$ptr), [{
        return isPrivateStore(dyn_cast<StoreSDNode>(N));
}]>;
def private_i16trunc_store : PatFrag<(ops node:$val, node:$ptr),
    (truncstorei16 node:$val, node:$ptr), [{
        return isPrivateStore(dyn_cast<StoreSDNode>(N));
}]>;
def private_i32trunc_store : PatFrag<(ops node:$val, node:$ptr),
    (truncstorei32 node:$val, node:$ptr), [{
        return isPrivateStore(dyn_cast<StoreSDNode>(N));
}]>;
def private_i64trunc_store : PatFrag<(ops node:$val, node:$ptr),
    (truncstorei64 node:$val, node:$ptr), [{
        return isPrivateStore(dyn_cast<StoreSDNode>(N));
}]>;
def private_f32trunc_store : PatFrag<(ops node:$val, node:$ptr),
    (truncstoref32 node:$val, node:$ptr), [{
        return isPrivateStore(dyn_cast<StoreSDNode>(N));
}]>;
def private_f64trunc_store : PatFrag<(ops node:$val, node:$ptr),
    (truncstoref64 node:$val, node:$ptr), [{
        return isPrivateStore(dyn_cast<StoreSDNode>(N));
}]>;
def private_v2i8trunc_store : PatFrag<(ops node:$val, node:$ptr),
    (truncstorev2i8 node:$val, node:$ptr), [{
        return isPrivateStore(dyn_cast<StoreSDNode>(N));
}]>;
def private_v2i16trunc_store : PatFrag<(ops node:$val, node:$ptr),
    (truncstorev2i16 node:$val, node:$ptr), [{
        return isPrivateStore(dyn_cast<StoreSDNode>(N));
}]>;
def private_v2i32trunc_store : PatFrag<(ops node:$val, node:$ptr),
    (truncstorev2i32 node:$val, node:$ptr), [{
        return isPrivateStore(dyn_cast<StoreSDNode>(N));
}]>;
def private_v2i64trunc_store : PatFrag<(ops node:$val, node:$ptr),
    (truncstorev2i64 node:$val, node:$ptr), [{
        return isPrivateStore(dyn_cast<StoreSDNode>(N));
}]>;
def private_v2f32trunc_store : PatFrag<(ops node:$val, node:$ptr),
    (truncstorev2f32 node:$val, node:$ptr), [{
        return isPrivateStore(dyn_cast<StoreSDNode>(N));
}]>;
def private_v2f64trunc_store : PatFrag<(ops node:$val, node:$ptr),
    (truncstorev2f64 node:$val, node:$ptr), [{
        return isPrivateStore(dyn_cast<StoreSDNode>(N));
}]>;
def private_v4i8trunc_store : PatFrag<(ops node:$val, node:$ptr),
    (truncstorev4i8 node:$val, node:$ptr), [{
        return isPrivateStore(dyn_cast<StoreSDNode>(N));
}]>;
def private_v4i16trunc_store : PatFrag<(ops node:$val, node:$ptr),
    (truncstorev4i16 node:$val, node:$ptr), [{
        return isPrivateStore(dyn_cast<StoreSDNode>(N));
}]>;
def private_v4i32trunc_store : PatFrag<(ops node:$val, node:$ptr),
    (truncstorev4i32 node:$val, node:$ptr), [{
        return isPrivateStore(dyn_cast<StoreSDNode>(N));
}]>;
def private_v4f32trunc_store : PatFrag<(ops node:$val, node:$ptr),
    (truncstorev4f32 node:$val, node:$ptr), [{
        return isPrivateStore(dyn_cast<StoreSDNode>(N));
}]>;

def local_trunc_store : PatFrag<(ops node:$val, node:$ptr),
    (truncstore node:$val, node:$ptr), [{
        return isLocalStore(dyn_cast<StoreSDNode>(N));
}]>;
def local_i8trunc_store : PatFrag<(ops node:$val, node:$ptr),
    (truncstorei8 node:$val, node:$ptr), [{
        return isLocalStore(dyn_cast<StoreSDNode>(N));
}]>;
def local_i16trunc_store : PatFrag<(ops node:$val, node:$ptr),
    (truncstorei16 node:$val, node:$ptr), [{
        return isLocalStore(dyn_cast<StoreSDNode>(N));
}]>;
def local_i32trunc_store : PatFrag<(ops node:$val, node:$ptr),
    (truncstorei32 node:$val, node:$ptr), [{
        return isLocalStore(dyn_cast<StoreSDNode>(N));
}]>;
def local_i64trunc_store : PatFrag<(ops node:$val, node:$ptr),
    (truncstorei64 node:$val, node:$ptr), [{
        return isLocalStore(dyn_cast<StoreSDNode>(N));
}]>;
def local_f32trunc_store : PatFrag<(ops node:$val, node:$ptr),
    (truncstoref32 node:$val, node:$ptr), [{
        return isLocalStore(dyn_cast<StoreSDNode>(N));
}]>;
def local_f64trunc_store : PatFrag<(ops node:$val, node:$ptr),
    (truncstoref64 node:$val, node:$ptr), [{
        return isLocalStore(dyn_cast<StoreSDNode>(N));
}]>;
def local_v2i8trunc_store : PatFrag<(ops node:$val, node:$ptr),
    (truncstorev2i8 node:$val, node:$ptr), [{
        return isLocalStore(dyn_cast<StoreSDNode>(N));
}]>;
def local_v2i16trunc_store : PatFrag<(ops node:$val, node:$ptr),
    (truncstorev2i16 node:$val, node:$ptr), [{
        return isLocalStore(dyn_cast<StoreSDNode>(N));
}]>;
def local_v2i32trunc_store : PatFrag<(ops node:$val, node:$ptr),
    (truncstorev2i32 node:$val, node:$ptr), [{
        return isLocalStore(dyn_cast<StoreSDNode>(N));
}]>;
def local_v2i64trunc_store : PatFrag<(ops node:$val, node:$ptr),
    (truncstorev2i64 node:$val, node:$ptr), [{
        return isLocalStore(dyn_cast<StoreSDNode>(N));
}]>;
def local_v2f32trunc_store : PatFrag<(ops node:$val, node:$ptr),
    (truncstorev2f32 node:$val, node:$ptr), [{
        return isLocalStore(dyn_cast<StoreSDNode>(N));
}]>;
def local_v2f64trunc_store : PatFrag<(ops node:$val, node:$ptr),
    (truncstorev2f64 node:$val, node:$ptr), [{
        return isLocalStore(dyn_cast<StoreSDNode>(N));
}]>;
def local_v4i8trunc_store : PatFrag<(ops node:$val, node:$ptr),
    (truncstorev4i8 node:$val, node:$ptr), [{
        return isLocalStore(dyn_cast<StoreSDNode>(N));
}]>;
def local_v4i16trunc_store : PatFrag<(ops node:$val, node:$ptr),
    (truncstorev4i16 node:$val, node:$ptr), [{
        return isLocalStore(dyn_cast<StoreSDNode>(N));
}]>;
def local_v4i32trunc_store : PatFrag<(ops node:$val, node:$ptr),
    (truncstorev4i32 node:$val, node:$ptr), [{
        return isLocalStore(dyn_cast<StoreSDNode>(N));
}]>;
def local_v4f32trunc_store : PatFrag<(ops node:$val, node:$ptr),
    (truncstorev4f32 node:$val, node:$ptr), [{
        return isLocalStore(dyn_cast<StoreSDNode>(N));
}]>;

def region_trunc_store : PatFrag<(ops node:$val, node:$ptr),
    (truncstore node:$val, node:$ptr), [{
        return isRegionStore(dyn_cast<StoreSDNode>(N));
}]>;
def region_i8trunc_store : PatFrag<(ops node:$val, node:$ptr),
    (truncstorei8 node:$val, node:$ptr), [{
        return isRegionStore(dyn_cast<StoreSDNode>(N));
}]>;
def region_i16trunc_store : PatFrag<(ops node:$val, node:$ptr),
    (truncstorei16 node:$val, node:$ptr), [{
        return isRegionStore(dyn_cast<StoreSDNode>(N));
}]>;
def region_i32trunc_store : PatFrag<(ops node:$val, node:$ptr),
    (truncstorei32 node:$val, node:$ptr), [{
        return isRegionStore(dyn_cast<StoreSDNode>(N));
}]>;
def region_i64trunc_store : PatFrag<(ops node:$val, node:$ptr),
    (truncstorei64 node:$val, node:$ptr), [{
        return isRegionStore(dyn_cast<StoreSDNode>(N));
}]>;
def region_f32trunc_store : PatFrag<(ops node:$val, node:$ptr),
    (truncstoref32 node:$val, node:$ptr), [{
        return isRegionStore(dyn_cast<StoreSDNode>(N));
}]>;
def region_f64trunc_store : PatFrag<(ops node:$val, node:$ptr),
    (truncstoref64 node:$val, node:$ptr), [{
        return isRegionStore(dyn_cast<StoreSDNode>(N));
}]>;
def region_v2i8trunc_store : PatFrag<(ops node:$val, node:$ptr),
    (truncstorev2i8 node:$val, node:$ptr), [{
        return isRegionStore(dyn_cast<StoreSDNode>(N));
}]>;
def region_v2i16trunc_store : PatFrag<(ops node:$val, node:$ptr),
    (truncstorev2i16 node:$val, node:$ptr), [{
        return isRegionStore(dyn_cast<StoreSDNode>(N));
}]>;
def region_v2i32trunc_store : PatFrag<(ops node:$val, node:$ptr),
    (truncstorev2i32 node:$val, node:$ptr), [{
        return isRegionStore(dyn_cast<StoreSDNode>(N));
}]>;
def region_v2i64trunc_store : PatFrag<(ops node:$val, node:$ptr),
    (truncstorev2i64 node:$val, node:$ptr), [{
        return isRegionStore(dyn_cast<StoreSDNode>(N));
}]>;
def region_v2f32trunc_store : PatFrag<(ops node:$val, node:$ptr),
    (truncstorev2f32 node:$val, node:$ptr), [{
        return isRegionStore(dyn_cast<StoreSDNode>(N));
}]>;
def region_v2f64trunc_store : PatFrag<(ops node:$val, node:$ptr),
    (truncstorev2f64 node:$val, node:$ptr), [{
        return isRegionStore(dyn_cast<StoreSDNode>(N));
}]>;
def region_v4i8trunc_store : PatFrag<(ops node:$val, node:$ptr),
    (truncstorev4i8 node:$val, node:$ptr), [{
        return isRegionStore(dyn_cast<StoreSDNode>(N));
}]>;
def region_v4i16trunc_store : PatFrag<(ops node:$val, node:$ptr),
    (truncstorev4i16 node:$val, node:$ptr), [{
        return isRegionStore(dyn_cast<StoreSDNode>(N));
}]>;
def region_v4i32trunc_store : PatFrag<(ops node:$val, node:$ptr),
    (truncstorev4i32 node:$val, node:$ptr), [{
        return isRegionStore(dyn_cast<StoreSDNode>(N));
}]>;
def region_v4f32trunc_store : PatFrag<(ops node:$val, node:$ptr),
    (truncstorev4f32 node:$val, node:$ptr), [{
        return isRegionStore(dyn_cast<StoreSDNode>(N));
}]>;

//===----------------------------------------------------------------------===//
// Load pattern fragments
//===----------------------------------------------------------------------===//
// Global address space loads
def global_load : PatFrag<(ops node:$ptr), (load node:$ptr), [{
    return isGlobalLoad(dyn_cast<LoadSDNode>(N));
}]>;
def global_sext_load : PatFrag<(ops node:$ptr), (sextload node:$ptr), [{
    return isGlobalLoad(dyn_cast<LoadSDNode>(N));
}]>;
def global_aext_load : PatFrag<(ops node:$ptr), (zextload node:$ptr), [{
    return isGlobalLoad(dyn_cast<LoadSDNode>(N));
}]>;
def global_zext_load : PatFrag<(ops node:$ptr), (extload node:$ptr), [{
    return isGlobalLoad(dyn_cast<LoadSDNode>(N));
}]>;
// Private address space loads
def private_load : PatFrag<(ops node:$ptr), (load node:$ptr), [{
    return isPrivateLoad(dyn_cast<LoadSDNode>(N));
}]>;
def private_sext_load : PatFrag<(ops node:$ptr), (sextload node:$ptr), [{
    return isPrivateLoad(dyn_cast<LoadSDNode>(N));
}]>;
def private_aext_load : PatFrag<(ops node:$ptr), (zextload node:$ptr), [{
    return isPrivateLoad(dyn_cast<LoadSDNode>(N));
}]>;
def private_zext_load : PatFrag<(ops node:$ptr), (extload node:$ptr), [{
    return isPrivateLoad(dyn_cast<LoadSDNode>(N));
}]>;
// Local address space loads
def local_load : PatFrag<(ops node:$ptr), (load node:$ptr), [{
    return isLocalLoad(dyn_cast<LoadSDNode>(N));
}]>;
def local_sext_load : PatFrag<(ops node:$ptr), (sextload node:$ptr), [{
    return isLocalLoad(dyn_cast<LoadSDNode>(N));
}]>;
def local_aext_load : PatFrag<(ops node:$ptr), (zextload node:$ptr), [{
    return isLocalLoad(dyn_cast<LoadSDNode>(N));
}]>;
def local_zext_load : PatFrag<(ops node:$ptr), (extload node:$ptr), [{
    return isLocalLoad(dyn_cast<LoadSDNode>(N));
}]>;
// Region address space loads
def region_load : PatFrag<(ops node:$ptr), (load node:$ptr), [{
    return isRegionLoad(dyn_cast<LoadSDNode>(N));
}]>;
def region_sext_load : PatFrag<(ops node:$ptr), (sextload node:$ptr), [{
    return isRegionLoad(dyn_cast<LoadSDNode>(N));
}]>;
def region_aext_load : PatFrag<(ops node:$ptr), (zextload node:$ptr), [{
    return isRegionLoad(dyn_cast<LoadSDNode>(N));
}]>;
def region_zext_load : PatFrag<(ops node:$ptr), (extload node:$ptr), [{
    return isRegionLoad(dyn_cast<LoadSDNode>(N));
}]>;
// Constant address space loads
def constant_load : PatFrag<(ops node:$ptr), (load node:$ptr), [{
    return isConstantLoad(dyn_cast<LoadSDNode>(N), -1);
}]>;
def constant_sext_load : PatFrag<(ops node:$ptr), (sextload node:$ptr), [{
    return isConstantLoad(dyn_cast<LoadSDNode>(N), -1);
}]>;
def constant_aext_load : PatFrag<(ops node:$ptr), (zextload node:$ptr), [{
    return isConstantLoad(dyn_cast<LoadSDNode>(N), -1);
}]>;
def constant_zext_load : PatFrag<(ops node:$ptr), (extload node:$ptr), [{
    return isConstantLoad(dyn_cast<LoadSDNode>(N), -1);
}]>;
// Constant pool loads
def cp_load : PatFrag<(ops node:$ptr), (load node:$ptr), [{
  return isCPLoad(dyn_cast<LoadSDNode>(N));
}]>;
def cp_sext_load : PatFrag<(ops node:$ptr), (sextload node:$ptr), [{
  return isCPLoad(dyn_cast<LoadSDNode>(N));
}]>;
def cp_zext_load : PatFrag<(ops node:$ptr), (zextload node:$ptr), [{
  return isCPLoad(dyn_cast<LoadSDNode>(N));
}]>;
def cp_aext_load : PatFrag<(ops node:$ptr), (extload node:$ptr), [{
  return isCPLoad(dyn_cast<LoadSDNode>(N));
}]>;

//===----------------------------------------------------------------------===//
// Complex addressing mode patterns
//===----------------------------------------------------------------------===//
def ADDR : ComplexPattern<i32, 2, "SelectADDR", [], []>;
def ADDRF : ComplexPattern<i32, 2, "SelectADDR", [frameindex], []>;
def ADDR64 : ComplexPattern<i64, 2, "SelectADDR64", [], []>;
def ADDR64F : ComplexPattern<i64, 2, "SelectADDR64", [frameindex], []>;


//===----------------------------------------------------------------------===//
// Conditional Instruction Pattern Leafs
//===----------------------------------------------------------------------===//
class IL_CC_Op<int N> : PatLeaf<(i32 N)>;
def IL_CC_D_EQ  : IL_CC_Op<0>;
def IL_CC_D_GE  : IL_CC_Op<1>;
def IL_CC_D_LT  : IL_CC_Op<2>;
def IL_CC_D_NE  : IL_CC_Op<3>;
def IL_CC_F_EQ  : IL_CC_Op<4>;
def IL_CC_F_GE  : IL_CC_Op<5>;
def IL_CC_F_LT  : IL_CC_Op<6>;
def IL_CC_F_NE  : IL_CC_Op<7>;
def IL_CC_I_EQ  : IL_CC_Op<8>;
def IL_CC_I_GE  : IL_CC_Op<9>;
def IL_CC_I_LT  : IL_CC_Op<10>;
def IL_CC_I_NE  : IL_CC_Op<11>;
def IL_CC_U_GE  : IL_CC_Op<12>;
def IL_CC_U_LT  : IL_CC_Op<13>;
// Pseudo IL comparison instructions that aren't natively supported
def IL_CC_F_GT  : IL_CC_Op<14>;
def IL_CC_U_GT  : IL_CC_Op<15>;
def IL_CC_I_GT  : IL_CC_Op<16>;
def IL_CC_D_GT  : IL_CC_Op<17>;
def IL_CC_F_LE  : IL_CC_Op<18>;
def IL_CC_U_LE  : IL_CC_Op<19>;
def IL_CC_I_LE  : IL_CC_Op<20>;
def IL_CC_D_LE  : IL_CC_Op<21>;
def IL_CC_F_UNE : IL_CC_Op<22>;
def IL_CC_F_UEQ : IL_CC_Op<23>;
def IL_CC_F_ULT : IL_CC_Op<24>;
def IL_CC_F_UGT : IL_CC_Op<25>;
def IL_CC_F_ULE : IL_CC_Op<26>;
def IL_CC_F_UGE : IL_CC_Op<27>;
def IL_CC_F_ONE : IL_CC_Op<28>;
def IL_CC_F_OEQ : IL_CC_Op<29>;
def IL_CC_F_OLT : IL_CC_Op<30>;
def IL_CC_F_OGT : IL_CC_Op<31>;
def IL_CC_F_OLE : IL_CC_Op<32>;
def IL_CC_F_OGE : IL_CC_Op<33>;
def IL_CC_D_UNE : IL_CC_Op<34>;
def IL_CC_D_UEQ : IL_CC_Op<35>;
def IL_CC_D_ULT : IL_CC_Op<36>;
def IL_CC_D_UGT : IL_CC_Op<37>;
def IL_CC_D_ULE : IL_CC_Op<38>;
def IL_CC_D_UGE : IL_CC_Op<39>;
def IL_CC_D_ONE : IL_CC_Op<30>;
def IL_CC_D_OEQ : IL_CC_Op<41>;
def IL_CC_D_OLT : IL_CC_Op<42>;
def IL_CC_D_OGT : IL_CC_Op<43>;
def IL_CC_D_OLE : IL_CC_Op<44>;
def IL_CC_D_OGE : IL_CC_Op<45>;
def IL_CC_U_EQ  : IL_CC_Op<46>;
def IL_CC_U_NE  : IL_CC_Op<47>;
def IL_CC_F_O   : IL_CC_Op<48>;
def IL_CC_D_O   : IL_CC_Op<49>;
def IL_CC_F_UO  : IL_CC_Op<50>;
def IL_CC_D_UO  : IL_CC_Op<51>;
def IL_CC_L_LE  : IL_CC_Op<52>;
def IL_CC_L_GE  : IL_CC_Op<53>;
def IL_CC_L_EQ  : IL_CC_Op<54>;
def IL_CC_L_NE  : IL_CC_Op<55>;
def IL_CC_L_LT  : IL_CC_Op<56>;
def IL_CC_L_GT  : IL_CC_Op<57>;
def IL_CC_UL_LE  : IL_CC_Op<58>;
def IL_CC_UL_GE  : IL_CC_Op<59>;
def IL_CC_UL_EQ  : IL_CC_Op<60>;
def IL_CC_UL_NE  : IL_CC_Op<61>;
def IL_CC_UL_LT  : IL_CC_Op<62>;
def IL_CC_UL_GT  : IL_CC_Op<63>;
