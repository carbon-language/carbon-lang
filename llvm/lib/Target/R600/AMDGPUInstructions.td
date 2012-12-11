//===-- AMDGPUInstructions.td - Common instruction defs ---*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains instruction defs that are common to all hw codegen
// targets.
//
//===----------------------------------------------------------------------===//

class AMDGPUInst <dag outs, dag ins, string asm, list<dag> pattern> : Instruction {
  field bits<16> AMDILOp = 0;
  field bits<3> Gen = 0;

  let Namespace = "AMDGPU";
  let OutOperandList = outs;
  let InOperandList = ins;
  let AsmString = asm;
  let Pattern = pattern;
  let Itinerary = NullALU;
  let TSFlags{42-40} = Gen;
  let TSFlags{63-48} = AMDILOp;
}

class AMDGPUShaderInst <dag outs, dag ins, string asm, list<dag> pattern>
    : AMDGPUInst<outs, ins, asm, pattern> {

  field bits<32> Inst = 0xffffffff;

}

def InstFlag : OperandWithDefaultOps <i32, (ops (i32 0))>;

def COND_EQ : PatLeaf <
  (cond),
  [{switch(N->get()){{default: return false;
                     case ISD::SETOEQ: case ISD::SETUEQ:
                     case ISD::SETEQ: return true;}}}]
>;

def COND_NE : PatLeaf <
  (cond),
  [{switch(N->get()){{default: return false;
                     case ISD::SETONE: case ISD::SETUNE:
                     case ISD::SETNE: return true;}}}]
>;
def COND_GT : PatLeaf <
  (cond),
  [{switch(N->get()){{default: return false;
                     case ISD::SETOGT: case ISD::SETUGT:
                     case ISD::SETGT: return true;}}}]
>;

def COND_GE : PatLeaf <
  (cond),
  [{switch(N->get()){{default: return false;
                     case ISD::SETOGE: case ISD::SETUGE:
                     case ISD::SETGE: return true;}}}]
>;

def COND_LT : PatLeaf <
  (cond),
  [{switch(N->get()){{default: return false;
                     case ISD::SETOLT: case ISD::SETULT:
                     case ISD::SETLT: return true;}}}]
>;

def COND_LE : PatLeaf <
  (cond),
  [{switch(N->get()){{default: return false;
                     case ISD::SETOLE: case ISD::SETULE:
                     case ISD::SETLE: return true;}}}]
>;

//===----------------------------------------------------------------------===//
// Load/Store Pattern Fragments
//===----------------------------------------------------------------------===//

def zextloadi8_global : PatFrag<(ops node:$ptr), (zextloadi8 node:$ptr), [{
    return isGlobalLoad(dyn_cast<LoadSDNode>(N));
}]>;

class Constants {
int TWO_PI = 0x40c90fdb;
int PI = 0x40490fdb;
int TWO_PI_INV = 0x3e22f983;
}
def CONST : Constants;

def FP_ZERO : PatLeaf <
  (fpimm),
  [{return N->getValueAPF().isZero();}]
>;

def FP_ONE : PatLeaf <
  (fpimm),
  [{return N->isExactlyValue(1.0);}]
>;

let isCodeGenOnly = 1, isPseudo = 1, usesCustomInserter = 1  in {

class CLAMP <RegisterClass rc> : AMDGPUShaderInst <
  (outs rc:$dst),
  (ins rc:$src0),
  "CLAMP $dst, $src0",
  [(set rc:$dst, (int_AMDIL_clamp rc:$src0, (f32 FP_ZERO), (f32 FP_ONE)))]
>;

class FABS <RegisterClass rc> : AMDGPUShaderInst <
  (outs rc:$dst),
  (ins rc:$src0),
  "FABS $dst, $src0",
  [(set rc:$dst, (fabs rc:$src0))]
>;

class FNEG <RegisterClass rc> : AMDGPUShaderInst <
  (outs rc:$dst),
  (ins rc:$src0),
  "FNEG $dst, $src0",
  [(set rc:$dst, (fneg rc:$src0))]
>;

def SHADER_TYPE : AMDGPUShaderInst <
  (outs),
  (ins i32imm:$type),
  "SHADER_TYPE $type",
  [(int_AMDGPU_shader_type imm:$type)]
>;

} // End isCodeGenOnly = 1, isPseudo = 1, hasCustomInserter = 1

/* Generic helper patterns for intrinsics */
/* -------------------------------------- */

class POW_Common <AMDGPUInst log_ieee, AMDGPUInst exp_ieee, AMDGPUInst mul,
                  RegisterClass rc> : Pat <
  (fpow rc:$src0, rc:$src1),
  (exp_ieee (mul rc:$src1, (log_ieee rc:$src0)))
>;

/* Other helper patterns */
/* --------------------- */

/* Extract element pattern */
class Extract_Element <ValueType sub_type, ValueType vec_type,
                     RegisterClass vec_class, int sub_idx, 
                     SubRegIndex sub_reg>: Pat<
  (sub_type (vector_extract (vec_type vec_class:$src), sub_idx)),
  (EXTRACT_SUBREG vec_class:$src, sub_reg)
>;

/* Insert element pattern */
class Insert_Element <ValueType elem_type, ValueType vec_type,
                      RegisterClass elem_class, RegisterClass vec_class,
                      int sub_idx, SubRegIndex sub_reg> : Pat <

  (vec_type (vector_insert (vec_type vec_class:$vec),
                           (elem_type elem_class:$elem), sub_idx)),
  (INSERT_SUBREG vec_class:$vec, elem_class:$elem, sub_reg)
>;

// Vector Build pattern
class Vector_Build <ValueType vecType, RegisterClass vectorClass,
                    ValueType elemType, RegisterClass elemClass> : Pat <
  (vecType (build_vector (elemType elemClass:$x), (elemType elemClass:$y),
                         (elemType elemClass:$z), (elemType elemClass:$w))),
  (INSERT_SUBREG (INSERT_SUBREG (INSERT_SUBREG (INSERT_SUBREG
  (vecType (IMPLICIT_DEF)), elemClass:$x, sel_x), elemClass:$y, sel_y),
                            elemClass:$z, sel_z), elemClass:$w, sel_w)
>;

// bitconvert pattern
class BitConvert <ValueType dt, ValueType st, RegisterClass rc> : Pat <
  (dt (bitconvert (st rc:$src0))),
  (dt rc:$src0)
>;

class DwordAddrPat<ValueType vt, RegisterClass rc> : Pat <
  (vt (AMDGPUdwordaddr (vt rc:$addr))),
  (vt rc:$addr)
>;

include "R600Instructions.td"

include "SIInstrInfo.td"

