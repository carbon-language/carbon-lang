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

} // End isCodeGenOnly = 1, isPseudo = 1, hasCustomInserter = 1

/* Generic helper patterns for intrinsics */
/* -------------------------------------- */

class POW_Common <AMDGPUInst log_ieee, AMDGPUInst exp_ieee, AMDGPUInst mul,
                  RegisterClass rc> : Pat <
  (int_AMDGPU_pow rc:$src0, rc:$src1),
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
class Vector_Build <ValueType vecType, RegisterClass elemClass> : Pat <
  (IL_vbuild elemClass:$src),
  (INSERT_SUBREG (vecType (IMPLICIT_DEF)), elemClass:$src, sel_x)
>;

// bitconvert pattern
class BitConvert <ValueType dt, ValueType st, RegisterClass rc> : Pat <
  (dt (bitconvert (st rc:$src0))),
  (dt rc:$src0)
>;

include "R600Instructions.td"

include "SIInstrInfo.td"

