//===-- SIRegisterInfo.td - SI Register defs ---------------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Declarations that describe the SI registers
//===----------------------------------------------------------------------===//

class SIReg <string n, bits<16> encoding = 0> : Register<n> {
  let Namespace = "AMDGPU";
  let HWEncoding = encoding;
}

// Special Registers
def VCC : SIReg<"VCC", 106>;
def EXEC : SIReg<"EXEC", 126>;
def SCC : SIReg<"SCC", 253>;
def M0 : SIReg <"M0", 124>;

// SGPR registers
foreach Index = 0-101 in {
  def SGPR#Index : SIReg <"SGPR"#Index, Index>;
}

// VGPR registers
foreach Index = 0-255 in {
  def VGPR#Index : SIReg <"VGPR"#Index, Index> {
    let HWEncoding{8} = 1;
  }
}

// virtual Interpolation registers
def PERSP_SAMPLE_I : SIReg <"PERSP_SAMPLE_I">;
def PERSP_SAMPLE_J : SIReg <"PERSP_SAMPLE_J">;
def PERSP_CENTER_I : SIReg <"PERSP_CENTER_I">;
def PERSP_CENTER_J : SIReg <"PERSP_CENTER_J">;
def PERSP_CENTROID_I : SIReg <"PERSP_CENTROID_I">;
def PERSP_CENTROID_J : SIReg <"PERP_CENTROID_J">;
def PERSP_I_W : SIReg <"PERSP_I_W">;
def PERSP_J_W : SIReg <"PERSP_J_W">;
def PERSP_1_W : SIReg <"PERSP_1_W">;
def LINEAR_SAMPLE_I : SIReg <"LINEAR_SAMPLE_I">;
def LINEAR_SAMPLE_J : SIReg <"LINEAR_SAMPLE_J">;
def LINEAR_CENTER_I : SIReg <"LINEAR_CENTER_I">;
def LINEAR_CENTER_J : SIReg <"LINEAR_CENTER_J">;
def LINEAR_CENTROID_I : SIReg <"LINEAR_CENTROID_I">;
def LINEAR_CENTROID_J : SIReg <"LINEAR_CENTROID_J">;
def LINE_STIPPLE_TEX_COORD : SIReg <"LINE_STIPPLE_TEX_COORD">;
def POS_X_FLOAT : SIReg <"POS_X_FLOAT">;
def POS_Y_FLOAT : SIReg <"POS_Y_FLOAT">;
def POS_Z_FLOAT : SIReg <"POS_Z_FLOAT">;
def POS_W_FLOAT : SIReg <"POS_W_FLOAT">;
def FRONT_FACE : SIReg <"FRONT_FACE">;
def ANCILLARY : SIReg <"ANCILLARY">;
def SAMPLE_COVERAGE : SIReg <"SAMPLE_COVERAGE">;
def POS_FIXED_PT : SIReg <"POS_FIXED_PT">;

//===----------------------------------------------------------------------===//
//  Groupings using register classes and tuples
//===----------------------------------------------------------------------===//

// SGPR 32-bit registers
def SGPR_32 : RegisterClass<"AMDGPU", [f32, i32], 32,
                            (add (sequence "SGPR%u", 0, 101))>;

// SGPR 64-bit registers
def SGPR_64 : RegisterTuples<[sub0, sub1],
                             [(add (decimate (trunc SGPR_32, 101), 2)),
                              (add (decimate (shl SGPR_32, 1), 2))]>;

// SGPR 128-bit registers
def SGPR_128 : RegisterTuples<[sub0, sub1, sub2, sub3],
                              [(add (decimate (trunc SGPR_32, 99), 4)),
                               (add (decimate (shl SGPR_32, 1), 4)),
                               (add (decimate (shl SGPR_32, 2), 4)),
                               (add (decimate (shl SGPR_32, 3), 4))]>;

// SGPR 256-bit registers
def SGPR_256 : RegisterTuples<[sub0, sub1, sub2, sub3, sub4, sub5, sub6, sub7],
                              [(add (decimate (trunc SGPR_32, 95), 4)),
                               (add (decimate (shl SGPR_32, 1), 4)),
                               (add (decimate (shl SGPR_32, 2), 4)),
                               (add (decimate (shl SGPR_32, 3), 4)),
                               (add (decimate (shl SGPR_32, 4), 4)),
                               (add (decimate (shl SGPR_32, 5), 4)),
                               (add (decimate (shl SGPR_32, 6), 4)),
                               (add (decimate (shl SGPR_32, 7), 4))]>;

// SGPR 512-bit registers
def SGPR_512 : RegisterTuples<[sub0, sub1, sub2, sub3, sub4, sub5, sub6, sub7,
                               sub8, sub9, sub10, sub11, sub12, sub13, sub14, sub15],
                              [(add (decimate (trunc SGPR_32, 87), 4)),
                               (add (decimate (shl SGPR_32, 1), 4)),
                               (add (decimate (shl SGPR_32, 2), 4)),
                               (add (decimate (shl SGPR_32, 3), 4)),
                               (add (decimate (shl SGPR_32, 4), 4)),
                               (add (decimate (shl SGPR_32, 5), 4)),
                               (add (decimate (shl SGPR_32, 6), 4)),
                               (add (decimate (shl SGPR_32, 7), 4)),
                               (add (decimate (shl SGPR_32, 8), 4)),
                               (add (decimate (shl SGPR_32, 9), 4)),
                               (add (decimate (shl SGPR_32, 10), 4)),
                               (add (decimate (shl SGPR_32, 11), 4)),
                               (add (decimate (shl SGPR_32, 12), 4)),
                               (add (decimate (shl SGPR_32, 13), 4)),
                               (add (decimate (shl SGPR_32, 14), 4)),
                               (add (decimate (shl SGPR_32, 15), 4))]>;

// VGPR 32-bit registers
def VGPR_32 : RegisterClass<"AMDGPU", [f32, i32], 32,
                            (add (sequence "VGPR%u", 0, 255))>;

// VGPR 64-bit registers
def VGPR_64 : RegisterTuples<[sub0, sub1],
                             [(add (trunc VGPR_32, 255)),
                              (add (shl VGPR_32, 1))]>;

// VGPR 128-bit registers
def VGPR_128 : RegisterTuples<[sub0, sub1, sub2, sub3],
                              [(add (trunc VGPR_32, 253)),
                               (add (shl VGPR_32, 1)),
                               (add (shl VGPR_32, 2)),
                               (add (shl VGPR_32, 3))]>;

// VGPR 256-bit registers
def VGPR_256 : RegisterTuples<[sub0, sub1, sub2, sub3, sub4, sub5, sub6, sub7],
                              [(add (trunc VGPR_32, 249)),
                               (add (shl VGPR_32, 1)),
                               (add (shl VGPR_32, 2)),
                               (add (shl VGPR_32, 3)),
                               (add (shl VGPR_32, 4)),
                               (add (shl VGPR_32, 5)),
                               (add (shl VGPR_32, 6)),
                               (add (shl VGPR_32, 7))]>;

// VGPR 512-bit registers
def VGPR_512 : RegisterTuples<[sub0, sub1, sub2, sub3, sub4, sub5, sub6, sub7,
                               sub8, sub9, sub10, sub11, sub12, sub13, sub14, sub15],
                              [(add (trunc VGPR_32, 241)),
                               (add (shl VGPR_32, 1)),
                               (add (shl VGPR_32, 2)),
                               (add (shl VGPR_32, 3)),
                               (add (shl VGPR_32, 4)),
                               (add (shl VGPR_32, 5)),
                               (add (shl VGPR_32, 6)),
                               (add (shl VGPR_32, 7)),
                               (add (shl VGPR_32, 8)),
                               (add (shl VGPR_32, 9)),
                               (add (shl VGPR_32, 10)),
                               (add (shl VGPR_32, 11)),
                               (add (shl VGPR_32, 12)),
                               (add (shl VGPR_32, 13)),
                               (add (shl VGPR_32, 14)),
                               (add (shl VGPR_32, 15))]>;

//===----------------------------------------------------------------------===//
//  Register classes used as source and destination
//===----------------------------------------------------------------------===//

// Special register classes for predicates and the M0 register
def SCCReg : RegisterClass<"AMDGPU", [i32, i1], 32, (add SCC)>;
def VCCReg : RegisterClass<"AMDGPU", [i64, i1], 64, (add VCC)>;
def EXECReg : RegisterClass<"AMDGPU", [i64, i1], 64, (add EXEC)>;
def M0Reg : RegisterClass<"AMDGPU", [i32], 32, (add M0)>;

// Register class for all scalar registers (SGPRs + Special Registers)
def SReg_32 : RegisterClass<"AMDGPU", [f32, i32], 32,
  (add SGPR_32, M0Reg)
>;

def SReg_64 : RegisterClass<"AMDGPU", [i64, i1], 64,
  (add SGPR_64, VCCReg, EXECReg)
>;

def SReg_128 : RegisterClass<"AMDGPU", [v16i8], 128, (add SGPR_128)>;

def SReg_256 : RegisterClass<"AMDGPU", [v32i8], 256, (add SGPR_256)>;

def SReg_512 : RegisterClass<"AMDGPU", [v64i8], 512, (add SGPR_512)>;

// Register class for all vector registers (VGPRs + Interploation Registers)
def VReg_32 : RegisterClass<"AMDGPU", [f32, i32, v1i32], 32, (add VGPR_32)>;

def VReg_64 : RegisterClass<"AMDGPU", [i64, v2i32], 64, (add VGPR_64)>;

def VReg_128 : RegisterClass<"AMDGPU", [v4f32, v4i32], 128, (add VGPR_128)>;

def VReg_256 : RegisterClass<"AMDGPU", [v8i32], 256, (add VGPR_256)>;

def VReg_512 : RegisterClass<"AMDGPU", [v16i32], 512, (add VGPR_512)>;

//===----------------------------------------------------------------------===//
//  [SV]Src_* register classes, can have either an immediate or an register
//===----------------------------------------------------------------------===//

def SSrc_32 : RegisterClass<"AMDGPU", [i32, f32], 32, (add SReg_32)>;

def SSrc_64 : RegisterClass<"AMDGPU", [i64, i1], 64, (add SReg_64)>;

def VSrc_32 : RegisterClass<"AMDGPU", [i32, f32], 32,
  (add VReg_32, SReg_32,
    PERSP_SAMPLE_I, PERSP_SAMPLE_J,
    PERSP_CENTER_I, PERSP_CENTER_J,
    PERSP_CENTROID_I, PERSP_CENTROID_J,
    PERSP_I_W, PERSP_J_W, PERSP_1_W,
    LINEAR_SAMPLE_I, LINEAR_SAMPLE_J,
    LINEAR_CENTER_I, LINEAR_CENTER_J,
    LINEAR_CENTROID_I, LINEAR_CENTROID_J,
    LINE_STIPPLE_TEX_COORD,
    POS_X_FLOAT,
    POS_Y_FLOAT,
    POS_Z_FLOAT,
    POS_W_FLOAT,
    FRONT_FACE,
    ANCILLARY,
    SAMPLE_COVERAGE,
    POS_FIXED_PT
  )
>;

def VSrc_64 : RegisterClass<"AMDGPU", [i64], 64, (add VReg_64, SReg_64)>;

