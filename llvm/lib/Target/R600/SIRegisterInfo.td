
let Namespace = "AMDGPU" in {
  def low : SubRegIndex;
  def high : SubRegIndex;

  def sub0 : SubRegIndex;
  def sub1 : SubRegIndex;
  def sub2 : SubRegIndex;
  def sub3 : SubRegIndex;
  def sub4 : SubRegIndex;
  def sub5 : SubRegIndex;
  def sub6 : SubRegIndex;
  def sub7 : SubRegIndex;
}

class SIReg <string n, bits<16> encoding = 0> : Register<n> {
  let Namespace = "AMDGPU";
  let HWEncoding = encoding;
}

class SI_64 <string n, list<Register> subregs, bits<16> encoding> : RegisterWithSubRegs<n, subregs> {
  let Namespace = "AMDGPU";
  let SubRegIndices = [low, high];
  let HWEncoding = encoding;
}

class SGPR_32 <bits<16> num, string name> : SIReg<name, num>;

class VGPR_32 <bits<16> num, string name> : SIReg<name, num>;

// Special Registers
def VCC : SIReg<"VCC", 106>;
def EXEC_LO : SIReg <"EXEC LO", 126>;
def EXEC_HI : SIReg <"EXEC HI", 127>;
def EXEC : SI_64<"EXEC", [EXEC_LO, EXEC_HI], 126>;
def SCC : SIReg<"SCC", 253>;
def SREG_LIT_0 : SIReg <"S LIT 0", 128>;
def SI_LITERAL_CONSTANT : SIReg<"LITERAL CONSTANT", 255>;
def M0 : SIReg <"M0", 124>;

//Interpolation registers
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

// SGPR 32-bit registers
foreach Index = 0-101 in {
  def SGPR#Index : SGPR_32 <Index, "SGPR"#Index>;
}

def SGPR_32 : RegisterClass<"AMDGPU", [f32, i32], 32,
                            (add (sequence "SGPR%u", 0, 101))>;

// SGPR 64-bit registers
def SGPR_64 : RegisterTuples<[low, high],
                             [(add (decimate SGPR_32, 2)),
                              (add(decimate (rotl SGPR_32, 1), 2))]>;

// SGPR 128-bit registers
def SGPR_128 : RegisterTuples<[sel_x, sel_y, sel_z, sel_w],
                              [(add (decimate SGPR_32, 4)),
                               (add (decimate (rotl SGPR_32, 1), 4)),
                               (add (decimate (rotl SGPR_32, 2), 4)),
                               (add (decimate (rotl SGPR_32, 3), 4))]>;

// SGPR 256-bit registers
def SGPR_256 : RegisterTuples<[sub0, sub1, sub2, sub3, sub4, sub5, sub6, sub7],
                              [(add (decimate SGPR_32, 8)),
                               (add (decimate (rotl SGPR_32, 1), 8)),
                               (add (decimate (rotl SGPR_32, 2), 8)),
                               (add (decimate (rotl SGPR_32, 3), 8)),
                               (add (decimate (rotl SGPR_32, 4), 8)),
                               (add (decimate (rotl SGPR_32, 5), 8)),
                               (add (decimate (rotl SGPR_32, 6), 8)),
                               (add (decimate (rotl SGPR_32, 7), 8))]>;

// VGPR 32-bit registers
foreach Index = 0-255 in {
  def VGPR#Index : VGPR_32 <Index, "VGPR"#Index>;
}

def VGPR_32 : RegisterClass<"AMDGPU", [f32, i32], 32,
                            (add (sequence "VGPR%u", 0, 255))>;

// VGPR 64-bit registers
def VGPR_64 : RegisterTuples<[low, high],
                             [(add (decimate VGPR_32, 2)),
                              (add (decimate (rotl VGPR_32, 1), 2))]>;

// VGPR 128-bit registers
def VGPR_128 : RegisterTuples<[sel_x, sel_y, sel_z, sel_w],
                              [(add (decimate VGPR_32, 4)),
                               (add (decimate (rotl VGPR_32, 1), 4)),
                               (add (decimate (rotl VGPR_32, 2), 4)),
                               (add (decimate (rotl VGPR_32, 3), 4))]>;

// Register class for all scalar registers (SGPRs + Special Registers)
def SReg_32 : RegisterClass<"AMDGPU", [f32, i32], 32,
    (add SGPR_32,  SREG_LIT_0, M0, EXEC_LO, EXEC_HI)
>;

def SReg_64 : RegisterClass<"AMDGPU", [i64], 64, (add SGPR_64, VCC, EXEC)>;

def SReg_1 : RegisterClass<"AMDGPU", [i1], 1, (add VCC, SGPR_64, EXEC)>;

def SReg_128 : RegisterClass<"AMDGPU", [v4f32, v4i32], 128, (add SGPR_128)>;

def SReg_256 : RegisterClass<"AMDGPU", [v8i32], 256, (add SGPR_256)>;

// Register class for all vector registers (VGPRs + Interploation Registers)
def VReg_32 : RegisterClass<"AMDGPU", [f32, i32], 32,
    (add VGPR_32,
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

def VReg_64 : RegisterClass<"AMDGPU", [i64], 64, (add VGPR_64)>;

def VReg_128 : RegisterClass<"AMDGPU", [v4f32], 128, (add VGPR_128)>;

// AllReg_* - A set of all scalar and vector registers of a given width.
def AllReg_32 : RegisterClass<"AMDGPU", [f32, i32], 32, (add VReg_32, SReg_32)>;

def AllReg_64 : RegisterClass<"AMDGPU", [f64, i64], 64, (add SReg_64, VReg_64)>;

// Special register classes for predicates and the M0 register
def SCCReg : RegisterClass<"AMDGPU", [i1], 1, (add SCC)>;
def VCCReg : RegisterClass<"AMDGPU", [i1], 1, (add VCC)>;
def EXECReg : RegisterClass<"AMDGPU", [i1], 1, (add EXEC)>;
def M0Reg : RegisterClass<"AMDGPU", [i32], 32, (add M0)>;

