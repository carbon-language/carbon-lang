
class R600Reg <string name, bits<16> encoding> : Register<name> {
  let Namespace = "AMDGPU";
  let HWEncoding = encoding;
}

class R600RegWithChan <string name, bits<9> sel, string chan> :
    Register <name> {

  field bits<2> chan_encoding = !if(!eq(chan, "X"), 0,
                                !if(!eq(chan, "Y"), 1,
                                !if(!eq(chan, "Z"), 2,
                                !if(!eq(chan, "W"), 3, 0))));
  let HWEncoding{8-0}  = sel;
  let HWEncoding{10-9} = chan_encoding;
  let Namespace = "AMDGPU";
}

class R600Reg_128<string n, list<Register> subregs, bits<16> encoding> :
    RegisterWithSubRegs<n, subregs> {
  let Namespace = "AMDGPU";
  let SubRegIndices = [sel_x, sel_y, sel_z, sel_w];
  let HWEncoding = encoding;
}

foreach Index = 0-127 in {
  foreach Chan = [ "X", "Y", "Z", "W" ] in {
    // 32-bit Temporary Registers
    def T#Index#_#Chan : R600RegWithChan <"T"#Index#"."#Chan, Index, Chan>;

    // 32-bit Constant Registers (There are more than 128, this the number
    // that is currently supported.
    def C#Index#_#Chan : R600RegWithChan <"C"#Index#"."#Chan, Index, Chan>;
  }
  // 128-bit Temporary Registers
  def T#Index#_XYZW : R600Reg_128 <"T"#Index#".XYZW",
                                   [!cast<Register>("T"#Index#"_X"),
                                    !cast<Register>("T"#Index#"_Y"),
                                    !cast<Register>("T"#Index#"_Z"),
                                    !cast<Register>("T"#Index#"_W")],
                                   Index>;
}

// Array Base Register holding input in FS
foreach Index = 448-464 in {
  def ArrayBase#Index :  R600Reg<"ARRAY_BASE", Index>;
}


// Special Registers

def ZERO : R600Reg<"0.0", 248>;
def ONE : R600Reg<"1.0", 249>;
def NEG_ONE : R600Reg<"-1.0", 249>;
def ONE_INT : R600Reg<"1", 250>;
def HALF : R600Reg<"0.5", 252>;
def NEG_HALF : R600Reg<"-0.5", 252>;
def ALU_LITERAL_X : R600Reg<"literal.x", 253>;
def PV_X : R600Reg<"pv.x", 254>;
def PREDICATE_BIT : R600Reg<"PredicateBit", 0>;
def PRED_SEL_OFF: R600Reg<"Pred_sel_off", 0>;
def PRED_SEL_ZERO : R600Reg<"Pred_sel_zero", 2>;
def PRED_SEL_ONE : R600Reg<"Pred_sel_one", 3>;

def R600_ArrayBase : RegisterClass <"AMDGPU", [f32, i32], 32,
                          (add (sequence "ArrayBase%u", 448, 464))>;

def R600_CReg32 : RegisterClass <"AMDGPU", [f32, i32], 32,
                          (add (interleave
                                  (interleave (sequence "C%u_X", 0, 127),
                                              (sequence "C%u_Z", 0, 127)),
                                  (interleave (sequence "C%u_Y", 0, 127),
                                              (sequence "C%u_W", 0, 127))))>;

def R600_TReg32_X : RegisterClass <"AMDGPU", [f32, i32], 32,
                                   (add (sequence "T%u_X", 0, 127))>;

def R600_TReg32_Y : RegisterClass <"AMDGPU", [f32, i32], 32,
                                   (add (sequence "T%u_Y", 0, 127))>;

def R600_TReg32_Z : RegisterClass <"AMDGPU", [f32, i32], 32,
                                   (add (sequence "T%u_Z", 0, 127))>;

def R600_TReg32_W : RegisterClass <"AMDGPU", [f32, i32], 32,
                                   (add (sequence "T%u_W", 0, 127))>;

def R600_TReg32 : RegisterClass <"AMDGPU", [f32, i32], 32,
                          (add (interleave
                                 (interleave R600_TReg32_X, R600_TReg32_Z),
                                 (interleave R600_TReg32_Y, R600_TReg32_W)))>;

def R600_Reg32 : RegisterClass <"AMDGPU", [f32, i32], 32, (add
    R600_TReg32,
    R600_CReg32,
    R600_ArrayBase,
    ZERO, HALF, ONE, ONE_INT, PV_X, ALU_LITERAL_X, NEG_ONE, NEG_HALF)>;

def R600_Predicate : RegisterClass <"AMDGPU", [i32], 32, (add
    PRED_SEL_OFF, PRED_SEL_ZERO, PRED_SEL_ONE)>;

def R600_Predicate_Bit: RegisterClass <"AMDGPU", [i32], 32, (add
    PREDICATE_BIT)>;

def R600_Reg128 : RegisterClass<"AMDGPU", [v4f32, v4i32], 128,
                                (add (sequence "T%u_XYZW", 0, 127))> {
  let CopyCost = -1;
}
