
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
  let SubRegIndices = [sub0, sub1, sub2, sub3];
  let HWEncoding = encoding;
}

class R600Reg_64<string n, list<Register> subregs, bits<16> encoding> :
    RegisterWithSubRegs<n, subregs> {
  let Namespace = "AMDGPU";
  let SubRegIndices = [sub0, sub1];
  let HWEncoding = encoding;
}


foreach Index = 0-127 in {
  foreach Chan = [ "X", "Y", "Z", "W" ] in {
    // 32-bit Temporary Registers
    def T#Index#_#Chan : R600RegWithChan <"T"#Index#"."#Chan, Index, Chan>;

    // Indirect addressing offset registers
    def Addr#Index#_#Chan : R600RegWithChan <"T("#Index#" + AR.x)."#Chan,
                                              Index, Chan>;
  }
  // 128-bit Temporary Registers
  def T#Index#_XYZW : R600Reg_128 <"T"#Index#"",
                                   [!cast<Register>("T"#Index#"_X"),
                                    !cast<Register>("T"#Index#"_Y"),
                                    !cast<Register>("T"#Index#"_Z"),
                                    !cast<Register>("T"#Index#"_W")],
                                   Index>;

  def T#Index#_XY : R600Reg_64 <"T"#Index#"",
                                   [!cast<Register>("T"#Index#"_X"),
                                    !cast<Register>("T"#Index#"_Y")],
                                   Index>;
}

// KCACHE_BANK0
foreach Index = 159-128 in {
  foreach Chan = [ "X", "Y", "Z", "W" ] in {
    // 32-bit Temporary Registers
    def KC0_#Index#_#Chan : R600RegWithChan <"KC0["#!add(Index,-128)#"]."#Chan, Index, Chan>;
  }
  // 128-bit Temporary Registers
  def KC0_#Index#_XYZW : R600Reg_128 <"KC0["#!add(Index, -128)#"].XYZW",
                                 [!cast<Register>("KC0_"#Index#"_X"),
                                  !cast<Register>("KC0_"#Index#"_Y"),
                                  !cast<Register>("KC0_"#Index#"_Z"),
                                  !cast<Register>("KC0_"#Index#"_W")],
                                 Index>;
}

// KCACHE_BANK1
foreach Index = 191-160 in {
  foreach Chan = [ "X", "Y", "Z", "W" ] in {
    // 32-bit Temporary Registers
    def KC1_#Index#_#Chan : R600RegWithChan <"KC1["#!add(Index,-160)#"]."#Chan, Index, Chan>;
  }
  // 128-bit Temporary Registers
  def KC1_#Index#_XYZW : R600Reg_128 <"KC1["#!add(Index, -160)#"].XYZW",
                                 [!cast<Register>("KC1_"#Index#"_X"),
                                  !cast<Register>("KC1_"#Index#"_Y"),
                                  !cast<Register>("KC1_"#Index#"_Z"),
                                  !cast<Register>("KC1_"#Index#"_W")],
                                 Index>;
}


// Array Base Register holding input in FS
foreach Index = 448-480 in {
  def ArrayBase#Index :  R600Reg<"ARRAY_BASE", Index>;
}


// Special Registers

def OQA : R600Reg<"OQA", 219>;
def OQB : R600Reg<"OQB", 220>;
def OQAP : R600Reg<"OQAP", 221>;
def OQBP : R600Reg<"OQAP", 222>;
def LDS_DIRECT_A : R600Reg<"LDS_DIRECT_A", 223>;
def LDS_DIRECT_B : R600Reg<"LDS_DIRECT_B", 224>;
def ZERO : R600Reg<"0.0", 248>;
def ONE : R600Reg<"1.0", 249>;
def NEG_ONE : R600Reg<"-1.0", 249>;
def ONE_INT : R600Reg<"1", 250>;
def HALF : R600Reg<"0.5", 252>;
def NEG_HALF : R600Reg<"-0.5", 252>;
def ALU_LITERAL_X : R600RegWithChan<"literal.x", 253, "X">;
def ALU_LITERAL_Y : R600RegWithChan<"literal.y", 253, "Y">;
def ALU_LITERAL_Z : R600RegWithChan<"literal.z", 253, "Z">;
def ALU_LITERAL_W : R600RegWithChan<"literal.w", 253, "W">;
def PV_X : R600RegWithChan<"PV.X", 254, "X">;
def PV_Y : R600RegWithChan<"PV.Y", 254, "Y">;
def PV_Z : R600RegWithChan<"PV.Z", 254, "Z">;
def PV_W : R600RegWithChan<"PV.W", 254, "W">;
def PS: R600Reg<"PS", 255>;
def PREDICATE_BIT : R600Reg<"PredicateBit", 0>;
def PRED_SEL_OFF: R600Reg<"Pred_sel_off", 0>;
def PRED_SEL_ZERO : R600Reg<"Pred_sel_zero", 2>;
def PRED_SEL_ONE : R600Reg<"Pred_sel_one", 3>;
def AR_X : R600Reg<"AR.x", 0>;

def R600_ArrayBase : RegisterClass <"AMDGPU", [f32, i32], 32,
                          (add (sequence "ArrayBase%u", 448, 480))>;
// special registers for ALU src operands
// const buffer reference, SRCx_SEL contains index
def ALU_CONST : R600Reg<"CBuf", 0>;
// interpolation param reference, SRCx_SEL contains index
def ALU_PARAM : R600Reg<"Param", 0>;

let isAllocatable = 0 in {

// XXX: Only use the X channel, until we support wider stack widths
def R600_Addr : RegisterClass <"AMDGPU", [i32], 127, (add (sequence "Addr%u_X", 0, 127))>;

def R600_LDS_SRC_REG : RegisterClass<"AMDGPU", [i32], 32,
  (add OQA, OQB, OQAP, OQBP, LDS_DIRECT_A, LDS_DIRECT_B)>;

def R600_KC0_X : RegisterClass <"AMDGPU", [f32, i32], 32,
                              (add (sequence "KC0_%u_X", 128, 159))>;

def R600_KC0_Y : RegisterClass <"AMDGPU", [f32, i32], 32,
                              (add (sequence "KC0_%u_Y", 128, 159))>;

def R600_KC0_Z : RegisterClass <"AMDGPU", [f32, i32], 32,
                              (add (sequence "KC0_%u_Z", 128, 159))>;

def R600_KC0_W : RegisterClass <"AMDGPU", [f32, i32], 32,
                              (add (sequence "KC0_%u_W", 128, 159))>;

def R600_KC0 : RegisterClass <"AMDGPU", [f32, i32], 32,
                                   (interleave R600_KC0_X, R600_KC0_Y,
                                               R600_KC0_Z, R600_KC0_W)>;

def R600_KC1_X : RegisterClass <"AMDGPU", [f32, i32], 32,
                              (add (sequence "KC1_%u_X", 160, 191))>;

def R600_KC1_Y : RegisterClass <"AMDGPU", [f32, i32], 32,
                              (add (sequence "KC1_%u_Y", 160, 191))>;

def R600_KC1_Z : RegisterClass <"AMDGPU", [f32, i32], 32,
                              (add (sequence "KC1_%u_Z", 160, 191))>;

def R600_KC1_W : RegisterClass <"AMDGPU", [f32, i32], 32,
                              (add (sequence "KC1_%u_W", 160, 191))>;

def R600_KC1 : RegisterClass <"AMDGPU", [f32, i32], 32,
                                   (interleave R600_KC1_X, R600_KC1_Y,
                                               R600_KC1_Z, R600_KC1_W)>;

} // End isAllocatable = 0

def R600_TReg32_X : RegisterClass <"AMDGPU", [f32, i32], 32,
                                   (add (sequence "T%u_X", 0, 127), AR_X)>;

def R600_TReg32_Y : RegisterClass <"AMDGPU", [f32, i32], 32,
                                   (add (sequence "T%u_Y", 0, 127))>;

def R600_TReg32_Z : RegisterClass <"AMDGPU", [f32, i32], 32,
                                   (add (sequence "T%u_Z", 0, 127))>;

def R600_TReg32_W : RegisterClass <"AMDGPU", [f32, i32], 32,
                                   (add (sequence "T%u_W", 0, 127))>;

def R600_TReg32 : RegisterClass <"AMDGPU", [f32, i32], 32,
                                   (interleave R600_TReg32_X, R600_TReg32_Y,
                                               R600_TReg32_Z, R600_TReg32_W)>;

def R600_Reg32 : RegisterClass <"AMDGPU", [f32, i32], 32, (add
    R600_TReg32,
    R600_ArrayBase,
    R600_Addr,
    R600_KC0, R600_KC1,
    ZERO, HALF, ONE, ONE_INT, PV_X, ALU_LITERAL_X, NEG_ONE, NEG_HALF,
    ALU_CONST, ALU_PARAM, OQAP
    )>;

def R600_Predicate : RegisterClass <"AMDGPU", [i32], 32, (add
    PRED_SEL_OFF, PRED_SEL_ZERO, PRED_SEL_ONE)>;

def R600_Predicate_Bit: RegisterClass <"AMDGPU", [i32], 32, (add
    PREDICATE_BIT)>;

def R600_Reg128 : RegisterClass<"AMDGPU", [v4f32, v4i32], 128,
                                (add (sequence "T%u_XYZW", 0, 127))> {
  let CopyCost = -1;
}

def R600_Reg64 : RegisterClass<"AMDGPU", [v2f32, v2i32], 64,
                                (add (sequence "T%u_XY", 0, 63))>;
