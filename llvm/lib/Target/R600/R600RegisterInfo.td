
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

    // Indirect addressing offset registers
    def Addr#Index#_#Chan : R600RegWithChan <"T("#Index#" + AR.x)."#Chan,
                                              Index, Chan>;
    def TRegMem#Index#_#Chan : R600RegWithChan <"T"#Index#"."#Chan, Index,
                                                Chan>;
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
def AR_X : R600Reg<"AR.x", 0>;

def R600_ArrayBase : RegisterClass <"AMDGPU", [f32, i32], 32,
                          (add (sequence "ArrayBase%u", 448, 464))>;
// special registers for ALU src operands
// const buffer reference, SRCx_SEL contains index
def ALU_CONST : R600Reg<"CBuf", 0>;
// interpolation param reference, SRCx_SEL contains index
def ALU_PARAM : R600Reg<"Param", 0>;

let isAllocatable = 0 in {

// XXX: Only use the X channel, until we support wider stack widths
def R600_Addr : RegisterClass <"AMDGPU", [i32], 127, (add (sequence "Addr%u_X", 0, 127))>;

} // End isAllocatable = 0

def R600_TReg32_X : RegisterClass <"AMDGPU", [f32, i32], 32,
                                   (add (sequence "T%u_X", 0, 127))>;

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
    ZERO, HALF, ONE, ONE_INT, PV_X, ALU_LITERAL_X, NEG_ONE, NEG_HALF,
    ALU_CONST, ALU_PARAM
    )>;

def R600_Predicate : RegisterClass <"AMDGPU", [i32], 32, (add
    PRED_SEL_OFF, PRED_SEL_ZERO, PRED_SEL_ONE)>;

def R600_Predicate_Bit: RegisterClass <"AMDGPU", [i32], 32, (add
    PREDICATE_BIT)>;

def R600_Reg128 : RegisterClass<"AMDGPU", [v4f32, v4i32], 128,
                                (add (sequence "T%u_XYZW", 0, 127))> {
  let CopyCost = -1;
}

//===----------------------------------------------------------------------===//
// Register classes for indirect addressing
//===----------------------------------------------------------------------===//

// Super register for all the Indirect Registers.  This register class is used
// by the REG_SEQUENCE instruction to specify the registers to use for direct
// reads / writes which may be written / read by an indirect address.
class IndirectSuper<string n, list<Register> subregs> :
    RegisterWithSubRegs<n, subregs> {
  let Namespace = "AMDGPU";
  let SubRegIndices =
 [indirect_0,indirect_1,indirect_2,indirect_3,indirect_4,indirect_5,indirect_6,
 indirect_7,indirect_8,indirect_9,indirect_10,indirect_11,indirect_12,
 indirect_13,indirect_14,indirect_15];
}

def IndirectSuperReg : IndirectSuper<"Indirect",
  [TRegMem0_X, TRegMem1_X, TRegMem2_X, TRegMem3_X, TRegMem4_X, TRegMem5_X,
   TRegMem6_X, TRegMem7_X, TRegMem8_X, TRegMem9_X, TRegMem10_X, TRegMem11_X,
   TRegMem12_X, TRegMem13_X, TRegMem14_X, TRegMem15_X]
>;

def IndirectReg : RegisterClass<"AMDGPU", [f32, i32], 32, (add IndirectSuperReg)>;

// This register class defines the registers that are the storage units for
// the "Indirect Addressing" pseudo memory space.
// XXX: Only use the X channel, until we support wider stack widths
def TRegMem : RegisterClass<"AMDGPU", [f32, i32], 32,
  (add (sequence "TRegMem%u_X", 0, 16))
>;
