//===- Sparc.td - Target Description for Sparc V9 Target --------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Target-Independent interface
//===----------------------------------------------------------------------===//

class Register {
  string Namespace = "";
  int Size;
}

class Instruction {
  string Name;          // The opcode string for this instruction
  string Namespace = "";

  list<Register> Uses = [];  // Default to using no non-operand registers
  list<Register> Defs = [];  // Default to modifying no non-operand registers

  // These bits capture information about the high-level semantics of the
  // instruction.
  bit isReturn     = 0;      // Is this instruction a return instruction?
  bit isBranch     = 0;      // Is this instruction a branch instruction?
  bit isCall       = 0;      // Is this instruction a call instruction?
}


//===----------------------------------------------------------------------===//
//  Declarations that describe the Sparc register file 
//===----------------------------------------------------------------------===//

class V9Reg : Register { set Namespace = "SparcV9"; }

// Ri - One of the 32 64 bit integer registers
class Ri<bits<5> num> : V9Reg { set Size = 64; field bits<5> Num = num; }

def G0 : Ri<0>;    def G1 : Ri<1>;    def G2 : Ri<2>;    def G3 : Ri<3>;
// ...


//===----------------------------------------------------------------------===//
// This is temporary testing stuff.....
//===----------------------------------------------------------------------===//

class InstV9 : Instruction {          // Sparc instruction baseline
  field bits<32> Inst;

  set Namespace = "SparcV9";

  bits<2> op;
  set Inst{31-30} = op;               // Top two bits are the 'op' field

  // Bit attributes specific to Sparc instructions
  bit isPasi       = 0; // Does this instruction affect an alternate addr space?
  bit isDeprecated = 0; // Is this instruction deprecated?
  bit isPrivileged = 0; // Is this a privileged instruction?
}


//===----------------------------------------------------------------------===//
// Format #2 classes
//
class F2 : InstV9 {                   // Format 2 instructions
  bits<3> op2;
  set op = 0;    // Op = 0
  set Inst{24-22} = op2;
}

class F2_br : F2 {                    // Format 2 Branch instruction
  bit annul;  // All branches have an annul bit
  set Inst{29} = annul;
  set isBranch    = 1;      // All instances are branch instructions
}

class F2_2<bits<4> cond, string name> : F2_br { // Format 2.2 instructions
  bits<22> disp;

  set Name        = name;
  set Inst{28-25} = cond;
  set Inst{21-0}  = disp;
}

class F2_3<bits<4> cond, string name> : F2_br { // Format 2.3 instructions
  bits<2> cc;
  bits<19> disp;
  bit predict;

  set Name        = name;
  set Inst{28-25} = cond;
  set Inst{21-20} = cc;
  set Inst{19}    = predict;
  set Inst{18-0}  = disp;
}

class F2_4<bits<3> rcond, string name> : F2_br { // Format 2.4 instructions
  // Variables exposed by the instruction...
  bit        predict;
  bits<5>    rs1;
  bits<16>   disp;

  set Name = name;
  set Inst{28}    = 0;
  set Inst{27-25} = rcond;
  //  Inst{24-22} = op2 field
  set Inst{21-20} = disp{15-14};
  set Inst{19}    = predict;
  set Inst{18-14} = rs1;
  set Inst{13-0 } = disp{13-0};
}


//===----------------------------------------------------------------------===//
// Format #3 classes
//

// F3 - Common superclass of all F3 instructions.  All instructions have an op3
// field.
class F3 : InstV9 {
  bits<6> op3;
  set op{1} = 1;   // Op = 2 or 3
  set Inst{24-19} = op3;
}

// F3_rs1 - Common superclass of instructions that use rs1
class F3_rs1 : F3 {
  bits<5> rs1;
  set Inst{18-14} = rs1;
}

// F3_rs1rd - Common superclass of instructions that use rs1 & rd...
class F3_rs1rd : F3_rs1 {
  bits<5> rd;
  set Inst{29-25} = rd;
}

// F3_rs1rdrs2 - Common superclass of instructions with rs1, rd, & rs2 fields
class F3_rs1rdrs2 : F3_rs1 {
  bits<5> rs2;
  set Inst{4-0} = rs2;
}

// Specific F3 classes...
//

class F3_1<bits<2> opVal,   bits<6> op3val,   string name> : F3_rs1rdrs2 {
  set op  = opVal;
  set op3 = op3val;
  set Name = name;
  set Inst{13} = 0;   // i field = 0
  //set Inst{12-5} = dontcare;
}

class F3_2<bits<2> opVal,   bits<6> op3val,   string name> : F3_rs1rd {
  bits<13> simm;

  set op  = opVal;
  set op3 = op3val;
  set Name = name;
  set Inst{13} = 1;   // i field = 1
  set Inst{12-0} = simm;
}

class F3_3<bits<2> opVal,   bits<6> op3val,   string name> : F3_rs1 {
  bits<5> rs2;
  set op   = opVal;
  set op3  = op3val;
  set Name = name;
  //set Inst{29-25} = dontcare;
  set Inst{13}  = 0;
  //set Inst{12-5} = dontcare;
  set Inst{4-0} = rs2;
}

class F3_4<bits<2> opVal,   bits<6> op3val,   string name> : F3_rs1 {
  bits<13> simm;
  set op   = opVal;
  set op3  = op3val;
  set Name = name;
  //set Inst{29-25} = dontcare;
  set Inst{13}  = 1;
  set Inst{12-0} = simm;
}



class F3_16<bits<2> opVal,  bits<6> op3val,
            bits<9> opfval, string name> : F3_rs1rdrs2 {
  set op  = opVal;
  set op3 = op3val;
  set Name = name;
  set Inst{13-5} = opfval;
}

class F3_18<bits<5> fcn, string name> : F3 {
  set op = 2;
  set op3 = 0b111110;
  set Name = name;
  set Inst{29-25} = fcn;
  //set Inst{18-0 } = dontcare;
}

//===----------------------------------------------------------------------===//
// Instruction list...
//

// Section A.2: p161
def ADDr    : F3_1<2, 0b000000, "add">;             // add r, r, r
def ADDi    : F3_2<2, 0b000000, "add">;             // add r, r, i
def ADDCCr  : F3_1<2, 0b010000, "addcc">;           // addcc r, r, r
def ADDCCi  : F3_2<2, 0b010000, "addcc">;           // addcc r, r, i
def ADDCr   : F3_1<2, 0b001000, "addC">;            // addC r, r, r
def ADDCi   : F3_2<2, 0b001000, "addC">;            // addC r, r, i
def ADDCCCr : F3_1<2, 0b011000, "addCcc">;          // addCcc r, r, r
def ADDCCCi : F3_2<2, 0b011000, "addCcc">;          // addCcc r, r, i

// Section A.3: p162
set op2 = 0b011 in {
  def BRZ     : F2_4<0b001, "brz">;                 // Branch on rs1 == 0
  def BRLEZ   : F2_4<0b010, "brlez">;               // Branch on rs1 <= 0
  def BRLZ    : F2_4<0b011, "brlz">;                // Branch on rs1 <  0
  def BRNZ    : F2_4<0b101, "brnz">;                // Branch on rs1 != 0
  def BRGZ    : F2_4<0b110, "brgz">;                // Branch on rs1 >  0
  def BRGEZ   : F2_4<0b111, "brgez">;               // Branch on rs1 >= 0
}

// Section A.4: p164
set isDeprecated = 1 in {
  set op2 = 0b110 in {
    def FBA     : F2_2<0b1000, "fba">;              // Branch always
    def FBN     : F2_2<0b0000, "fbn">;              // Branch never
    def FBU     : F2_2<0b0111, "fbu">;              // Branch on unordered
    def FBG     : F2_2<0b0110, "fbg">;              // Branch >
    def FBUG    : F2_2<0b0101, "fbug">;             // Branch on unordered or >
    def FBL     : F2_2<0b0100, "fbl">;              // Branch <
    def FBUL    : F2_2<0b0011, "fbul">;             // Branch on unordered or <
    def FBLG    : F2_2<0b0010, "fblg">;             // Branch < or >
    def FBNE    : F2_2<0b0001, "fbne">;             // Branch !=
    def FBE     : F2_2<0b1001, "fbe">;              // Branch ==
    def FBUE    : F2_2<0b1010, "fbue">;             // Branch on unordered or ==
    def FBGE    : F2_2<0b1011, "fbge">;             // Branch > or ==
    def FBUGE   : F2_2<0b1100, "fbuge">;            // Branch unord or > or ==
    def FBLE    : F2_2<0b1101, "fble">;             // Branch < or ==
    def FBULE   : F2_2<0b1110, "fbule">;            // Branch unord or < or ==
    def FBO     : F2_2<0b1111, "fbo">;              // Branch on ordered
  }
}

// Section A.5: p167
set op2 = 0b101 in {
  def FBPA     : F2_3<0b1000, "fbpa">;              // Branch always
  def FBPN     : F2_3<0b0000, "fbpn">;              // Branch never
  def FBPU     : F2_3<0b0111, "fbpu">;              // Branch on unordered
  def FBPG     : F2_3<0b0110, "fbpg">;              // Branch >
  def FBPUG    : F2_3<0b0101, "fbpug">;             // Branch on unordered or >
  def FBPL     : F2_3<0b0100, "fbpl">;              // Branch <
  def FBPUL    : F2_3<0b0011, "fbpul">;             // Branch on unordered or <
  def FBPLG    : F2_3<0b0010, "fbplg">;             // Branch < or >
  def FBPNE    : F2_3<0b0001, "fbpne">;             // Branch !=
  def FBPE     : F2_3<0b1001, "fbpe">;              // Branch ==
  def FBPUE    : F2_3<0b1010, "fbpue">;             // Branch on unordered or ==
  def FBPGE    : F2_3<0b1011, "fbpge">;             // Branch > or ==
  def FBPUGE   : F2_3<0b1100, "fbpuge">;            // Branch unord or > or ==
  def FBPLE    : F2_3<0b1101, "fbple">;             // Branch < or ==
  def FBPULE   : F2_3<0b1110, "fbpule">;            // Branch unord or < or ==
  def FBPO     : F2_3<0b1111, "fbpo">;              // Branch on ordered
}

// Section A.6: p170: Bicc
set isDeprecated = 1 in {
  set op2 = 0b010 in {
    def BA     : F2_2<0b1000, "ba">;              // Branch always
    def BN     : F2_2<0b0000, "bn">;              // Branch never
    def BNE    : F2_2<0b1001, "bne">;             // Branch !=
    def BE     : F2_2<0b0001, "be">;              // Branch ==
    def BG     : F2_2<0b1010, "bg">;              // Branch >
    def BLE    : F2_2<0b0010, "ble">;             // Branch <=
    def BGE    : F2_2<0b1011, "bge">;             // Branch >=
    def BL     : F2_2<0b0011, "bl">;              // Branch <
    def BGU    : F2_2<0b1100, "bgu">;             // Branch unsigned >
    def BLEU   : F2_2<0b0100, "bleu">;            // Branch unsigned <=
    def BCC    : F2_2<0b1101, "bcc">;             // Branch unsigned >=
    def BCS    : F2_2<0b0101, "bcs">;             // Branch unsigned <=
    def BPOS   : F2_2<0b1110, "bpos">;            // Branch on positive
    def BNEG   : F2_2<0b0110, "bneg">;            // Branch on negative
    def BVC    : F2_2<0b1111, "bvc">;             // Branch on overflow clear
    def BVS    : F2_2<0b0111, "bvs">;             // Branch on overflow set
  }
}

// Section A.7: p172
set op2 = 0b001 in {
  def BPA     : F2_3<0b1000, "bpa">;              // Branch always
  def BPN     : F2_3<0b0000, "bpn">;              // Branch never
  def BPNE    : F2_3<0b1001, "bpne">;             // Branch !=
  def BPE     : F2_3<0b0001, "bpe">;              // Branch ==
  def BPG     : F2_3<0b1010, "bpg">;              // Branch >
  def BPLE    : F2_3<0b0010, "bple">;             // Branch <=
  def BPGE    : F2_3<0b1011, "bpge">;             // Branch >=
  def BPL     : F2_3<0b0011, "bpl">;              // Branch <
  def BPGU    : F2_3<0b1100, "bpgu">;             // Branch unsigned >

  def BPLEU   : F2_3<0b0100, "bpleu">;            // Branch unsigned <=
  def BPCC    : F2_3<0b1101, "bpcc">;             // Branch unsigned >=
  def BPCS    : F2_3<0b0101, "bpcs">;             // Branch unsigned <=
  def BPPOS   : F2_3<0b1110, "bppos">;            // Branch on positive
  def BPNEG   : F2_3<0b0110, "bpneg">;            // Branch on negative
  def BPVC    : F2_3<0b1111, "bpvc">;             // Branch on overflow clear
  def BPVS    : F2_3<0b0111, "bpvs">;             // Branch on overflow set
}

// Section A.8: p175 - CALL - the only Format #1 instruction
def CALL : InstV9 {
  bits<30> disp;
  set op = 1;
  set Inst{29-0} = disp;
  set Name = "call";
  set isCall = 1;
}

// Section A.9: Compare and Swap - p176
// CASA/CASXA: are for alternate address spaces!  Ignore them


// Section A.10: Divide (64-bit / 32-bit) - p178
set isDeprecated = 1 in {
  def UDIVr   : F3_1<2, 0b001110, "udiv">;        // udiv r, r, r
  def UDIVi   : F3_2<2, 0b001110, "udiv">;        // udiv r, r, i
  def SDIVr   : F3_1<2, 0b001111, "sdiv">;        // sdiv r, r, r
  def SDIVi   : F3_2<2, 0b001111, "sdiv">;        // sdiv r, r, i
  def UDIVCCr : F3_1<2, 0b011110, "udivcc">;      // udivcc r, r, r
  def UDIVCCi : F3_2<2, 0b011110, "udivcc">;      // udivcc r, r, i
  def SDIVCCr : F3_1<2, 0b011111, "sdivcc">;      // sdivcc r, r, r
  def SDIVCCi : F3_2<2, 0b011111, "sdivcc">;      // sdivcc r, r, i
}

// Section A.11: DONE and RETRY - p181
set isPrivileged = 1 in {
  def DONE    : F3_18<0, "done">;                 // done
  def RETRY   : F3_18<1, "retry">;                // retry
}

// Section A.12: Floating-Point Add and Subtract - p182
def FADDS : F3_16<2, 0b110100, 0x41, "fadds">;    // fadds f, f, f
def FADDD : F3_16<2, 0b110100, 0x42, "faddd">;    // faddd f, f, f
def FADDQ : F3_16<2, 0b110100, 0x43, "faddq">;    // faddq f, f, f
def FSUBS : F3_16<2, 0b110100, 0x45, "fsubs">;    // fsubs f, f, f
def FSUBD : F3_16<2, 0b110100, 0x46, "fsubd">;    // fsubd f, f, f
def FSUBQ : F3_16<2, 0b110100, 0x47, "fsubq">;    // fsubq f, f, f

//
// ...
//

// Section A.45: RETURN - p240
set isReturn = 1 in {
  def RETURNr : F3_3<2, 0b111001, "return">;      // return
  def RETURNi : F3_4<2, 0b111001, "return">;      // return
}
