//===- Sparc.td - Target Description for Sparc V9 Target --------*- C++ -*-===//
// vim:ft=cpp
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

def G0 : Ri< 0>;    def G1 : Ri< 1>;    def G2 : Ri< 2>;    def G3 : Ri< 3>;
def G4 : Ri< 4>;    def G5 : Ri< 5>;    def G6 : Ri< 6>;    def G7 : Ri< 7>;
def O0 : Ri< 8>;    def O1 : Ri< 9>;    def O2 : Ri<10>;    def O3 : Ri<11>;
def O4 : Ri<12>;    def O5 : Ri<13>;    def O6 : Ri<14>;    def O7 : Ri<15>;
def L0 : Ri<16>;    def L1 : Ri<17>;    def L2 : Ri<18>;    def L3 : Ri<19>;
def L4 : Ri<20>;    def L5 : Ri<21>;    def L6 : Ri<22>;    def L7 : Ri<23>;
def I0 : Ri<24>;    def I1 : Ri<25>;    def I2 : Ri<26>;    def I3 : Ri<27>;
def I4 : Ri<28>;    def I5 : Ri<29>;    def I6 : Ri<30>;    def I7 : Ri<31>;
// Floating-point registers?
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

// Format 2.1 instructions
class F2_1<string name> : F2 {
  bits<5> rd;
  bits<22> imm;

  set Name = name;
  set Inst{29-25} = rd;
  set Inst{21-0}  = imm;
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
class F3_rs1rd : F3 {
  // Added rs1 here manually to have rd appear before rs1
  // Formerly inherited directly from F3_rs1
  bits<5> rd;
  bits<5> rs1;
  set Inst{29-25} = rd;
  set Inst{18-14} = rs1;
}

// F3_rs1rdrs2 - Common superclass of instructions with rs1, rd, & rs2 fields
class F3_rs1rdrs2 : F3_rs1rd {
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

class F3_4<bits<2> opVal,   bits<6> op3Val,   string name> : F3_rs1 {
  bits<13> simm;
  set op   = opVal;
  set op3  = op3Val;
  set Name = name;
  //set Inst{29-25} = dontcare;
  set Inst{13}  = 1;
  set Inst{12-0} = simm;
}

class F3_11<bits<2> opVal, bits<6> op3Val,    string name> : F3_rs1rdrs2 {
  bit x;
  set op = opVal;
  set op3 = op3Val;
  set Name = name;
  set Inst{13} = 0; // i field = 0
  set Inst{12} = x;
  //set Inst{11-5} = dontcare;
}

class F3_12<bits<2> opVal, bits<6> op3Val,    string name> : F3 {
  bits<5> shcnt;

  set Name = name;
  set Inst{13} = 1; // i field = 1
  set Inst{12} = 0; // x field = 0
  //set Inst{11-5} = dontcare;
  set Inst{4-0} = shcnt;
}

class F3_13<bits<2> opVal, bits<6> op3Val,    string name> : F3 {
  bits<6> shcnt;

  set Name = name;
  set Inst{13} = 1; // i field = 1
  set Inst{12} = 1; // x field = 1
  //set Inst{11-6} = dontcare;
  set Inst{5-0} = shcnt;
}

class F3_14<bits<2> opVal, bits<6> op3val, 
            bits<9> opfval, string name> : F3_rs1rdrs2 {
  set op  = opVal;
  set op3 = op3val;
  set Name = name;
  //set Inst{18-14} = dontcare;
  set Inst{13-5} = opfval;
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
def ADDccr  : F3_1<2, 0b010000, "addcc">;           // addcc r, r, r
def ADDcci  : F3_2<2, 0b010000, "addcc">;           // addcc r, r, i
def ADDCr   : F3_1<2, 0b001000, "addC">;            // addC r, r, r
def ADDCi   : F3_2<2, 0b001000, "addC">;            // addC r, r, i
def ADDCccr : F3_1<2, 0b011000, "addCcc">;          // addCcc r, r, r
def ADDCcci : F3_2<2, 0b011000, "addCcc">;          // addCcc r, r, i

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
//set op2 = 0b101 in {
  //def FBPA     : F2_3<0b1000, "fbpa">;              // Branch always
  //def FBPN     : F2_3<0b0000, "fbpn">;              // Branch never
  //def FBPU     : F2_3<0b0111, "fbpu">;              // Branch on unordered
  //def FBPG     : F2_3<0b0110, "fbpg">;              // Branch >
  //def FBPUG    : F2_3<0b0101, "fbpug">;             // Branch on unordered or >
  //def FBPL     : F2_3<0b0100, "fbpl">;              // Branch <
  //def FBPUL    : F2_3<0b0011, "fbpul">;             // Branch on unordered or <
  //def FBPLG    : F2_3<0b0010, "fbplg">;             // Branch < or >
  //def FBPNE    : F2_3<0b0001, "fbpne">;             // Branch !=
  //def FBPE     : F2_3<0b1001, "fbpe">;              // Branch ==
  //def FBPUE    : F2_3<0b1010, "fbpue">;             // Branch on unordered or ==
  //def FBPGE    : F2_3<0b1011, "fbpge">;             // Branch > or ==
  //def FBPUGE   : F2_3<0b1100, "fbpuge">;            // Branch unord or > or ==
  //def FBPLE    : F2_3<0b1101, "fbple">;             // Branch < or ==
  //def FBPULE   : F2_3<0b1110, "fbpule">;            // Branch unord or < or ==
  //def FBPO     : F2_3<0b1111, "fbpo">;              // Branch on ordered
//}

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
//set op2 = 0b001 in {
//  def BPA     : F2_3<0b1000, "bpa">;              // Branch always
//  def BPN     : F2_3<0b0000, "bpn">;              // Branch never
//  def BPNE    : F2_3<0b1001, "bpne">;             // Branch !=
//  def BPE     : F2_3<0b0001, "bpe">;              // Branch ==
//  def BPG     : F2_3<0b1010, "bpg">;              // Branch >
//  def BPLE    : F2_3<0b0010, "bple">;             // Branch <=
//  def BPGE    : F2_3<0b1011, "bpge">;             // Branch >=
//  def BPL     : F2_3<0b0011, "bpl">;              // Branch <
//  def BPGU    : F2_3<0b1100, "bpgu">;             // Branch unsigned >
//  def BPLEU   : F2_3<0b0100, "bpleu">;            // Branch unsigned <=
//  def BPCC    : F2_3<0b1101, "bpcc">;             // Branch unsigned >=
//  def BPCS    : F2_3<0b0101, "bpcs">;             // Branch unsigned <=
//  def BPPOS   : F2_3<0b1110, "bppos">;            // Branch on positive
//  def BPNEG   : F2_3<0b0110, "bpneg">;            // Branch on negative
//  def BPVC    : F2_3<0b1111, "bpvc">;             // Branch on overflow clear
//  def BPVS    : F2_3<0b0111, "bpvs">;             // Branch on overflow set
//}

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
// Not used in the Sparc backend
//set isDeprecated = 1 in {
  //def UDIVr   : F3_1<2, 0b001110, "udiv">;        // udiv r, r, r
  //def UDIVi   : F3_2<2, 0b001110, "udiv">;        // udiv r, r, i
  //def SDIVr   : F3_1<2, 0b001111, "sdiv">;        // sdiv r, r, r
  //def SDIVi   : F3_2<2, 0b001111, "sdiv">;        // sdiv r, r, i
  //def UDIVCCr : F3_1<2, 0b011110, "udivcc">;      // udivcc r, r, r
  //def UDIVCCi : F3_2<2, 0b011110, "udivcc">;      // udivcc r, r, i
  //def SDIVCCr : F3_1<2, 0b011111, "sdivcc">;      // sdivcc r, r, r
  //def SDIVCCi : F3_2<2, 0b011111, "sdivcc">;      // sdivcc r, r, i
//}

// Section A.11: DONE and RETRY - p181
//set isPrivileged = 1 in {
  //def DONE    : F3_18<0, "done">;                 // done
  //def RETRY   : F3_18<1, "retry">;                // retry
//}

// Section A.12: Floating-Point Add and Subtract - p182
def FADDS : F3_16<2, 0b110100, 0x41, "fadds">;    // fadds f, f, f
def FADDD : F3_16<2, 0b110100, 0x42, "faddd">;    // faddd f, f, f
def FADDQ : F3_16<2, 0b110100, 0x43, "faddq">;    // faddq f, f, f
def FSUBS : F3_16<2, 0b110100, 0x45, "fsubs">;    // fsubs f, f, f
def FSUBD : F3_16<2, 0b110100, 0x46, "fsubd">;    // fsubd f, f, f
def FSUBQ : F3_16<2, 0b110100, 0x47, "fsubq">;    // fsubq f, f, f

// Section A.17: Floating-Point Move - p164
def FMOVS : F3_14<2, 0b110100, 0b000000001, "fmovs">;   // fmovs r, r
def FMOVD : F3_14<2, 0b110100, 0b000000010, "fmovs">;   // fmovd r, r
//def FMOVQ : F3_14<2, 0b110100, 0b000000011, "fmovs">;   // fmovq r, r
def FNEGS : F3_14<2, 0b110100, 0b000000101, "fnegs">;   // fnegs r, r
def FNEGD : F3_14<2, 0b110100, 0b000000110, "fnegs">;   // fnegs r, r
//def FNEGQ : F3_14<2, 0b110100, 0b000000111, "fnegs">;   // fnegs r, r
def FABSS : F3_14<2, 0b110100, 0b000001001, "fabss">;   // fabss r, r
def FABSD : F3_14<2, 0b110100, 0b000001010, "fabss">;   // fabss r, r
//def FABSQ : F3_14<2, 0b110100, 0b000001011, "fabss">;   // fabss r, r

// Section A.18: Floating-Point Multiply and Divide - p165
def FMULS  : F3_16<2, 0b110100, 0b001001001, "fmuls">;   // fmuls r, r, r
def FMULD  : F3_16<2, 0b110100, 0b001001010, "fmuld">;   // fmuld r, r, r
def FMULQ  : F3_16<2, 0b110100, 0b001001011, "fmulq">;   // fmulq r, r, r
def FSMULD : F3_16<2, 0b110100, 0b001101001, "fsmuld">;  // fsmuls r, r, r
def FDMULQ : F3_16<2, 0b110100, 0b001101110, "fdmulq">;  // fdmuls r, r, r
def FDIVS  : F3_16<2, 0b110100, 0b001001101, "fdivs">;   // fdivs r, r, r
def FDIVD  : F3_16<2, 0b110100, 0b001001110, "fdivs">;   // fdivd r, r, r
def FDIVQ  : F3_16<2, 0b110100, 0b001001111, "fdivs">;   // fdivq r, r, r

// Section A.19: Floating-Point Square Root - p166
def FSQRTS : F3_14<2, 0b110100, 0b000101001, "fsqrts">;  // fsqrts r, r
def FSQRTD : F3_14<2, 0b110100, 0b000101010, "fsqrts">;  // fsqrts r, r
def FSQRTQ : F3_14<2, 0b110100, 0b000101011, "fsqrts">;  // fsqrts r, r

// Section A.24: Jump and Link
// Mimicking the Sparc's instr def...
def JMPLCALLr : F3_1<2, 0b111000, "jmpl">;              // jmpl [r+r], r
def JMPLCALLi : F3_1<2, 0b111000, "jmpl">;              // jmpl [r+i], r
def JMPLRETr : F3_1<2, 0b111000, "jmpl">;              // jmpl [r+r], r
def JMPLRETi : F3_1<2, 0b111000, "jmpl">;              // jmpl [r+i], r

// FIXME: FCMPS, FCMPD, FCMPQ !!!
// FIXME: FMULS, FMULD, FMULQ, ...

// Section A.25: Load Floating-Point - p173
def LDFr  : F3_1<3, 0b100000, "ld">;             // ld [r+r], r
def LDFi  : F3_2<3, 0b100000, "ld">;             // ld [r+i], r
def LDDFr : F3_1<3, 0b100011, "ldd">;            // ldd [r+r], r
def LDDFi : F3_2<3, 0b100011, "ldd">;            // ldd [r+i], r
def LDQFr : F3_1<3, 0b100010, "ldq">;            // ldq [r+r], r
def LDQFi : F3_2<3, 0b100010, "ldq">;            // ldq [r+i], r
set isDeprecated = 1 in {
  set rd = 0 in {
    def LDFSRr : F3_1<3, 0b100001, "ld">;        // ld [r+r], r
    def LDFSRi : F3_2<3, 0b100001, "ld">;        // ld [r+i], r
  }
}
set rd = 1 in {
  def LDXFSRr : F3_1<3, 0b100001, "ldx">;         // ldx [r+r], r
  def LDXFSRi : F3_2<3, 0b100001, "ldx">;         // ldx [r+i], r
}

// Section A.27: Load Integer - p178
def LDSBr : F3_1<3, 0b001001, "ldsb">;           // ldsb [r+r], r
def LDSBi : F3_2<3, 0b001001, "ldsb">;           // ldsb [r+i], r
def LDSHr : F3_1<3, 0b001010, "ldsh">;           // ldsh [r+r], r
def LDSHi : F3_2<3, 0b001010, "ldsh">;           // ldsh [r+i], r
def LDSWr : F3_1<3, 0b001000, "ldsw">;           // ldsh [r+r], r 
def LDSWi : F3_2<3, 0b001000, "ldsw">;           // ldsh [r+i], r
def LDUBr : F3_1<3, 0b000001, "ldub">;           // ldub [r+r], r
def LDUBi : F3_2<3, 0b000001, "ldub">;           // ldub [r+i], r
def LDUHr : F3_1<3, 0b000010, "lduh">;           // lduh [r+r], r
def LDUHi : F3_2<3, 0b000010, "lduh">;           // lduh [r+i], r
// synonym: LD
def LDUWr : F3_1<3, 0b000000, "lduw">;           // lduw [r+r], r
def LDUWi : F3_2<3, 0b000000, "lduw">;           // lduw [r+i], r
// LDD should no longer be used, LDX should be used instead
def LDXr : F3_1<3, 0b001011, "ldx">;              // ldx [r+r], r
def LDXi : F3_2<3, 0b001011, "ldx">;              // ldx [r+i], r
//set isDeprecated = 1 in {
//  def LDDr : F3_1<3, 0b000011, "ldd">;            // ldd [r+r], r
//  def LDDi : F3_2<3, 0b000011, "ldd">;            // ldd [r+i], r
//}

// Section A.31: Logical operations
def ANDr    : F3_1<2, 0b000001, "and">;          // and r, r, r
def ANDi    : F3_2<2, 0b000001, "and">;          // and r, r, i
def ANDccr  : F3_1<2, 0b010001, "andcc">;        // andcc r, r, r
def ANDcci  : F3_2<2, 0b010001, "andcc">;        // andcc r, r, i
def ANDNr   : F3_1<2, 0b000101, "andn">;         // andn r, r, r
def ANDNi   : F3_2<2, 0b000101, "andn">;         // andn r, r, i
def ANDNccr : F3_1<2, 0b010101, "andncc">;       // andncc r, r, r
def ANDNcci : F3_2<2, 0b010101, "andncc">;       // andncc r, r, i

def ORr    : F3_1<2, 0b000010, "or">;            // or r, r, r
def ORi    : F3_2<2, 0b000010, "or">;            // or r, r, i
def ORccr  : F3_1<2, 0b010010, "orcc">;          // orcc r, r, r
def ORcci  : F3_2<2, 0b010010, "orcc">;          // orcc r, r, i
def ORNr   : F3_1<2, 0b000110, "orn">;           // orn r, r, r
def ORNi   : F3_2<2, 0b000110, "orn">;           // orn r, r, i
def ORNccr : F3_1<2, 0b010110, "orncc">;         // orncc r, r, r
def ORNcci : F3_2<2, 0b010110, "orncc">;         // orncc r, r, i

def XORr    : F3_1<2, 0b000011, "xor">;          // xor r, r, r
def XORi    : F3_2<2, 0b000011, "xor">;          // xor r, r, i
def XORccr  : F3_1<2, 0b010011, "xorcc">;        // xorcc r, r, r
def XORcci  : F3_2<2, 0b010011, "xorcc">;        // xorcc r, r, i
def XNORr   : F3_1<2, 0b000111, "xnor">;         // xnor r, r, r
def XNORi   : F3_2<2, 0b000111, "xnor">;         // xnor r, r, i
def XNORccr : F3_1<2, 0b010111, "xnorcc">;       // xnorcc r, r, r
def XNORcci : F3_2<2, 0b010111, "xnorcc">;       // xnorcc r, r, i

#if 0
// Section A.33: Move Floating-Point Register on Condition (FMOVcc)
// For integer condition codes
def FMOVA   : F4_7<2, 0b110101, 0b1000, "fmova">;        // fmova r, r
def FMOVN   : F4_7<2, 0b110101, 0b0000, "fmovn">;        // fmovn r, r
def FMOVNE  : F4_7<2, 0b110101, 0b1001, "fmovne">;       // fmovne r, r
def FMOVE   : F4_7<2, 0b110101, 0b0000, "fmove">;        // fmove r, r
def FMOVG   : F4_7<2, 0b110101, 0b1010, "fmovg">;        // fmovg r, r
def FMOVLE  : F4_7<2, 0b110101, 0b0000, "fmovle">;       // fmovle r, r
def FMOVGE  : F4_7<2, 0b110101, 0b1011, "fmovge">;       // fmovge r, r
def FMOVL   : F4_7<2, 0b110101, 0b0011, "fmovl">;        // fmovl r, r
def FMOVGU  : F4_7<2, 0b110101, 0b1100, "fmovgu">;       // fmovgu r, r
def FMOVLEU : F4_7<2, 0b110101, 0b0100, "fmovleu">;      // fmovleu r, r
def FMOVCC  : F4_7<2, 0b110101, 0b1101, "fmovcc">;       // fmovcc r, r
def FMOVCS  : F4_7<2, 0b110101, 0b0101, "fmovcs">;       // fmovcs r, r
def FMOVPOS : F4_7<2, 0b110101, 0b1110, "fmovpos">;      // fmovpos r, r
def FMOVNEG : F4_7<2, 0b110101, 0b0110, "fmovneg">;      // fmovneg r, r
def FMOVVC  : F4_7<2, 0b110101, 0b1111, "fmovvc">;       // fmovvc r, r
def FMOVVS  : F4_7<2, 0b110101, 0b0111, "fmovvs">;       // fmovvs r, r

// For floating-point condition codes
def FMOVFA   : F4_7<2, 0b110101, 0b0100, "fmovfa">;      // fmovfa r, r
def FMOVFN   : F4_7<2, 0b110101, 0b0000, "fmovfn">;      // fmovfa r, r
def FMOVFU   : F4_7<2, 0b110101, 0b0111, "fmovfu">;      // fmovfu r, r
def FMOVFG   : F4_7<2, 0b110101, 0b0110, "fmovfg">;      // fmovfg r, r
def FMOVFUG  : F4_7<2, 0b110101, 0b0101, "fmovfug">;     // fmovfug r, r
def FMOVFL   : F4_7<2, 0b110101, 0b0100, "fmovfl">;      // fmovfl r, r
def FMOVFUL  : F4_7<2, 0b110101, 0b0011, "fmovful">;     // fmovful r, r
def FMOVFLG  : F4_7<2, 0b110101, 0b0010, "fmovflg">;     // fmovflg r, r
def FMOVFNE  : F4_7<2, 0b110101, 0b0001, "fmovfne">;     // fmovfne r, r
def FMOVFE   : F4_7<2, 0b110101, 0b1001, "fmovfe">;      // fmovfe r, r
def FMOVFUE  : F4_7<2, 0b110101, 0b1010, "fmovfue">;     // fmovfue r, r
def FMOVGE   : F4_7<2, 0b110101, 0b1011, "fmovge">;      // fmovge r, r
def FMOVFUGE : F4_7<2, 0b110101, 0b1100, "fmovfuge">;    // fmovfuge r, r
def FMOVFLE  : F4_7<2, 0b110101, 0b1101, "fmovfle">;     // fmovfle r, r
def FMOVFULE : F4_7<2, 0b110101, 0b1110, "fmovfule">;    // fmovfule r, r
def FMOVFO   : F4_7<2, 0b110101, 0b1111, "fmovfo">;      // fmovfo r, r
#endif

// Section A.37: Multiply and Divide (64-bit) - p199
def MULXr  : F3_1<2, 0b001001, "mulx">;        // mulx  r, r, r
def SDIVXr : F3_1<2, 0b101101, "sdivx">;       // mulx  r, r, r
def UDIVXr : F3_1<2, 0b001101, "udivx">;       // mulx  r, r, r
def MULXi  : F3_2<2, 0b001001, "mulx">;        // mulx  r, i, r
def SDIVXi : F3_2<2, 0b101101, "sdivx">;       // mulx  r, i, r
def UDIVXi : F3_2<2, 0b001101, "udivx">;       // mulx  r, i, r

// Section A.38: Multiply (32-bit) - p200
// Not used in the Sparc backend?
//set Inst{13} = 0 in {
//  def UMULr   : F3_1<2, 0b001010, "umul">;        // umul   r, r, r
//  def SMULr   : F3_1<2, 0b001011, "smul">;        // smul   r, r, r
//  def UMULCCr : F3_1<2, 0b011010, "umulcc">;      // mulcc  r, r, r
//  def SMULCCr : F3_1<2, 0b011011, "smulcc">;      // smulcc r, r, r
//}
//set Inst{13} = 1 in {
//  def UMULi   : F3_1<2, 0b001010, "umul">;        // umul   r, i, r
//  def SMULi   : F3_1<2, 0b001011, "smul">;        // smul   r, i, r
//  def UMULCCi : F3_1<2, 0b011010, "umulcc">;      // umulcc r, i, r
//  def SMULCCi : F3_1<2, 0b011011, "smulcc">;      // smulcc r, i, r
//}

// Section A.40: No operation - p204
// NOP is really a pseudo-instruction (special case of SETHI)
set op2 = 0b100 in {
  set rd = 0 in {
    set imm = 0 in {
      def NOP : F2_1<"nop">;                     // nop
    }
  }
}

// Section A.45: RETURN - p216
set isReturn = 1 in {
  def RETURNr : F3_3<2, 0b111001, "return">;      // return
  def RETURNi : F3_4<2, 0b111001, "return">;      // return
}

// Section A.46: SAVE and RESTORE - p217
def SAVEr    : F3_1<2, 0b111100, "save">;       // save    r, r, r
def SAVEi    : F3_2<2, 0b111100, "save">;       // save    r, i, r
def RESTOREr : F3_1<2, 0b111101, "restore">;    // restore r, r, r
def RESTOREi : F3_2<2, 0b111101, "restore">;    // restore r, i, r

// Section A.47: SAVED and RESTORED - p219
// FIXME: add these instrs

// Section A.48: SETHI - p220
set op2 = 0b100 in {
  def SETHI : F2_1<"sethi">;                      // sethi
}

// Section A.49: Shift - p221
// uses 5 least significant bits of rs2
//set x = 0 in {
//  def SLLr5  : F3_11<2, 0b100101, "sll">;                // sll r, r, r
//  def SRLr5  : F3_11<2, 0b100110, "srl">;                // srl r, r, r
//  def SRAr5  : F3_11<2, 0b100111, "sra">;                // sra r, r, r
//  def SLLXr5 : F3_11<2, 0b100101, "sllx">;               // sllx r, r, r
//  def SRLXr5 : F3_11<2, 0b100110, "srlx">;               // srlx r, r, r
//  def SRAXr5 : F3_11<2, 0b100111, "srax">;               // srax r, r, r
//}
// uses 6 least significant bits of rs2
set x = 1 in {
  //  def SLLr6  : F3_11<2, 0b100101, "sll">;                // sll r, r, r
  //  def SRLr6  : F3_11<2, 0b100110, "srl">;                // srl r, r, r
  //  def SRAr6  : F3_11<2, 0b100111, "sra">;                // sra r, r, r
  def SLLXr6 : F3_11<2, 0b100101, "sllx">;               // sllx r, r, r
  def SRLXr6 : F3_11<2, 0b100110, "srlx">;               // srlx r, r, r
  def SRAXr6 : F3_11<2, 0b100111, "srax">;               // srax r, r, r
}

//def SLLi5  : F3_12<2, 0b100101, "sll">;                // sll r, shcnt32, r
//def SRLi5  : F3_12<2, 0b100110, "srl">;                // srl r, shcnt32, r
//def SRAi5  : F3_12<2, 0b100111, "sra">;                // sra r, shcnt32, r
//def SLLXi5 : F3_12<2, 0b100101, "sllx">;               // sllx r, shcnt32, r
//def SRLXi5 : F3_12<2, 0b100110, "srlx">;               // srlx r, shcnt32, r
//def SRAXi5 : F3_12<2, 0b100111, "srax">;               // srax r, shcnt32, r

//def SLLi6  : F3_13<2, 0b100101, "sll">;                // sll r, shcnt64, r
//def SRLi6  : F3_13<2, 0b100110, "srl">;                // srl r, shcnt64, r
//def SRAi6  : F3_13<2, 0b100111, "sra">;                // sra r, shcnt64, r
def SLLXi6 : F3_13<2, 0b100101, "sllx">;                 // sllx r, shcnt64, r
def SRLXi6 : F3_13<2, 0b100110, "srlx">;                 // srlx r, shcnt64, r
def SRAXi6 : F3_13<2, 0b100111, "srax">;                 // srax r, shcnt64, r

// Section A.52: Store Floating-point -p225
def STFr  : F3_1<3, 0b100100, "st">;                      // st r, [r+r]
def STFi  : F3_2<3, 0b100100, "st">;                      // st r, [r+i]
def STDFr : F3_1<3, 0b100111, "std">;                     // std r, [r+r]
def STDFi : F3_2<3, 0b100111, "std">;                     // std r, [r+i]
// Not currently used in the Sparc backend
//def STQFr : F3_1<3, 0b100110, "stq">;                     // stq r, [r+r]
//def STQFi : F3_2<3, 0b100110, "stq">;                     // stq r, [r+i]
set isDeprecated = 1 in {
  def STFSRr : F3_1<3, 0b100101, "st">;                   // st r, [r+r]
  def STFSRi : F3_2<3, 0b100101, "st">;                   // st r, [r+i]
}
def STXFSRr : F3_1<3, 0b100101, "stq">;                   // stx r, [r+r]
def STXFSRi : F3_2<3, 0b100101, "stq">;                   // stx r, [r+i]

// Section A.54: Store Integer - p229
def STBr : F3_1<3, 0b000101, "stb">;                     // stb r, [r+r]
def STBi : F3_2<3, 0b000101, "stb">;                     // stb r, [r+i]
def STHr : F3_1<3, 0b000110, "stb">;                     // stb r, [r+r]
def STHi : F3_2<3, 0b000110, "stb">;                     // stb r, [r+i]
def STWr : F3_1<3, 0b000100, "stb">;                     // stb r, [r+r]
def STWi : F3_2<3, 0b000100, "stb">;                     // stb r, [r+i]
def STXr : F3_1<3, 0b001110, "stb">;                     // stb r, [r+r]
def STXi : F3_2<3, 0b001110, "stb">;                     // stb r, [r+i]

// Floating point store...

// Section A.56: Subtract - p233
def SUBr    : F3_1<2, 0b000100, "sub">;                   // sub r, r, r
def SUBi    : F3_1<2, 0b000100, "sub">;                   // sub r, i, r
def SUBccr  : F3_1<2, 0b010100, "subcc">;                 // subcc r, r, r
def SUBcci  : F3_1<2, 0b010100, "subcc">;                 // subcc r, i, r
def SUBCr   : F3_1<2, 0b001100, "subc">;                  // subc r, r, r
def SUBCi   : F3_1<2, 0b001100, "subc">;                  // subc r, i, r
def SUBCccr : F3_1<2, 0b011100, "subccc">;                // subccc r, r, r
def SUBCcci : F3_1<2, 0b011100, "subccc">;                // subccc r, i, r
