//===- Sparc.td - Target Description for Sparc V9 Target --------*- C++ -*-===//
// vim:ft=cpp
//===----------------------------------------------------------------------===//

#include "../Target.td"

#include "SparcV9_Reg.td"

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

class InstV9 : Instruction {          // Sparc instruction baseline
  field bits<32> Inst;

  set Namespace = "V9";

  bits<2> op;
  set Inst{31-30} = op;               // Top two bits are the 'op' field

  // Bit attributes specific to Sparc instructions
  bit isPasi       = 0; // Does this instruction affect an alternate addr space?
  bit isDeprecated = 0; // Is this instruction deprecated?
  bit isPrivileged = 0; // Is this a privileged instruction?
}

#include "SparcV9_F2.td"
#include "SparcV9_F3.td"
#include "SparcV9_F4.td"

//===----------------------------------------------------------------------===//
// Instruction list...
//

// Section A.2: Add - p137
def ADDr    : F3_1<2, 0b000000, "add">;             // add r, r, r
def ADDi    : F3_2<2, 0b000000, "add">;             // add r, i, r
def ADDccr  : F3_1<2, 0b010000, "addcc">;           // addcc r, r, r
def ADDcci  : F3_2<2, 0b010000, "addcc">;           // addcc r, i, r
def ADDCr   : F3_1<2, 0b001000, "addC">;            // addC r, r, r
def ADDCi   : F3_2<2, 0b001000, "addC">;            // addC r, i, r
def ADDCccr : F3_1<2, 0b011000, "addCcc">;          // addCcc r, r, r
def ADDCcci : F3_2<2, 0b011000, "addCcc">;          // addCcc r, i, r

// Section A.3: Branch on Integer Register with Prediction - p162
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
// Not used in the Sparc backend
#if 0
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
#endif

// Section A.6: Branch on Integer condition codes (Bicc) - p146
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

// Section A.7: Branch on integer condition codes with prediction - p148
// Not used in the Sparc backend
#if 0
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
#endif

// Section A.8: CALL - p151, the only Format #1 instruction
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
#if 0
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
#endif

// Section A.11: DONE and RETRY - p181
// Not used in the Sparc backend
#if 0
set isPrivileged = 1 in {
  def DONE    : F3_18<0, "done">;                 // done
  def RETRY   : F3_18<1, "retry">;                // retry
}
#endif

// Section A.12: Floating-Point Add and Subtract - p182
def FADDS : F3_16<2, 0b110100, 0x41, "fadds">;    // fadds f, f, f
def FADDD : F3_16<2, 0b110100, 0x42, "faddd">;    // faddd f, f, f
def FADDQ : F3_16<2, 0b110100, 0x43, "faddq">;    // faddq f, f, f
def FSUBS : F3_16<2, 0b110100, 0x45, "fsubs">;    // fsubs f, f, f
def FSUBD : F3_16<2, 0b110100, 0x46, "fsubd">;    // fsubd f, f, f
def FSUBQ : F3_16<2, 0b110100, 0x47, "fsubq">;    // fsubq f, f, f

// Section A.13: Floating-point compare - p159
def FCMPS  : F3_15<2, 0b110101, 0b010100001, "fcmps">;   // fcmps  %fcc, r1, r2
def FCMPD  : F3_15<2, 0b110101, 0b010100010, "fcmpd">;   // fcmpd  %fcc, r1, r2
def FCMPQ  : F3_15<2, 0b110101, 0b010100011, "fcmpq">;   // fcmpq  %fcc, r1, r2
def FCMPES : F3_15<2, 0b110101, 0b010100101, "fcmpes">;  // fcmpes %fcc, r1, r2
def FCMPED : F3_15<2, 0b110101, 0b010100110, "fcmped">;  // fcmped %fcc, r1, r2
def FCMPEQ : F3_15<2, 0b110101, 0b010100111, "fcmpeq">;  // fcmpeq %fcc, r1, r2

// Section A.14: Convert floating-point to integer - p161
def FSTOX : F3_14<2, 0b110100, 0b010000001, "fstox">;   // fstox rs2, rd
def FDTOX : F3_14<2, 0b110100, 0b010000010, "fstox">;   // fstox rs2, rd
def FQTOX : F3_14<2, 0b110100, 0b010000011, "fstox">;   // fstox rs2, rd
def FSTOI : F3_14<2, 0b110100, 0b011010001, "fstoi">;   // fstoi rs2, rd
def FDTOI : F3_14<2, 0b110100, 0b011010010, "fdtoi">;   // fdtoi rs2, rd
def FQTOI : F3_14<2, 0b110100, 0b011010011, "fqtoi">;   // fqtoi rs2, rd

// Section A.15: Convert between floating-point formats - p162
def FSTOD : F3_14<2, 0b110100, 0b011001001, "fstod">;   // fstod rs2, rd
def FSTOQ : F3_14<2, 0b110100, 0b011001101, "fstoq">;   // fstoq rs2, rd
def FDTOS : F3_14<2, 0b110100, 0b011000110, "fstos">;   // fstos rs2, rd
def FDTOQ : F3_14<2, 0b110100, 0b011001110, "fdtoq">;   // fdtoq rs2, rd
def FQTOS : F3_14<2, 0b110100, 0b011000111, "fqtos">;   // fqtos rs2, rd
def FQTOD : F3_14<2, 0b110100, 0b011001011, "fqtod">;   // fqtod rs2, rd

// Section A.16: Convert integer to floating-point - p163
def FXTOS : F3_14<2, 0b110100, 0b010000100, "fxtos">;   // fxtos rs2, rd
def FXTOD : F3_14<2, 0b110100, 0b010001000, "fxtod">;   // fxtod rs2, rd
def FXTOQ : F3_14<2, 0b110100, 0b010001100, "fxtoq">;   // fxtoq rs2, rd
def FITOS : F3_14<2, 0b110100, 0b011000100, "fitos">;   // fitos rs2, rd
def FITOD : F3_14<2, 0b110100, 0b011001000, "fitod">;   // fitod rs2, rd
def FITOQ : F3_14<2, 0b110100, 0b011001100, "fitoq">;   // fitoq rs2, rd

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

// FIXME: A.20: Flush Instruction Memory - p167
// FIXME: A.21: Flush Register Windows - p169

// A.22: Illegal instruction Trap - p170
// Not currently used

// A.23: Implementation-Dependent Instructions - p171
// Not currently used

// Section A.24: Jump and Link - p172
// Mimicking the Sparc's instr def...
def JMPLCALLr : F3_1<2, 0b111000, "jmpl">;              // jmpl [r+r], r
def JMPLCALLi : F3_2<2, 0b111000, "jmpl">;              // jmpl [r+i], r
def JMPLRETr  : F3_1<2, 0b111000, "jmpl">;              // jmpl [r+r], r
def JMPLRETi  : F3_2<2, 0b111000, "jmpl">;              // jmpl [r+i], r

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
#if 0
set isDeprecated = 1 in {
  def LDDr : F3_1<3, 0b000011, "ldd">;            // ldd [r+r], r
  def LDDi : F3_2<3, 0b000011, "ldd">;            // ldd [r+i], r
}
#endif

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

// Section A.32: Memory Barrier - p186
// Not currently used in the Sparc backend

// Section A.33: Move Floating-Point Register on Condition (FMOVcc)
#if 0
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

// Section A.34: Move FP Register on Integer Register condition (FMOVr) - p192
def FMOVRSZ   : F4_6<2, 0b110101, 0b001, 0b00101, "fmovrsz">;  //fmovsrz r,r,rd
def FMOVRSLEZ : F4_6<2, 0b110101, 0b010, 0b00101, "fmovrslez">;//fmovsrz r,r,rd
def FMOVRSLZ  : F4_6<2, 0b110101, 0b011, 0b00101, "fmovrslz">; //fmovsrz r,r,rd
def FMOVRSNZ  : F4_6<2, 0b110101, 0b101, 0b00101, "fmovrsne">; //fmovsrz r,r,rd
def FMOVRSGZ  : F4_6<2, 0b110101, 0b110, 0b00101, "fmovrsgz">; //fmovsrz r,r,rd
def FMOVRSGEZ : F4_6<2, 0b110101, 0b111, 0b00101, "fmovrsgez">;//fmovsrz r,r,rd

def FMOVRDZ   : F4_6<2, 0b110101, 0b001, 0b00110, "fmovrdz">;  //fmovsrz r,r,rd
def FMOVRDLEZ : F4_6<2, 0b110101, 0b010, 0b00110, "fmovrdlez">;//fmovsrz r,r,rd
def FMOVRDLZ  : F4_6<2, 0b110101, 0b011, 0b00110, "fmovrdlz">; //fmovsrz r,r,rd
def FMOVRDNZ  : F4_6<2, 0b110101, 0b101, 0b00110, "fmovrdne">; //fmovsrz r,r,rd
def FMOVRDGZ  : F4_6<2, 0b110101, 0b110, 0b00110, "fmovrdgz">; //fmovsrz r,r,rd
def FMOVRDGEZ : F4_6<2, 0b110101, 0b111, 0b00110, "fmovrdgez">;//fmovsrz r,r,rd

def FMOVRQZ   : F4_6<2, 0b110101, 0b001, 0b00111, "fmovrqz">;  //fmovsrz r,r,rd
def FMOVRQLEZ : F4_6<2, 0b110101, 0b010, 0b00111, "fmovrqlez">;//fmovsrz r,r,rd
def FMOVRQLZ  : F4_6<2, 0b110101, 0b011, 0b00111, "fmovrqlz">; //fmovsrz r,r,rd
def FMOVRQNZ  : F4_6<2, 0b110101, 0b101, 0b00111, "fmovrqne">; //fmovsrz r,r,rd
def FMOVRQGZ  : F4_6<2, 0b110101, 0b110, 0b00111, "fmovrqgz">; //fmovsrz r,r,rd
def FMOVRQGEZ : F4_6<2, 0b110101, 0b111, 0b00111, "fmovrqgez">;//fmovsrz r,r,rd


// Section A.35: Move Integer Register on Condition (MOVcc) - p194
// For integer condition codes
def MOVAr    : F4_3<2, 0b101100, 0b1000, "mova">;      // mova i/xcc, rs2, rd
def MOVAi    : F4_4<2, 0b101100, 0b1000, "mova">;      // mova i/xcc, rs2, rd
def MOVNr    : F4_3<2, 0b101100, 0b0000, "movn">;      // mova i/xcc, rs2, rd
def MOVNi    : F4_4<2, 0b101100, 0b0000, "movn">;      // mova i/xcc, rs2, rd
def MOVNEr   : F4_3<2, 0b101100, 0b1001, "movne">;     // mova i/xcc, rs2, rd
def MOVNEi   : F4_4<2, 0b101100, 0b1001, "movne">;     // mova i/xcc, rs2, rd
def MOVEr    : F4_3<2, 0b101100, 0b0001, "move">;      // mova i/xcc, rs2, rd
def MOVEi    : F4_4<2, 0b101100, 0b0001, "move">;      // mova i/xcc, rs2, rd
def MOVGr    : F4_3<2, 0b101100, 0b1010, "movg">;      // mova i/xcc, rs2, rd
def MOVGi    : F4_4<2, 0b101100, 0b1010, "movg">;      // mova i/xcc, rs2, rd
def MOVLEr   : F4_3<2, 0b101100, 0b0010, "movle">;     // mova i/xcc, rs2, rd
def MOVLEi   : F4_4<2, 0b101100, 0b0010, "movle">;     // mova i/xcc, rs2, rd
def MOVGEr   : F4_3<2, 0b101100, 0b1011, "movge">;     // mova i/xcc, rs2, rd
def MOVGEi   : F4_4<2, 0b101100, 0b1011, "movge">;     // mova i/xcc, rs2, rd
def MOVLr    : F4_3<2, 0b101100, 0b0011, "movl">;      // mova i/xcc, rs2, rd
def MOVLi    : F4_4<2, 0b101100, 0b0011, "movl">;      // mova i/xcc, rs2, rd
def MOVGUr   : F4_3<2, 0b101100, 0b1100, "movgu">;     // mova i/xcc, rs2, rd
def MOVGUi   : F4_4<2, 0b101100, 0b1100, "movgu">;     // mova i/xcc, rs2, rd
def MOVLEUr  : F4_3<2, 0b101100, 0b0100, "movleu">;    // mova i/xcc, rs2, rd
def MOVLEUi  : F4_4<2, 0b101100, 0b0100, "movleu">;    // mova i/xcc, rs2, rd
def MOVCCr   : F4_3<2, 0b101100, 0b1101, "movcc">;     // mova i/xcc, rs2, rd
def MOVCCi   : F4_4<2, 0b101100, 0b1101, "movcc">;     // mova i/xcc, rs2, rd
def MOVCSr   : F4_3<2, 0b101100, 0b0101, "movcs">;     // mova i/xcc, rs2, rd
def MOVCSi   : F4_4<2, 0b101100, 0b0101, "movcs">;     // mova i/xcc, rs2, rd
def MOVPOSr  : F4_3<2, 0b101100, 0b1110, "movpos">;    // mova i/xcc, rs2, rd
def MOVPOSi  : F4_4<2, 0b101100, 0b1110, "movpos">;    // mova i/xcc, rs2, rd
def MOVNEGr  : F4_3<2, 0b101100, 0b0110, "movneg">;    // mova i/xcc, rs2, rd
def MOVNEGi  : F4_4<2, 0b101100, 0b0110, "movneg">;    // mova i/xcc, rs2, rd
def MOVVCr   : F4_3<2, 0b101100, 0b1111, "movvc">;     // mova i/xcc, rs2, rd
def MOVVCi   : F4_4<2, 0b101100, 0b1111, "movvc">;     // mova i/xcc, rs2, rd
def MOVVSr   : F4_3<2, 0b101100, 0b0111, "movvs">;     // mova i/xcc, rs2, rd
def MOVVSi   : F4_4<2, 0b101100, 0b0111, "movvs">;     // mova i/xcc, rs2, rd

// For floating-point condition codes
def MOVFAr   : F4_3<2, 0b101100, 0b1000, "movfa">;     // mova i/xcc, rs2, rd
def MOVFAi   : F4_4<2, 0b101100, 0b1000, "movfa">;     // mova i/xcc, rs2, rd
def MOVFNr   : F4_3<2, 0b101100, 0b0000, "movfn">;     // mova i/xcc, rs2, rd
def MOVFNi   : F4_4<2, 0b101100, 0b0000, "movfn">;     // mova i/xcc, rs2, rd
def MOVFUr   : F4_3<2, 0b101100, 0b0111, "movfu">;     // mova i/xcc, rs2, rd
def MOVFUi   : F4_4<2, 0b101100, 0b0111, "movfu">;     // mova i/xcc, rs2, rd
def MOVFGr   : F4_3<2, 0b101100, 0b0110, "movfg">;     // mova i/xcc, rs2, rd
def MOVFGi   : F4_4<2, 0b101100, 0b0110, "movfg">;     // mova i/xcc, rs2, rd
def MOVFUGr  : F4_3<2, 0b101100, 0b0101, "movfug">;    // mova i/xcc, rs2, rd
def MOVFUGi  : F4_4<2, 0b101100, 0b0101, "movfug">;    // mova i/xcc, rs2, rd
def MOVFLr   : F4_3<2, 0b101100, 0b0100, "movfl">;     // mova i/xcc, rs2, rd
def MOVFLi   : F4_4<2, 0b101100, 0b0100, "movfl">;     // mova i/xcc, rs2, rd
def MOVFULr  : F4_3<2, 0b101100, 0b0011, "movful">;    // mova i/xcc, rs2, rd
def MOVFULi  : F4_4<2, 0b101100, 0b0011, "movful">;    // mova i/xcc, rs2, rd
def MOVFLGr  : F4_3<2, 0b101100, 0b0010, "movflg">;    // mova i/xcc, rs2, rd
def MOVFLGi  : F4_4<2, 0b101100, 0b0010, "movflg">;    // mova i/xcc, rs2, rd
def MOVFNEr  : F4_3<2, 0b101100, 0b0001, "movfne">;    // mova i/xcc, rs2, rd
def MOVFNEi  : F4_4<2, 0b101100, 0b0001, "movfne">;    // mova i/xcc, rs2, rd
def MOVFEr   : F4_3<2, 0b101100, 0b1001, "movfe">;     // mova i/xcc, rs2, rd
def MOVFEi   : F4_4<2, 0b101100, 0b1001, "movfe">;     // mova i/xcc, rs2, rd
def MOVFUEr  : F4_3<2, 0b101100, 0b1010, "movfue">;    // mova i/xcc, rs2, rd
def MOVFUEi  : F4_4<2, 0b101100, 0b1010, "movfue">;    // mova i/xcc, rs2, rd
def MOVFGEr  : F4_3<2, 0b101100, 0b1011, "movfge">;    // mova i/xcc, rs2, rd
def MOVFGEi  : F4_4<2, 0b101100, 0b1011, "movfge">;    // mova i/xcc, rs2, rd
def MOVFUGEr : F4_3<2, 0b101100, 0b1100, "movfuge">;   // mova i/xcc, rs2, rd
def MOVFUGEi : F4_4<2, 0b101100, 0b1100, "movfuge">;   // mova i/xcc, rs2, rd
def MOVFLEr  : F4_3<2, 0b101100, 0b1101, "movfle">;    // mova i/xcc, rs2, rd
def MOVFLEi  : F4_4<2, 0b101100, 0b1101, "movfle">;    // mova i/xcc, rs2, rd
def MOVFULEr : F4_3<2, 0b101100, 0b1110, "movfule">;   // mova i/xcc, rs2, rd
def MOVFULEi : F4_4<2, 0b101100, 0b1110, "movfule">;   // mova i/xcc, rs2, rd
def MOVFOr   : F4_3<2, 0b101100, 0b1111, "movfo">;     // mova i/xcc, rs2, rd
def MOVFOi   : F4_4<2, 0b101100, 0b1111, "movfo">;     // mova i/xcc, rs2, rd

// Section A.36: Move Integer Register on Register Condition (MOVR)
def MOVRZr   : F3_5<2, 0b101111, 0b001, "movrz">;      // movrz rs1, rs2, rd
def MOVRZi   : F3_6<2, 0b101111, 0b001, "movrz">;      // movrz rs1, imm, rd
def MOVRLEZr : F3_5<2, 0b101111, 0b010, "movrlez">;    // movrz rs1, rs2, rd
def MOVRLEZi : F3_6<2, 0b101111, 0b010, "movrlez">;    // movrz rs1, imm, rd
def MOVRLZr  : F3_5<2, 0b101111, 0b011, "movrlz">;     // movrz rs1, rs2, rd
def MOVRLZi  : F3_6<2, 0b101111, 0b011, "movrlz">;     // movrz rs1, imm, rd
def MOVRNZr  : F3_5<2, 0b101111, 0b101, "movrnz">;     // movrz rs1, rs2, rd
def MOVRNZi  : F3_6<2, 0b101111, 0b101, "movrnz">;     // movrz rs1, imm, rd
def MOVRGZr  : F3_5<2, 0b101111, 0b110, "movrgz">;     // movrz rs1, rs2, rd
def MOVRGZi  : F3_6<2, 0b101111, 0b110, "movrgz">;     // movrz rs1, imm, rd
def MOVRGEZr : F3_5<2, 0b101111, 0b111, "movrgez">;    // movrz rs1, rs2, rd
def MOVRGEZi : F3_6<2, 0b101111, 0b111, "movrgez">;    // movrz rs1, imm, rd

// Section A.37: Multiply and Divide (64-bit) - p199
def MULXr  : F3_1<2, 0b001001, "mulx">;        // mulx  r, r, r
def SDIVXr : F3_1<2, 0b101101, "sdivx">;       // mulx  r, r, r
def UDIVXr : F3_1<2, 0b001101, "udivx">;       // mulx  r, r, r
def MULXi  : F3_2<2, 0b001001, "mulx">;        // mulx  r, i, r
def SDIVXi : F3_2<2, 0b101101, "sdivx">;       // mulx  r, i, r
def UDIVXi : F3_2<2, 0b001101, "udivx">;       // mulx  r, i, r

// Section A.38: Multiply (32-bit) - p200
// Not used in the Sparc backend
#if 0
set Inst{13} = 0 in {
  def UMULr   : F3_1<2, 0b001010, "umul">;        // umul   r, r, r
  def SMULr   : F3_1<2, 0b001011, "smul">;        // smul   r, r, r
  def UMULCCr : F3_1<2, 0b011010, "umulcc">;      // mulcc  r, r, r
  def SMULCCr : F3_1<2, 0b011011, "smulcc">;      // smulcc r, r, r
}
set Inst{13} = 1 in {
  def UMULi   : F3_1<2, 0b001010, "umul">;        // umul   r, i, r
  def SMULi   : F3_1<2, 0b001011, "smul">;        // smul   r, i, r
  def UMULCCi : F3_1<2, 0b011010, "umulcc">;      // umulcc r, i, r
  def SMULCCi : F3_1<2, 0b011011, "smulcc">;      // smulcc r, i, r
}
#endif

// Section A.39: Multiply Step - p202
// Not currently used in the Sparc backend

// Section A.40: No operation - p204
// NOP is really a pseudo-instruction (special case of SETHI)
set op2 = 0b100 in {
  set rd = 0 in {
    set imm = 0 in {
      def NOP : F2_1<"nop">;                     // nop
    }
  }
}

// Section A.41: Population Count - p205
// Not currently used in the Sparc backend

// Section A.42: Prefetch Data - p206
// Not currently used in the Sparc backend

// Section A.43: Read Privileged Register - p211
// Not currently used in the Sparc backend

// Section A.44: Read State Register
// The only instr from this section currently used is RDCCR
set rs1 = 2 in {
  def RDCCR : F3_17<2, 0b101000, "rd">;             // rd %ccr, r
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
// Not currently used in Sparc backend

// Section A.48: SETHI - p220
set op2 = 0b100 in {
  def SETHI : F2_1<"sethi">;                      // sethi
}

// Section A.49: Shift - p221
// Not currently used in the Sparc backend
#if 0
 uses 5 least significant bits of rs2
set x = 0 in {
  def SLLr5  : F3_11<2, 0b100101, "sll">;                // sll r, r, r
  def SRLr5  : F3_11<2, 0b100110, "srl">;                // srl r, r, r
  def SRAr5  : F3_11<2, 0b100111, "sra">;                // sra r, r, r
  def SLLXr5 : F3_11<2, 0b100101, "sllx">;               // sllx r, r, r
  def SRLXr5 : F3_11<2, 0b100110, "srlx">;               // srlx r, r, r
  def SRAXr5 : F3_11<2, 0b100111, "srax">;               // srax r, r, r
}
#endif

// uses 6 least significant bits of rs2
set x = 1 in {
  def SLLr6  : F3_11<2, 0b100101, "sll">;                // sll r, r, r
  def SRLr6  : F3_11<2, 0b100110, "srl">;                // srl r, r, r
  def SRAr6  : F3_11<2, 0b100111, "sra">;                // sra r, r, r
  def SLLXr6 : F3_11<2, 0b100101, "sllx">;               // sllx r, r, r
  def SRLXr6 : F3_11<2, 0b100110, "srlx">;               // srlx r, r, r
  def SRAXr6 : F3_11<2, 0b100111, "srax">;               // srax r, r, r
}

// Not currently used in the Sparc backend
#if 0
def SLLi5  : F3_12<2, 0b100101, "sll">;                // sll r, shcnt32, r
def SRLi5  : F3_12<2, 0b100110, "srl">;                // srl r, shcnt32, r
def SRAi5  : F3_12<2, 0b100111, "sra">;                // sra r, shcnt32, r
def SLLXi5 : F3_12<2, 0b100101, "sllx">;               // sllx r, shcnt32, r
def SRLXi5 : F3_12<2, 0b100110, "srlx">;               // srlx r, shcnt32, r
def SRAXi5 : F3_12<2, 0b100111, "srax">;               // srax r, shcnt32, r
#endif

def SLLi6  : F3_13<2, 0b100101, "sll">;                  // sll r, shcnt64, r
def SRLi6  : F3_13<2, 0b100110, "srl">;                  // srl r, shcnt64, r
def SRAi6  : F3_13<2, 0b100111, "sra">;                  // sra r, shcnt64, r
def SLLXi6 : F3_13<2, 0b100101, "sllx">;                 // sllx r, shcnt64, r
def SRLXi6 : F3_13<2, 0b100110, "srlx">;                 // srlx r, shcnt64, r
def SRAXi6 : F3_13<2, 0b100111, "srax">;                 // srax r, shcnt64, r

// Section A.50: Sofware-Initiated Reset - p223
// Not currently used in the Sparc backend

// Section A.51: Store Barrier - p224
// Not currently used in the Sparc backend

// Section A.52: Store Floating-point -p225
def STFr  : F3_1rd<3, 0b100100, "st">;                      // st r, [r+r]
def STFi  : F3_2rd<3, 0b100100, "st">;                      // st r, [r+i]
def STDFr : F3_1rd<3, 0b100111, "std">;                     // std r, [r+r]
def STDFi : F3_2rd<3, 0b100111, "std">;                     // std r, [r+i]

// Not currently used in the Sparc backend
#if 0
def STQFr : F3_1rd<3, 0b100110, "stq">;                     // stq r, [r+r]
def STQFi : F3_2rd<3, 0b100110, "stq">;                     // stq r, [r+i]
#endif

set isDeprecated = 1 in {
  def STFSRr : F3_1<3, 0b100101, "st">;                   // st r, [r+r]
  def STFSRi : F3_2<3, 0b100101, "st">;                   // st r, [r+i]
}
def STXFSRr : F3_1<3, 0b100101, "stq">;                   // stx r, [r+r]
def STXFSRi : F3_2<3, 0b100101, "stq">;                   // stx r, [r+i]

// Section A.53: Store Floating-Point into Alternate Space - p227
// Not currently used in the Sparc backend

// Section A.54: Store Integer - p229
def STBr : F3_1<3, 0b000101, "stb">;                     // stb r, [r+r]
def STBi : F3_2<3, 0b000101, "stb">;                     // stb r, [r+i]
def STHr : F3_1<3, 0b000110, "stb">;                     // stb r, [r+r]
def STHi : F3_2<3, 0b000110, "stb">;                     // stb r, [r+i]
def STWr : F3_1<3, 0b000100, "stb">;                     // stb r, [r+r]
def STWi : F3_2<3, 0b000100, "stb">;                     // stb r, [r+i]
def STXr : F3_1<3, 0b001110, "stb">;                     // stb r, [r+r]
def STXi : F3_2<3, 0b001110, "stb">;                     // stb r, [r+i]

// Section A.55: Store Integer into Alternate Space - p231
// Not currently used in the Sparc backend

// Section A.56: Subtract - p233
def SUBr    : F3_1<2, 0b000100, "sub">;                   // sub r, r, r
def SUBi    : F3_1<2, 0b000100, "sub">;                   // sub r, i, r
def SUBccr  : F3_1<2, 0b010100, "subcc">;                 // subcc r, r, r
def SUBcci  : F3_1<2, 0b010100, "subcc">;                 // subcc r, i, r
def SUBCr   : F3_1<2, 0b001100, "subc">;                  // subc r, r, r
def SUBCi   : F3_1<2, 0b001100, "subc">;                  // subc r, i, r
def SUBCccr : F3_1<2, 0b011100, "subccc">;                // subccc r, r, r
def SUBCcci : F3_1<2, 0b011100, "subccc">;                // subccc r, i, r

// FIXME: More...?
