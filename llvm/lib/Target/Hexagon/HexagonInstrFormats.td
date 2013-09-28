//==- HexagonInstrFormats.td - Hexagon Instruction Formats --*- tablegen -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//                         Hexagon Intruction Flags +
//
//                    *** Must match HexagonBaseInfo.h ***
//===----------------------------------------------------------------------===//

class IType<bits<5> t> {
  bits<5> Value = t;
}
def TypePSEUDO : IType<0>;
def TypeALU32  : IType<1>;
def TypeCR     : IType<2>;
def TypeJR     : IType<3>;
def TypeJ      : IType<4>;
def TypeLD     : IType<5>;
def TypeST     : IType<6>;
def TypeSYSTEM : IType<7>;
def TypeXTYPE  : IType<8>;
def TypeENDLOOP: IType<31>;

// Maintain list of valid subtargets for each instruction.
class SubTarget<bits<4> value> {
  bits<4> Value = value;
}

def HasV2SubT     : SubTarget<0xf>;
def HasV2SubTOnly : SubTarget<0x1>;
def NoV2SubT      : SubTarget<0x0>;
def HasV3SubT     : SubTarget<0xe>;
def HasV3SubTOnly : SubTarget<0x2>;
def NoV3SubT      : SubTarget<0x1>;
def HasV4SubT     : SubTarget<0xc>;
def NoV4SubT      : SubTarget<0x3>;
def HasV5SubT     : SubTarget<0x8>;
def NoV5SubT      : SubTarget<0x7>;

// Addressing modes for load/store instructions
class AddrModeType<bits<3> value> {
  bits<3> Value = value;
}

def NoAddrMode     : AddrModeType<0>;  // No addressing mode
def Absolute       : AddrModeType<1>;  // Absolute addressing mode
def AbsoluteSet    : AddrModeType<2>;  // Absolute set addressing mode
def BaseImmOffset  : AddrModeType<3>;  // Indirect with offset
def BaseLongOffset : AddrModeType<4>;  // Indirect with long offset
def BaseRegOffset  : AddrModeType<5>;  // Indirect with register offset
def PostInc        : AddrModeType<6>;  // Post increment addressing mode

class MemAccessSize<bits<3> value> {
  bits<3> Value = value;
}

def NoMemAccess      : MemAccessSize<0>;// Not a memory acces instruction.
def ByteAccess       : MemAccessSize<1>;// Byte access instruction (memb).
def HalfWordAccess   : MemAccessSize<2>;// Half word access instruction (memh).
def WordAccess       : MemAccessSize<3>;// Word access instruction (memw).
def DoubleWordAccess : MemAccessSize<4>;// Double word access instruction (memd)


//===----------------------------------------------------------------------===//
//                         Intruction Class Declaration +
//===----------------------------------------------------------------------===//

class OpcodeHexagon {
  field bits<32> Inst = ?; // Default to an invalid insn.
  bits<4> IClass = 0; // ICLASS
  bits<2> IParse = 0; // Parse bits.

  let Inst{31-28} = IClass;
  let Inst{15-14} = IParse;

  bits<1> zero = 0;
}

class InstHexagon<dag outs, dag ins, string asmstr, list<dag> pattern,
                  string cstr, InstrItinClass itin, IType type>
  : Instruction, OpcodeHexagon {
  let Namespace = "Hexagon";

  dag OutOperandList = outs;
  dag InOperandList = ins;
  let AsmString = asmstr;
  let Pattern = pattern;
  let Constraints = cstr;
  let Itinerary = itin;
  let Size = 4;

  // *** Must match MCTargetDesc/HexagonBaseInfo.h ***

  // Instruction type according to the ISA.
  IType Type = type;
  let TSFlags{4-0} = Type.Value;

  // Solo instructions, i.e., those that cannot be in a packet with others.
  bits<1> isSolo = 0;
  let TSFlags{5} = isSolo;

  // Predicated instructions.
  bits<1> isPredicated = 0;
  let TSFlags{6} = isPredicated;
  bits<1> isPredicatedFalse = 0;
  let TSFlags{7} = isPredicatedFalse;
  bits<1> isPredicatedNew = 0;
  let TSFlags{8} = isPredicatedNew;

  // New-value insn helper fields.
  bits<1> isNewValue = 0;
  let TSFlags{9} = isNewValue; // New-value consumer insn.
  bits<1> hasNewValue = 0;
  let TSFlags{10} = hasNewValue; // New-value producer insn.
  bits<3> opNewValue = 0;
  let TSFlags{13-11} = opNewValue; // New-value produced operand.
  bits<2> opNewBits = 0;
  let TSFlags{15-14} = opNewBits; // New-value opcode bits location: 0, 8, 16.
  bits<1> isNVStorable = 0;
  let TSFlags{16} = isNVStorable; // Store that can become new-value store.
  bits<1> isNVStore = 0;
  let TSFlags{17} = isNVStore; // New-value store insn.

  // Immediate extender helper fields.
  bits<1> isExtendable = 0;
  let TSFlags{18} = isExtendable; // Insn may be extended.
  bits<1> isExtended = 0;
  let TSFlags{19} = isExtended; // Insn must be extended.
  bits<3> opExtendable = 0;
  let TSFlags{22-20} = opExtendable; // Which operand may be extended.
  bits<1> isExtentSigned = 0;
  let TSFlags{23} = isExtentSigned; // Signed or unsigned range.
  bits<5> opExtentBits = 0;
  let TSFlags{28-24} = opExtentBits; //Number of bits of range before extending.

  // If an instruction is valid on a subtarget (v2-v5), set the corresponding
  // bit from validSubTargets. v2 is the least significant bit.
  // By default, instruction is valid on all subtargets.
  SubTarget validSubTargets = HasV2SubT;
  let TSFlags{32-29} = validSubTargets.Value;

  // Addressing mode for load/store instructions.
  AddrModeType addrMode = NoAddrMode;
  let TSFlags{35-33} = addrMode.Value;

  // Memory access size for mem access instructions (load/store)
  MemAccessSize accessSize = NoMemAccess;
  let TSFlags{38-36} = accessSize.Value;

  // Fields used for relation models.
  string BaseOpcode = "";
  string CextOpcode = "";
  string PredSense = "";
  string PNewValue = "";
  string NValueST  = "";    // Set to "true" for new-value stores.
  string InputType = "";    // Input is "imm" or "reg" type.
  string isMEMri = "false"; // Set to "true" for load/store with MEMri operand.
  string isFloat = "false"; // Set to "true" for the floating-point load/store.
  string isBrTaken = ""; // Set to "true"/"false" for jump instructions

  let PredSense = !if(isPredicated, !if(isPredicatedFalse, "false", "true"),
                                    "");
  let PNewValue = !if(isPredicatedNew, "new", "");
  let NValueST = !if(isNVStore, "true", "false");

  // *** Must match MCTargetDesc/HexagonBaseInfo.h ***
}

//===----------------------------------------------------------------------===//
//                         Intruction Classes Definitions +
//===----------------------------------------------------------------------===//

// LD Instruction Class in V2/V3/V4.
// Definition of the instruction class NOT CHANGED.
class LDInst<dag outs, dag ins, string asmstr, list<dag> pattern = [],
             string cstr = "">
  : InstHexagon<outs, ins, asmstr, pattern, cstr, LD, TypeLD>;

let mayLoad = 1 in
class LDInst2<dag outs, dag ins, string asmstr, list<dag> pattern = [],
              string cstr = "">
  : LDInst<outs, ins, asmstr, pattern, cstr>;

class CONSTLDInst<dag outs, dag ins, string asmstr, list<dag> pattern = [],
                  string cstr = "">
  : LDInst<outs, ins, asmstr, pattern, cstr>;

// LD Instruction Class in V2/V3/V4.
// Definition of the instruction class NOT CHANGED.
class LDInstPost<dag outs, dag ins, string asmstr, list<dag> pattern = [],
                 string cstr = "">
  : LDInst<outs, ins, asmstr, pattern, cstr>;

let mayLoad = 1 in
class LD0Inst<dag outs, dag ins, string asmstr, list<dag> pattern = [],
              string cstr = "">
  : LDInst<outs, ins, asmstr, pattern, cstr>;

// ST Instruction Class in V2/V3 can take SLOT0 only.
// ST Instruction Class in V4    can take SLOT0 & SLOT1.
// Definition of the instruction class CHANGED from V2/V3 to V4.
let mayStore = 1 in
class STInst<dag outs, dag ins, string asmstr, list<dag> pattern = [],
             string cstr = "">
  : InstHexagon<outs, ins, asmstr, pattern, cstr, ST, TypeST>;

class STInst2<dag outs, dag ins, string asmstr, list<dag> pattern = [],
              string cstr = "">
  : STInst<outs, ins, asmstr, pattern, cstr>;

let mayStore = 1 in
class ST0Inst<dag outs, dag ins, string asmstr, list<dag> pattern = [],
              string cstr = "">
  : InstHexagon<outs, ins, asmstr, pattern, cstr, ST0, TypeST>;

// ST Instruction Class in V2/V3 can take SLOT0 only.
// ST Instruction Class in V4    can take SLOT0 & SLOT1.
// Definition of the instruction class CHANGED from V2/V3 to V4.
class STInstPost<dag outs, dag ins, string asmstr, list<dag> pattern = [],
                 string cstr = "">
  : STInst<outs, ins, asmstr, pattern, cstr>;

// SYSTEM Instruction Class in V4 can take SLOT0 only
// In V2/V3 we used ST for this but in v4 ST can take SLOT0 or SLOT1.
class SYSInst<dag outs, dag ins, string asmstr, list<dag> pattern = [],
              string cstr = "">
  : InstHexagon<outs, ins, asmstr, pattern, cstr, SYS, TypeSYSTEM>;

// ALU32 Instruction Class in V2/V3/V4.
// Definition of the instruction class NOT CHANGED.
class ALU32Inst<dag outs, dag ins, string asmstr, list<dag> pattern = [],
                string cstr = "">
   : InstHexagon<outs, ins, asmstr, pattern, cstr, ALU32, TypeALU32>;

// ALU64 Instruction Class in V2/V3.
// XTYPE Instruction Class in V4.
// Definition of the instruction class NOT CHANGED.
// Name of the Instruction Class changed from ALU64 to XTYPE from V2/V3 to V4.
class ALU64Inst<dag outs, dag ins, string asmstr, list<dag> pattern = [],
                string cstr = "">
   : InstHexagon<outs, ins, asmstr, pattern, cstr, ALU64, TypeXTYPE>;

class ALU64_acc<dag outs, dag ins, string asmstr, list<dag> pattern = [],
                string cstr = "">
  : ALU64Inst<outs, ins, asmstr, pattern, cstr>;


// M Instruction Class in V2/V3.
// XTYPE Instruction Class in V4.
// Definition of the instruction class NOT CHANGED.
// Name of the Instruction Class changed from M to XTYPE from V2/V3 to V4.
class MInst<dag outs, dag ins, string asmstr, list<dag> pattern = [],
            string cstr = "">
  : InstHexagon<outs, ins, asmstr, pattern, cstr, M, TypeXTYPE>;

// M Instruction Class in V2/V3.
// XTYPE Instruction Class in V4.
// Definition of the instruction class NOT CHANGED.
// Name of the Instruction Class changed from M to XTYPE from V2/V3 to V4.
class MInst_acc<dag outs, dag ins, string asmstr, list<dag> pattern = [],
                string cstr = "">
    : MInst<outs, ins, asmstr, pattern, cstr>;

// S Instruction Class in V2/V3.
// XTYPE Instruction Class in V4.
// Definition of the instruction class NOT CHANGED.
// Name of the Instruction Class changed from S to XTYPE from V2/V3 to V4.
class SInst<dag outs, dag ins, string asmstr, list<dag> pattern = [],
            string cstr = "">
  : InstHexagon<outs, ins, asmstr, pattern, cstr, S, TypeXTYPE>;

// S Instruction Class in V2/V3.
// XTYPE Instruction Class in V4.
// Definition of the instruction class NOT CHANGED.
// Name of the Instruction Class changed from S to XTYPE from V2/V3 to V4.
class SInst_acc<dag outs, dag ins, string asmstr, list<dag> pattern = [],
                string cstr = "">
  : SInst<outs, ins, asmstr, pattern, cstr>;

// J Instruction Class in V2/V3/V4.
// Definition of the instruction class NOT CHANGED.
class JInst<dag outs, dag ins, string asmstr, list<dag> pattern = [],
            string cstr = "">
  : InstHexagon<outs, ins, asmstr, pattern, cstr, J, TypeJ>;

// JR Instruction Class in V2/V3/V4.
// Definition of the instruction class NOT CHANGED.
class JRInst<dag outs, dag ins, string asmstr, list<dag> pattern = [],
             string cstr = "">
  : InstHexagon<outs, ins, asmstr, pattern, cstr, JR, TypeJR>;

// CR Instruction Class in V2/V3/V4.
// Definition of the instruction class NOT CHANGED.
class CRInst<dag outs, dag ins, string asmstr, list<dag> pattern = [],
             string cstr = "">
  : InstHexagon<outs, ins, asmstr, pattern, cstr, CR, TypeCR>;

let isCodeGenOnly = 1, isPseudo = 1 in
class Endloop<dag outs, dag ins, string asmstr, list<dag> pattern = [],
              string cstr = "">
  : InstHexagon<outs, ins, asmstr, pattern, cstr, ENDLOOP, TypeENDLOOP>;

let isCodeGenOnly = 1, isPseudo = 1 in
class Pseudo<dag outs, dag ins, string asmstr, list<dag> pattern = [],
             string cstr = "">
  : InstHexagon<outs, ins, asmstr, pattern, cstr, PSEUDO, TypePSEUDO>;

let isCodeGenOnly = 1, isPseudo = 1 in
class PseudoM<dag outs, dag ins, string asmstr, list<dag> pattern = [],
              string cstr="">
  : InstHexagon<outs, ins, asmstr, pattern, cstr, PSEUDOM, TypePSEUDO>;

//===----------------------------------------------------------------------===//
//                         Intruction Classes Definitions -
//===----------------------------------------------------------------------===//


//
// ALU32 patterns
//.
class ALU32_rr<dag outs, dag ins, string asmstr, list<dag> pattern,
               string cstr = "">
   : ALU32Inst<outs, ins, asmstr, pattern, cstr>;

class ALU32_ir<dag outs, dag ins, string asmstr, list<dag> pattern,
               string cstr = "">
   : ALU32Inst<outs, ins, asmstr, pattern, cstr>;

class ALU32_ri<dag outs, dag ins, string asmstr, list<dag> pattern,
               string cstr = "">
   : ALU32Inst<outs, ins, asmstr, pattern, cstr>;

class ALU32_ii<dag outs, dag ins, string asmstr, list<dag> pattern,
               string cstr = "">
   : ALU32Inst<outs, ins, asmstr, pattern, cstr>;

//
// ALU64 patterns.
//
class ALU64_rr<dag outs, dag ins, string asmstr, list<dag> pattern,
               string cstr = "">
   : ALU64Inst<outs, ins, asmstr, pattern, cstr>;

class ALU64_ri<dag outs, dag ins, string asmstr, list<dag> pattern,
               string cstr = "">
   : ALU64Inst<outs, ins, asmstr, pattern, cstr>;

// Post increment ST Instruction.
class STInstPI<dag outs, dag ins, string asmstr, list<dag> pattern = [],
               string cstr = "">
  : STInst<outs, ins, asmstr, pattern, cstr>;

let mayStore = 1 in
class STInst2PI<dag outs, dag ins, string asmstr, list<dag> pattern = [],
                string cstr = "">
  : STInst<outs, ins, asmstr, pattern, cstr>;

// Post increment LD Instruction.
class LDInstPI<dag outs, dag ins, string asmstr, list<dag> pattern = [],
               string cstr = "">
  : LDInst<outs, ins, asmstr, pattern, cstr>;

let mayLoad = 1 in
class LDInst2PI<dag outs, dag ins, string asmstr, list<dag> pattern = [],
                string cstr = "">
  : LDInst<outs, ins, asmstr, pattern, cstr>;

//===----------------------------------------------------------------------===//
// V4 Instruction Format Definitions +
//===----------------------------------------------------------------------===//

include "HexagonInstrFormatsV4.td"

//===----------------------------------------------------------------------===//
// V4 Instruction Format Definitions +
//===----------------------------------------------------------------------===//
