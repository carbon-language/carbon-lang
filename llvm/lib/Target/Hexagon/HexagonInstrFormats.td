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


//===----------------------------------------------------------------------===//
//                         Intruction Class Declaration +
//===----------------------------------------------------------------------===//

class InstHexagon<dag outs, dag ins, string asmstr, list<dag> pattern,
                  string cstr, InstrItinClass itin> : Instruction {
  field bits<32> Inst;

  let Namespace = "Hexagon";

  dag OutOperandList = outs;
  dag InOperandList = ins;
  let AsmString   = asmstr;
  let Pattern = pattern;
  let Constraints = cstr;
  let Itinerary   = itin;

  // *** The code below must match HexagonBaseInfo.h ***

  // Predicated instructions.
  bits<1> isPredicated = 0;
  let TSFlags{1} = isPredicated;

  // *** The code above must match HexagonBaseInfo.h ***
}

//===----------------------------------------------------------------------===//
//                         Intruction Classes Definitions +
//===----------------------------------------------------------------------===//

// LD Instruction Class in V2/V3/V4.
// Definition of the instruction class NOT CHANGED.
class LDInst<dag outs, dag ins, string asmstr, list<dag> pattern>
  : InstHexagon<outs, ins, asmstr, pattern, "", LD> {
  bits<5> rd;
  bits<5> rs;
  bits<13> imm13;
}

// LD Instruction Class in V2/V3/V4.
// Definition of the instruction class NOT CHANGED.
class LDInstPost<dag outs, dag ins, string asmstr, list<dag> pattern,
                 string cstr>
  : InstHexagon<outs, ins, asmstr, pattern, cstr,  LD> {
  bits<5> rd;
  bits<5> rs;
  bits<5> rt;
  bits<13> imm13;
}

// ST Instruction Class in V2/V3 can take SLOT0 only.
// ST Instruction Class in V4    can take SLOT0 & SLOT1.
// Definition of the instruction class CHANGED from V2/V3 to V4.
class STInst<dag outs, dag ins, string asmstr, list<dag> pattern>
  : InstHexagon<outs, ins, asmstr, pattern, "",  ST> {
  bits<5> rd;
  bits<5> rs;
  bits<13> imm13;
}

// ST Instruction Class in V2/V3 can take SLOT0 only.
// ST Instruction Class in V4    can take SLOT0 & SLOT1.
// Definition of the instruction class CHANGED from V2/V3 to V4.
class STInstPost<dag outs, dag ins, string asmstr, list<dag> pattern,
                 string cstr>
  : InstHexagon<outs, ins, asmstr, pattern, cstr,  ST> {
  bits<5> rd;
  bits<5> rs;
  bits<5> rt;
  bits<13> imm13;
}

// ALU32 Instruction Class in V2/V3/V4.
// Definition of the instruction class NOT CHANGED.
class ALU32Type<dag outs, dag ins, string asmstr, list<dag> pattern>
   : InstHexagon<outs, ins, asmstr, pattern, "",  ALU32> {
  bits<5>  rd;
  bits<5>  rs;
  bits<5>  rt;
  bits<16> imm16;
  bits<16> imm16_2;
}

// ALU64 Instruction Class in V2/V3.
// XTYPE Instruction Class in V4.
// Definition of the instruction class NOT CHANGED.
// Name of the Instruction Class changed from ALU64 to XTYPE from V2/V3 to V4.
class ALU64Type<dag outs, dag ins, string asmstr, list<dag> pattern>
   : InstHexagon<outs, ins, asmstr, pattern, "",  ALU64> {
  bits<5>  rd;
  bits<5>  rs;
  bits<5>  rt;
  bits<16> imm16;
  bits<16> imm16_2;
}

// M Instruction Class in V2/V3.
// XTYPE Instruction Class in V4.
// Definition of the instruction class NOT CHANGED.
// Name of the Instruction Class changed from M to XTYPE from V2/V3 to V4.
class MInst<dag outs, dag ins, string asmstr, list<dag> pattern>
  : InstHexagon<outs, ins, asmstr, pattern, "",  M> {
  bits<5> rd;
  bits<5> rs;
  bits<5> rt;
}

// M Instruction Class in V2/V3.
// XTYPE Instruction Class in V4.
// Definition of the instruction class NOT CHANGED.
// Name of the Instruction Class changed from M to XTYPE from V2/V3 to V4.
class MInst_acc<dag outs, dag ins, string asmstr, list<dag> pattern,
                string cstr>
    : InstHexagon<outs, ins, asmstr, pattern, cstr,  M> {
  bits<5> rd;
  bits<5> rs;
  bits<5> rt;
}

// S Instruction Class in V2/V3.
// XTYPE Instruction Class in V4.
// Definition of the instruction class NOT CHANGED.
// Name of the Instruction Class changed from S to XTYPE from V2/V3 to V4.
class SInst<dag outs, dag ins, string asmstr, list<dag> pattern>
//: InstHexagon<outs, ins, asmstr, pattern, cstr, !if(V4T, XTYPE_V4, M)> {
  : InstHexagon<outs, ins, asmstr, pattern, "",  S> {
//  : InstHexagon<outs, ins, asmstr, pattern, "", S> {
  bits<5> rd;
  bits<5> rs;
  bits<5> rt;
}

// S Instruction Class in V2/V3.
// XTYPE Instruction Class in V4.
// Definition of the instruction class NOT CHANGED.
// Name of the Instruction Class changed from S to XTYPE from V2/V3 to V4.
class SInst_acc<dag outs, dag ins, string asmstr, list<dag> pattern,
                string cstr>
  : InstHexagon<outs, ins, asmstr, pattern, cstr,  S> {
//  : InstHexagon<outs, ins, asmstr, pattern, cstr,  S> {
//  : InstHexagon<outs, ins, asmstr, pattern, cstr, !if(V4T, XTYPE_V4, S)> {
  bits<5> rd;
  bits<5> rs;
  bits<5> rt;
}

// J Instruction Class in V2/V3/V4.
// Definition of the instruction class NOT CHANGED.
class JType<dag outs, dag ins, string asmstr, list<dag> pattern>
  : InstHexagon<outs, ins, asmstr, pattern, "",  J> {
  bits<16> imm16;
}

// JR Instruction Class in V2/V3/V4.
// Definition of the instruction class NOT CHANGED.
class JRType<dag outs, dag ins, string asmstr, list<dag> pattern>
  : InstHexagon<outs, ins, asmstr, pattern, "", JR> {
  bits<5>  rs;
  bits<5>  pu; // Predicate register
}

// CR Instruction Class in V2/V3/V4.
// Definition of the instruction class NOT CHANGED.
class CRInst<dag outs, dag ins, string asmstr, list<dag> pattern>
  : InstHexagon<outs, ins, asmstr, pattern, "", CR> {
  bits<5> rs;
  bits<10> imm10;
}


class Pseudo<dag outs, dag ins, string asmstr, list<dag> pattern>
 : InstHexagon<outs, ins, asmstr, pattern, "", PSEUDO>;


//===----------------------------------------------------------------------===//
//                         Intruction Classes Definitions -
//===----------------------------------------------------------------------===//


//
// ALU32 patterns
//.
class ALU32_rr<dag outs, dag ins, string asmstr, list<dag> pattern>
   : ALU32Type<outs, ins, asmstr, pattern> {
}

class ALU32_ir<dag outs, dag ins, string asmstr, list<dag> pattern>
   : ALU32Type<outs, ins, asmstr, pattern> {
   let rt{0-4} = 0;
}

class ALU32_ri<dag outs, dag ins, string asmstr, list<dag> pattern>
   : ALU32Type<outs, ins, asmstr, pattern> {
  let rt{0-4} = 0;
}

class ALU32_ii<dag outs, dag ins, string asmstr, list<dag> pattern>
   : ALU32Type<outs, ins, asmstr, pattern> {
  let rt{0-4} = 0;
}

//
// ALU64 patterns.
//
class ALU64_rr<dag outs, dag ins, string asmstr, list<dag> pattern>
   : ALU64Type<outs, ins, asmstr, pattern> {
}

// J Type Instructions.
class JInst<dag outs, dag ins, string asmstr, list<dag> pattern>
  : JType<outs, ins, asmstr, pattern> {
}

// JR type Instructions.
class JRInst<dag outs, dag ins, string asmstr, list<dag> pattern>
  : JRType<outs, ins, asmstr, pattern> {
}


// Post increment ST Instruction.
class STInstPI<dag outs, dag ins, string asmstr, list<dag> pattern, string cstr>
  : STInstPost<outs, ins, asmstr, pattern, cstr> {
  let rt{0-4} = 0;
}

// Post increment LD Instruction.
class LDInstPI<dag outs, dag ins, string asmstr, list<dag> pattern, string cstr>
  : LDInstPost<outs, ins, asmstr, pattern, cstr> {
  let rt{0-4} = 0;
}

//===----------------------------------------------------------------------===//
// V4 Instruction Format Definitions +
//===----------------------------------------------------------------------===//

include "HexagonInstrFormatsV4.td"

//===----------------------------------------------------------------------===//
// V4 Instruction Format Definitions +
//===----------------------------------------------------------------------===//
