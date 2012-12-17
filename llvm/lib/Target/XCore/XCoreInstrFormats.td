//===-- XCoreInstrFormats.td - XCore Instruction Formats ---*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instruction format superclass
//===----------------------------------------------------------------------===//
class InstXCore<int sz, dag outs, dag ins, string asmstr, list<dag> pattern>
    : Instruction {
  field bits<32> Inst;

  let Namespace = "XCore";
  dag OutOperandList = outs;
  dag InOperandList = ins;
  let AsmString   = asmstr;
  let Pattern = pattern;
  let Size = sz;
  field bits<32> SoftFail = 0;
}

// XCore pseudo instructions format
class PseudoInstXCore<dag outs, dag ins, string asmstr, list<dag> pattern>
   : InstXCore<0, outs, ins, asmstr, pattern> {
  let isPseudo = 1;
}

//===----------------------------------------------------------------------===//
// Instruction formats
//===----------------------------------------------------------------------===//

class _F3R<dag outs, dag ins, string asmstr, list<dag> pattern>
    : InstXCore<2, outs, ins, asmstr, pattern> {
}

class _FL3R<dag outs, dag ins, string asmstr, list<dag> pattern>
    : InstXCore<4, outs, ins, asmstr, pattern> {
}

class _F2RUS<dag outs, dag ins, string asmstr, list<dag> pattern>
    : InstXCore<2, outs, ins, asmstr, pattern> {
}

class _FL2RUS<dag outs, dag ins, string asmstr, list<dag> pattern>
    : InstXCore<4, outs, ins, asmstr, pattern> {
}

class _FRU6<dag outs, dag ins, string asmstr, list<dag> pattern>
    : InstXCore<2, outs, ins, asmstr, pattern> {
}

class _FLRU6<dag outs, dag ins, string asmstr, list<dag> pattern>
    : InstXCore<4, outs, ins, asmstr, pattern> {
}

class _FU6<dag outs, dag ins, string asmstr, list<dag> pattern>
    : InstXCore<2, outs, ins, asmstr, pattern> {
}

class _FLU6<dag outs, dag ins, string asmstr, list<dag> pattern>
    : InstXCore<4, outs, ins, asmstr, pattern> {
}

class _FU10<dag outs, dag ins, string asmstr, list<dag> pattern>
    : InstXCore<2, outs, ins, asmstr, pattern> {
}

class _FLU10<dag outs, dag ins, string asmstr, list<dag> pattern>
    : InstXCore<4, outs, ins, asmstr, pattern> {
}

class _F2R<bits<6> opc, dag outs, dag ins, string asmstr, list<dag> pattern>
    : InstXCore<2, outs, ins, asmstr, pattern> {
  let Inst{15-11} = opc{5-1};
  let Inst{4} = opc{0};
  let DecoderMethod = "Decode2RInstruction";
}

// 2R with first operand as both a source and a destination.
class _F2RSrcDst<bits<6> opc, dag outs, dag ins, string asmstr,
                 list<dag> pattern> : _F2R<opc, outs, ins, asmstr, pattern> {
  let DecoderMethod = "Decode2RSrcDstInstruction";
}

// Same as 2R with last two operands swapped
class _FR2R<bits<6> opc, dag outs, dag ins, string asmstr, list<dag> pattern>
    : _F2R<opc, outs, ins, asmstr, pattern> {
  let DecoderMethod = "DecodeR2RInstruction";
}

class _FRUS<bits<6> opc, dag outs, dag ins, string asmstr, list<dag> pattern>
    : InstXCore<2, outs, ins, asmstr, pattern> {
  let Inst{15-11} = opc{5-1};
  let Inst{4} = opc{0};
  let DecoderMethod = "DecodeRUSInstruction";
}

// RUS with bitp operand
class _FRUSBitp<bits<6> opc, dag outs, dag ins, string asmstr,
                list<dag> pattern>
    : _FRUS<opc, outs, ins, asmstr, pattern> {
  let DecoderMethod = "DecodeRUSBitpInstruction";
}

// RUS with first operand as both a source and a destination and a bitp second
// operand
class _FRUSSrcDstBitp<bits<6> opc, dag outs, dag ins, string asmstr,
                      list<dag> pattern>
    : _FRUS<opc, outs, ins, asmstr, pattern> {
  let DecoderMethod = "DecodeRUSSrcDstBitpInstruction";
}

class _FL2R<bits<10> opc, dag outs, dag ins, string asmstr, list<dag> pattern>
    : InstXCore<4, outs, ins, asmstr, pattern> {
  let Inst{31-27} = opc{9-5};
  let Inst{26-20} = 0b1111110;
  let Inst{19-16} = opc{4-1};

  let Inst{15-11} = 0b11111;
  let Inst{4} = opc{0};
  let DecoderMethod = "DecodeL2RInstruction";
}

// Same as L2R with last two operands swapped
class _FLR2R<bits<10> opc, dag outs, dag ins, string asmstr, list<dag> pattern>
    : _FL2R<opc, outs, ins, asmstr, pattern> {
  let DecoderMethod = "DecodeLR2RInstruction";
}

class _F1R<bits<6> opc, dag outs, dag ins, string asmstr, list<dag> pattern>
    : InstXCore<2, outs, ins, asmstr, pattern> {
  bits<4> a;

  let Inst{15-11} = opc{5-1};
  let Inst{10-5} = 0b111111;
  let Inst{4} = opc{0};
  let Inst{3-0} = a;
}

class _F0R<bits<10> opc, dag outs, dag ins, string asmstr, list<dag> pattern>
    : InstXCore<2, outs, ins, asmstr, pattern> {
  let Inst{15-11} = opc{9-5};
  let Inst{10-5} = 0b111111;
  let Inst{4-0} = opc{4-0};
}

class _L4R<dag outs, dag ins, string asmstr, list<dag> pattern>
    : InstXCore<4, outs, ins, asmstr, pattern> {
}

class _L5R<dag outs, dag ins, string asmstr, list<dag> pattern>
    : InstXCore<4, outs, ins, asmstr, pattern> {
}

class _L6R<dag outs, dag ins, string asmstr, list<dag> pattern>
    : InstXCore<4, outs, ins, asmstr, pattern> {
}
