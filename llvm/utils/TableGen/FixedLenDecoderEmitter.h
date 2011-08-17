//===------------ FixedLenDecoderEmitter.h - Decoder Generator --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// It contains the tablegen backend that emits the decoder functions for
// targets with fixed length instruction set.
//
//===----------------------------------------------------------------------===//

#ifndef FixedLenDECODEREMITTER_H
#define FixedLenDECODEREMITTER_H

#include "CodeGenTarget.h"
#include "TableGenBackend.h"

#include "llvm/Support/DataTypes.h"

namespace llvm {

struct EncodingField {
  unsigned Base, Width, Offset;
  EncodingField(unsigned B, unsigned W, unsigned O)
    : Base(B), Width(W), Offset(O) { }
};

struct OperandInfo {
  std::vector<EncodingField> Fields;
  std::string Decoder;

  OperandInfo(std::string D)
    : Decoder(D) { }

  void addField(unsigned Base, unsigned Width, unsigned Offset) {
    Fields.push_back(EncodingField(Base, Width, Offset));
  }

  unsigned numFields() { return Fields.size(); }

  typedef std::vector<EncodingField>::iterator iterator;

  iterator begin() { return Fields.begin(); }
  iterator end()   { return Fields.end();   }
};

class FixedLenDecoderEmitter : public TableGenBackend {
public:
  FixedLenDecoderEmitter(RecordKeeper &R,
                         std::string GPrefix  = "if (",
                         std::string GPostfix = " == MCDisassembler::Fail) return MCDisassembler::Fail;",
                         std::string ROK      = "MCDisassembler::Success",
                         std::string RFail    = "MCDisassembler::Fail",
                         std::string L        = "") :
    Records(R), Target(R),
    NumberedInstructions(Target.getInstructionsByEnumValue()),
    GuardPrefix(GPrefix), GuardPostfix(GPostfix),
    ReturnOK(ROK), ReturnFail(RFail), Locals(L) {}

  // run - Output the code emitter
  void run(raw_ostream &o);

private:
  RecordKeeper &Records;
  CodeGenTarget Target;
  std::vector<const CodeGenInstruction*> NumberedInstructions;
  std::vector<unsigned> Opcodes;
  std::map<unsigned, std::vector<OperandInfo> > Operands;
public:
  std::string GuardPrefix, GuardPostfix;
  std::string ReturnOK, ReturnFail;
  std::string Locals;
};

} // end llvm namespace

#endif
