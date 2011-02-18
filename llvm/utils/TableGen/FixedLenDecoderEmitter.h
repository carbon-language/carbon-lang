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

struct OperandInfo {
  unsigned FieldBase;
  unsigned FieldLength;
  std::string Decoder;

  OperandInfo(unsigned FB, unsigned FL, std::string D)
    : FieldBase(FB), FieldLength(FL), Decoder(D) { }
};

class FixedLenDecoderEmitter : public TableGenBackend {
public:
  FixedLenDecoderEmitter(RecordKeeper &R) :
    Records(R), Target(R),
    NumberedInstructions(Target.getInstructionsByEnumValue()) {}

  // run - Output the code emitter
  void run(raw_ostream &o);

private:
  RecordKeeper &Records;
  CodeGenTarget Target;
  std::vector<const CodeGenInstruction*> NumberedInstructions;
  std::vector<unsigned> Opcodes;
  std::map<unsigned, std::vector<OperandInfo> > Operands;

  bool populateInstruction(const CodeGenInstruction &CGI, unsigned Opc);
  void populateInstructions();
};

} // end llvm namespace

#endif
