//===- CallingConvEmitter.cpp - Generate calling conventions --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend is responsible for emitting descriptions of the calling
// conventions supported by this target.
//
//===----------------------------------------------------------------------===//

#include "CodeGenTarget.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include <cassert>
using namespace llvm;

namespace {
class CallingConvEmitter {
  RecordKeeper &Records;
public:
  explicit CallingConvEmitter(RecordKeeper &R) : Records(R) {}

  void run(raw_ostream &o);

private:
  void EmitCallingConv(Record *CC, raw_ostream &O);
  void EmitAction(Record *Action, unsigned Indent, raw_ostream &O);
  unsigned Counter;
};
} // End anonymous namespace

void CallingConvEmitter::run(raw_ostream &O) {
  std::vector<Record*> CCs = Records.getAllDerivedDefinitions("CallingConv");

  // Emit prototypes for all of the non-custom CC's so that they can forward ref
  // each other.
  Records.startTimer("Emit prototypes");
  for (Record *CC : CCs) {
    if (!CC->getValueAsBit("Custom")) {
      unsigned Pad = CC->getName().size();
      if (CC->getValueAsBit("Entry")) {
        O << "bool llvm::";
        Pad += 12;
      } else {
        O << "static bool ";
        Pad += 13;
      }
      O << CC->getName() << "(unsigned ValNo, MVT ValVT,\n"
        << std::string(Pad, ' ') << "MVT LocVT, CCValAssign::LocInfo LocInfo,\n"
        << std::string(Pad, ' ')
        << "ISD::ArgFlagsTy ArgFlags, CCState &State);\n";
    }
  }

  // Emit each non-custom calling convention description in full.
  Records.startTimer("Emit full descriptions");
  for (Record *CC : CCs) {
    if (!CC->getValueAsBit("Custom"))
      EmitCallingConv(CC, O);
  }
}


void CallingConvEmitter::EmitCallingConv(Record *CC, raw_ostream &O) {
  ListInit *CCActions = CC->getValueAsListInit("Actions");
  Counter = 0;

  O << "\n\n";
  unsigned Pad = CC->getName().size();
  if (CC->getValueAsBit("Entry")) {
    O << "bool llvm::";
    Pad += 12;
  } else {
    O << "static bool ";
    Pad += 13;
  }
  O << CC->getName() << "(unsigned ValNo, MVT ValVT,\n"
    << std::string(Pad, ' ') << "MVT LocVT, CCValAssign::LocInfo LocInfo,\n"
    << std::string(Pad, ' ') << "ISD::ArgFlagsTy ArgFlags, CCState &State) {\n";
  // Emit all of the actions, in order.
  for (unsigned i = 0, e = CCActions->size(); i != e; ++i) {
    O << "\n";
    EmitAction(CCActions->getElementAsRecord(i), 2, O);
  }
  
  O << "\n  return true; // CC didn't match.\n";
  O << "}\n";
}

void CallingConvEmitter::EmitAction(Record *Action,
                                    unsigned Indent, raw_ostream &O) {
  std::string IndentStr = std::string(Indent, ' ');
  
  if (Action->isSubClassOf("CCPredicateAction")) {
    O << IndentStr << "if (";
    
    if (Action->isSubClassOf("CCIfType")) {
      ListInit *VTs = Action->getValueAsListInit("VTs");
      for (unsigned i = 0, e = VTs->size(); i != e; ++i) {
        Record *VT = VTs->getElementAsRecord(i);
        if (i != 0) O << " ||\n    " << IndentStr;
        O << "LocVT == " << getEnumName(getValueType(VT));
      }

    } else if (Action->isSubClassOf("CCIf")) {
      O << Action->getValueAsString("Predicate");
    } else {
      errs() << *Action;
      PrintFatalError(Action->getLoc(), "Unknown CCPredicateAction!");
    }
    
    O << ") {\n";
    EmitAction(Action->getValueAsDef("SubAction"), Indent+2, O);
    O << IndentStr << "}\n";
  } else {
    if (Action->isSubClassOf("CCDelegateTo")) {
      Record *CC = Action->getValueAsDef("CC");
      O << IndentStr << "if (!" << CC->getName()
        << "(ValNo, ValVT, LocVT, LocInfo, ArgFlags, State))\n"
        << IndentStr << "  return false;\n";
    } else if (Action->isSubClassOf("CCAssignToReg")) {
      ListInit *RegList = Action->getValueAsListInit("RegList");
      if (RegList->size() == 1) {
        O << IndentStr << "if (unsigned Reg = State.AllocateReg(";
        O << getQualifiedName(RegList->getElementAsRecord(0)) << ")) {\n";
      } else {
        O << IndentStr << "static const MCPhysReg RegList" << ++Counter
          << "[] = {\n";
        O << IndentStr << "  ";
        ListSeparator LS;
        for (unsigned i = 0, e = RegList->size(); i != e; ++i)
          O << LS << getQualifiedName(RegList->getElementAsRecord(i));
        O << "\n" << IndentStr << "};\n";
        O << IndentStr << "if (unsigned Reg = State.AllocateReg(RegList"
          << Counter << ")) {\n";
      }
      O << IndentStr << "  State.addLoc(CCValAssign::getReg(ValNo, ValVT, "
        << "Reg, LocVT, LocInfo));\n";
      O << IndentStr << "  return false;\n";
      O << IndentStr << "}\n";
    } else if (Action->isSubClassOf("CCAssignToRegWithShadow")) {
      ListInit *RegList = Action->getValueAsListInit("RegList");
      ListInit *ShadowRegList = Action->getValueAsListInit("ShadowRegList");
      if (!ShadowRegList->empty() && ShadowRegList->size() != RegList->size())
        PrintFatalError(Action->getLoc(),
                        "Invalid length of list of shadowed registers");

      if (RegList->size() == 1) {
        O << IndentStr << "if (unsigned Reg = State.AllocateReg(";
        O << getQualifiedName(RegList->getElementAsRecord(0));
        O << ", " << getQualifiedName(ShadowRegList->getElementAsRecord(0));
        O << ")) {\n";
      } else {
        unsigned RegListNumber = ++Counter;
        unsigned ShadowRegListNumber = ++Counter;

        O << IndentStr << "static const MCPhysReg RegList" << RegListNumber
          << "[] = {\n";
        O << IndentStr << "  ";
        ListSeparator LS;
        for (unsigned i = 0, e = RegList->size(); i != e; ++i)
          O << LS << getQualifiedName(RegList->getElementAsRecord(i));
        O << "\n" << IndentStr << "};\n";

        O << IndentStr << "static const MCPhysReg RegList"
          << ShadowRegListNumber << "[] = {\n";
        O << IndentStr << "  ";
        ListSeparator LSS;
        for (unsigned i = 0, e = ShadowRegList->size(); i != e; ++i)
          O << LSS << getQualifiedName(ShadowRegList->getElementAsRecord(i));
        O << "\n" << IndentStr << "};\n";

        O << IndentStr << "if (unsigned Reg = State.AllocateReg(RegList"
          << RegListNumber << ", " << "RegList" << ShadowRegListNumber
          << ")) {\n";
      }
      O << IndentStr << "  State.addLoc(CCValAssign::getReg(ValNo, ValVT, "
        << "Reg, LocVT, LocInfo));\n";
      O << IndentStr << "  return false;\n";
      O << IndentStr << "}\n";
    } else if (Action->isSubClassOf("CCAssignToStack")) {
      int Size = Action->getValueAsInt("Size");
      int Align = Action->getValueAsInt("Align");

      O << IndentStr << "unsigned Offset" << ++Counter
        << " = State.AllocateStack(";
      if (Size)
        O << Size << ", ";
      else
        O << "\n" << IndentStr
          << "  State.getMachineFunction().getDataLayout()."
             "getTypeAllocSize(EVT(LocVT).getTypeForEVT(State.getContext())),"
             " ";
      if (Align)
        O << "Align(" << Align << ")";
      else
        O << "\n"
          << IndentStr
          << "  State.getMachineFunction().getDataLayout()."
             "getABITypeAlign(EVT(LocVT).getTypeForEVT(State.getContext()"
             "))";
      O << ");\n" << IndentStr
        << "State.addLoc(CCValAssign::getMem(ValNo, ValVT, Offset"
        << Counter << ", LocVT, LocInfo));\n";
      O << IndentStr << "return false;\n";
    } else if (Action->isSubClassOf("CCAssignToStackWithShadow")) {
      int Size = Action->getValueAsInt("Size");
      int Align = Action->getValueAsInt("Align");
      ListInit *ShadowRegList = Action->getValueAsListInit("ShadowRegList");

      unsigned ShadowRegListNumber = ++Counter;

      O << IndentStr << "static const MCPhysReg ShadowRegList"
          << ShadowRegListNumber << "[] = {\n";
      O << IndentStr << "  ";
      ListSeparator LS;
      for (unsigned i = 0, e = ShadowRegList->size(); i != e; ++i)
        O << LS << getQualifiedName(ShadowRegList->getElementAsRecord(i));
      O << "\n" << IndentStr << "};\n";

      O << IndentStr << "unsigned Offset" << ++Counter
        << " = State.AllocateStack(" << Size << ", Align(" << Align << "), "
        << "ShadowRegList" << ShadowRegListNumber << ");\n";
      O << IndentStr << "State.addLoc(CCValAssign::getMem(ValNo, ValVT, Offset"
        << Counter << ", LocVT, LocInfo));\n";
      O << IndentStr << "return false;\n";
    } else if (Action->isSubClassOf("CCPromoteToType")) {
      Record *DestTy = Action->getValueAsDef("DestTy");
      MVT::SimpleValueType DestVT = getValueType(DestTy);
      O << IndentStr << "LocVT = " << getEnumName(DestVT) <<";\n";
      if (MVT(DestVT).isFloatingPoint()) {
        O << IndentStr << "LocInfo = CCValAssign::FPExt;\n";
      } else {
        O << IndentStr << "if (ArgFlags.isSExt())\n"
          << IndentStr << "  LocInfo = CCValAssign::SExt;\n"
          << IndentStr << "else if (ArgFlags.isZExt())\n"
          << IndentStr << "  LocInfo = CCValAssign::ZExt;\n"
          << IndentStr << "else\n"
          << IndentStr << "  LocInfo = CCValAssign::AExt;\n";
      }
    } else if (Action->isSubClassOf("CCPromoteToUpperBitsInType")) {
      Record *DestTy = Action->getValueAsDef("DestTy");
      MVT::SimpleValueType DestVT = getValueType(DestTy);
      O << IndentStr << "LocVT = " << getEnumName(DestVT) << ";\n";
      if (MVT(DestVT).isFloatingPoint()) {
        PrintFatalError(Action->getLoc(),
                        "CCPromoteToUpperBitsInType does not handle floating "
                        "point");
      } else {
        O << IndentStr << "if (ArgFlags.isSExt())\n"
          << IndentStr << "  LocInfo = CCValAssign::SExtUpper;\n"
          << IndentStr << "else if (ArgFlags.isZExt())\n"
          << IndentStr << "  LocInfo = CCValAssign::ZExtUpper;\n"
          << IndentStr << "else\n"
          << IndentStr << "  LocInfo = CCValAssign::AExtUpper;\n";
      }
    } else if (Action->isSubClassOf("CCBitConvertToType")) {
      Record *DestTy = Action->getValueAsDef("DestTy");
      O << IndentStr << "LocVT = " << getEnumName(getValueType(DestTy)) <<";\n";
      O << IndentStr << "LocInfo = CCValAssign::BCvt;\n";
    } else if (Action->isSubClassOf("CCTruncToType")) {
      Record *DestTy = Action->getValueAsDef("DestTy");
      O << IndentStr << "LocVT = " << getEnumName(getValueType(DestTy)) <<";\n";
      O << IndentStr << "LocInfo = CCValAssign::Trunc;\n";
    } else if (Action->isSubClassOf("CCPassIndirect")) {
      Record *DestTy = Action->getValueAsDef("DestTy");
      O << IndentStr << "LocVT = " << getEnumName(getValueType(DestTy)) <<";\n";
      O << IndentStr << "LocInfo = CCValAssign::Indirect;\n";
    } else if (Action->isSubClassOf("CCPassByVal")) {
      int Size = Action->getValueAsInt("Size");
      int Align = Action->getValueAsInt("Align");
      O << IndentStr << "State.HandleByVal(ValNo, ValVT, LocVT, LocInfo, "
        << Size << ", Align(" << Align << "), ArgFlags);\n";
      O << IndentStr << "return false;\n";
    } else if (Action->isSubClassOf("CCCustom")) {
      O << IndentStr
        << "if (" << Action->getValueAsString("FuncName") << "(ValNo, ValVT, "
        << "LocVT, LocInfo, ArgFlags, State))\n";
      O << IndentStr << "  return false;\n";
    } else {
      errs() << *Action;
      PrintFatalError(Action->getLoc(), "Unknown CCAction!");
    }
  }
}

namespace llvm {

void EmitCallingConv(RecordKeeper &RK, raw_ostream &OS) {
  emitSourceFileHeader("Calling Convention Implementation Fragment", OS);
  CallingConvEmitter(RK).run(OS);
}

} // End llvm namespace
