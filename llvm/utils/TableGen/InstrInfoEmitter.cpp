//===- InstrInfoEmitter.cpp - Generate a Instruction Set Desc. --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend is responsible for emitting a description of the target
// instruction set for the code generator.
//
//===----------------------------------------------------------------------===//

#include "CodeGenDAGPatterns.h"
#include "CodeGenInstruction.h"
#include "CodeGenSchedule.h"
#include "CodeGenTarget.h"
#include "PredicateExpander.h"
#include "SequenceToOffsetTable.h"
#include "TableGenBackends.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include <cassert>
#include <cstdint>
#include <iterator>
#include <map>
#include <string>
#include <utility>
#include <vector>

using namespace llvm;

namespace {

class InstrInfoEmitter {
  RecordKeeper &Records;
  CodeGenDAGPatterns CDP;
  const CodeGenSchedModels &SchedModels;

public:
  InstrInfoEmitter(RecordKeeper &R):
    Records(R), CDP(R), SchedModels(CDP.getTargetInfo().getSchedModels()) {}

  // run - Output the instruction set description.
  void run(raw_ostream &OS);

private:
  void emitEnums(raw_ostream &OS);

  typedef std::map<std::vector<std::string>, unsigned> OperandInfoMapTy;

  /// The keys of this map are maps which have OpName enum values as their keys
  /// and instruction operand indices as their values.  The values of this map
  /// are lists of instruction names.
  typedef std::map<std::map<unsigned, unsigned>,
                   std::vector<std::string>> OpNameMapTy;
  typedef std::map<std::string, unsigned>::iterator StrUintMapIter;

  /// Generate member functions in the target-specific GenInstrInfo class.
  ///
  /// This method is used to custom expand TIIPredicate definitions.
  /// See file llvm/Target/TargetInstPredicates.td for a description of what is
  /// a TIIPredicate and how to use it.
  void emitTIIHelperMethods(raw_ostream &OS, StringRef TargetName,
                            bool ExpandDefinition = true);

  /// Expand TIIPredicate definitions to functions that accept a const MCInst
  /// reference.
  void emitMCIIHelperMethods(raw_ostream &OS, StringRef TargetName);
  void emitRecord(const CodeGenInstruction &Inst, unsigned Num,
                  Record *InstrInfo,
                  std::map<std::vector<Record*>, unsigned> &EL,
                  const OperandInfoMapTy &OpInfo,
                  raw_ostream &OS);
  void emitOperandTypeMappings(
      raw_ostream &OS, const CodeGenTarget &Target,
      ArrayRef<const CodeGenInstruction *> NumberedInstructions);
  void initOperandMapData(
            ArrayRef<const CodeGenInstruction *> NumberedInstructions,
            StringRef Namespace,
            std::map<std::string, unsigned> &Operands,
            OpNameMapTy &OperandMap);
  void emitOperandNameMappings(raw_ostream &OS, const CodeGenTarget &Target,
            ArrayRef<const CodeGenInstruction*> NumberedInstructions);

  void emitLogicalOperandSizeMappings(
      raw_ostream &OS, StringRef Namespace,
      ArrayRef<const CodeGenInstruction *> NumberedInstructions);
  void emitLogicalOperandTypeMappings(
      raw_ostream &OS, StringRef Namespace,
      ArrayRef<const CodeGenInstruction *> NumberedInstructions);

  // Operand information.
  void EmitOperandInfo(raw_ostream &OS, OperandInfoMapTy &OperandInfoIDs);
  std::vector<std::string> GetOperandInfo(const CodeGenInstruction &Inst);
};

} // end anonymous namespace

static void PrintDefList(const std::vector<Record*> &Uses,
                         unsigned Num, raw_ostream &OS) {
  OS << "static const MCPhysReg ImplicitList" << Num << "[] = { ";
  for (Record *U : Uses)
    OS << getQualifiedName(U) << ", ";
  OS << "0 };\n";
}

//===----------------------------------------------------------------------===//
// Operand Info Emission.
//===----------------------------------------------------------------------===//

std::vector<std::string>
InstrInfoEmitter::GetOperandInfo(const CodeGenInstruction &Inst) {
  std::vector<std::string> Result;

  for (auto &Op : Inst.Operands) {
    // Handle aggregate operands and normal operands the same way by expanding
    // either case into a list of operands for this op.
    std::vector<CGIOperandList::OperandInfo> OperandList;

    // This might be a multiple operand thing.  Targets like X86 have
    // registers in their multi-operand operands.  It may also be an anonymous
    // operand, which has a single operand, but no declared class for the
    // operand.
    DagInit *MIOI = Op.MIOperandInfo;

    if (!MIOI || MIOI->getNumArgs() == 0) {
      // Single, anonymous, operand.
      OperandList.push_back(Op);
    } else {
      for (unsigned j = 0, e = Op.MINumOperands; j != e; ++j) {
        OperandList.push_back(Op);

        auto *OpR = cast<DefInit>(MIOI->getArg(j))->getDef();
        OperandList.back().Rec = OpR;
      }
    }

    for (unsigned j = 0, e = OperandList.size(); j != e; ++j) {
      Record *OpR = OperandList[j].Rec;
      std::string Res;

      if (OpR->isSubClassOf("RegisterOperand"))
        OpR = OpR->getValueAsDef("RegClass");
      if (OpR->isSubClassOf("RegisterClass"))
        Res += getQualifiedName(OpR) + "RegClassID, ";
      else if (OpR->isSubClassOf("PointerLikeRegClass"))
        Res += utostr(OpR->getValueAsInt("RegClassKind")) + ", ";
      else
        // -1 means the operand does not have a fixed register class.
        Res += "-1, ";

      // Fill in applicable flags.
      Res += "0";

      // Ptr value whose register class is resolved via callback.
      if (OpR->isSubClassOf("PointerLikeRegClass"))
        Res += "|(1<<MCOI::LookupPtrRegClass)";

      // Predicate operands.  Check to see if the original unexpanded operand
      // was of type PredicateOp.
      if (Op.Rec->isSubClassOf("PredicateOp"))
        Res += "|(1<<MCOI::Predicate)";

      // Optional def operands.  Check to see if the original unexpanded operand
      // was of type OptionalDefOperand.
      if (Op.Rec->isSubClassOf("OptionalDefOperand"))
        Res += "|(1<<MCOI::OptionalDef)";

      // Branch target operands.  Check to see if the original unexpanded
      // operand was of type BranchTargetOperand.
      if (Op.Rec->isSubClassOf("BranchTargetOperand"))
        Res += "|(1<<MCOI::BranchTarget)";

      // Fill in operand type.
      Res += ", ";
      assert(!Op.OperandType.empty() && "Invalid operand type.");
      Res += Op.OperandType;

      // Fill in constraint info.
      Res += ", ";

      const CGIOperandList::ConstraintInfo &Constraint =
        Op.Constraints[j];
      if (Constraint.isNone())
        Res += "0";
      else if (Constraint.isEarlyClobber())
        Res += "MCOI_EARLY_CLOBBER";
      else {
        assert(Constraint.isTied());
        Res += "MCOI_TIED_TO(" + utostr(Constraint.getTiedOperand()) + ")";
      }

      Result.push_back(Res);
    }
  }

  return Result;
}

void InstrInfoEmitter::EmitOperandInfo(raw_ostream &OS,
                                       OperandInfoMapTy &OperandInfoIDs) {
  // ID #0 is for no operand info.
  unsigned OperandListNum = 0;
  OperandInfoIDs[std::vector<std::string>()] = ++OperandListNum;

  OS << "\n";
  const CodeGenTarget &Target = CDP.getTargetInfo();
  for (const CodeGenInstruction *Inst : Target.getInstructionsByEnumValue()) {
    std::vector<std::string> OperandInfo = GetOperandInfo(*Inst);
    unsigned &N = OperandInfoIDs[OperandInfo];
    if (N != 0) continue;

    N = ++OperandListNum;
    OS << "static const MCOperandInfo OperandInfo" << N << "[] = { ";
    for (const std::string &Info : OperandInfo)
      OS << "{ " << Info << " }, ";
    OS << "};\n";
  }
}

/// Initialize data structures for generating operand name mappings.
///
/// \param Operands [out] A map used to generate the OpName enum with operand
///        names as its keys and operand enum values as its values.
/// \param OperandMap [out] A map for representing the operand name mappings for
///        each instructions.  This is used to generate the OperandMap table as
///        well as the getNamedOperandIdx() function.
void InstrInfoEmitter::initOperandMapData(
        ArrayRef<const CodeGenInstruction *> NumberedInstructions,
        StringRef Namespace,
        std::map<std::string, unsigned> &Operands,
        OpNameMapTy &OperandMap) {
  unsigned NumOperands = 0;
  for (const CodeGenInstruction *Inst : NumberedInstructions) {
    if (!Inst->TheDef->getValueAsBit("UseNamedOperandTable"))
      continue;
    std::map<unsigned, unsigned> OpList;
    for (const auto &Info : Inst->Operands) {
      StrUintMapIter I = Operands.find(Info.Name);

      if (I == Operands.end()) {
        I = Operands.insert(Operands.begin(),
                    std::pair<std::string, unsigned>(Info.Name, NumOperands++));
      }
      OpList[I->second] = Info.MIOperandNo;
    }
    OperandMap[OpList].push_back(Namespace.str() + "::" +
                                 Inst->TheDef->getName().str());
  }
}

/// Generate a table and function for looking up the indices of operands by
/// name.
///
/// This code generates:
/// - An enum in the llvm::TargetNamespace::OpName namespace, with one entry
///   for each operand name.
/// - A 2-dimensional table called OperandMap for mapping OpName enum values to
///   operand indices.
/// - A function called getNamedOperandIdx(uint16_t Opcode, uint16_t NamedIdx)
///   for looking up the operand index for an instruction, given a value from
///   OpName enum
void InstrInfoEmitter::emitOperandNameMappings(raw_ostream &OS,
           const CodeGenTarget &Target,
           ArrayRef<const CodeGenInstruction*> NumberedInstructions) {
  StringRef Namespace = Target.getInstNamespace();
  std::string OpNameNS = "OpName";
  // Map of operand names to their enumeration value.  This will be used to
  // generate the OpName enum.
  std::map<std::string, unsigned> Operands;
  OpNameMapTy OperandMap;

  initOperandMapData(NumberedInstructions, Namespace, Operands, OperandMap);

  OS << "#ifdef GET_INSTRINFO_OPERAND_ENUM\n";
  OS << "#undef GET_INSTRINFO_OPERAND_ENUM\n";
  OS << "namespace llvm {\n";
  OS << "namespace " << Namespace << " {\n";
  OS << "namespace " << OpNameNS << " {\n";
  OS << "enum {\n";
  for (const auto &Op : Operands)
    OS << "  " << Op.first << " = " << Op.second << ",\n";

  OS << "  OPERAND_LAST";
  OS << "\n};\n";
  OS << "} // end namespace OpName\n";
  OS << "} // end namespace " << Namespace << "\n";
  OS << "} // end namespace llvm\n";
  OS << "#endif //GET_INSTRINFO_OPERAND_ENUM\n\n";

  OS << "#ifdef GET_INSTRINFO_NAMED_OPS\n";
  OS << "#undef GET_INSTRINFO_NAMED_OPS\n";
  OS << "namespace llvm {\n";
  OS << "namespace " << Namespace << " {\n";
  OS << "LLVM_READONLY\n";
  OS << "int16_t getNamedOperandIdx(uint16_t Opcode, uint16_t NamedIdx) {\n";
  if (!Operands.empty()) {
    OS << "  static const int16_t OperandMap [][" << Operands.size()
       << "] = {\n";
    for (const auto &Entry : OperandMap) {
      const std::map<unsigned, unsigned> &OpList = Entry.first;
      OS << "{";

      // Emit a row of the OperandMap table
      for (unsigned i = 0, e = Operands.size(); i != e; ++i)
        OS << (OpList.count(i) == 0 ? -1 : (int)OpList.find(i)->second) << ", ";

      OS << "},\n";
    }
    OS << "};\n";

    OS << "  switch(Opcode) {\n";
    unsigned TableIndex = 0;
    for (const auto &Entry : OperandMap) {
      for (const std::string &Name : Entry.second)
        OS << "  case " << Name << ":\n";

      OS << "    return OperandMap[" << TableIndex++ << "][NamedIdx];\n";
    }
    OS << "  default: return -1;\n";
    OS << "  }\n";
  } else {
    // There are no operands, so no need to emit anything
    OS << "  return -1;\n";
  }
  OS << "}\n";
  OS << "} // end namespace " << Namespace << "\n";
  OS << "} // end namespace llvm\n";
  OS << "#endif //GET_INSTRINFO_NAMED_OPS\n\n";
}

/// Generate an enum for all the operand types for this target, under the
/// llvm::TargetNamespace::OpTypes namespace.
/// Operand types are all definitions derived of the Operand Target.td class.
void InstrInfoEmitter::emitOperandTypeMappings(
    raw_ostream &OS, const CodeGenTarget &Target,
    ArrayRef<const CodeGenInstruction *> NumberedInstructions) {

  StringRef Namespace = Target.getInstNamespace();
  std::vector<Record *> Operands = Records.getAllDerivedDefinitions("Operand");
  std::vector<Record *> RegisterOperands =
      Records.getAllDerivedDefinitions("RegisterOperand");
  std::vector<Record *> RegisterClasses =
      Records.getAllDerivedDefinitions("RegisterClass");

  OS << "#ifdef GET_INSTRINFO_OPERAND_TYPES_ENUM\n";
  OS << "#undef GET_INSTRINFO_OPERAND_TYPES_ENUM\n";
  OS << "namespace llvm {\n";
  OS << "namespace " << Namespace << " {\n";
  OS << "namespace OpTypes {\n";
  OS << "enum OperandType {\n";

  unsigned EnumVal = 0;
  for (const std::vector<Record *> *RecordsToAdd :
       {&Operands, &RegisterOperands, &RegisterClasses}) {
    for (const Record *Op : *RecordsToAdd) {
      if (!Op->isAnonymous())
        OS << "  " << Op->getName() << " = " << EnumVal << ",\n";
      ++EnumVal;
    }
  }

  OS << "  OPERAND_TYPE_LIST_END" << "\n};\n";
  OS << "} // end namespace OpTypes\n";
  OS << "} // end namespace " << Namespace << "\n";
  OS << "} // end namespace llvm\n";
  OS << "#endif // GET_INSTRINFO_OPERAND_TYPES_ENUM\n\n";

  OS << "#ifdef GET_INSTRINFO_OPERAND_TYPE\n";
  OS << "#undef GET_INSTRINFO_OPERAND_TYPE\n";
  OS << "namespace llvm {\n";
  OS << "namespace " << Namespace << " {\n";
  OS << "LLVM_READONLY\n";
  OS << "static int getOperandType(uint16_t Opcode, uint16_t OpIdx) {\n";
  // TODO: Factor out duplicate operand lists to compress the tables.
  if (!NumberedInstructions.empty()) {
    std::vector<int> OperandOffsets;
    std::vector<Record *> OperandRecords;
    int CurrentOffset = 0;
    for (const CodeGenInstruction *Inst : NumberedInstructions) {
      OperandOffsets.push_back(CurrentOffset);
      for (const auto &Op : Inst->Operands) {
        const DagInit *MIOI = Op.MIOperandInfo;
        if (!MIOI || MIOI->getNumArgs() == 0) {
          // Single, anonymous, operand.
          OperandRecords.push_back(Op.Rec);
          ++CurrentOffset;
        } else {
          for (Init *Arg : MIOI->getArgs()) {
            OperandRecords.push_back(cast<DefInit>(Arg)->getDef());
            ++CurrentOffset;
          }
        }
      }
    }

    // Emit the table of offsets (indexes) into the operand type table.
    // Size the unsigned integer offset to save space.
    assert(OperandRecords.size() <= UINT32_MAX &&
           "Too many operands for offset table");
    OS << ((OperandRecords.size() <= UINT16_MAX) ? "  const uint16_t"
                                                 : "  const uint32_t");
    OS << " Offsets[] = {\n";
    for (int I = 0, E = OperandOffsets.size(); I != E; ++I)
      OS << "    " << OperandOffsets[I] << ",\n";
    OS << "  };\n";

    // Add an entry for the end so that we don't need to special case it below.
    OperandOffsets.push_back(OperandRecords.size());

    // Emit the actual operand types in a flat table.
    // Size the signed integer operand type to save space.
    assert(EnumVal <= INT16_MAX &&
           "Too many operand types for operand types table");
    OS << ((EnumVal <= INT8_MAX) ? "  const int8_t" : "  const int16_t");
    OS << " OpcodeOperandTypes[] = {\n    ";
    for (int I = 0, E = OperandRecords.size(), CurOffset = 1; I != E; ++I) {
      // We print each Opcode's operands in its own row.
      if (I == OperandOffsets[CurOffset]) {
        OS << "\n    ";
        // If there are empty rows, mark them with an empty comment.
        while (OperandOffsets[++CurOffset] == I)
          OS << "/**/\n    ";
      }
      Record *OpR = OperandRecords[I];
      if ((OpR->isSubClassOf("Operand") ||
           OpR->isSubClassOf("RegisterOperand") ||
           OpR->isSubClassOf("RegisterClass")) &&
          !OpR->isAnonymous())
        OS << "OpTypes::" << OpR->getName();
      else
        OS << -1;
      OS << ", ";
    }
    OS << "\n  };\n";

    OS << "  return OpcodeOperandTypes[Offsets[Opcode] + OpIdx];\n";
  } else {
    OS << "  llvm_unreachable(\"No instructions defined\");\n";
  }
  OS << "}\n";
  OS << "} // end namespace " << Namespace << "\n";
  OS << "} // end namespace llvm\n";
  OS << "#endif // GET_INSTRINFO_OPERAND_TYPE\n\n";
}

void InstrInfoEmitter::emitLogicalOperandSizeMappings(
    raw_ostream &OS, StringRef Namespace,
    ArrayRef<const CodeGenInstruction *> NumberedInstructions) {
  std::map<std::vector<unsigned>, unsigned> LogicalOpSizeMap;

  std::map<unsigned, std::vector<std::string>> InstMap;

  size_t LogicalOpListSize = 0U;
  std::vector<unsigned> LogicalOpList;
  for (const auto *Inst : NumberedInstructions) {
    if (!Inst->TheDef->getValueAsBit("UseLogicalOperandMappings"))
      continue;

    LogicalOpList.clear();
    llvm::transform(Inst->Operands, std::back_inserter(LogicalOpList),
                    [](const CGIOperandList::OperandInfo &Op) -> unsigned {
                      auto *MIOI = Op.MIOperandInfo;
                      if (!MIOI || MIOI->getNumArgs() == 0)
                        return 1;
                      return MIOI->getNumArgs();
                    });
    LogicalOpListSize = std::max(LogicalOpList.size(), LogicalOpListSize);

    auto I =
        LogicalOpSizeMap.insert({LogicalOpList, LogicalOpSizeMap.size()}).first;
    InstMap[I->second].push_back(
        (Namespace + "::" + Inst->TheDef->getName()).str());
  }

  OS << "#ifdef GET_INSTRINFO_LOGICAL_OPERAND_SIZE_MAP\n";
  OS << "#undef GET_INSTRINFO_LOGICAL_OPERAND_SIZE_MAP\n";
  OS << "namespace llvm {\n";
  OS << "namespace " << Namespace << " {\n";
  OS << "LLVM_READONLY static unsigned\n";
  OS << "getLogicalOperandSize(uint16_t Opcode, uint16_t LogicalOpIdx) {\n";
  if (!InstMap.empty()) {
    std::vector<const std::vector<unsigned> *> LogicalOpSizeList(
        LogicalOpSizeMap.size());
    for (auto &P : LogicalOpSizeMap) {
      LogicalOpSizeList[P.second] = &P.first;
    }
    OS << "  static const unsigned SizeMap[][" << LogicalOpListSize
       << "] = {\n";
    for (auto &R : LogicalOpSizeList) {
      const auto &Row = *R;
      OS << "   {";
      int i;
      for (i = 0; i < static_cast<int>(Row.size()); ++i) {
        OS << Row[i] << ", ";
      }
      for (; i < static_cast<int>(LogicalOpListSize); ++i) {
        OS << "0, ";
      }
      OS << "}, ";
      OS << "\n";
    }
    OS << "  };\n";

    OS << "  switch (Opcode) {\n";
    OS << "  default: return LogicalOpIdx;\n";
    for (auto &P : InstMap) {
      auto OpMapIdx = P.first;
      const auto &Insts = P.second;
      for (const auto &Inst : Insts) {
        OS << "  case " << Inst << ":\n";
      }
      OS << "    return SizeMap[" << OpMapIdx << "][LogicalOpIdx];\n";
    }
    OS << "  }\n";
  } else {
    OS << "  return LogicalOpIdx;\n";
  }
  OS << "}\n";

  OS << "LLVM_READONLY static inline unsigned\n";
  OS << "getLogicalOperandIdx(uint16_t Opcode, uint16_t LogicalOpIdx) {\n";
  OS << "  auto S = 0U;\n";
  OS << "  for (auto i = 0U; i < LogicalOpIdx; ++i)\n";
  OS << "    S += getLogicalOperandSize(Opcode, i);\n";
  OS << "  return S;\n";
  OS << "}\n";

  OS << "} // end namespace " << Namespace << "\n";
  OS << "} // end namespace llvm\n";
  OS << "#endif // GET_INSTRINFO_LOGICAL_OPERAND_SIZE_MAP\n\n";
}

void InstrInfoEmitter::emitLogicalOperandTypeMappings(
    raw_ostream &OS, StringRef Namespace,
    ArrayRef<const CodeGenInstruction *> NumberedInstructions) {
  std::map<std::vector<std::string>, unsigned> LogicalOpTypeMap;

  std::map<unsigned, std::vector<std::string>> InstMap;

  size_t OpTypeListSize = 0U;
  std::vector<std::string> LogicalOpTypeList;
  for (const auto *Inst : NumberedInstructions) {
    if (!Inst->TheDef->getValueAsBit("UseLogicalOperandMappings"))
      continue;

    LogicalOpTypeList.clear();
    for (const auto &Op : Inst->Operands) {
      auto *OpR = Op.Rec;
      if ((OpR->isSubClassOf("Operand") ||
           OpR->isSubClassOf("RegisterOperand") ||
           OpR->isSubClassOf("RegisterClass")) &&
          !OpR->isAnonymous()) {
        LogicalOpTypeList.push_back(
            (Namespace + "::OpTypes::" + Op.Rec->getName()).str());
      } else {
        LogicalOpTypeList.push_back("-1");
      }
    }
    OpTypeListSize = std::max(LogicalOpTypeList.size(), OpTypeListSize);

    auto I =
        LogicalOpTypeMap.insert({LogicalOpTypeList, LogicalOpTypeMap.size()})
            .first;
    InstMap[I->second].push_back(
        (Namespace + "::" + Inst->TheDef->getName()).str());
  }

  OS << "#ifdef GET_INSTRINFO_LOGICAL_OPERAND_TYPE_MAP\n";
  OS << "#undef GET_INSTRINFO_LOGICAL_OPERAND_TYPE_MAP\n";
  OS << "namespace llvm {\n";
  OS << "namespace " << Namespace << " {\n";
  OS << "LLVM_READONLY static int\n";
  OS << "getLogicalOperandType(uint16_t Opcode, uint16_t LogicalOpIdx) {\n";
  if (!InstMap.empty()) {
    std::vector<const std::vector<std::string> *> LogicalOpTypeList(
        LogicalOpTypeMap.size());
    for (auto &P : LogicalOpTypeMap) {
      LogicalOpTypeList[P.second] = &P.first;
    }
    OS << "  static const int TypeMap[][" << OpTypeListSize << "] = {\n";
    for (int r = 0, rs = LogicalOpTypeList.size(); r < rs; ++r) {
      const auto &Row = *LogicalOpTypeList[r];
      OS << "   {";
      int i, s = Row.size();
      for (i = 0; i < s; ++i) {
        if (i > 0)
          OS << ", ";
        OS << Row[i];
      }
      for (; i < static_cast<int>(OpTypeListSize); ++i) {
        if (i > 0)
          OS << ", ";
        OS << "-1";
      }
      OS << "}";
      if (r != rs - 1)
        OS << ",";
      OS << "\n";
    }
    OS << "  };\n";

    OS << "  switch (Opcode) {\n";
    OS << "  default: return -1;\n";
    for (auto &P : InstMap) {
      auto OpMapIdx = P.first;
      const auto &Insts = P.second;
      for (const auto &Inst : Insts) {
        OS << "  case " << Inst << ":\n";
      }
      OS << "    return TypeMap[" << OpMapIdx << "][LogicalOpIdx];\n";
    }
    OS << "  }\n";
  } else {
    OS << "  return -1;\n";
  }
  OS << "}\n";
  OS << "} // end namespace " << Namespace << "\n";
  OS << "} // end namespace llvm\n";
  OS << "#endif // GET_INSTRINFO_LOGICAL_OPERAND_TYPE_MAP\n\n";
}

void InstrInfoEmitter::emitMCIIHelperMethods(raw_ostream &OS,
                                             StringRef TargetName) {
  RecVec TIIPredicates = Records.getAllDerivedDefinitions("TIIPredicate");
  if (TIIPredicates.empty())
    return;

  OS << "#ifdef GET_INSTRINFO_MC_HELPER_DECLS\n";
  OS << "#undef GET_INSTRINFO_MC_HELPER_DECLS\n\n";

  OS << "namespace llvm {\n";
  OS << "class MCInst;\n\n";

  OS << "namespace " << TargetName << "_MC {\n\n";

  for (const Record *Rec : TIIPredicates) {
    OS << "bool " << Rec->getValueAsString("FunctionName")
        << "(const MCInst &MI);\n";
  }

  OS << "\n} // end namespace " << TargetName << "_MC\n";
  OS << "} // end namespace llvm\n\n";

  OS << "#endif // GET_INSTRINFO_MC_HELPER_DECLS\n\n";

  OS << "#ifdef GET_INSTRINFO_MC_HELPERS\n";
  OS << "#undef GET_INSTRINFO_MC_HELPERS\n\n";

  OS << "namespace llvm {\n";
  OS << "namespace " << TargetName << "_MC {\n\n";

  PredicateExpander PE(TargetName);
  PE.setExpandForMC(true);

  for (const Record *Rec : TIIPredicates) {
    OS << "bool " << Rec->getValueAsString("FunctionName");
    OS << "(const MCInst &MI) {\n";

    OS.indent(PE.getIndentLevel() * 2);
    PE.expandStatement(OS, Rec->getValueAsDef("Body"));
    OS << "\n}\n\n";
  }

  OS << "} // end namespace " << TargetName << "_MC\n";
  OS << "} // end namespace llvm\n\n";

  OS << "#endif // GET_GENISTRINFO_MC_HELPERS\n";
}

void InstrInfoEmitter::emitTIIHelperMethods(raw_ostream &OS,
                                            StringRef TargetName,
                                            bool ExpandDefinition) {
  RecVec TIIPredicates = Records.getAllDerivedDefinitions("TIIPredicate");
  if (TIIPredicates.empty())
    return;

  PredicateExpander PE(TargetName);
  PE.setExpandForMC(false);

  for (const Record *Rec : TIIPredicates) {
    OS << (ExpandDefinition ? "" : "static ") << "bool ";
    if (ExpandDefinition)
      OS << TargetName << "InstrInfo::";
    OS << Rec->getValueAsString("FunctionName");
    OS << "(const MachineInstr &MI)";
    if (!ExpandDefinition) {
      OS << ";\n";
      continue;
    }

    OS << " {\n";
    OS.indent(PE.getIndentLevel() * 2);
    PE.expandStatement(OS, Rec->getValueAsDef("Body"));
    OS << "\n}\n\n";
  }
}

//===----------------------------------------------------------------------===//
// Main Output.
//===----------------------------------------------------------------------===//

// run - Emit the main instruction description records for the target...
void InstrInfoEmitter::run(raw_ostream &OS) {
  emitSourceFileHeader("Target Instruction Enum Values and Descriptors", OS);
  emitEnums(OS);

  OS << "#ifdef GET_INSTRINFO_MC_DESC\n";
  OS << "#undef GET_INSTRINFO_MC_DESC\n";

  OS << "namespace llvm {\n\n";

  CodeGenTarget &Target = CDP.getTargetInfo();
  const std::string &TargetName = std::string(Target.getName());
  Record *InstrInfo = Target.getInstructionSet();

  // Keep track of all of the def lists we have emitted already.
  std::map<std::vector<Record*>, unsigned> EmittedLists;
  unsigned ListNumber = 0;

  // Emit all of the instruction's implicit uses and defs.
  Records.startTimer("Emit uses/defs");
  for (const CodeGenInstruction *II : Target.getInstructionsByEnumValue()) {
    Record *Inst = II->TheDef;
    std::vector<Record*> Uses = Inst->getValueAsListOfDefs("Uses");
    if (!Uses.empty()) {
      unsigned &IL = EmittedLists[Uses];
      if (!IL) PrintDefList(Uses, IL = ++ListNumber, OS);
    }
    std::vector<Record*> Defs = Inst->getValueAsListOfDefs("Defs");
    if (!Defs.empty()) {
      unsigned &IL = EmittedLists[Defs];
      if (!IL) PrintDefList(Defs, IL = ++ListNumber, OS);
    }
  }

  OperandInfoMapTy OperandInfoIDs;

  // Emit all of the operand info records.
  Records.startTimer("Emit operand info");
  EmitOperandInfo(OS, OperandInfoIDs);

  // Emit all of the MCInstrDesc records in their ENUM ordering.
  //
  Records.startTimer("Emit InstrDesc records");
  OS << "\nextern const MCInstrDesc " << TargetName << "Insts[] = {\n";
  ArrayRef<const CodeGenInstruction*> NumberedInstructions =
    Target.getInstructionsByEnumValue();

  SequenceToOffsetTable<std::string> InstrNames;
  unsigned Num = 0;
  for (const CodeGenInstruction *Inst : NumberedInstructions) {
    // Keep a list of the instruction names.
    InstrNames.add(std::string(Inst->TheDef->getName()));
    // Emit the record into the table.
    emitRecord(*Inst, Num++, InstrInfo, EmittedLists, OperandInfoIDs, OS);
  }
  OS << "};\n\n";

  // Emit the array of instruction names.
  Records.startTimer("Emit instruction names");
  InstrNames.layout();
  InstrNames.emitStringLiteralDef(OS, Twine("extern const char ") + TargetName +
                                          "InstrNameData[]");

  OS << "extern const unsigned " << TargetName <<"InstrNameIndices[] = {";
  Num = 0;
  for (const CodeGenInstruction *Inst : NumberedInstructions) {
    // Newline every eight entries.
    if (Num % 8 == 0)
      OS << "\n    ";
    OS << InstrNames.get(std::string(Inst->TheDef->getName())) << "U, ";
    ++Num;
  }
  OS << "\n};\n\n";

  bool HasDeprecationFeatures =
      llvm::any_of(NumberedInstructions, [](const CodeGenInstruction *Inst) {
        return !Inst->HasComplexDeprecationPredicate &&
               !Inst->DeprecatedReason.empty();
      });
  if (HasDeprecationFeatures) {
    OS << "extern const uint8_t " << TargetName
       << "InstrDeprecationFeatures[] = {";
    Num = 0;
    for (const CodeGenInstruction *Inst : NumberedInstructions) {
      if (Num % 8 == 0)
        OS << "\n    ";
      if (!Inst->HasComplexDeprecationPredicate &&
          !Inst->DeprecatedReason.empty())
        OS << Target.getInstNamespace() << "::" << Inst->DeprecatedReason
           << ", ";
      else
        OS << "uint8_t(-1), ";
      ++Num;
    }
    OS << "\n};\n\n";
  }

  bool HasComplexDeprecationInfos =
      llvm::any_of(NumberedInstructions, [](const CodeGenInstruction *Inst) {
        return Inst->HasComplexDeprecationPredicate;
      });
  if (HasComplexDeprecationInfos) {
    OS << "extern const MCInstrInfo::ComplexDeprecationPredicate " << TargetName
       << "InstrComplexDeprecationInfos[] = {";
    Num = 0;
    for (const CodeGenInstruction *Inst : NumberedInstructions) {
      if (Num % 8 == 0)
        OS << "\n    ";
      if (Inst->HasComplexDeprecationPredicate)
        // Emit a function pointer to the complex predicate method.
        OS << "&get" << Inst->DeprecatedReason << "DeprecationInfo, ";
      else
        OS << "nullptr, ";
      ++Num;
    }
    OS << "\n};\n\n";
  }

  // MCInstrInfo initialization routine.
  Records.startTimer("Emit initialization routine");
  OS << "static inline void Init" << TargetName
     << "MCInstrInfo(MCInstrInfo *II) {\n";
  OS << "  II->InitMCInstrInfo(" << TargetName << "Insts, " << TargetName
     << "InstrNameIndices, " << TargetName << "InstrNameData, ";
  if (HasDeprecationFeatures)
    OS << TargetName << "InstrDeprecationFeatures, ";
  else
    OS << "nullptr, ";
  if (HasComplexDeprecationInfos)
    OS << TargetName << "InstrComplexDeprecationInfos, ";
  else
    OS << "nullptr, ";
  OS << NumberedInstructions.size() << ");\n}\n\n";

  OS << "} // end namespace llvm\n";

  OS << "#endif // GET_INSTRINFO_MC_DESC\n\n";

  // Create a TargetInstrInfo subclass to hide the MC layer initialization.
  OS << "#ifdef GET_INSTRINFO_HEADER\n";
  OS << "#undef GET_INSTRINFO_HEADER\n";

  std::string ClassName = TargetName + "GenInstrInfo";
  OS << "namespace llvm {\n";
  OS << "struct " << ClassName << " : public TargetInstrInfo {\n"
     << "  explicit " << ClassName
     << "(int CFSetupOpcode = -1, int CFDestroyOpcode = -1, int CatchRetOpcode = -1, int ReturnOpcode = -1);\n"
     << "  ~" << ClassName << "() override = default;\n";


  OS << "\n};\n} // end namespace llvm\n";

  OS << "#endif // GET_INSTRINFO_HEADER\n\n";

  OS << "#ifdef GET_INSTRINFO_HELPER_DECLS\n";
  OS << "#undef GET_INSTRINFO_HELPER_DECLS\n\n";
  emitTIIHelperMethods(OS, TargetName, /* ExpandDefinition = */ false);
  OS << "\n";
  OS << "#endif // GET_INSTRINFO_HELPER_DECLS\n\n";

  OS << "#ifdef GET_INSTRINFO_HELPERS\n";
  OS << "#undef GET_INSTRINFO_HELPERS\n\n";
  emitTIIHelperMethods(OS, TargetName, /* ExpandDefinition = */ true);
  OS << "#endif // GET_INSTRINFO_HELPERS\n\n";

  OS << "#ifdef GET_INSTRINFO_CTOR_DTOR\n";
  OS << "#undef GET_INSTRINFO_CTOR_DTOR\n";

  OS << "namespace llvm {\n";
  OS << "extern const MCInstrDesc " << TargetName << "Insts[];\n";
  OS << "extern const unsigned " << TargetName << "InstrNameIndices[];\n";
  OS << "extern const char " << TargetName << "InstrNameData[];\n";
  if (HasDeprecationFeatures)
    OS << "extern const uint8_t " << TargetName
       << "InstrDeprecationFeatures[];\n";
  if (HasComplexDeprecationInfos)
    OS << "extern const MCInstrInfo::ComplexDeprecationPredicate " << TargetName
       << "InstrComplexDeprecationInfos[];\n";
  OS << ClassName << "::" << ClassName
     << "(int CFSetupOpcode, int CFDestroyOpcode, int CatchRetOpcode, int "
        "ReturnOpcode)\n"
     << "  : TargetInstrInfo(CFSetupOpcode, CFDestroyOpcode, CatchRetOpcode, "
        "ReturnOpcode) {\n"
     << "  InitMCInstrInfo(" << TargetName << "Insts, " << TargetName
     << "InstrNameIndices, " << TargetName << "InstrNameData, ";
  if (HasDeprecationFeatures)
    OS << TargetName << "InstrDeprecationFeatures, ";
  else
    OS << "nullptr, ";
  if (HasComplexDeprecationInfos)
    OS << TargetName << "InstrComplexDeprecationInfos, ";
  else
    OS << "nullptr, ";
  OS << NumberedInstructions.size() << ");\n}\n";
  OS << "} // end namespace llvm\n";

  OS << "#endif // GET_INSTRINFO_CTOR_DTOR\n\n";

  Records.startTimer("Emit operand name mappings");
  emitOperandNameMappings(OS, Target, NumberedInstructions);

  Records.startTimer("Emit operand type mappings");
  emitOperandTypeMappings(OS, Target, NumberedInstructions);

  Records.startTimer("Emit logical operand size mappings");
  emitLogicalOperandSizeMappings(OS, TargetName, NumberedInstructions);

  Records.startTimer("Emit logical operand type mappings");
  emitLogicalOperandTypeMappings(OS, TargetName, NumberedInstructions);

  Records.startTimer("Emit helper methods");
  emitMCIIHelperMethods(OS, TargetName);
}

void InstrInfoEmitter::emitRecord(const CodeGenInstruction &Inst, unsigned Num,
                                  Record *InstrInfo,
                         std::map<std::vector<Record*>, unsigned> &EmittedLists,
                                  const OperandInfoMapTy &OpInfo,
                                  raw_ostream &OS) {
  int MinOperands = 0;
  if (!Inst.Operands.empty())
    // Each logical operand can be multiple MI operands.
    MinOperands = Inst.Operands.back().MIOperandNo +
                  Inst.Operands.back().MINumOperands;

  OS << "  { ";
  OS << Num << ",\t" << MinOperands << ",\t"
     << Inst.Operands.NumDefs << ",\t"
     << Inst.TheDef->getValueAsInt("Size") << ",\t"
     << SchedModels.getSchedClassIdx(Inst) << ",\t0";

  CodeGenTarget &Target = CDP.getTargetInfo();

  // Emit all of the target independent flags...
  if (Inst.isPreISelOpcode)    OS << "|(1ULL<<MCID::PreISelOpcode)";
  if (Inst.isPseudo)           OS << "|(1ULL<<MCID::Pseudo)";
  if (Inst.isReturn)           OS << "|(1ULL<<MCID::Return)";
  if (Inst.isEHScopeReturn)    OS << "|(1ULL<<MCID::EHScopeReturn)";
  if (Inst.isBranch)           OS << "|(1ULL<<MCID::Branch)";
  if (Inst.isIndirectBranch)   OS << "|(1ULL<<MCID::IndirectBranch)";
  if (Inst.isCompare)          OS << "|(1ULL<<MCID::Compare)";
  if (Inst.isMoveImm)          OS << "|(1ULL<<MCID::MoveImm)";
  if (Inst.isMoveReg)          OS << "|(1ULL<<MCID::MoveReg)";
  if (Inst.isBitcast)          OS << "|(1ULL<<MCID::Bitcast)";
  if (Inst.isAdd)              OS << "|(1ULL<<MCID::Add)";
  if (Inst.isTrap)             OS << "|(1ULL<<MCID::Trap)";
  if (Inst.isSelect)           OS << "|(1ULL<<MCID::Select)";
  if (Inst.isBarrier)          OS << "|(1ULL<<MCID::Barrier)";
  if (Inst.hasDelaySlot)       OS << "|(1ULL<<MCID::DelaySlot)";
  if (Inst.isCall)             OS << "|(1ULL<<MCID::Call)";
  if (Inst.canFoldAsLoad)      OS << "|(1ULL<<MCID::FoldableAsLoad)";
  if (Inst.mayLoad)            OS << "|(1ULL<<MCID::MayLoad)";
  if (Inst.mayStore)           OS << "|(1ULL<<MCID::MayStore)";
  if (Inst.mayRaiseFPException) OS << "|(1ULL<<MCID::MayRaiseFPException)";
  if (Inst.isPredicable)       OS << "|(1ULL<<MCID::Predicable)";
  if (Inst.isConvertibleToThreeAddress) OS << "|(1ULL<<MCID::ConvertibleTo3Addr)";
  if (Inst.isCommutable)       OS << "|(1ULL<<MCID::Commutable)";
  if (Inst.isTerminator)       OS << "|(1ULL<<MCID::Terminator)";
  if (Inst.isReMaterializable) OS << "|(1ULL<<MCID::Rematerializable)";
  if (Inst.isNotDuplicable)    OS << "|(1ULL<<MCID::NotDuplicable)";
  if (Inst.Operands.hasOptionalDef) OS << "|(1ULL<<MCID::HasOptionalDef)";
  if (Inst.usesCustomInserter) OS << "|(1ULL<<MCID::UsesCustomInserter)";
  if (Inst.hasPostISelHook)    OS << "|(1ULL<<MCID::HasPostISelHook)";
  if (Inst.Operands.isVariadic)OS << "|(1ULL<<MCID::Variadic)";
  if (Inst.hasSideEffects)     OS << "|(1ULL<<MCID::UnmodeledSideEffects)";
  if (Inst.isAsCheapAsAMove)   OS << "|(1ULL<<MCID::CheapAsAMove)";
  if (!Target.getAllowRegisterRenaming() || Inst.hasExtraSrcRegAllocReq)
    OS << "|(1ULL<<MCID::ExtraSrcRegAllocReq)";
  if (!Target.getAllowRegisterRenaming() || Inst.hasExtraDefRegAllocReq)
    OS << "|(1ULL<<MCID::ExtraDefRegAllocReq)";
  if (Inst.isRegSequence) OS << "|(1ULL<<MCID::RegSequence)";
  if (Inst.isExtractSubreg) OS << "|(1ULL<<MCID::ExtractSubreg)";
  if (Inst.isInsertSubreg) OS << "|(1ULL<<MCID::InsertSubreg)";
  if (Inst.isConvergent) OS << "|(1ULL<<MCID::Convergent)";
  if (Inst.variadicOpsAreDefs) OS << "|(1ULL<<MCID::VariadicOpsAreDefs)";
  if (Inst.isAuthenticated) OS << "|(1ULL<<MCID::Authenticated)";

  // Emit all of the target-specific flags...
  BitsInit *TSF = Inst.TheDef->getValueAsBitsInit("TSFlags");
  if (!TSF)
    PrintFatalError(Inst.TheDef->getLoc(), "no TSFlags?");
  uint64_t Value = 0;
  for (unsigned i = 0, e = TSF->getNumBits(); i != e; ++i) {
    if (const auto *Bit = dyn_cast<BitInit>(TSF->getBit(i)))
      Value |= uint64_t(Bit->getValue()) << i;
    else
      PrintFatalError(Inst.TheDef->getLoc(),
                      "Invalid TSFlags bit in " + Inst.TheDef->getName());
  }
  OS << ", 0x";
  OS.write_hex(Value);
  OS << "ULL, ";

  // Emit the implicit uses and defs lists...
  std::vector<Record*> UseList = Inst.TheDef->getValueAsListOfDefs("Uses");
  if (UseList.empty())
    OS << "nullptr, ";
  else
    OS << "ImplicitList" << EmittedLists[UseList] << ", ";

  std::vector<Record*> DefList = Inst.TheDef->getValueAsListOfDefs("Defs");
  if (DefList.empty())
    OS << "nullptr, ";
  else
    OS << "ImplicitList" << EmittedLists[DefList] << ", ";

  // Emit the operand info.
  std::vector<std::string> OperandInfo = GetOperandInfo(Inst);
  if (OperandInfo.empty())
    OS << "nullptr";
  else
    OS << "OperandInfo" << OpInfo.find(OperandInfo)->second;

  OS << " },  // Inst #" << Num << " = " << Inst.TheDef->getName() << "\n";
}

// emitEnums - Print out enum values for all of the instructions.
void InstrInfoEmitter::emitEnums(raw_ostream &OS) {
  OS << "#ifdef GET_INSTRINFO_ENUM\n";
  OS << "#undef GET_INSTRINFO_ENUM\n";

  OS << "namespace llvm {\n\n";

  const CodeGenTarget &Target = CDP.getTargetInfo();

  // We must emit the PHI opcode first...
  StringRef Namespace = Target.getInstNamespace();

  if (Namespace.empty())
    PrintFatalError("No instructions defined!");

  OS << "namespace " << Namespace << " {\n";
  OS << "  enum {\n";
  unsigned Num = 0;
  for (const CodeGenInstruction *Inst : Target.getInstructionsByEnumValue())
    OS << "    " << Inst->TheDef->getName() << "\t= " << Num++ << ",\n";
  OS << "    INSTRUCTION_LIST_END = " << Num << "\n";
  OS << "  };\n\n";
  OS << "} // end namespace " << Namespace << "\n";
  OS << "} // end namespace llvm\n";
  OS << "#endif // GET_INSTRINFO_ENUM\n\n";

  OS << "#ifdef GET_INSTRINFO_SCHED_ENUM\n";
  OS << "#undef GET_INSTRINFO_SCHED_ENUM\n";
  OS << "namespace llvm {\n\n";
  OS << "namespace " << Namespace << " {\n";
  OS << "namespace Sched {\n";
  OS << "  enum {\n";
  Num = 0;
  for (const auto &Class : SchedModels.explicit_classes())
    OS << "    " << Class.Name << "\t= " << Num++ << ",\n";
  OS << "    SCHED_LIST_END = " << Num << "\n";
  OS << "  };\n";
  OS << "} // end namespace Sched\n";
  OS << "} // end namespace " << Namespace << "\n";
  OS << "} // end namespace llvm\n";

  OS << "#endif // GET_INSTRINFO_SCHED_ENUM\n\n";
}

namespace llvm {

void EmitInstrInfo(RecordKeeper &RK, raw_ostream &OS) {
  RK.startTimer("Analyze DAG patterns");
  InstrInfoEmitter(RK).run(OS);
  RK.startTimer("Emit map table");
  EmitMapTable(RK, OS);
}

} // end namespace llvm
