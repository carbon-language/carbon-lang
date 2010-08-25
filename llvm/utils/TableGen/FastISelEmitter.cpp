//===- FastISelEmitter.cpp - Generate an instruction selector -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend emits code for use by the "fast" instruction
// selection algorithm. See the comments at the top of
// lib/CodeGen/SelectionDAG/FastISel.cpp for background.
//
// This file scans through the target's tablegen instruction-info files
// and extracts instructions with obvious-looking patterns, and it emits
// code to look up these instructions by type and operator.
//
//===----------------------------------------------------------------------===//

#include "FastISelEmitter.h"
#include "Record.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/VectorExtras.h"
using namespace llvm;

namespace {

/// InstructionMemo - This class holds additional information about an
/// instruction needed to emit code for it.
///
struct InstructionMemo {
  std::string Name;
  const CodeGenRegisterClass *RC;
  std::string SubRegNo;
  std::vector<std::string>* PhysRegs;
};

/// OperandsSignature - This class holds a description of a list of operand
/// types. It has utility methods for emitting text based on the operands.
///
struct OperandsSignature {
  std::vector<std::string> Operands;

  bool operator<(const OperandsSignature &O) const {
    return Operands < O.Operands;
  }

  bool empty() const { return Operands.empty(); }

  /// initialize - Examine the given pattern and initialize the contents
  /// of the Operands array accordingly. Return true if all the operands
  /// are supported, false otherwise.
  ///
  bool initialize(TreePatternNode *InstPatNode,
                  const CodeGenTarget &Target,
                  MVT::SimpleValueType VT) {

    if (!InstPatNode->isLeaf()) {
      if (InstPatNode->getOperator()->getName() == "imm") {
        Operands.push_back("i");
        return true;
      }
      if (InstPatNode->getOperator()->getName() == "fpimm") {
        Operands.push_back("f");
        return true;
      }
    }
    
    const CodeGenRegisterClass *DstRC = 0;
    
    for (unsigned i = 0, e = InstPatNode->getNumChildren(); i != e; ++i) {
      TreePatternNode *Op = InstPatNode->getChild(i);
      
      // For now, filter out any operand with a predicate.
      // For now, filter out any operand with multiple values.
      if (!Op->getPredicateFns().empty() ||
          Op->getNumTypes() != 1)
        return false;
      
      assert(Op->hasTypeSet(0) && "Type infererence not done?");
      // For now, all the operands must have the same type.
      if (Op->getType(0) != VT)
        return false;
      
      if (!Op->isLeaf()) {
        if (Op->getOperator()->getName() == "imm") {
          Operands.push_back("i");
          continue;
        }
        if (Op->getOperator()->getName() == "fpimm") {
          Operands.push_back("f");
          continue;
        }
        // For now, ignore other non-leaf nodes.
        return false;
      }
      DefInit *OpDI = dynamic_cast<DefInit*>(Op->getLeafValue());
      if (!OpDI)
        return false;
      Record *OpLeafRec = OpDI->getDef();
      // For now, the only other thing we accept is register operands.

      const CodeGenRegisterClass *RC = 0;
      if (OpLeafRec->isSubClassOf("RegisterClass"))
        RC = &Target.getRegisterClass(OpLeafRec);
      else if (OpLeafRec->isSubClassOf("Register"))
        RC = Target.getRegisterClassForRegister(OpLeafRec);
      else
        return false;
        
      // For now, this needs to be a register class of some sort.
      if (!RC)
        return false;

      // For now, all the operands must have the same register class or be
      // a strict subclass of the destination.
      if (DstRC) {
        if (DstRC != RC && !DstRC->hasSubClass(RC))
          return false;
      } else
        DstRC = RC;
      Operands.push_back("r");
    }
    return true;
  }

  void PrintParameters(raw_ostream &OS) const {
    for (unsigned i = 0, e = Operands.size(); i != e; ++i) {
      if (Operands[i] == "r") {
        OS << "unsigned Op" << i << ", bool Op" << i << "IsKill";
      } else if (Operands[i] == "i") {
        OS << "uint64_t imm" << i;
      } else if (Operands[i] == "f") {
        OS << "ConstantFP *f" << i;
      } else {
        assert("Unknown operand kind!");
        abort();
      }
      if (i + 1 != e)
        OS << ", ";
    }
  }

  void PrintArguments(raw_ostream &OS,
                      const std::vector<std::string>& PR) const {
    assert(PR.size() == Operands.size());
    bool PrintedArg = false;
    for (unsigned i = 0, e = Operands.size(); i != e; ++i) {
      if (PR[i] != "")
        // Implicit physical register operand.
        continue;

      if (PrintedArg)
        OS << ", ";
      if (Operands[i] == "r") {
        OS << "Op" << i << ", Op" << i << "IsKill";
        PrintedArg = true;
      } else if (Operands[i] == "i") {
        OS << "imm" << i;
        PrintedArg = true;
      } else if (Operands[i] == "f") {
        OS << "f" << i;
        PrintedArg = true;
      } else {
        assert("Unknown operand kind!");
        abort();
      }
    }
  }

  void PrintArguments(raw_ostream &OS) const {
    for (unsigned i = 0, e = Operands.size(); i != e; ++i) {
      if (Operands[i] == "r") {
        OS << "Op" << i << ", Op" << i << "IsKill";
      } else if (Operands[i] == "i") {
        OS << "imm" << i;
      } else if (Operands[i] == "f") {
        OS << "f" << i;
      } else {
        assert("Unknown operand kind!");
        abort();
      }
      if (i + 1 != e)
        OS << ", ";
    }
  }


  void PrintManglingSuffix(raw_ostream &OS,
                           const std::vector<std::string>& PR) const {
    for (unsigned i = 0, e = Operands.size(); i != e; ++i) {
      if (PR[i] != "")
        // Implicit physical register operand. e.g. Instruction::Mul expect to
        // select to a binary op. On x86, mul may take a single operand with
        // the other operand being implicit. We must emit something that looks
        // like a binary instruction except for the very inner FastEmitInst_*
        // call.
        continue;
      OS << Operands[i];
    }
  }

  void PrintManglingSuffix(raw_ostream &OS) const {
    for (unsigned i = 0, e = Operands.size(); i != e; ++i) {
      OS << Operands[i];
    }
  }
};

class FastISelMap {
  typedef std::map<std::string, InstructionMemo> PredMap;
  typedef std::map<MVT::SimpleValueType, PredMap> RetPredMap;
  typedef std::map<MVT::SimpleValueType, RetPredMap> TypeRetPredMap;
  typedef std::map<std::string, TypeRetPredMap> OpcodeTypeRetPredMap;
  typedef std::map<OperandsSignature, OpcodeTypeRetPredMap> 
            OperandsOpcodeTypeRetPredMap;

  OperandsOpcodeTypeRetPredMap SimplePatterns;

  std::string InstNS;

public:
  explicit FastISelMap(std::string InstNS);

  void CollectPatterns(CodeGenDAGPatterns &CGP);
  void PrintFunctionDefinitions(raw_ostream &OS);
};

}

static std::string getOpcodeName(Record *Op, CodeGenDAGPatterns &CGP) {
  return CGP.getSDNodeInfo(Op).getEnumName();
}

static std::string getLegalCName(std::string OpName) {
  std::string::size_type pos = OpName.find("::");
  if (pos != std::string::npos)
    OpName.replace(pos, 2, "_");
  return OpName;
}

FastISelMap::FastISelMap(std::string instns)
  : InstNS(instns) {
}

void FastISelMap::CollectPatterns(CodeGenDAGPatterns &CGP) {
  const CodeGenTarget &Target = CGP.getTargetInfo();

  // Determine the target's namespace name.
  InstNS = Target.getInstNamespace() + "::";
  assert(InstNS.size() > 2 && "Can't determine target-specific namespace!");

  // Scan through all the patterns and record the simple ones.
  for (CodeGenDAGPatterns::ptm_iterator I = CGP.ptm_begin(),
       E = CGP.ptm_end(); I != E; ++I) {
    const PatternToMatch &Pattern = *I;

    // For now, just look at Instructions, so that we don't have to worry
    // about emitting multiple instructions for a pattern.
    TreePatternNode *Dst = Pattern.getDstPattern();
    if (Dst->isLeaf()) continue;
    Record *Op = Dst->getOperator();
    if (!Op->isSubClassOf("Instruction"))
      continue;
    CodeGenInstruction &II = CGP.getTargetInfo().getInstruction(Op);
    if (II.OperandList.empty())
      continue;
      
    // For now, ignore multi-instruction patterns.
    bool MultiInsts = false;
    for (unsigned i = 0, e = Dst->getNumChildren(); i != e; ++i) {
      TreePatternNode *ChildOp = Dst->getChild(i);
      if (ChildOp->isLeaf())
        continue;
      if (ChildOp->getOperator()->isSubClassOf("Instruction")) {
        MultiInsts = true;
        break;
      }
    }
    if (MultiInsts)
      continue;

    // For now, ignore instructions where the first operand is not an
    // output register.
    const CodeGenRegisterClass *DstRC = 0;
    std::string SubRegNo;
    if (Op->getName() != "EXTRACT_SUBREG") {
      Record *Op0Rec = II.OperandList[0].Rec;
      if (!Op0Rec->isSubClassOf("RegisterClass"))
        continue;
      DstRC = &Target.getRegisterClass(Op0Rec);
      if (!DstRC)
        continue;
    } else {
      // If this isn't a leaf, then continue since the register classes are
      // a bit too complicated for now.
      if (!Dst->getChild(1)->isLeaf()) continue;
      
      DefInit *SR = dynamic_cast<DefInit*>(Dst->getChild(1)->getLeafValue());
      if (SR)
        SubRegNo = getQualifiedName(SR->getDef());
      else
        SubRegNo = Dst->getChild(1)->getLeafValue()->getAsString();
    }

    // Inspect the pattern.
    TreePatternNode *InstPatNode = Pattern.getSrcPattern();
    if (!InstPatNode) continue;
    if (InstPatNode->isLeaf()) continue;

    // Ignore multiple result nodes for now.
    if (InstPatNode->getNumTypes() > 1) continue;
    
    Record *InstPatOp = InstPatNode->getOperator();
    std::string OpcodeName = getOpcodeName(InstPatOp, CGP);
    MVT::SimpleValueType RetVT = MVT::isVoid;
    if (InstPatNode->getNumTypes()) RetVT = InstPatNode->getType(0);
    MVT::SimpleValueType VT = RetVT;
    if (InstPatNode->getNumChildren()) {
      assert(InstPatNode->getChild(0)->getNumTypes() == 1);
      VT = InstPatNode->getChild(0)->getType(0);
    }

    // For now, filter out instructions which just set a register to
    // an Operand or an immediate, like MOV32ri.
    if (InstPatOp->isSubClassOf("Operand"))
      continue;

    // For now, filter out any instructions with predicates.
    if (!InstPatNode->getPredicateFns().empty())
      continue;

    // Check all the operands.
    OperandsSignature Operands;
    if (!Operands.initialize(InstPatNode, Target, VT))
      continue;
    
    std::vector<std::string>* PhysRegInputs = new std::vector<std::string>();
    if (!InstPatNode->isLeaf() &&
        (InstPatNode->getOperator()->getName() == "imm" ||
         InstPatNode->getOperator()->getName() == "fpimmm"))
      PhysRegInputs->push_back("");
    else if (!InstPatNode->isLeaf()) {
      for (unsigned i = 0, e = InstPatNode->getNumChildren(); i != e; ++i) {
        TreePatternNode *Op = InstPatNode->getChild(i);
        if (!Op->isLeaf()) {
          PhysRegInputs->push_back("");
          continue;
        }
        
        DefInit *OpDI = dynamic_cast<DefInit*>(Op->getLeafValue());
        Record *OpLeafRec = OpDI->getDef();
        std::string PhysReg;
        if (OpLeafRec->isSubClassOf("Register")) {
          PhysReg += static_cast<StringInit*>(OpLeafRec->getValue( \
                     "Namespace")->getValue())->getValue();
          PhysReg += "::";
          
          std::vector<CodeGenRegister> Regs = Target.getRegisters();
          for (unsigned i = 0; i < Regs.size(); ++i) {
            if (Regs[i].TheDef == OpLeafRec) {
              PhysReg += Regs[i].getName();
              break;
            }
          }
        }
      
        PhysRegInputs->push_back(PhysReg);
      }
    } else
      PhysRegInputs->push_back("");

    // Get the predicate that guards this pattern.
    std::string PredicateCheck = Pattern.getPredicateCheck();

    // Ok, we found a pattern that we can handle. Remember it.
    InstructionMemo Memo = {
      Pattern.getDstPattern()->getOperator()->getName(),
      DstRC,
      SubRegNo,
      PhysRegInputs
    };
    assert(!SimplePatterns[Operands][OpcodeName][VT][RetVT]
            .count(PredicateCheck) &&
           "Duplicate pattern!");
    SimplePatterns[Operands][OpcodeName][VT][RetVT][PredicateCheck] = Memo;
  }
}

void FastISelMap::PrintFunctionDefinitions(raw_ostream &OS) {
  // Now emit code for all the patterns that we collected.
  for (OperandsOpcodeTypeRetPredMap::const_iterator OI = SimplePatterns.begin(),
       OE = SimplePatterns.end(); OI != OE; ++OI) {
    const OperandsSignature &Operands = OI->first;
    const OpcodeTypeRetPredMap &OTM = OI->second;

    for (OpcodeTypeRetPredMap::const_iterator I = OTM.begin(), E = OTM.end();
         I != E; ++I) {
      const std::string &Opcode = I->first;
      const TypeRetPredMap &TM = I->second;

      OS << "// FastEmit functions for " << Opcode << ".\n";
      OS << "\n";

      // Emit one function for each opcode,type pair.
      for (TypeRetPredMap::const_iterator TI = TM.begin(), TE = TM.end();
           TI != TE; ++TI) {
        MVT::SimpleValueType VT = TI->first;
        const RetPredMap &RM = TI->second;
        if (RM.size() != 1) {
          for (RetPredMap::const_iterator RI = RM.begin(), RE = RM.end();
               RI != RE; ++RI) {
            MVT::SimpleValueType RetVT = RI->first;
            const PredMap &PM = RI->second;
            bool HasPred = false;

            OS << "unsigned FastEmit_"
               << getLegalCName(Opcode)
               << "_" << getLegalCName(getName(VT))
               << "_" << getLegalCName(getName(RetVT)) << "_";
            Operands.PrintManglingSuffix(OS);
            OS << "(";
            Operands.PrintParameters(OS);
            OS << ") {\n";

            // Emit code for each possible instruction. There may be
            // multiple if there are subtarget concerns.
            for (PredMap::const_iterator PI = PM.begin(), PE = PM.end();
                 PI != PE; ++PI) {
              std::string PredicateCheck = PI->first;
              const InstructionMemo &Memo = PI->second;
  
              if (PredicateCheck.empty()) {
                assert(!HasPred &&
                       "Multiple instructions match, at least one has "
                       "a predicate and at least one doesn't!");
              } else {
                OS << "  if (" + PredicateCheck + ") {\n";
                OS << "  ";
                HasPred = true;
              }
              
              for (unsigned i = 0; i < Memo.PhysRegs->size(); ++i) {
                if ((*Memo.PhysRegs)[i] != "")
                  OS << "  BuildMI(*FuncInfo.MBB, FuncInfo.InsertPt, DL, "
                     << "TII.get(TargetOpcode::COPY), "
                     << (*Memo.PhysRegs)[i] << ").addReg(Op" << i << ");\n";
              }
              
              OS << "  return FastEmitInst_";
              if (Memo.SubRegNo.empty()) {
                Operands.PrintManglingSuffix(OS, *Memo.PhysRegs);
                OS << "(" << InstNS << Memo.Name << ", ";
                OS << InstNS << Memo.RC->getName() << "RegisterClass";
                if (!Operands.empty())
                  OS << ", ";
                Operands.PrintArguments(OS, *Memo.PhysRegs);
                OS << ");\n";
              } else {
                OS << "extractsubreg(" << getName(RetVT);
                OS << ", Op0, Op0IsKill, ";
                OS << Memo.SubRegNo;
                OS << ");\n";
              }
              
              if (HasPred)
                OS << "  }\n";
              
            }
            // Return 0 if none of the predicates were satisfied.
            if (HasPred)
              OS << "  return 0;\n";
            OS << "}\n";
            OS << "\n";
          }
          
          // Emit one function for the type that demultiplexes on return type.
          OS << "unsigned FastEmit_"
             << getLegalCName(Opcode) << "_"
             << getLegalCName(getName(VT)) << "_";
          Operands.PrintManglingSuffix(OS);
          OS << "(MVT RetVT";
          if (!Operands.empty())
            OS << ", ";
          Operands.PrintParameters(OS);
          OS << ") {\nswitch (RetVT.SimpleTy) {\n";
          for (RetPredMap::const_iterator RI = RM.begin(), RE = RM.end();
               RI != RE; ++RI) {
            MVT::SimpleValueType RetVT = RI->first;
            OS << "  case " << getName(RetVT) << ": return FastEmit_"
               << getLegalCName(Opcode) << "_" << getLegalCName(getName(VT))
               << "_" << getLegalCName(getName(RetVT)) << "_";
            Operands.PrintManglingSuffix(OS);
            OS << "(";
            Operands.PrintArguments(OS);
            OS << ");\n";
          }
          OS << "  default: return 0;\n}\n}\n\n";
          
        } else {
          // Non-variadic return type.
          OS << "unsigned FastEmit_"
             << getLegalCName(Opcode) << "_"
             << getLegalCName(getName(VT)) << "_";
          Operands.PrintManglingSuffix(OS);
          OS << "(MVT RetVT";
          if (!Operands.empty())
            OS << ", ";
          Operands.PrintParameters(OS);
          OS << ") {\n";
          
          OS << "  if (RetVT.SimpleTy != " << getName(RM.begin()->first)
             << ")\n    return 0;\n";
          
          const PredMap &PM = RM.begin()->second;
          bool HasPred = false;
          
          // Emit code for each possible instruction. There may be
          // multiple if there are subtarget concerns.
          for (PredMap::const_iterator PI = PM.begin(), PE = PM.end(); PI != PE;
               ++PI) {
            std::string PredicateCheck = PI->first;
            const InstructionMemo &Memo = PI->second;

            if (PredicateCheck.empty()) {
              assert(!HasPred &&
                     "Multiple instructions match, at least one has "
                     "a predicate and at least one doesn't!");
            } else {
              OS << "  if (" + PredicateCheck + ") {\n";
              OS << "  ";
              HasPred = true;
            }
            
            for (unsigned i = 0; i < Memo.PhysRegs->size(); ++i) {
              if ((*Memo.PhysRegs)[i] != "")
                OS << "  BuildMI(*FuncInfo.MBB, FuncInfo.InsertPt, DL, "
                   << "TII.get(TargetOpcode::COPY), "
                   << (*Memo.PhysRegs)[i] << ").addReg(Op" << i << ");\n";
            }
            
            OS << "  return FastEmitInst_";
            
            if (Memo.SubRegNo.empty()) {
              Operands.PrintManglingSuffix(OS, *Memo.PhysRegs);
              OS << "(" << InstNS << Memo.Name << ", ";
              OS << InstNS << Memo.RC->getName() << "RegisterClass";
              if (!Operands.empty())
                OS << ", ";
              Operands.PrintArguments(OS, *Memo.PhysRegs);
              OS << ");\n";
            } else {
              OS << "extractsubreg(RetVT, Op0, Op0IsKill, ";
              OS << Memo.SubRegNo;
              OS << ");\n";
            }
            
             if (HasPred)
               OS << "  }\n";
          }
          
          // Return 0 if none of the predicates were satisfied.
          if (HasPred)
            OS << "  return 0;\n";
          OS << "}\n";
          OS << "\n";
        }
      }

      // Emit one function for the opcode that demultiplexes based on the type.
      OS << "unsigned FastEmit_"
         << getLegalCName(Opcode) << "_";
      Operands.PrintManglingSuffix(OS);
      OS << "(MVT VT, MVT RetVT";
      if (!Operands.empty())
        OS << ", ";
      Operands.PrintParameters(OS);
      OS << ") {\n";
      OS << "  switch (VT.SimpleTy) {\n";
      for (TypeRetPredMap::const_iterator TI = TM.begin(), TE = TM.end();
           TI != TE; ++TI) {
        MVT::SimpleValueType VT = TI->first;
        std::string TypeName = getName(VT);
        OS << "  case " << TypeName << ": return FastEmit_"
           << getLegalCName(Opcode) << "_" << getLegalCName(TypeName) << "_";
        Operands.PrintManglingSuffix(OS);
        OS << "(RetVT";
        if (!Operands.empty())
          OS << ", ";
        Operands.PrintArguments(OS);
        OS << ");\n";
      }
      OS << "  default: return 0;\n";
      OS << "  }\n";
      OS << "}\n";
      OS << "\n";
    }

    OS << "// Top-level FastEmit function.\n";
    OS << "\n";

    // Emit one function for the operand signature that demultiplexes based
    // on opcode and type.
    OS << "unsigned FastEmit_";
    Operands.PrintManglingSuffix(OS);
    OS << "(MVT VT, MVT RetVT, unsigned Opcode";
    if (!Operands.empty())
      OS << ", ";
    Operands.PrintParameters(OS);
    OS << ") {\n";
    OS << "  switch (Opcode) {\n";
    for (OpcodeTypeRetPredMap::const_iterator I = OTM.begin(), E = OTM.end();
         I != E; ++I) {
      const std::string &Opcode = I->first;

      OS << "  case " << Opcode << ": return FastEmit_"
         << getLegalCName(Opcode) << "_";
      Operands.PrintManglingSuffix(OS);
      OS << "(VT, RetVT";
      if (!Operands.empty())
        OS << ", ";
      Operands.PrintArguments(OS);
      OS << ");\n";
    }
    OS << "  default: return 0;\n";
    OS << "  }\n";
    OS << "}\n";
    OS << "\n";
  }
}

void FastISelEmitter::run(raw_ostream &OS) {
  const CodeGenTarget &Target = CGP.getTargetInfo();

  // Determine the target's namespace name.
  std::string InstNS = Target.getInstNamespace() + "::";
  assert(InstNS.size() > 2 && "Can't determine target-specific namespace!");

  EmitSourceFileHeader("\"Fast\" Instruction Selector for the " +
                       Target.getName() + " target", OS);

  FastISelMap F(InstNS);
  F.CollectPatterns(CGP);
  F.PrintFunctionDefinitions(OS);
}

FastISelEmitter::FastISelEmitter(RecordKeeper &R)
  : Records(R),
    CGP(R) {
}

