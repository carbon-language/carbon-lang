//===- FastISelEmitter.cpp - Generate an instruction selector -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend emits a "fast" instruction selector.
//
// This instruction selection method is designed to emit very poor code
// quickly. Also, it is not designed to do much lowering, so most illegal
// types (e.g. i64 on 32-bit targets) and operations (e.g. calls) are not
// supported and cannot easily be added. Blocks containing operations
// that are not supported need to be handled by a more capable selector,
// such as the SelectionDAG selector.
//
// The intended use for "fast" instruction selection is "-O0" mode
// compilation, where the quality of the generated code is irrelevant when
// weighed against the speed at which the code can be generated.
//
// If compile time is so important, you might wonder why we don't just
// skip codegen all-together, emit LLVM bytecode files, and execute them
// with an interpreter. The answer is that it would complicate linking and
// debugging, and also because that isn't how a compiler is expected to
// work in some circles.
//
// If you need better generated code or more lowering than what this
// instruction selector provides, use the SelectionDAG (DAGISel) instruction
// selector instead. If you're looking here because SelectionDAG isn't fast
// enough, consider looking into improving the SelectionDAG infastructure
// instead. At the time of this writing there remain several major
// opportunities for improvement.
// 
//===----------------------------------------------------------------------===//

#include "FastISelEmitter.h"
#include "Record.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Streams.h"
#include "llvm/ADT/VectorExtras.h"
using namespace llvm;

namespace {

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
                  MVT::SimpleValueType VT,
                  const CodeGenRegisterClass *DstRC) {
    for (unsigned i = 0, e = InstPatNode->getNumChildren(); i != e; ++i) {
      TreePatternNode *Op = InstPatNode->getChild(i);
      // For now, filter out any operand with a predicate.
      if (!Op->getPredicateFn().empty())
        return false;
      // For now, filter out any operand with multiple values.
      if (Op->getExtTypes().size() != 1)
        return false;
      // For now, all the operands must have the same type.
      if (Op->getTypeNum(0) != VT)
        return false;
      if (!Op->isLeaf()) {
        if (Op->getOperator()->getName() == "imm") {
          Operands.push_back("i");
          return true;
        }
        // For now, ignore fpimm and other non-leaf nodes.
        return false;
      }
      DefInit *OpDI = dynamic_cast<DefInit*>(Op->getLeafValue());
      if (!OpDI)
        return false;
      Record *OpLeafRec = OpDI->getDef();
      // TODO: handle instructions which have physreg operands.
      if (OpLeafRec->isSubClassOf("Register"))
        return false;
      // For now, the only other thing we accept is register operands.
      if (!OpLeafRec->isSubClassOf("RegisterClass"))
        return false;
      // For now, require the register operands' register classes to all
      // be the same.
      const CodeGenRegisterClass *RC = &Target.getRegisterClass(OpLeafRec);
      if (!RC)
        return false;
      // For now, all the operands must have the same register class.
      if (DstRC != RC)
        return false;
      Operands.push_back("r");
    }
    return true;
  }

  void PrintParameters(std::ostream &OS) const {
    for (unsigned i = 0, e = Operands.size(); i != e; ++i) {
      if (Operands[i] == "r") {
        OS << "unsigned Op" << i;
      } else if (Operands[i] == "i") {
        OS << "uint64_t imm" << i;
      } else {
        assert("Unknown operand kind!");
        abort();
      }
      if (i + 1 != e)
        OS << ", ";
    }
  }

  void PrintArguments(std::ostream &OS) const {
    for (unsigned i = 0, e = Operands.size(); i != e; ++i) {
      if (Operands[i] == "r") {
        OS << "Op" << i;
      } else if (Operands[i] == "i") {
        OS << "imm" << i;
      } else {
        assert("Unknown operand kind!");
        abort();
      }
      if (i + 1 != e)
        OS << ", ";
    }
  }

  void PrintManglingSuffix(std::ostream &OS) const {
    for (unsigned i = 0, e = Operands.size(); i != e; ++i) {
      OS << Operands[i];
    }
  }
};

/// InstructionMemo - This class holds additional information about an
/// instruction needed to emit code for it.
///
struct InstructionMemo {
  std::string Name;
  const CodeGenRegisterClass *RC;
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

void FastISelEmitter::run(std::ostream &OS) {
  EmitSourceFileHeader("\"Fast\" Instruction Selector for the " +
                       Target.getName() + " target", OS);

  OS << "#include \"llvm/CodeGen/FastISel.h\"\n";
  OS << "\n";
  OS << "namespace llvm {\n";
  OS << "\n";
  OS << "namespace " << InstNS.substr(0, InstNS.size() - 2) << " {\n";
  OS << "\n";
  
  typedef std::map<std::string, InstructionMemo> PredMap;
  typedef std::map<MVT::SimpleValueType, PredMap> TypePredMap;
  typedef std::map<std::string, TypePredMap> OpcodeTypePredMap;
  typedef std::map<OperandsSignature, OpcodeTypePredMap> OperandsOpcodeTypePredMap;
  OperandsOpcodeTypePredMap SimplePatterns;

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
    CodeGenInstruction &II = CGP.getTargetInfo().getInstruction(Op->getName());
    if (II.OperandList.empty())
      continue;

    // For now, ignore instructions where the first operand is not an
    // output register.
    Record *Op0Rec = II.OperandList[0].Rec;
    if (!Op0Rec->isSubClassOf("RegisterClass"))
      continue;
    const CodeGenRegisterClass *DstRC = &Target.getRegisterClass(Op0Rec);
    if (!DstRC)
      continue;

    // Inspect the pattern.
    TreePatternNode *InstPatNode = Pattern.getSrcPattern();
    if (!InstPatNode) continue;
    if (InstPatNode->isLeaf()) continue;

    Record *InstPatOp = InstPatNode->getOperator();
    std::string OpcodeName = getOpcodeName(InstPatOp, CGP);
    MVT::SimpleValueType VT = InstPatNode->getTypeNum(0);

    // For now, filter out instructions which just set a register to
    // an Operand or an immediate, like MOV32ri.
    if (InstPatOp->isSubClassOf("Operand"))
      continue;
    if (InstPatOp->getName() == "imm" ||
        InstPatOp->getName() == "fpimm")
      continue;

    // For now, filter out any instructions with predicates.
    if (!InstPatNode->getPredicateFn().empty())
      continue;

    // Check all the operands.
    OperandsSignature Operands;
    if (!Operands.initialize(InstPatNode, Target, VT, DstRC))
      continue;

    // Get the predicate that guards this pattern.
    std::string PredicateCheck = Pattern.getPredicateCheck();

    // Ok, we found a pattern that we can handle. Remember it.
    InstructionMemo Memo = {
      Pattern.getDstPattern()->getOperator()->getName(),
      DstRC
    };
    assert(!SimplePatterns[Operands][OpcodeName][VT].count(PredicateCheck) &&
           "Duplicate pattern!");
    SimplePatterns[Operands][OpcodeName][VT][PredicateCheck] = Memo;
  }

  // Declare the target FastISel class.
  OS << "class FastISel : public llvm::FastISel {\n";
  for (OperandsOpcodeTypePredMap::const_iterator OI = SimplePatterns.begin(),
       OE = SimplePatterns.end(); OI != OE; ++OI) {
    const OperandsSignature &Operands = OI->first;
    const OpcodeTypePredMap &OTM = OI->second;

    for (OpcodeTypePredMap::const_iterator I = OTM.begin(), E = OTM.end();
         I != E; ++I) {
      const std::string &Opcode = I->first;
      const TypePredMap &TM = I->second;

      for (TypePredMap::const_iterator TI = TM.begin(), TE = TM.end();
           TI != TE; ++TI) {
        MVT::SimpleValueType VT = TI->first;

        OS << "  unsigned FastEmit_" << getLegalCName(Opcode)
           << "_" << getLegalCName(getName(VT)) << "_";
        Operands.PrintManglingSuffix(OS);
        OS << "(";
        Operands.PrintParameters(OS);
        OS << ");\n";
      }

      OS << "  unsigned FastEmit_" << getLegalCName(Opcode) << "_";
      Operands.PrintManglingSuffix(OS);
      OS << "(MVT::SimpleValueType VT";
      if (!Operands.empty())
        OS << ", ";
      Operands.PrintParameters(OS);
      OS << ");\n";
    }

    OS << "  unsigned FastEmit_";
    Operands.PrintManglingSuffix(OS);
    OS << "(MVT::SimpleValueType VT, ISD::NodeType Opcode";
    if (!Operands.empty())
      OS << ", ";
    Operands.PrintParameters(OS);
    OS << ");\n";
  }
  OS << "\n";

  // Declare the Subtarget member, which is used for predicate checks.
  OS << "  const " << InstNS.substr(0, InstNS.size() - 2)
     << "Subtarget *Subtarget;\n";
  OS << "\n";

  // Declare the constructor.
  OS << "public:\n";
  OS << "  explicit FastISel(MachineFunction &mf)\n";
  OS << "     : llvm::FastISel(mf),\n";
  OS << "       Subtarget(&TM.getSubtarget<" << InstNS.substr(0, InstNS.size() - 2)
     << "Subtarget>()) {}\n";
  OS << "};\n";
  OS << "\n";

  // Define the target FastISel creation function.
  OS << "llvm::FastISel *createFastISel(MachineFunction &mf) {\n";
  OS << "  return new FastISel(mf);\n";
  OS << "}\n";
  OS << "\n";

  // Now emit code for all the patterns that we collected.
  for (OperandsOpcodeTypePredMap::const_iterator OI = SimplePatterns.begin(),
       OE = SimplePatterns.end(); OI != OE; ++OI) {
    const OperandsSignature &Operands = OI->first;
    const OpcodeTypePredMap &OTM = OI->second;

    for (OpcodeTypePredMap::const_iterator I = OTM.begin(), E = OTM.end();
         I != E; ++I) {
      const std::string &Opcode = I->first;
      const TypePredMap &TM = I->second;

      OS << "// FastEmit functions for " << Opcode << ".\n";
      OS << "\n";

      // Emit one function for each opcode,type pair.
      for (TypePredMap::const_iterator TI = TM.begin(), TE = TM.end();
           TI != TE; ++TI) {
        MVT::SimpleValueType VT = TI->first;
        const PredMap &PM = TI->second;
        bool HasPred = false;

        OS << "unsigned FastISel::FastEmit_"
           << getLegalCName(Opcode)
           << "_" << getLegalCName(getName(VT)) << "_";
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
            assert(!HasPred && "Multiple instructions match, at least one has "
                               "a predicate and at least one doesn't!");
          } else {
            OS << "  if (" + PredicateCheck + ")\n";
            OS << "  ";
            HasPred = true;
          }
          OS << "  return FastEmitInst_";
          Operands.PrintManglingSuffix(OS);
          OS << "(" << InstNS << Memo.Name << ", ";
          OS << InstNS << Memo.RC->getName() << "RegisterClass";
          if (!Operands.empty())
            OS << ", ";
          Operands.PrintArguments(OS);
          OS << ");\n";
        }
        // Return 0 if none of the predicates were satisfied.
        if (HasPred)
          OS << "  return 0;\n";
        OS << "}\n";
        OS << "\n";
      }

      // Emit one function for the opcode that demultiplexes based on the type.
      OS << "unsigned FastISel::FastEmit_"
         << getLegalCName(Opcode) << "_";
      Operands.PrintManglingSuffix(OS);
      OS << "(MVT::SimpleValueType VT";
      if (!Operands.empty())
        OS << ", ";
      Operands.PrintParameters(OS);
      OS << ") {\n";
      OS << "  switch (VT) {\n";
      for (TypePredMap::const_iterator TI = TM.begin(), TE = TM.end();
           TI != TE; ++TI) {
        MVT::SimpleValueType VT = TI->first;
        std::string TypeName = getName(VT);
        OS << "  case " << TypeName << ": return FastEmit_"
           << getLegalCName(Opcode) << "_" << getLegalCName(TypeName) << "_";
        Operands.PrintManglingSuffix(OS);
        OS << "(";
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
    OS << "unsigned FastISel::FastEmit_";
    Operands.PrintManglingSuffix(OS);
    OS << "(MVT::SimpleValueType VT, ISD::NodeType Opcode";
    if (!Operands.empty())
      OS << ", ";
    Operands.PrintParameters(OS);
    OS << ") {\n";
    OS << "  switch (Opcode) {\n";
    for (OpcodeTypePredMap::const_iterator I = OTM.begin(), E = OTM.end();
         I != E; ++I) {
      const std::string &Opcode = I->first;

      OS << "  case " << Opcode << ": return FastEmit_"
         << getLegalCName(Opcode) << "_";
      Operands.PrintManglingSuffix(OS);
      OS << "(VT";
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

  OS << "} // namespace X86\n";
  OS << "\n";
  OS << "} // namespace llvm\n";
}

FastISelEmitter::FastISelEmitter(RecordKeeper &R)
  : Records(R),
    CGP(R),
    Target(CGP.getTargetInfo()),
    InstNS(Target.getInstNamespace() + "::") {

  assert(InstNS.size() > 2 && "Can't determine target-specific namespace!");
}
