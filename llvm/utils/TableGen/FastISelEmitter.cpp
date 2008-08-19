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

  void PrintParameters(std::ostream &OS) const {
    for (unsigned i = 0, e = Operands.size(); i != e; ++i) {
      if (Operands[i] == "r") {
        OS << "unsigned Op" << i;
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
                       CGP.getTargetInfo().getName() + " target", OS);
  
  const CodeGenTarget &Target = CGP.getTargetInfo();
  
  // Get the namespace to insert instructions into.  Make sure not to pick up
  // "TargetInstrInfo" by accidentally getting the namespace off the PHI
  // instruction or something.
  std::string InstNS;
  for (CodeGenTarget::inst_iterator i = Target.inst_begin(),
       e = Target.inst_end(); i != e; ++i) {
    InstNS = i->second.Namespace;
    if (InstNS != "TargetInstrInfo")
      break;
  }

  OS << "namespace llvm {\n";
  OS << "namespace " << InstNS << " {\n";
  OS << "class FastISel;\n";
  OS << "}\n";
  OS << "}\n";
  OS << "\n";
  
  if (!InstNS.empty()) InstNS += "::";

  typedef std::map<MVT::SimpleValueType, InstructionMemo> TypeMap;
  typedef std::map<std::string, TypeMap> OpcodeTypeMap;
  typedef std::map<OperandsSignature, OpcodeTypeMap> OperandsOpcodeTypeMap;
  OperandsOpcodeTypeMap SimplePatterns;

  // Create the supported type signatures.
  OperandsSignature KnownOperands;
  SimplePatterns[KnownOperands] = OpcodeTypeMap();
  KnownOperands.Operands.push_back("r");
  SimplePatterns[KnownOperands] = OpcodeTypeMap();
  KnownOperands.Operands.push_back("r");
  SimplePatterns[KnownOperands] = OpcodeTypeMap();

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
    // an Operand, like MOV32ri.
    if (InstPatOp->isSubClassOf("Operand"))
      continue;

    // Check all the operands. For now only accept register operands.
    OperandsSignature Operands;
    for (unsigned i = 0, e = InstPatNode->getNumChildren(); i != e; ++i) {
      TreePatternNode *Op = InstPatNode->getChild(i);
      if (!Op->isLeaf())
        goto continue_label;
      DefInit *OpDI = dynamic_cast<DefInit*>(Op->getLeafValue());
      if (!OpDI)
        goto continue_label;
      Record *OpLeafRec = OpDI->getDef();
      if (!OpLeafRec->isSubClassOf("RegisterClass"))
        goto continue_label;
      const CodeGenRegisterClass *RC = &Target.getRegisterClass(OpLeafRec);
      if (!RC)
        goto continue_label;
      if (Op->getTypeNum(0) != VT)
        goto continue_label;
      Operands.Operands.push_back("r");
    }

    // If it's not a known signature, ignore it.
    if (!SimplePatterns.count(Operands))
      continue;

    // Ok, we found a pattern that we can handle. Remember it.
    {
      InstructionMemo Memo = { Pattern.getDstPattern()->getOperator()->getName(),
                               DstRC };
      SimplePatterns[Operands][OpcodeName][VT] = Memo;
    }

  continue_label:;
  }

  OS << "#include \"llvm/CodeGen/FastISel.h\"\n";
  OS << "\n";
  OS << "namespace llvm {\n";
  OS << "\n";

  // Declare the target FastISel class.
  OS << "class " << InstNS << "FastISel : public llvm::FastISel {\n";
  for (OperandsOpcodeTypeMap::const_iterator OI = SimplePatterns.begin(),
       OE = SimplePatterns.end(); OI != OE; ++OI) {
    const OperandsSignature &Operands = OI->first;
    const OpcodeTypeMap &OTM = OI->second;

    for (OpcodeTypeMap::const_iterator I = OTM.begin(), E = OTM.end();
         I != E; ++I) {
      const std::string &Opcode = I->first;
      const TypeMap &TM = I->second;

      for (TypeMap::const_iterator TI = TM.begin(), TE = TM.end();
           TI != TE; ++TI) {
        MVT::SimpleValueType VT = TI->first;

        OS << "  unsigned FastEmit_" << getLegalCName(Opcode)
           << "_" << getLegalCName(getName(VT)) << "(";
        Operands.PrintParameters(OS);
        OS << ");\n";
      }

      OS << "  unsigned FastEmit_" << getLegalCName(Opcode)
         << "(MVT::SimpleValueType VT";
      if (!Operands.empty())
        OS << ", ";
      Operands.PrintParameters(OS);
      OS << ");\n";
    }

    OS << "unsigned FastEmit_";
    Operands.PrintManglingSuffix(OS);
    OS << "(MVT::SimpleValueType VT, ISD::NodeType Opcode";
    if (!Operands.empty())
      OS << ", ";
    Operands.PrintParameters(OS);
    OS << ");\n";
  }
  OS << "public:\n";
  OS << "  FastISel(MachineBasicBlock *mbb, MachineFunction *mf, ";
  OS << "const TargetInstrInfo *tii) : llvm::FastISel(mbb, mf, tii) {}\n";
  OS << "};\n";
  OS << "\n";

  // Define the target FastISel creation function.
  OS << "llvm::FastISel *" << InstNS
     << "createFastISel(MachineBasicBlock *mbb, MachineFunction *mf, ";
  OS << "const TargetInstrInfo *tii) {\n";
  OS << "  return new " << InstNS << "FastISel(mbb, mf, tii);\n";
  OS << "}\n";
  OS << "\n";

  // Now emit code for all the patterns that we collected.
  for (OperandsOpcodeTypeMap::const_iterator OI = SimplePatterns.begin(),
       OE = SimplePatterns.end(); OI != OE; ++OI) {
    const OperandsSignature &Operands = OI->first;
    const OpcodeTypeMap &OTM = OI->second;

    for (OpcodeTypeMap::const_iterator I = OTM.begin(), E = OTM.end();
         I != E; ++I) {
      const std::string &Opcode = I->first;
      const TypeMap &TM = I->second;

      OS << "// FastEmit functions for " << Opcode << ".\n";
      OS << "\n";

      // Emit one function for each opcode,type pair.
      for (TypeMap::const_iterator TI = TM.begin(), TE = TM.end();
           TI != TE; ++TI) {
        MVT::SimpleValueType VT = TI->first;
        const InstructionMemo &Memo = TI->second;
  
        OS << "unsigned " << InstNS << "FastISel::FastEmit_"
           << getLegalCName(Opcode)
           << "_" << getLegalCName(getName(VT)) << "(";
        Operands.PrintParameters(OS);
        OS << ") {\n";
        OS << "  return FastEmitInst_";
        Operands.PrintManglingSuffix(OS);
        OS << "(" << InstNS << Memo.Name << ", ";
        OS << InstNS << Memo.RC->getName() << "RegisterClass";
        if (!Operands.empty())
          OS << ", ";
        Operands.PrintArguments(OS);
        OS << ");\n";
        OS << "}\n";
        OS << "\n";
      }

      // Emit one function for the opcode that demultiplexes based on the type.
      OS << "unsigned " << InstNS << "FastISel::FastEmit_"
         << getLegalCName(Opcode) << "(MVT::SimpleValueType VT";
      if (!Operands.empty())
        OS << ", ";
      Operands.PrintParameters(OS);
      OS << ") {\n";
      OS << "  switch (VT) {\n";
      for (TypeMap::const_iterator TI = TM.begin(), TE = TM.end();
           TI != TE; ++TI) {
        MVT::SimpleValueType VT = TI->first;
        std::string TypeName = getName(VT);
        OS << "  case " << TypeName << ": return FastEmit_"
           << getLegalCName(Opcode) << "_" << getLegalCName(TypeName) << "(";
        Operands.PrintArguments(OS);
        OS << ");\n";
      }
      OS << "  default: return 0;\n";
      OS << "  }\n";
      OS << "}\n";
      OS << "\n";
    }

    // Emit one function for the operand signature that demultiplexes based
    // on opcode and type.
    OS << "unsigned " << InstNS << "FastISel::FastEmit_";
    Operands.PrintManglingSuffix(OS);
    OS << "(MVT::SimpleValueType VT, ISD::NodeType Opcode";
    if (!Operands.empty())
      OS << ", ";
    Operands.PrintParameters(OS);
    OS << ") {\n";
    OS << "  switch (Opcode) {\n";
    for (OpcodeTypeMap::const_iterator I = OTM.begin(), E = OTM.end();
         I != E; ++I) {
      const std::string &Opcode = I->first;

      OS << "  case " << Opcode << ": return FastEmit_"
         << getLegalCName(Opcode) << "(VT";
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

  OS << "}\n";
}

// todo: really filter out Constants
