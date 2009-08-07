//===- AsmMatcherEmitter.cpp - Generate an assembly matcher ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend emits a target specifier matcher for converting parsed
// assembly operands in the MCInst structures.
//
// The input to the target specific matcher is a list of literal tokens and
// operands. The target specific parser should generally eliminate any syntax
// which is not relevant for matching; for example, comma tokens should have
// already been consumed and eliminated by the parser. Most instructions will
// end up with a single literal token (the instruction name) and some number of
// operands.
//
// Some example inputs, for X86:
//   'addl' (immediate ...) (register ...)
//   'add' (immediate ...) (memory ...)
//   'call' '*' %epc 
//
// The assembly matcher is responsible for converting this input into a precise
// machine instruction (i.e., an instruction with a well defined encoding). This
// mapping has several properties which complicate matching:
//
//  - It may be ambiguous; many architectures can legally encode particular
//    variants of an instruction in different ways (for example, using a smaller
//    encoding for small immediates). Such ambiguities should never be
//    arbitrarily resolved by the assembler, the assembler is always responsible
//    for choosing the "best" available instruction.
//
//  - It may depend on the subtarget or the assembler context. Instructions
//    which are invalid for the current mode, but otherwise unambiguous (e.g.,
//    an SSE instruction in a file being assembled for i486) should be accepted
//    and rejected by the assembler front end. However, if the proper encoding
//    for an instruction is dependent on the assembler context then the matcher
//    is responsible for selecting the correct machine instruction for the
//    current mode.
//
// The core matching algorithm attempts to exploit the regularity in most
// instruction sets to quickly determine the set of possibly matching
// instructions, and the simplify the generated code. Additionally, this helps
// to ensure that the ambiguities are intentionally resolved by the user.
//
// The matching is divided into two distinct phases:
//
//   1. Classification: Each operand is mapped to the unique set which (a)
//      contains it, and (b) is the largest such subset for which a single
//      instruction could match all members.
//
//      For register classes, we can generate these subgroups automatically. For
//      arbitrary operands, we expect the user to define the classes and their
//      relations to one another (for example, 8-bit signed immediates as a
//      subset of 32-bit immediates).
//
//      By partitioning the operands in this way, we guarantee that for any
//      tuple of classes, any single instruction must match either all or none
//      of the sets of operands which could classify to that tuple.
//
//      In addition, the subset relation amongst classes induces a partial order
//      on such tuples, which we use to resolve ambiguities.
//
//      FIXME: What do we do if a crazy case shows up where this is the wrong
//      resolution?
//
//   2. The input can now be treated as a tuple of classes (static tokens are
//      simple singleton sets). Each such tuple should generally map to a single
//      instruction (we currently ignore cases where this isn't true, whee!!!),
//      which we can emit a simple matcher for.
//
//===----------------------------------------------------------------------===//

#include "AsmMatcherEmitter.h"
#include "CodeGenTarget.h"
#include "Record.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include <set>
#include <list>
using namespace llvm;

namespace {
  cl::opt<std::string>
  MatchOneInstr("match-one-instr", cl::desc("Match only the named instruction"),
              cl::init(""));
}

/// FlattenVariants - Flatten an .td file assembly string by selecting the
/// variant at index \arg N.
static std::string FlattenVariants(const std::string &AsmString,
                                   unsigned N) {
  StringRef Cur = AsmString;
  std::string Res = "";
  
  for (;;) {
    // Find the start of the next variant string.
    size_t VariantsStart = 0;
    for (size_t e = Cur.size(); VariantsStart != e; ++VariantsStart)
      if (Cur[VariantsStart] == '{' && 
          (VariantsStart == 0 || (Cur[VariantsStart-1] != '$' &&
                                  Cur[VariantsStart-1] != '\\')))
        break;

    // Add the prefix to the result.
    Res += Cur.slice(0, VariantsStart);
    if (VariantsStart == Cur.size())
      break;

    ++VariantsStart; // Skip the '{'.

    // Scan to the end of the variants string.
    size_t VariantsEnd = VariantsStart;
    unsigned NestedBraces = 1;
    for (size_t e = Cur.size(); VariantsEnd != e; ++VariantsEnd) {
      if (Cur[VariantsEnd] == '}' && Cur[VariantsEnd-1] != '\\') {
        if (--NestedBraces == 0)
          break;
      } else if (Cur[VariantsEnd] == '{')
        ++NestedBraces;
    }

    // Select the Nth variant (or empty).
    StringRef Selection = Cur.slice(VariantsStart, VariantsEnd);
    for (unsigned i = 0; i != N; ++i)
      Selection = Selection.split('|').second;
    Res += Selection.split('|').first;

    assert(VariantsEnd != Cur.size() && 
           "Unterminated variants in assembly string!");
    Cur = Cur.substr(VariantsEnd + 1);
  } 

  return Res;
}

/// TokenizeAsmString - Tokenize a simplified assembly string.
static void TokenizeAsmString(const StringRef &AsmString, 
                              SmallVectorImpl<StringRef> &Tokens) {
  unsigned Prev = 0;
  bool InTok = true;
  for (unsigned i = 0, e = AsmString.size(); i != e; ++i) {
    switch (AsmString[i]) {
    case '[':
    case ']':
    case '*':
    case '!':
    case ' ':
    case '\t':
    case ',':
      if (InTok) {
        Tokens.push_back(AsmString.slice(Prev, i));
        InTok = false;
      }
      if (!isspace(AsmString[i]) && AsmString[i] != ',')
        Tokens.push_back(AsmString.substr(i, 1));
      Prev = i + 1;
      break;
      
    case '\\':
      if (InTok) {
        Tokens.push_back(AsmString.slice(Prev, i));
        InTok = false;
      }
      ++i;
      assert(i != AsmString.size() && "Invalid quoted character");
      Tokens.push_back(AsmString.substr(i, 1));
      Prev = i + 1;
      break;

    case '$': {
      // If this isn't "${", treat like a normal token.
      if (i + 1 == AsmString.size() || AsmString[i + 1] != '{') {
        if (InTok) {
          Tokens.push_back(AsmString.slice(Prev, i));
          InTok = false;
        }
        Prev = i;
        break;
      }

      if (InTok) {
        Tokens.push_back(AsmString.slice(Prev, i));
        InTok = false;
      }

      StringRef::iterator End =
        std::find(AsmString.begin() + i, AsmString.end(), '}');
      assert(End != AsmString.end() && "Missing brace in operand reference!");
      size_t EndPos = End - AsmString.begin();
      Tokens.push_back(AsmString.slice(i, EndPos+1));
      Prev = EndPos + 1;
      i = EndPos;
      break;
    }

    default:
      InTok = true;
    }
  }
  if (InTok && Prev != AsmString.size())
    Tokens.push_back(AsmString.substr(Prev));
}

static bool IsAssemblerInstruction(const StringRef &Name,
                                   const CodeGenInstruction &CGI, 
                                   const SmallVectorImpl<StringRef> &Tokens) {
  // Ignore psuedo ops.
  //
  // FIXME: This is a hack.
  if (const RecordVal *Form = CGI.TheDef->getValue("Form"))
    if (Form->getValue()->getAsString() == "Pseudo")
      return false;
  
  // Ignore "PHI" node.
  //
  // FIXME: This is also a hack.
  if (Name == "PHI")
    return false;

  // Ignore instructions with no .s string.
  //
  // FIXME: What are these?
  if (CGI.AsmString.empty())
    return false;

  // FIXME: Hack; ignore any instructions with a newline in them.
  if (std::find(CGI.AsmString.begin(), 
                CGI.AsmString.end(), '\n') != CGI.AsmString.end())
    return false;
  
  // Ignore instructions with attributes, these are always fake instructions for
  // simplifying codegen.
  //
  // FIXME: Is this true?
  //
  // Also, we ignore instructions which reference the operand multiple times;
  // this implies a constraint we would not currently honor. These are
  // currently always fake instructions for simplifying codegen.
  //
  // FIXME: Encode this assumption in the .td, so we can error out here.
  std::set<std::string> OperandNames;
  for (unsigned i = 1, e = Tokens.size(); i < e; ++i) {
    if (Tokens[i][0] == '$' && 
        std::find(Tokens[i].begin(), 
                  Tokens[i].end(), ':') != Tokens[i].end()) {
      DEBUG({
          errs() << "warning: '" << Name << "': "
                 << "ignoring instruction; operand with attribute '" 
                 << Tokens[i] << "', \n";
        });
      return false;
    }

    if (Tokens[i][0] == '$' && !OperandNames.insert(Tokens[i]).second) {
      DEBUG({
          errs() << "warning: '" << Name << "': "
                 << "ignoring instruction; tied operand '" 
                 << Tokens[i] << "', \n";
        });
      return false;
    }
  }

  return true;
}

namespace {

struct OperandListLess {
  bool operator()(const
                  std::pair<const CodeGenInstruction::OperandInfo*, unsigned> &
                  A,
                  const
                  std::pair<const CodeGenInstruction::OperandInfo*, unsigned> &
                  B) {
    return A.first->MIOperandNo < B.first->MIOperandNo;
  }
                  
};

struct InstructionInfo {
  struct Operand {
    enum {
      Token,
      Class
    } Kind;

    struct ClassData {
      /// Operand - The tablegen operand this class corresponds to.
      const CodeGenInstruction::OperandInfo *Operand;

      /// ClassName - The name of this operand's class.
      std::string ClassName;

      /// PredicateMethod - The name of the operand method to test whether the
      /// operand matches this class.
      std::string PredicateMethod;

      /// RenderMethod - The name of the operand method to add this operand to
      /// an MCInst.
      std::string RenderMethod;
    } AsClass;
  };

  /// InstrName - The target name for this instruction.
  std::string InstrName;

  /// Instr - The instruction this matches.
  const CodeGenInstruction *Instr;

  /// AsmString - The assembly string for this instruction (with variants
  /// removed).
  std::string AsmString;

  /// Tokens - The tokenized assembly pattern that this instruction matches.
  SmallVector<StringRef, 4> Tokens;

  /// Operands - The operands that this instruction matches.
  SmallVector<Operand, 4> Operands;

  /// ConversionFn - The name of the conversion function to convert parsed
  /// operands into an MCInst for this function.
  std::string ConversionFn;

  /// OrderedClassOperands - The indices of the class operands, ordered by their
  /// MIOperandNo order (which is the order they should be passed to the
  /// conversion function).
  SmallVector<unsigned, 4> OrderedClassOperands;

public:
  void dump();
};

}

void InstructionInfo::dump() {
  errs() << InstrName << " -- " << "flattened:\"" << AsmString << '\"'
         << ", tokens:[";
  for (unsigned i = 0, e = Tokens.size(); i != e; ++i) {
    errs() << Tokens[i];
    if (i + 1 != e)
      errs() << ", ";
  }
  errs() << "]\n";

  for (unsigned i = 0, e = Operands.size(); i != e; ++i) {
    Operand &Op = Operands[i];
    errs() << "  op[" << i << "] = ";
    if (Op.Kind == Operand::Token) {
      errs() << '\"' << Tokens[i] << "\"\n";
      continue;
    }

    assert(Op.Kind == Operand::Class && "Invalid kind!");
    const CodeGenInstruction::OperandInfo &OI = *Op.AsClass.Operand;
    errs() << OI.Name << " " << OI.Rec->getName()
           << " (" << OI.MIOperandNo << ", " << OI.MINumOperands << ")\n";
  }
}

static void BuildInstructionInfos(CodeGenTarget &Target,
                                  std::vector<InstructionInfo*> &Infos) {
  const std::map<std::string, CodeGenInstruction> &Instructions =
    Target.getInstructions();

  for (std::map<std::string, CodeGenInstruction>::const_iterator 
         it = Instructions.begin(), ie = Instructions.end(); it != ie; ++it) {
    const CodeGenInstruction &CGI = it->second;

    if (!MatchOneInstr.empty() && it->first != MatchOneInstr)
      continue;

    OwningPtr<InstructionInfo> II(new InstructionInfo);
    
    II->InstrName = it->first;
    II->Instr = &it->second;
    II->AsmString = FlattenVariants(CGI.AsmString, 0);

    TokenizeAsmString(II->AsmString, II->Tokens);

    // Ignore instructions which shouldn't be matched.
    if (!IsAssemblerInstruction(it->first, CGI, II->Tokens))
      continue;

    for (unsigned i = 0, e = II->Tokens.size(); i != e; ++i) {
      StringRef Token = II->Tokens[i];

      // Check for simple tokens.
      if (Token[0] != '$') {
        InstructionInfo::Operand Op;
        Op.Kind = InstructionInfo::Operand::Token;
        II->Operands.push_back(Op);
        continue;
      }

      // Otherwise this is an operand reference.
      InstructionInfo::Operand Op;
      Op.Kind = InstructionInfo::Operand::Class;

      StringRef OperandName;
      if (Token[1] == '{')
        OperandName = Token.substr(2, Token.size() - 3);
      else
        OperandName = Token.substr(1);

      // Map this token to an operand. FIXME: Move elsewhere.
      unsigned Idx;
      try {
        Idx = CGI.getOperandNamed(OperandName);
      } catch(...) {
        errs() << "error: unable to find operand: '" << OperandName << "'!\n";
        break;
      }

      const CodeGenInstruction::OperandInfo &OI = CGI.OperandList[Idx];      
      Op.AsClass.Operand = &OI;

      if (OI.Rec->isSubClassOf("RegisterClass")) {
        Op.AsClass.ClassName = "Reg";
        Op.AsClass.PredicateMethod = "isReg";
        Op.AsClass.RenderMethod = "addRegOperands";
      } else if (OI.Rec->isSubClassOf("Operand")) {
        // FIXME: This should not be hard coded.
        const RecordVal *RV = OI.Rec->getValue("Type");

        // FIXME: Yet another total hack.
        if (RV->getValue()->getAsString() == "iPTR" ||
            OI.Rec->getName() == "lea32mem" ||
            OI.Rec->getName() == "lea64_32mem") {
          Op.AsClass.ClassName = "Mem";
          Op.AsClass.PredicateMethod = "isMem";
          Op.AsClass.RenderMethod = "addMemOperands";
        } else {
          Op.AsClass.ClassName = "Imm";
          Op.AsClass.PredicateMethod = "isImm";
          Op.AsClass.RenderMethod = "addImmOperands";
        }
      } else {
        OI.Rec->dump();
        assert(0 && "Unexpected instruction operand record!");
      }

      II->Operands.push_back(Op);
    }

    // If we broke out, ignore the instruction.
    if (II->Operands.size() != II->Tokens.size())
      continue;

    Infos.push_back(II.take());
  }
}

static void ConstructConversionFunctions(CodeGenTarget &Target,
                                         std::vector<InstructionInfo*> &Infos,
                                         raw_ostream &OS) {
  // Function we have already generated.
  std::set<std::string> GeneratedFns;

  for (std::vector<InstructionInfo*>::const_iterator it = Infos.begin(),
         ie = Infos.end(); it != ie; ++it) {
    InstructionInfo &II = **it;

    // Order the (class) operands by the order to convert them into an MCInst.
    SmallVector<std::pair<unsigned, unsigned>, 4> MIOperandList;
    for (unsigned i = 0, e = II.Operands.size(); i != e; ++i) {
      InstructionInfo::Operand &Op = II.Operands[i];
      if (Op.Kind == InstructionInfo::Operand::Class)
        MIOperandList.push_back(std::make_pair(Op.AsClass.Operand->MIOperandNo,
                                               i));
    }
    std::sort(MIOperandList.begin(), MIOperandList.end());

    // Compute the total number of operands.
    unsigned NumMIOperands = 0;
    for (unsigned i = 0, e = II.Instr->OperandList.size(); i != e; ++i) {
      const CodeGenInstruction::OperandInfo &OI = II.Instr->OperandList[i];
      NumMIOperands = std::max(NumMIOperands, 
                               OI.MIOperandNo + OI.MINumOperands);
    }

    // Build the conversion function signature.
    std::string Signature = "Convert";
    unsigned CurIndex = 0;
    for (unsigned i = 0, e = MIOperandList.size(); i != e; ++i) {
      InstructionInfo::Operand &Op = II.Operands[MIOperandList[i].second];
      assert(CurIndex <= Op.AsClass.Operand->MIOperandNo &&
             "Duplicate match for instruction operand!");

      // Save the conversion index, for use by the matcher.
      II.OrderedClassOperands.push_back(MIOperandList[i].second);
      
      // Skip operands which weren't matched by anything, this occurs when the
      // .td file encodes "implicit" operands as explicit ones.
      //
      // FIXME: This should be removed from the MCInst structure.
      for (; CurIndex != Op.AsClass.Operand->MIOperandNo; ++CurIndex)
        Signature += "Imp";

      Signature += Op.AsClass.ClassName;
      Signature += utostr(Op.AsClass.Operand->MINumOperands);
      CurIndex += Op.AsClass.Operand->MINumOperands;
    }

    // Add any trailing implicit operands.
    for (; CurIndex != NumMIOperands; ++CurIndex)
      Signature += "Imp";

    // Save the conversion function, for use by the matcher.
    II.ConversionFn = Signature;

    // Check if we have already generated this function.
    if (!GeneratedFns.insert(Signature).second)
      continue;

    // If not, emit it now.
    //
    // FIXME: There should be no need to pass the number of operands to fill;
    // this should always be implicit in the class.
    OS << "static bool " << Signature << "(MCInst &Inst, unsigned Opcode";
    for (unsigned i = 0, e = MIOperandList.size(); i != e; ++i)
      OS << ", " << Target.getName() << "Operand Op" << i;
    OS << ") {\n";
    OS << "  Inst.setOpcode(Opcode);\n";
    CurIndex = 0;
    for (unsigned i = 0, e = MIOperandList.size(); i != e; ++i) {
      InstructionInfo::Operand &Op = II.Operands[MIOperandList[i].second];

      // Add the implicit operands.
      for (; CurIndex != Op.AsClass.Operand->MIOperandNo; ++CurIndex)
        OS << "  Inst.addOperand(MCOperand::CreateReg(0));\n";

      OS << "  Op" << i << "." << Op.AsClass.RenderMethod 
         << "(Inst, " << Op.AsClass.Operand->MINumOperands << ");\n";
      CurIndex += Op.AsClass.Operand->MINumOperands;
    }
    
    // And add trailing implicit operands.
    for (; CurIndex != NumMIOperands; ++CurIndex)
      OS << "  Inst.addOperand(MCOperand::CreateReg(0));\n";

    OS << "  return false;\n";
    OS << "}\n\n";
  }
}

void AsmMatcherEmitter::run(raw_ostream &OS) {
  CodeGenTarget Target;
  const std::vector<CodeGenRegister> &Registers = Target.getRegisters();
  Record *AsmParser = Target.getAsmParser();
  std::string ClassName = AsmParser->getValueAsString("AsmParserClassName");

  std::string Namespace = Registers[0].TheDef->getValueAsString("Namespace");

  EmitSourceFileHeader("Assembly Matcher Source Fragment", OS);

  // Emit the function to match a register name to number.

  OS << "bool " << Target.getName() << ClassName
     << "::MatchRegisterName(const StringRef &Name, unsigned &RegNo) {\n";

  // FIXME: TableGen should have a fast string matcher generator.
  for (unsigned i = 0, e = Registers.size(); i != e; ++i) {
    const CodeGenRegister &Reg = Registers[i];
    if (Reg.TheDef->getValueAsString("AsmName").empty())
      continue;

    OS << "  if (Name == \"" 
       << Reg.TheDef->getValueAsString("AsmName") << "\")\n"
       << "    return RegNo=" << i + 1 << ", false;\n";
  }
  OS << "  return true;\n";
  OS << "}\n\n";

  std::vector<InstructionInfo*> Infos;
  BuildInstructionInfos(Target, Infos);

#undef DEBUG_TYPE
#define DEBUG_TYPE "instruction_info"
  DEBUG({
      for (std::vector<InstructionInfo*>::iterator it = Infos.begin(),
             ie = Infos.end(); it != ie; ++it)
        (*it)->dump();
    });
#undef DEBUG_TYPE
#define DEBUG_TYPE ""

  // FIXME: At this point we should be able to totally order Infos, if not then
  // we have an ambiguity which the .td file should be forced to resolve.

  // Generate the terminal actions to convert operands into an MCInst. We still
  // pass the operands in to these functions individually (as opposed to the
  // array) so that we do not need to worry about the operand order.
  ConstructConversionFunctions(Target, Infos, OS);

  // Build a very stupid version of the match function which just checks each
  // instruction in order.

  OS << "bool " << Target.getName() << ClassName
     << "::MatchInstruction(" 
     << "SmallVectorImpl<" << Target.getName() << "Operand> &Operands, "
     << "MCInst &Inst) {\n";

  for (std::vector<InstructionInfo*>::const_iterator it = Infos.begin(),
         ie = Infos.end(); it != ie; ++it) {
    InstructionInfo &II = **it;

    // The parser is expected to arrange things so that each "token" matches
    // exactly one target specific operand.
    OS << "  if (Operands.size() == " << II.Operands.size();
    for (unsigned i = 0, e = II.Operands.size(); i != e; ++i) {
      InstructionInfo::Operand &Op = II.Operands[i];
      
      OS << " &&\n";
      OS << "      ";

      if (Op.Kind == InstructionInfo::Operand::Token)
        OS << "Operands[" << i << "].isToken(\"" << II.Tokens[i] << "\")";
      else
        OS << "Operands[" << i << "]." 
           << Op.AsClass.PredicateMethod << "()";
    }
    OS << ")\n";
    OS << "    return " << II.ConversionFn << "(Inst, " 
       << Target.getName() << "::" << II.InstrName;
    for (unsigned i = 0, e = II.OrderedClassOperands.size(); i != e; ++i)
      OS << ", Operands[" << II.OrderedClassOperands[i] << "]";
    OS << ");\n\n";
  }

  OS << "  return true;\n";
  OS << "}\n\n";
}
