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
#include <list>
#include <map>
#include <set>
using namespace llvm;

namespace {
static cl::opt<std::string>
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

/// ClassInfo - Helper class for storing the information about a particular
/// class of operands which can be matched.
struct ClassInfo {
  enum {
    Token, ///< The class for a particular token.
    Register, ///< A register class.
    User ///< A user defined class.
  } Kind;

  /// Name - The class name, suitable for use as an enum.
  std::string Name;

  /// ValueName - The name of the value this class represents; for a token this
  /// is the literal token string, for an operand it is the TableGen class (or
  /// empty if this is a derived class).
  std::string ValueName;

  /// PredicateMethod - The name of the operand method to test whether the
  /// operand matches this class; this is not valid for Token kinds.
  std::string PredicateMethod;

  /// RenderMethod - The name of the operand method to add this operand to an
  /// MCInst; this is not valid for Token kinds.
  std::string RenderMethod;
};

/// InstructionInfo - Helper class for storing the necessary information for an
/// instruction which is capable of being matched.
struct InstructionInfo {
  struct Operand {
    /// The unique class instance this operand should match.
    ClassInfo *Class;

    /// The original operand this corresponds to, if any.
    const CodeGenInstruction::OperandInfo *OperandInfo;
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

  /// ConversionFnKind - The enum value which is passed to the generated
  /// ConvertToMCInst to convert parsed operands into an MCInst for this
  /// function.
  std::string ConversionFnKind;

public:
  void dump();
};

class AsmMatcherInfo {
public:
  /// The classes which are needed for matching.
  std::vector<ClassInfo*> Classes;
  
  /// The information on the instruction to match.
  std::vector<InstructionInfo*> Instructions;

private:
  /// Map of token to class information which has already been constructed.
  std::map<std::string, ClassInfo*> TokenClasses;

  /// Map of operand name to class information which has already been
  /// constructed.
  std::map<std::string, ClassInfo*> OperandClasses;

private:
  /// getTokenClass - Lookup or create the class for the given token.
  ClassInfo *getTokenClass(const StringRef &Token);

  /// getOperandClass - Lookup or create the class for the given operand.
  ClassInfo *getOperandClass(const StringRef &Token,
                             const CodeGenInstruction::OperandInfo &OI);

public:
  /// BuildInfo - Construct the various tables used during matching.
  void BuildInfo(CodeGenTarget &Target);
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
    if (Op.Class->Kind == ClassInfo::Token) {
      errs() << '\"' << Tokens[i] << "\"\n";
      continue;
    }

    const CodeGenInstruction::OperandInfo &OI = *Op.OperandInfo;
    errs() << OI.Name << " " << OI.Rec->getName()
           << " (" << OI.MIOperandNo << ", " << OI.MINumOperands << ")\n";
  }
}

static std::string getEnumNameForToken(const StringRef &Str) {
  std::string Res;
  
  for (StringRef::iterator it = Str.begin(), ie = Str.end(); it != ie; ++it) {
    switch (*it) {
    case '*': Res += "_STAR_"; break;
    case '%': Res += "_PCT_"; break;
    case ':': Res += "_COLON_"; break;

    default:
      if (isalnum(*it))  {
        Res += *it;
      } else {
        Res += "_" + utostr((unsigned) *it) + "_";
      }
    }
  }

  return Res;
}

ClassInfo *AsmMatcherInfo::getTokenClass(const StringRef &Token) {
  ClassInfo *&Entry = TokenClasses[Token];
  
  if (!Entry) {
    Entry = new ClassInfo();
    Entry->Kind = ClassInfo::Token;
    Entry->Name = "MCK_" + getEnumNameForToken(Token);
    Entry->ValueName = Token;
    Entry->PredicateMethod = "<invalid>";
    Entry->RenderMethod = "<invalid>";
    Classes.push_back(Entry);
  }

  return Entry;
}

ClassInfo *
AsmMatcherInfo::getOperandClass(const StringRef &Token,
                                const CodeGenInstruction::OperandInfo &OI) {
  std::string ClassName;
  if (OI.Rec->isSubClassOf("RegisterClass")) {
    ClassName = "Reg";
  } else if (OI.Rec->isSubClassOf("Operand")) {
    // FIXME: This should not be hard coded.
    const RecordVal *RV = OI.Rec->getValue("Type");
    
    // FIXME: Yet another total hack.
    if (RV->getValue()->getAsString() == "iPTR" ||
        OI.Rec->getName() == "i8mem_NOREX" ||
        OI.Rec->getName() == "lea32mem" ||
        OI.Rec->getName() == "lea64mem" ||
        OI.Rec->getName() == "i128mem" ||
        OI.Rec->getName() == "sdmem" ||
        OI.Rec->getName() == "ssmem" ||
        OI.Rec->getName() == "lea64_32mem") {
      ClassName = "Mem";
    } else {
      ClassName = "Imm";
    }
  }

  ClassInfo *&Entry = OperandClasses[ClassName];
  
  if (!Entry) {
    Entry = new ClassInfo();
    // FIXME: Hack.
    if (ClassName == "Reg") {
      Entry->Kind = ClassInfo::Register;
    } else {
      Entry->Kind = ClassInfo::User;
    }
    Entry->Name = "MCK_" + ClassName;
    Entry->ValueName = OI.Rec->getName();
    Entry->PredicateMethod = "is" + ClassName;
    Entry->RenderMethod = "add" + ClassName + "Operands";
    Classes.push_back(Entry);
  }
  
  return Entry;
}

void AsmMatcherInfo::BuildInfo(CodeGenTarget &Target) {
  for (std::map<std::string, CodeGenInstruction>::const_iterator 
         it = Target.getInstructions().begin(), 
         ie = Target.getInstructions().end(); 
       it != ie; ++it) {
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
        Op.Class = getTokenClass(Token);
        Op.OperandInfo = 0;
        II->Operands.push_back(Op);
        continue;
      }

      // Otherwise this is an operand reference.
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
      InstructionInfo::Operand Op;
      Op.Class = getOperandClass(Token, OI);
      Op.OperandInfo = &OI;
      II->Operands.push_back(Op);
    }

    // If we broke out, ignore the instruction.
    if (II->Operands.size() != II->Tokens.size())
      continue;

    Instructions.push_back(II.take());
  }
}

static void ConstructConversionFunctions(CodeGenTarget &Target,
                                         std::vector<InstructionInfo*> &Infos,
                                         raw_ostream &OS) {
  // Write the convert function to a separate stream, so we can drop it after
  // the enum.
  std::string ConvertFnBody;
  raw_string_ostream CvtOS(ConvertFnBody);

  // Function we have already generated.
  std::set<std::string> GeneratedFns;

  // Start the unified conversion function.

  CvtOS << "static bool ConvertToMCInst(ConversionKind Kind, MCInst &Inst, "
        << "unsigned Opcode,\n"
        << "                            SmallVectorImpl<"
        << Target.getName() << "Operand> &Operands) {\n";
  CvtOS << "  Inst.setOpcode(Opcode);\n";
  CvtOS << "  switch (Kind) {\n";
  CvtOS << "  default:\n";

  // Start the enum, which we will generate inline.

  OS << "// Unified function for converting operants to MCInst instances.\n\n";
  OS << "enum ConversionKind {\n";
  
  for (std::vector<InstructionInfo*>::const_iterator it = Infos.begin(),
         ie = Infos.end(); it != ie; ++it) {
    InstructionInfo &II = **it;

    // Order the (class) operands by the order to convert them into an MCInst.
    SmallVector<std::pair<unsigned, unsigned>, 4> MIOperandList;
    for (unsigned i = 0, e = II.Operands.size(); i != e; ++i) {
      InstructionInfo::Operand &Op = II.Operands[i];
      if (Op.OperandInfo)
        MIOperandList.push_back(std::make_pair(Op.OperandInfo->MIOperandNo, i));
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
      assert(CurIndex <= Op.OperandInfo->MIOperandNo &&
             "Duplicate match for instruction operand!");
      
      Signature += "_";

      // Skip operands which weren't matched by anything, this occurs when the
      // .td file encodes "implicit" operands as explicit ones.
      //
      // FIXME: This should be removed from the MCInst structure.
      for (; CurIndex != Op.OperandInfo->MIOperandNo; ++CurIndex)
        Signature += "Imp";

      Signature += Op.Class->Name;
      Signature += utostr(Op.OperandInfo->MINumOperands);
      Signature += "_" + utostr(MIOperandList[i].second);

      CurIndex += Op.OperandInfo->MINumOperands;
    }

    // Add any trailing implicit operands.
    for (; CurIndex != NumMIOperands; ++CurIndex)
      Signature += "Imp";

    II.ConversionFnKind = Signature;

    // Check if we have already generated this signature.
    if (!GeneratedFns.insert(Signature).second)
      continue;

    // If not, emit it now.

    // Add to the enum list.
    OS << "  " << Signature << ",\n";

    // And to the convert function.
    CvtOS << "  case " << Signature << ":\n";
    CurIndex = 0;
    for (unsigned i = 0, e = MIOperandList.size(); i != e; ++i) {
      InstructionInfo::Operand &Op = II.Operands[MIOperandList[i].second];

      // Add the implicit operands.
      for (; CurIndex != Op.OperandInfo->MIOperandNo; ++CurIndex)
        CvtOS << "    Inst.addOperand(MCOperand::CreateReg(0));\n";

      CvtOS << "    Operands[" << MIOperandList[i].second 
         << "]." << Op.Class->RenderMethod 
         << "(Inst, " << Op.OperandInfo->MINumOperands << ");\n";
      CurIndex += Op.OperandInfo->MINumOperands;
    }
    
    // And add trailing implicit operands.
    for (; CurIndex != NumMIOperands; ++CurIndex)
      CvtOS << "    Inst.addOperand(MCOperand::CreateReg(0));\n";
    CvtOS << "    break;\n";
  }

  // Finish the convert function.

  CvtOS << "  }\n";
  CvtOS << "  return false;\n";
  CvtOS << "}\n\n";

  // Finish the enum, and drop the convert function after it.

  OS << "  NumConversionVariants\n";
  OS << "};\n\n";
  
  OS << CvtOS.str();
}

/// EmitMatchClassEnumeration - Emit the enumeration for match class kinds.
static void EmitMatchClassEnumeration(CodeGenTarget &Target,
                                      std::vector<ClassInfo*> &Infos,
                                      raw_ostream &OS) {
  OS << "namespace {\n\n";

  OS << "/// MatchClassKind - The kinds of classes which participate in\n"
     << "/// instruction matching.\n";
  OS << "enum MatchClassKind {\n";
  OS << "  InvalidMatchClass = 0,\n";
  for (std::vector<ClassInfo*>::iterator it = Infos.begin(), 
         ie = Infos.end(); it != ie; ++it) {
    ClassInfo &CI = **it;
    OS << "  " << CI.Name << ", // ";
    if (CI.Kind == ClassInfo::Token) {
      OS << "'" << CI.ValueName << "'\n";
    } else if (CI.Kind == ClassInfo::Register) {
      if (!CI.ValueName.empty())
        OS << "register class '" << CI.ValueName << "'\n";
      else
        OS << "derived register class\n";
    } else {
      OS << "user defined class '" << CI.ValueName << "'\n";
    }
  }
  OS << "  NumMatchClassKinds\n";
  OS << "};\n\n";

  OS << "}\n\n";
}

/// EmitClassifyOperand - Emit the function to classify an operand.
static void EmitClassifyOperand(CodeGenTarget &Target,
                                std::vector<ClassInfo*> &Infos,
                                raw_ostream &OS) {
  OS << "static MatchClassKind ClassifyOperand("
     << Target.getName() << "Operand &Operand) {\n";
  OS << "  if (Operand.isToken())\n";
  OS << "    return MatchTokenString(Operand.getToken());\n\n";
  for (std::vector<ClassInfo*>::iterator it = Infos.begin(), 
         ie = Infos.end(); it != ie; ++it) {
    ClassInfo &CI = **it;

    if (CI.Kind != ClassInfo::Token) {
      OS << "  if (Operand." << CI.PredicateMethod << "())\n";
      OS << "    return " << CI.Name << ";\n\n";
    }
  }
  OS << "  return InvalidMatchClass;\n";
  OS << "}\n\n";
}

typedef std::pair<std::string, std::string> StringPair;

/// FindFirstNonCommonLetter - Find the first character in the keys of the
/// string pairs that is not shared across the whole set of strings.  All
/// strings are assumed to have the same length.
static unsigned 
FindFirstNonCommonLetter(const std::vector<const StringPair*> &Matches) {
  assert(!Matches.empty());
  for (unsigned i = 0, e = Matches[0]->first.size(); i != e; ++i) {
    // Check to see if letter i is the same across the set.
    char Letter = Matches[0]->first[i];
    
    for (unsigned str = 0, e = Matches.size(); str != e; ++str)
      if (Matches[str]->first[i] != Letter)
        return i;
  }
  
  return Matches[0]->first.size();
}

/// EmitStringMatcherForChar - Given a set of strings that are known to be the
/// same length and whose characters leading up to CharNo are the same, emit
/// code to verify that CharNo and later are the same.
static void EmitStringMatcherForChar(const std::string &StrVariableName,
                                  const std::vector<const StringPair*> &Matches,
                                     unsigned CharNo, unsigned IndentCount,
                                     raw_ostream &OS) {
  assert(!Matches.empty() && "Must have at least one string to match!");
  std::string Indent(IndentCount*2+4, ' ');

  // If we have verified that the entire string matches, we're done: output the
  // matching code.
  if (CharNo == Matches[0]->first.size()) {
    assert(Matches.size() == 1 && "Had duplicate keys to match on");
    
    // FIXME: If Matches[0].first has embeded \n, this will be bad.
    OS << Indent << Matches[0]->second << "\t // \"" << Matches[0]->first
       << "\"\n";
    return;
  }
  
  // Bucket the matches by the character we are comparing.
  std::map<char, std::vector<const StringPair*> > MatchesByLetter;
  
  for (unsigned i = 0, e = Matches.size(); i != e; ++i)
    MatchesByLetter[Matches[i]->first[CharNo]].push_back(Matches[i]);
  

  // If we have exactly one bucket to match, see how many characters are common
  // across the whole set and match all of them at once.
  // length, just verify the rest of it with one if.
  if (MatchesByLetter.size() == 1) {
    unsigned FirstNonCommonLetter = FindFirstNonCommonLetter(Matches);
    unsigned NumChars = FirstNonCommonLetter-CharNo;
    
    if (NumChars == 1) {
      // Do the comparison with if (Str[1] == 'f')
      // FIXME: Need to escape general characters.
      OS << Indent << "if (" << StrVariableName << "[" << CharNo << "] == '"
         << Matches[0]->first[CharNo] << "') {\n";
    } else {
      // Do the comparison with if (Str.substr(1,3) == "foo").
      OS << Indent << "if (" << StrVariableName << ".substr(" << CharNo << ","
         << NumChars << ") == \"";
    
      // FIXME: Need to escape general strings.
      OS << Matches[0]->first.substr(CharNo, NumChars) << "\") {\n";
    }
    
    EmitStringMatcherForChar(StrVariableName, Matches, FirstNonCommonLetter,
                             IndentCount+1, OS);
    OS << Indent << "}\n";
    return;
  }
  
  // Otherwise, we have multiple possible things, emit a switch on the
  // character.
  OS << Indent << "switch (" << StrVariableName << "[" << CharNo << "]) {\n";
  OS << Indent << "default: break;\n";
  
  for (std::map<char, std::vector<const StringPair*> >::iterator LI = 
       MatchesByLetter.begin(), E = MatchesByLetter.end(); LI != E; ++LI) {
    // TODO: escape hard stuff (like \n) if we ever care about it.
    OS << Indent << "case '" << LI->first << "':\t // "
       << LI->second.size() << " strings to match.\n";
    EmitStringMatcherForChar(StrVariableName, LI->second, CharNo+1,
                             IndentCount+1, OS);
    OS << Indent << "  break;\n";
  }
  
  OS << Indent << "}\n";
  
}


/// EmitStringMatcher - Given a list of strings and code to execute when they
/// match, output a simple switch tree to classify the input string.  If a
/// match is found, the code in Vals[i].second is executed.  This code should do
/// a return to avoid falling through.  If nothing matches, execution falls
/// through.  StrVariableName is the name of teh variable to test.
static void EmitStringMatcher(const std::string &StrVariableName,
                              const std::vector<StringPair> &Matches,
                              raw_ostream &OS) {
  // First level categorization: group strings by length.
  std::map<unsigned, std::vector<const StringPair*> > MatchesByLength;
  
  for (unsigned i = 0, e = Matches.size(); i != e; ++i)
    MatchesByLength[Matches[i].first.size()].push_back(&Matches[i]);
  
  // Output a switch statement on length and categorize the elements within each
  // bin.
  OS << "  switch (" << StrVariableName << ".size()) {\n";
  OS << "  default: break;\n";
  
  
  for (std::map<unsigned, std::vector<const StringPair*> >::iterator LI =
       MatchesByLength.begin(), E = MatchesByLength.end(); LI != E; ++LI) {
    OS << "  case " << LI->first << ":\t // " << LI->second.size()
       << " strings to match.\n";
    EmitStringMatcherForChar(StrVariableName, LI->second, 0, 0, OS);
    OS << "    break;\n";
  }
  
  
  OS << "  }\n";
}


/// EmitMatchTokenString - Emit the function to match a token string to the
/// appropriate match class value.
static void EmitMatchTokenString(CodeGenTarget &Target,
                                 std::vector<ClassInfo*> &Infos,
                                 raw_ostream &OS) {
  // Construct the match list.
  std::vector<StringPair> Matches;
  for (std::vector<ClassInfo*>::iterator it = Infos.begin(), 
         ie = Infos.end(); it != ie; ++it) {
    ClassInfo &CI = **it;

    if (CI.Kind == ClassInfo::Token)
      Matches.push_back(StringPair(CI.ValueName, "return " + CI.Name + ";"));
  }

  OS << "static MatchClassKind MatchTokenString(const StringRef &Name) {\n";

  EmitStringMatcher("Name", Matches, OS);

  OS << "  return InvalidMatchClass;\n";
  OS << "}\n\n";
}

/// EmitMatchRegisterName - Emit the function to match a string to the target
/// specific register enum.
static void EmitMatchRegisterName(CodeGenTarget &Target, Record *AsmParser,
                                  raw_ostream &OS) {
  // Construct the match list.
  std::vector<StringPair> Matches;
  for (unsigned i = 0, e = Target.getRegisters().size(); i != e; ++i) {
    const CodeGenRegister &Reg = Target.getRegisters()[i];
    if (Reg.TheDef->getValueAsString("AsmName").empty())
      continue;

    Matches.push_back(StringPair(Reg.TheDef->getValueAsString("AsmName"),
                                 "return " + utostr(i + 1) + ";"));
  }
  
  OS << "unsigned " << Target.getName() 
     << AsmParser->getValueAsString("AsmParserClassName")
     << "::MatchRegisterName(const StringRef &Name) {\n";

  EmitStringMatcher("Name", Matches, OS);
  
  OS << "  return 0;\n";
  OS << "}\n\n";
}

void AsmMatcherEmitter::run(raw_ostream &OS) {
  CodeGenTarget Target;
  Record *AsmParser = Target.getAsmParser();
  std::string ClassName = AsmParser->getValueAsString("AsmParserClassName");

  EmitSourceFileHeader("Assembly Matcher Source Fragment", OS);

  // Emit the function to match a register name to number.
  EmitMatchRegisterName(Target, AsmParser, OS);

  // Compute the information on the instructions to match.
  AsmMatcherInfo Info;
  Info.BuildInfo(Target);

  DEBUG_WITH_TYPE("instruction_info", {
      for (std::vector<InstructionInfo*>::iterator 
             it = Info.Instructions.begin(), ie = Info.Instructions.end(); 
           it != ie; ++it)
        (*it)->dump();
    });

  // FIXME: At this point we should be able to totally order Infos, if not then
  // we have an ambiguity which the .td file should be forced to resolve.

  // Generate the terminal actions to convert operands into an MCInst.
  ConstructConversionFunctions(Target, Info.Instructions, OS);

  // Emit the enumeration for classes which participate in matching.
  EmitMatchClassEnumeration(Target, Info.Classes, OS);

  // Emit the routine to match token strings to their match class.
  EmitMatchTokenString(Target, Info.Classes, OS);

  // Emit the routine to classify an operand.
  EmitClassifyOperand(Target, Info.Classes, OS);

  // Finally, build the match function.

  size_t MaxNumOperands = 0;
  for (std::vector<InstructionInfo*>::const_iterator it =
         Info.Instructions.begin(), ie = Info.Instructions.end();
       it != ie; ++it)
    MaxNumOperands = std::max(MaxNumOperands, (*it)->Operands.size());
  
  OS << "bool " << Target.getName() << ClassName
     << "::MatchInstruction(" 
     << "SmallVectorImpl<" << Target.getName() << "Operand> &Operands, "
     << "MCInst &Inst) {\n";

  // Emit the static match table; unused classes get initalized to 0 which is
  // guaranteed to be InvalidMatchClass.
  //
  // FIXME: We can reduce the size of this table very easily. First, we change
  // it so that store the kinds in separate bit-fields for each index, which
  // only needs to be the max width used for classes at that index (we also need
  // to reject based on this during classification). If we then make sure to
  // order the match kinds appropriately (putting mnemonics last), then we
  // should only end up using a few bits for each class, especially the ones
  // following the mnemonic.
  OS << "  static const struct MatchEntry {\n";
  OS << "    unsigned Opcode;\n";
  OS << "    ConversionKind ConvertFn;\n";
  OS << "    MatchClassKind Classes[" << MaxNumOperands << "];\n";
  OS << "  } MatchTable[" << Info.Instructions.size() << "] = {\n";

  for (std::vector<InstructionInfo*>::const_iterator it =
         Info.Instructions.begin(), ie = Info.Instructions.end();
       it != ie; ++it) {
    InstructionInfo &II = **it;

    OS << "    { " << Target.getName() << "::" << II.InstrName
       << ", " << II.ConversionFnKind << ", { ";
    for (unsigned i = 0, e = II.Operands.size(); i != e; ++i) {
      InstructionInfo::Operand &Op = II.Operands[i];
      
      if (i) OS << ", ";
      OS << Op.Class->Name;
    }
    OS << " } },\n";
  }

  OS << "  };\n\n";

  // Emit code to compute the class list for this operand vector.
  OS << "  // Eliminate obvious mismatches.\n";
  OS << "  if (Operands.size() > " << MaxNumOperands << ")\n";
  OS << "    return true;\n\n";

  OS << "  // Compute the class list for this operand vector.\n";
  OS << "  MatchClassKind Classes[" << MaxNumOperands << "];\n";
  OS << "  for (unsigned i = 0, e = Operands.size(); i != e; ++i) {\n";
  OS << "    Classes[i] = ClassifyOperand(Operands[i]);\n\n";

  OS << "    // Check for invalid operands before matching.\n";
  OS << "    if (Classes[i] == InvalidMatchClass)\n";
  OS << "      return true;\n";
  OS << "  }\n\n";

  OS << "  // Mark unused classes.\n";
  OS << "  for (unsigned i = Operands.size(), e = " << MaxNumOperands << "; "
     << "i != e; ++i)\n";
  OS << "    Classes[i] = InvalidMatchClass;\n\n";

  // Emit code to search the table.
  OS << "  // Search the table.\n";
  OS << "  for (const MatchEntry *it = MatchTable, "
     << "*ie = MatchTable + " << Info.Instructions.size()
     << "; it != ie; ++it) {\n";
  for (unsigned i = 0; i != MaxNumOperands; ++i) {
    OS << "    if (Classes[" << i << "] != it->Classes[" << i << "])\n";
    OS << "      continue;\n";
  }
  OS << "\n";
  OS << "    return ConvertToMCInst(it->ConvertFn, Inst, "
     << "it->Opcode, Operands);\n";
  OS << "  }\n\n";

  OS << "  return true;\n";
  OS << "}\n\n";
}
