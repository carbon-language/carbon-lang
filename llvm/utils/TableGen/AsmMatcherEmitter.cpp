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
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include <list>
#include <map>
#include <set>
using namespace llvm;

static cl::opt<std::string>
MatchPrefix("match-prefix", cl::init(""),
            cl::desc("Only match instructions with the given prefix"));

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

  // Ignore "Int_*" and "*_Int" instructions, which are internal aliases.
  //
  // FIXME: This is a total hack.
  if (StringRef(Name).startswith("Int_") || StringRef(Name).endswith("_Int"))
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
                 << Tokens[i] << "'\n";
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
  enum ClassInfoKind {
    /// Invalid kind, for use as a sentinel value.
    Invalid = 0,

    /// The class for a particular token.
    Token,

    /// The (first) register class, subsequent register classes are
    /// RegisterClass0+1, and so on.
    RegisterClass0,

    /// The (first) user defined class, subsequent user defined classes are
    /// UserClass0+1, and so on.
    UserClass0 = 1<<16
  };

  /// Kind - The class kind, which is either a predefined kind, or (UserClass0 +
  /// N) for the Nth user defined class.
  unsigned Kind;

  /// SuperClasses - The super classes of this class. Note that for simplicities
  /// sake user operands only record their immediate super class, while register
  /// operands include all superclasses.
  std::vector<ClassInfo*> SuperClasses;

  /// Name - The full class name, suitable for use in an enum.
  std::string Name;

  /// ClassName - The unadorned generic name for this class (e.g., Token).
  std::string ClassName;

  /// ValueName - The name of the value this class represents; for a token this
  /// is the literal token string, for an operand it is the TableGen class (or
  /// empty if this is a derived class).
  std::string ValueName;

  /// PredicateMethod - The name of the operand method to test whether the
  /// operand matches this class; this is not valid for Token or register kinds.
  std::string PredicateMethod;

  /// RenderMethod - The name of the operand method to add this operand to an
  /// MCInst; this is not valid for Token or register kinds.
  std::string RenderMethod;

  /// For register classes, the records for all the registers in this class.
  std::set<Record*> Registers;

public:
  /// isRegisterClass() - Check if this is a register class.
  bool isRegisterClass() const {
    return Kind >= RegisterClass0 && Kind < UserClass0;
  }

  /// isUserClass() - Check if this is a user defined class.
  bool isUserClass() const {
    return Kind >= UserClass0;
  }

  /// isRelatedTo - Check whether this class is "related" to \arg RHS. Classes
  /// are related if they are in the same class hierarchy.
  bool isRelatedTo(const ClassInfo &RHS) const {
    // Tokens are only related to tokens.
    if (Kind == Token || RHS.Kind == Token)
      return Kind == Token && RHS.Kind == Token;

    // Registers classes are only related to registers classes, and only if
    // their intersection is non-empty.
    if (isRegisterClass() || RHS.isRegisterClass()) {
      if (!isRegisterClass() || !RHS.isRegisterClass())
        return false;

      std::set<Record*> Tmp;
      std::insert_iterator< std::set<Record*> > II(Tmp, Tmp.begin());
      std::set_intersection(Registers.begin(), Registers.end(), 
                            RHS.Registers.begin(), RHS.Registers.end(),
                            II);

      return !Tmp.empty();
    }

    // Otherwise we have two users operands; they are related if they are in the
    // same class hierarchy.
    //
    // FIXME: This is an oversimplification, they should only be related if they
    // intersect, however we don't have that information.
    assert(isUserClass() && RHS.isUserClass() && "Unexpected class!");
    const ClassInfo *Root = this;
    while (!Root->SuperClasses.empty())
      Root = Root->SuperClasses.front();

    const ClassInfo *RHSRoot = &RHS;
    while (!RHSRoot->SuperClasses.empty())
      RHSRoot = RHSRoot->SuperClasses.front();
    
    return Root == RHSRoot;
  }

  /// isSubsetOf - Test whether this class is a subset of \arg RHS; 
  bool isSubsetOf(const ClassInfo &RHS) const {
    // This is a subset of RHS if it is the same class...
    if (this == &RHS)
      return true;

    // ... or if any of its super classes are a subset of RHS.
    for (std::vector<ClassInfo*>::const_iterator it = SuperClasses.begin(),
           ie = SuperClasses.end(); it != ie; ++it)
      if ((*it)->isSubsetOf(RHS))
        return true;

    return false;
  }

  /// operator< - Compare two classes.
  bool operator<(const ClassInfo &RHS) const {
    // Unrelated classes can be ordered by kind.
    if (!isRelatedTo(RHS))
      return Kind < RHS.Kind;

    switch (Kind) {
    case Invalid:
      assert(0 && "Invalid kind!");
    case Token:
      // Tokens are comparable by value.
      //
      // FIXME: Compare by enum value.
      return ValueName < RHS.ValueName;

    default:
      // This class preceeds the RHS if it is a proper subset of the RHS.
      return this != &RHS && isSubsetOf(RHS);
    }
  }
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

  /// operator< - Compare two instructions.
  bool operator<(const InstructionInfo &RHS) const {
    if (Operands.size() != RHS.Operands.size())
      return Operands.size() < RHS.Operands.size();

    // Compare lexicographically by operand. The matcher validates that other
    // orderings wouldn't be ambiguous using \see CouldMatchAmiguouslyWith().
    for (unsigned i = 0, e = Operands.size(); i != e; ++i) {
      if (*Operands[i].Class < *RHS.Operands[i].Class)
        return true;
      if (*RHS.Operands[i].Class < *Operands[i].Class)
        return false;
    }

    return false;
  }

  /// CouldMatchAmiguouslyWith - Check whether this instruction could
  /// ambiguously match the same set of operands as \arg RHS (without being a
  /// strictly superior match).
  bool CouldMatchAmiguouslyWith(const InstructionInfo &RHS) {
    // The number of operands is unambiguous.
    if (Operands.size() != RHS.Operands.size())
      return false;

    // Tokens and operand kinds are unambiguous (assuming a correct target
    // specific parser).
    for (unsigned i = 0, e = Operands.size(); i != e; ++i)
      if (Operands[i].Class->Kind != RHS.Operands[i].Class->Kind ||
          Operands[i].Class->Kind == ClassInfo::Token)
        if (*Operands[i].Class < *RHS.Operands[i].Class ||
            *RHS.Operands[i].Class < *Operands[i].Class)
          return false;
    
    // Otherwise, this operand could commute if all operands are equivalent, or
    // there is a pair of operands that compare less than and a pair that
    // compare greater than.
    bool HasLT = false, HasGT = false;
    for (unsigned i = 0, e = Operands.size(); i != e; ++i) {
      if (*Operands[i].Class < *RHS.Operands[i].Class)
        HasLT = true;
      if (*RHS.Operands[i].Class < *Operands[i].Class)
        HasGT = true;
    }

    return !(HasLT ^ HasGT);
  }

public:
  void dump();
};

class AsmMatcherInfo {
public:
  /// The tablegen AsmParser record.
  Record *AsmParser;

  /// The AsmParser "CommentDelimiter" value.
  std::string CommentDelimiter;

  /// The AsmParser "RegisterPrefix" value.
  std::string RegisterPrefix;

  /// The classes which are needed for matching.
  std::vector<ClassInfo*> Classes;
  
  /// The information on the instruction to match.
  std::vector<InstructionInfo*> Instructions;

  /// Map of Register records to their class information.
  std::map<Record*, ClassInfo*> RegisterClasses;

private:
  /// Map of token to class information which has already been constructed.
  std::map<std::string, ClassInfo*> TokenClasses;

  /// Map of RegisterClass records to their class information.
  std::map<Record*, ClassInfo*> RegisterClassClasses;

  /// Map of AsmOperandClass records to their class information.
  std::map<Record*, ClassInfo*> AsmOperandClasses;

private:
  /// getTokenClass - Lookup or create the class for the given token.
  ClassInfo *getTokenClass(const StringRef &Token);

  /// getOperandClass - Lookup or create the class for the given operand.
  ClassInfo *getOperandClass(const StringRef &Token,
                             const CodeGenInstruction::OperandInfo &OI);

  /// BuildRegisterClasses - Build the ClassInfo* instances for register
  /// classes.
  void BuildRegisterClasses(CodeGenTarget &Target);

  /// BuildOperandClasses - Build the ClassInfo* instances for user defined
  /// operand classes.
  void BuildOperandClasses(CodeGenTarget &Target);

public:
  AsmMatcherInfo(Record *_AsmParser);

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
    errs() << "  op[" << i << "] = " << Op.Class->ClassName << " - ";
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
    Entry->ClassName = "Token";
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
  if (OI.Rec->isSubClassOf("RegisterClass")) {
    ClassInfo *CI = RegisterClassClasses[OI.Rec];

    if (!CI) {
      PrintError(OI.Rec->getLoc(), "register class has no class info!");
      throw std::string("ERROR: Missing register class!");
    }

    return CI;
  }

  assert(OI.Rec->isSubClassOf("Operand") && "Unexpected operand!");
  Record *MatchClass = OI.Rec->getValueAsDef("ParserMatchClass");
  ClassInfo *CI = AsmOperandClasses[MatchClass];

  if (!CI) {
    PrintError(OI.Rec->getLoc(), "operand has no match class!");
    throw std::string("ERROR: Missing match class!");
  }

  return CI;
}

void AsmMatcherInfo::BuildRegisterClasses(CodeGenTarget &Target) {
  std::vector<CodeGenRegisterClass> RegisterClasses;
  std::vector<CodeGenRegister> Registers;

  RegisterClasses = Target.getRegisterClasses();
  Registers = Target.getRegisters();

  // The register sets used for matching.
  std::set< std::set<Record*> > RegisterSets;

  // Gather the defined sets.  
  for (std::vector<CodeGenRegisterClass>::iterator it = RegisterClasses.begin(),
         ie = RegisterClasses.end(); it != ie; ++it)
    RegisterSets.insert(std::set<Record*>(it->Elements.begin(),
                                          it->Elements.end()));
  
  // Introduce derived sets where necessary (when a register does not determine
  // a unique register set class), and build the mapping of registers to the set
  // they should classify to.
  std::map<Record*, std::set<Record*> > RegisterMap;
  for (std::vector<CodeGenRegister>::iterator it = Registers.begin(),
         ie = Registers.end(); it != ie; ++it) {
    CodeGenRegister &CGR = *it;
    // Compute the intersection of all sets containing this register.
    std::set<Record*> ContainingSet;
    
    for (std::set< std::set<Record*> >::iterator it = RegisterSets.begin(),
           ie = RegisterSets.end(); it != ie; ++it) {
      if (!it->count(CGR.TheDef))
        continue;

      if (ContainingSet.empty()) {
        ContainingSet = *it;
      } else {
        std::set<Record*> Tmp;
        std::swap(Tmp, ContainingSet);
        std::insert_iterator< std::set<Record*> > II(ContainingSet,
                                                     ContainingSet.begin());
        std::set_intersection(Tmp.begin(), Tmp.end(), it->begin(), it->end(),
                              II);
      }
    }

    if (!ContainingSet.empty()) {
      RegisterSets.insert(ContainingSet);
      RegisterMap.insert(std::make_pair(CGR.TheDef, ContainingSet));
    }
  }

  // Construct the register classes.
  std::map<std::set<Record*>, ClassInfo*> RegisterSetClasses;
  unsigned Index = 0;
  for (std::set< std::set<Record*> >::iterator it = RegisterSets.begin(),
         ie = RegisterSets.end(); it != ie; ++it, ++Index) {
    ClassInfo *CI = new ClassInfo();
    CI->Kind = ClassInfo::RegisterClass0 + Index;
    CI->ClassName = "Reg" + utostr(Index);
    CI->Name = "MCK_Reg" + utostr(Index);
    CI->ValueName = "";
    CI->PredicateMethod = ""; // unused
    CI->RenderMethod = "addRegOperands";
    CI->Registers = *it;
    Classes.push_back(CI);
    RegisterSetClasses.insert(std::make_pair(*it, CI));
  }

  // Find the superclasses; we could compute only the subgroup lattice edges,
  // but there isn't really a point.
  for (std::set< std::set<Record*> >::iterator it = RegisterSets.begin(),
         ie = RegisterSets.end(); it != ie; ++it) {
    ClassInfo *CI = RegisterSetClasses[*it];
    for (std::set< std::set<Record*> >::iterator it2 = RegisterSets.begin(),
           ie2 = RegisterSets.end(); it2 != ie2; ++it2)
      if (*it != *it2 && 
          std::includes(it2->begin(), it2->end(), it->begin(), it->end()))
        CI->SuperClasses.push_back(RegisterSetClasses[*it2]);
  }

  // Name the register classes which correspond to a user defined RegisterClass.
  for (std::vector<CodeGenRegisterClass>::iterator it = RegisterClasses.begin(),
         ie = RegisterClasses.end(); it != ie; ++it) {
    ClassInfo *CI = RegisterSetClasses[std::set<Record*>(it->Elements.begin(),
                                                         it->Elements.end())];
    if (CI->ValueName.empty()) {
      CI->ClassName = it->getName();
      CI->Name = "MCK_" + it->getName();
      CI->ValueName = it->getName();
    } else
      CI->ValueName = CI->ValueName + "," + it->getName();

    RegisterClassClasses.insert(std::make_pair(it->TheDef, CI));
  }

  // Populate the map for individual registers.
  for (std::map<Record*, std::set<Record*> >::iterator it = RegisterMap.begin(),
         ie = RegisterMap.end(); it != ie; ++it)
    this->RegisterClasses[it->first] = RegisterSetClasses[it->second];
}

void AsmMatcherInfo::BuildOperandClasses(CodeGenTarget &Target) {
  std::vector<Record*> AsmOperands;
  AsmOperands = Records.getAllDerivedDefinitions("AsmOperandClass");
  unsigned Index = 0;
  for (std::vector<Record*>::iterator it = AsmOperands.begin(), 
         ie = AsmOperands.end(); it != ie; ++it, ++Index) {
    ClassInfo *CI = new ClassInfo();
    CI->Kind = ClassInfo::UserClass0 + Index;

    Init *Super = (*it)->getValueInit("SuperClass");
    if (DefInit *DI = dynamic_cast<DefInit*>(Super)) {
      ClassInfo *SC = AsmOperandClasses[DI->getDef()];
      if (!SC)
        PrintError((*it)->getLoc(), "Invalid super class reference!");
      else
        CI->SuperClasses.push_back(SC);
    } else {
      assert(dynamic_cast<UnsetInit*>(Super) && "Unexpected SuperClass field!");
    }
    CI->ClassName = (*it)->getValueAsString("Name");
    CI->Name = "MCK_" + CI->ClassName;
    CI->ValueName = (*it)->getName();

    // Get or construct the predicate method name.
    Init *PMName = (*it)->getValueInit("PredicateMethod");
    if (StringInit *SI = dynamic_cast<StringInit*>(PMName)) {
      CI->PredicateMethod = SI->getValue();
    } else {
      assert(dynamic_cast<UnsetInit*>(PMName) && 
             "Unexpected PredicateMethod field!");
      CI->PredicateMethod = "is" + CI->ClassName;
    }

    // Get or construct the render method name.
    Init *RMName = (*it)->getValueInit("RenderMethod");
    if (StringInit *SI = dynamic_cast<StringInit*>(RMName)) {
      CI->RenderMethod = SI->getValue();
    } else {
      assert(dynamic_cast<UnsetInit*>(RMName) &&
             "Unexpected RenderMethod field!");
      CI->RenderMethod = "add" + CI->ClassName + "Operands";
    }

    AsmOperandClasses[*it] = CI;
    Classes.push_back(CI);
  }
}

AsmMatcherInfo::AsmMatcherInfo(Record *_AsmParser) 
  : AsmParser(_AsmParser),
    CommentDelimiter(AsmParser->getValueAsString("CommentDelimiter")),
    RegisterPrefix(AsmParser->getValueAsString("RegisterPrefix"))
{
}

void AsmMatcherInfo::BuildInfo(CodeGenTarget &Target) {
  // Build info for the register classes.
  BuildRegisterClasses(Target);

  // Build info for the user defined assembly operand classes.
  BuildOperandClasses(Target);

  // Build the instruction information.
  for (std::map<std::string, CodeGenInstruction>::const_iterator 
         it = Target.getInstructions().begin(), 
         ie = Target.getInstructions().end(); 
       it != ie; ++it) {
    const CodeGenInstruction &CGI = it->second;

    if (!StringRef(it->first).startswith(MatchPrefix))
      continue;

    OwningPtr<InstructionInfo> II(new InstructionInfo);
    
    II->InstrName = it->first;
    II->Instr = &it->second;
    II->AsmString = FlattenVariants(CGI.AsmString, 0);

    // Remove comments from the asm string.
    if (!CommentDelimiter.empty()) {
      size_t Idx = StringRef(II->AsmString).find(CommentDelimiter);
      if (Idx != StringRef::npos)
        II->AsmString = II->AsmString.substr(0, Idx);
    }

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

  // Reorder classes so that classes preceed super classes.
  std::sort(Classes.begin(), Classes.end(), less_ptr<ClassInfo>());
}

static void EmitConvertToMCInst(CodeGenTarget &Target,
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

      // Registers are always converted the same, don't duplicate the conversion
      // function based on them.
      //
      // FIXME: We could generalize this based on the render method, if it
      // mattered.
      if (Op.Class->isRegisterClass())
        Signature += "Reg";
      else
        Signature += Op.Class->ClassName;
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
    } else if (CI.isRegisterClass()) {
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
                                AsmMatcherInfo &Info,
                                raw_ostream &OS) {
  OS << "static MatchClassKind ClassifyOperand("
     << Target.getName() << "Operand &Operand) {\n";

  // Classify tokens.
  OS << "  if (Operand.isToken())\n";
  OS << "    return MatchTokenString(Operand.getToken());\n\n";

  // Classify registers.
  //
  // FIXME: Don't hardcode isReg, getReg.
  OS << "  if (Operand.isReg()) {\n";
  OS << "    switch (Operand.getReg()) {\n";
  OS << "    default: return InvalidMatchClass;\n";
  for (std::map<Record*, ClassInfo*>::iterator 
         it = Info.RegisterClasses.begin(), ie = Info.RegisterClasses.end();
       it != ie; ++it)
    OS << "    case " << Target.getName() << "::" 
       << it->first->getName() << ": return " << it->second->Name << ";\n";
  OS << "    }\n";
  OS << "  }\n\n";

  // Classify user defined operands.
  for (std::vector<ClassInfo*>::iterator it = Info.Classes.begin(), 
         ie = Info.Classes.end(); it != ie; ++it) {
    ClassInfo &CI = **it;

    if (!CI.isUserClass())
      continue;

    OS << "  // '" << CI.ClassName << "' class";
    if (!CI.SuperClasses.empty()) {
      OS << ", subclass of ";
      for (unsigned i = 0, e = CI.SuperClasses.size(); i != e; ++i) {
        if (i) OS << ", ";
        OS << "'" << CI.SuperClasses[i]->ClassName << "'";
        assert(CI < *CI.SuperClasses[i] && "Invalid class relation!");
      }
    }
    OS << "\n";

    OS << "  if (Operand." << CI.PredicateMethod << "()) {\n";
      
    // Validate subclass relationships.
    if (!CI.SuperClasses.empty()) {
      for (unsigned i = 0, e = CI.SuperClasses.size(); i != e; ++i)
        OS << "    assert(Operand." << CI.SuperClasses[i]->PredicateMethod
           << "() && \"Invalid class relationship!\");\n";
    }

    OS << "    return " << CI.Name << ";\n";
    OS << "  }\n\n";
  }
  OS << "  return InvalidMatchClass;\n";
  OS << "}\n\n";
}

/// EmitIsSubclass - Emit the subclass predicate function.
static void EmitIsSubclass(CodeGenTarget &Target,
                           std::vector<ClassInfo*> &Infos,
                           raw_ostream &OS) {
  OS << "/// IsSubclass - Compute whether \\arg A is a subclass of \\arg B.\n";
  OS << "static bool IsSubclass(MatchClassKind A, MatchClassKind B) {\n";
  OS << "  if (A == B)\n";
  OS << "    return true;\n\n";

  OS << "  switch (A) {\n";
  OS << "  default:\n";
  OS << "    return false;\n";
  for (std::vector<ClassInfo*>::iterator it = Infos.begin(), 
         ie = Infos.end(); it != ie; ++it) {
    ClassInfo &A = **it;

    if (A.Kind != ClassInfo::Token) {
      std::vector<StringRef> SuperClasses;
      for (std::vector<ClassInfo*>::iterator it = Infos.begin(), 
             ie = Infos.end(); it != ie; ++it) {
        ClassInfo &B = **it;

        if (&A != &B && A.isSubsetOf(B))
          SuperClasses.push_back(B.Name);
      }

      if (SuperClasses.empty())
        continue;

      OS << "\n  case " << A.Name << ":\n";

      if (SuperClasses.size() == 1) {
        OS << "    return B == " << SuperClasses.back() << ";\n";
        continue;
      }

      OS << "    switch (B) {\n";
      OS << "    default: return false;\n";
      for (unsigned i = 0, e = SuperClasses.size(); i != e; ++i)
        OS << "    case " << SuperClasses[i] << ": return true;\n";
      OS << "    }\n";
    }
  }
  OS << "  }\n";
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
///
/// \return - True if control can leave the emitted code fragment.
static bool EmitStringMatcherForChar(const std::string &StrVariableName,
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
    return false;
  }
  
  // Bucket the matches by the character we are comparing.
  std::map<char, std::vector<const StringPair*> > MatchesByLetter;
  
  for (unsigned i = 0, e = Matches.size(); i != e; ++i)
    MatchesByLetter[Matches[i]->first[CharNo]].push_back(Matches[i]);
  

  // If we have exactly one bucket to match, see how many characters are common
  // across the whole set and match all of them at once.
  if (MatchesByLetter.size() == 1) {
    unsigned FirstNonCommonLetter = FindFirstNonCommonLetter(Matches);
    unsigned NumChars = FirstNonCommonLetter-CharNo;
    
    // Emit code to break out if the prefix doesn't match.
    if (NumChars == 1) {
      // Do the comparison with if (Str[1] != 'f')
      // FIXME: Need to escape general characters.
      OS << Indent << "if (" << StrVariableName << "[" << CharNo << "] != '"
         << Matches[0]->first[CharNo] << "')\n";
      OS << Indent << "  break;\n";
    } else {
      // Do the comparison with if (Str.substr(1,3) != "foo").    
      // FIXME: Need to escape general strings.
      OS << Indent << "if (" << StrVariableName << ".substr(" << CharNo << ","
         << NumChars << ") != \"";
      OS << Matches[0]->first.substr(CharNo, NumChars) << "\")\n";
      OS << Indent << "  break;\n";
    }
    
    return EmitStringMatcherForChar(StrVariableName, Matches, 
                                    FirstNonCommonLetter, IndentCount, OS);
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
    if (EmitStringMatcherForChar(StrVariableName, LI->second, CharNo+1,
                                 IndentCount+1, OS))
      OS << Indent << "  break;\n";
  }
  
  OS << Indent << "}\n";
  return true;
}


/// EmitStringMatcher - Given a list of strings and code to execute when they
/// match, output a simple switch tree to classify the input string.
/// 
/// If a match is found, the code in Vals[i].second is executed; control must
/// not exit this code fragment.  If nothing matches, execution falls through.
///
/// \param StrVariableName - The name of the variable to test.
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
    if (EmitStringMatcherForChar(StrVariableName, LI->second, 0, 0, OS))
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
  AsmMatcherInfo Info(AsmParser);
  Info.BuildInfo(Target);

  // Sort the instruction table using the partial order on classes.
  std::sort(Info.Instructions.begin(), Info.Instructions.end(),
            less_ptr<InstructionInfo>());
  
  DEBUG_WITH_TYPE("instruction_info", {
      for (std::vector<InstructionInfo*>::iterator 
             it = Info.Instructions.begin(), ie = Info.Instructions.end(); 
           it != ie; ++it)
        (*it)->dump();
    });

  // Check for ambiguous instructions.
  unsigned NumAmbiguous = 0;
  for (unsigned i = 0, e = Info.Instructions.size(); i != e; ++i) {
    for (unsigned j = i + 1; j != e; ++j) {
      InstructionInfo &A = *Info.Instructions[i];
      InstructionInfo &B = *Info.Instructions[j];
    
      if (A.CouldMatchAmiguouslyWith(B)) {
        DEBUG_WITH_TYPE("ambiguous_instrs", {
            errs() << "warning: ambiguous instruction match:\n";
            A.dump();
            errs() << "\nis incomparable with:\n";
            B.dump();
            errs() << "\n\n";
          });
        ++NumAmbiguous;
      }
    }
  }
  if (NumAmbiguous)
    DEBUG_WITH_TYPE("ambiguous_instrs", {
        errs() << "warning: " << NumAmbiguous 
               << " ambiguous instructions!\n";
      });

  // Generate the unified function to convert operands into an MCInst.
  EmitConvertToMCInst(Target, Info.Instructions, OS);

  // Emit the enumeration for classes which participate in matching.
  EmitMatchClassEnumeration(Target, Info.Classes, OS);

  // Emit the routine to match token strings to their match class.
  EmitMatchTokenString(Target, Info.Classes, OS);

  // Emit the routine to classify an operand.
  EmitClassifyOperand(Target, Info, OS);

  // Emit the subclass predicate routine.
  EmitIsSubclass(Target, Info.Classes, OS);

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
    OS << "    if (!IsSubclass(Classes[" 
       << i << "], it->Classes[" << i << "]))\n";
    OS << "      continue;\n";
  }
  OS << "\n";
  OS << "    return ConvertToMCInst(it->ConvertFn, Inst, "
     << "it->Opcode, Operands);\n";
  OS << "  }\n\n";

  OS << "  return true;\n";
  OS << "}\n\n";
}
