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
#include "StringMatcher.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/SmallPtrSet.h"
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


namespace {
  class AsmMatcherInfo;
struct SubtargetFeatureInfo;

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
    if (this == &RHS)
      return false;

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
      if (isSubsetOf(RHS))
        return true;
      if (RHS.isSubsetOf(*this))
        return false;

      // Otherwise, order by name to ensure we have a total ordering.
      return ValueName < RHS.ValueName;
    }
  }
};

/// MatchableInfo - Helper class for storing the necessary information for an
/// instruction or alias which is capable of being matched.
struct MatchableInfo {
  struct Operand {
    /// Token - This is the token that the operand came from.
    StringRef Token;
    
    /// The unique class instance this operand should match.
    ClassInfo *Class;

    /// The original operand this corresponds to.  This is unset for singleton
    /// registers and tokens, because they don't have a list in the ins/outs
    /// list.  If an operand is tied ($a=$b), this refers to source operand: $b.
    const CGIOperandList::OperandInfo *OperandInfo;
    
    explicit Operand(StringRef T) : Token(T), Class(0), OperandInfo(0) {}
  };

  /// InstrName - The target name for this instruction.
  std::string InstrName;

  /// TheDef - This is the definition of the instruction or InstAlias that this
  /// matchable came from.
  Record *const TheDef;
  
  /// OperandList - This is the operand list that came from the (ins) and (outs)
  /// list of the alias or instruction.
  const CGIOperandList &OperandList;

  /// AsmString - The assembly string for this instruction (with variants
  /// removed), e.g. "movsx $src, $dst".
  std::string AsmString;

  /// Mnemonic - This is the first token of the matched instruction, its
  /// mnemonic.
  StringRef Mnemonic;
  
  /// AsmOperands - The textual operands that this instruction matches,
  /// annotated with a class and where in the OperandList they were defined.
  /// This directly corresponds to the tokenized AsmString after the mnemonic is
  /// removed.
  SmallVector<Operand, 4> AsmOperands;

  /// Predicates - The required subtarget features to match this instruction.
  SmallVector<SubtargetFeatureInfo*, 4> RequiredFeatures;

  /// ConversionFnKind - The enum value which is passed to the generated
  /// ConvertToMCInst to convert parsed operands into an MCInst for this
  /// function.
  std::string ConversionFnKind;
  
  MatchableInfo(const CodeGenInstruction &CGI)
    : TheDef(CGI.TheDef), OperandList(CGI.Operands), AsmString(CGI.AsmString) {
    InstrName = TheDef->getName();
  }

  MatchableInfo(const CodeGenInstAlias *Alias)
    : TheDef(Alias->TheDef), OperandList(Alias->Operands),
      AsmString(Alias->AsmString) {
        
    // FIXME: Huge hack.
    DefInit *DI = dynamic_cast<DefInit*>(Alias->Result->getOperator());
    assert(DI);
        
    InstrName = DI->getDef()->getName();
  }
  
  void Initialize(const AsmMatcherInfo &Info,
                  SmallPtrSet<Record*, 16> &SingletonRegisters);
  
  /// Validate - Return true if this matchable is a valid thing to match against
  /// and perform a bunch of validity checking.
  bool Validate(StringRef CommentDelimiter, bool Hack) const;
  
  /// getSingletonRegisterForAsmOperand - If the specified token is a singleton
  /// register, return the Record for it, otherwise return null.
  Record *getSingletonRegisterForAsmOperand(unsigned i,
                                            const AsmMatcherInfo &Info) const;  

  /// operator< - Compare two matchables.
  bool operator<(const MatchableInfo &RHS) const {
    // The primary comparator is the instruction mnemonic.
    if (Mnemonic != RHS.Mnemonic)
      return Mnemonic < RHS.Mnemonic;

    if (AsmOperands.size() != RHS.AsmOperands.size())
      return AsmOperands.size() < RHS.AsmOperands.size();

    // Compare lexicographically by operand. The matcher validates that other
    // orderings wouldn't be ambiguous using \see CouldMatchAmiguouslyWith().
    for (unsigned i = 0, e = AsmOperands.size(); i != e; ++i) {
      if (*AsmOperands[i].Class < *RHS.AsmOperands[i].Class)
        return true;
      if (*RHS.AsmOperands[i].Class < *AsmOperands[i].Class)
        return false;
    }

    return false;
  }

  /// CouldMatchAmiguouslyWith - Check whether this matchable could
  /// ambiguously match the same set of operands as \arg RHS (without being a
  /// strictly superior match).
  bool CouldMatchAmiguouslyWith(const MatchableInfo &RHS) {
    // The primary comparator is the instruction mnemonic.
    if (Mnemonic != RHS.Mnemonic)
      return false;
    
    // The number of operands is unambiguous.
    if (AsmOperands.size() != RHS.AsmOperands.size())
      return false;

    // Otherwise, make sure the ordering of the two instructions is unambiguous
    // by checking that either (a) a token or operand kind discriminates them,
    // or (b) the ordering among equivalent kinds is consistent.

    // Tokens and operand kinds are unambiguous (assuming a correct target
    // specific parser).
    for (unsigned i = 0, e = AsmOperands.size(); i != e; ++i)
      if (AsmOperands[i].Class->Kind != RHS.AsmOperands[i].Class->Kind ||
          AsmOperands[i].Class->Kind == ClassInfo::Token)
        if (*AsmOperands[i].Class < *RHS.AsmOperands[i].Class ||
            *RHS.AsmOperands[i].Class < *AsmOperands[i].Class)
          return false;

    // Otherwise, this operand could commute if all operands are equivalent, or
    // there is a pair of operands that compare less than and a pair that
    // compare greater than.
    bool HasLT = false, HasGT = false;
    for (unsigned i = 0, e = AsmOperands.size(); i != e; ++i) {
      if (*AsmOperands[i].Class < *RHS.AsmOperands[i].Class)
        HasLT = true;
      if (*RHS.AsmOperands[i].Class < *AsmOperands[i].Class)
        HasGT = true;
    }

    return !(HasLT ^ HasGT);
  }

  void dump();
  
private:
  void TokenizeAsmString(const AsmMatcherInfo &Info);
};

/// SubtargetFeatureInfo - Helper class for storing information on a subtarget
/// feature which participates in instruction matching.
struct SubtargetFeatureInfo {
  /// \brief The predicate record for this feature.
  Record *TheDef;

  /// \brief An unique index assigned to represent this feature.
  unsigned Index;

  SubtargetFeatureInfo(Record *D, unsigned Idx) : TheDef(D), Index(Idx) {}
  
  /// \brief The name of the enumerated constant identifying this feature.
  std::string getEnumName() const {
    return "Feature_" + TheDef->getName();
  }
};

class AsmMatcherInfo {
public:
  /// The tablegen AsmParser record.
  Record *AsmParser;

  /// Target - The target information.
  CodeGenTarget &Target;

  /// The AsmParser "RegisterPrefix" value.
  std::string RegisterPrefix;

  /// The classes which are needed for matching.
  std::vector<ClassInfo*> Classes;

  /// The information on the matchables to match.
  std::vector<MatchableInfo*> Matchables;

  /// Map of Register records to their class information.
  std::map<Record*, ClassInfo*> RegisterClasses;

  /// Map of Predicate records to their subtarget information.
  std::map<Record*, SubtargetFeatureInfo*> SubtargetFeatures;
  
private:
  /// Map of token to class information which has already been constructed.
  std::map<std::string, ClassInfo*> TokenClasses;

  /// Map of RegisterClass records to their class information.
  std::map<Record*, ClassInfo*> RegisterClassClasses;

  /// Map of AsmOperandClass records to their class information.
  std::map<Record*, ClassInfo*> AsmOperandClasses;

private:
  /// getTokenClass - Lookup or create the class for the given token.
  ClassInfo *getTokenClass(StringRef Token);

  /// getOperandClass - Lookup or create the class for the given operand.
  ClassInfo *getOperandClass(StringRef Token,
                             const CGIOperandList::OperandInfo &OI);

  /// BuildRegisterClasses - Build the ClassInfo* instances for register
  /// classes.
  void BuildRegisterClasses(SmallPtrSet<Record*, 16> &SingletonRegisters);

  /// BuildOperandClasses - Build the ClassInfo* instances for user defined
  /// operand classes.
  void BuildOperandClasses();

public:
  AsmMatcherInfo(Record *AsmParser, CodeGenTarget &Target);

  /// BuildInfo - Construct the various tables used during matching.
  void BuildInfo();
  
  /// getSubtargetFeature - Lookup or create the subtarget feature info for the
  /// given operand.
  SubtargetFeatureInfo *getSubtargetFeature(Record *Def) const {
    assert(Def->isSubClassOf("Predicate") && "Invalid predicate type!");
    std::map<Record*, SubtargetFeatureInfo*>::const_iterator I =
      SubtargetFeatures.find(Def);
    return I == SubtargetFeatures.end() ? 0 : I->second;
  }
};

}

void MatchableInfo::dump() {
  errs() << InstrName << " -- " << "flattened:\"" << AsmString << "\"\n";

  for (unsigned i = 0, e = AsmOperands.size(); i != e; ++i) {
    Operand &Op = AsmOperands[i];
    errs() << "  op[" << i << "] = " << Op.Class->ClassName << " - ";
    if (Op.Class->Kind == ClassInfo::Token) {
      errs() << '\"' << Op.Token << "\"\n";
      continue;
    }

    if (!Op.OperandInfo) {
      errs() << "(singleton register)\n";
      continue;
    }

    const CGIOperandList::OperandInfo &OI = *Op.OperandInfo;
    errs() << OI.Name << " " << OI.Rec->getName()
           << " (" << OI.MIOperandNo << ", " << OI.MINumOperands << ")\n";
  }
}

void MatchableInfo::Initialize(const AsmMatcherInfo &Info,
                               SmallPtrSet<Record*, 16> &SingletonRegisters) {
  // TODO: Eventually support asmparser for Variant != 0.
  AsmString = CodeGenInstruction::FlattenAsmStringVariants(AsmString, 0);
  
  TokenizeAsmString(Info);
  
  // Compute the require features.
  std::vector<Record*> Predicates =TheDef->getValueAsListOfDefs("Predicates");
  for (unsigned i = 0, e = Predicates.size(); i != e; ++i)
    if (SubtargetFeatureInfo *Feature =
        Info.getSubtargetFeature(Predicates[i]))
      RequiredFeatures.push_back(Feature);
  
  // Collect singleton registers, if used.
  for (unsigned i = 0, e = AsmOperands.size(); i != e; ++i) {
    if (Record *Reg = getSingletonRegisterForAsmOperand(i, Info))
      SingletonRegisters.insert(Reg);
  }
}

/// TokenizeAsmString - Tokenize a simplified assembly string.
void MatchableInfo::TokenizeAsmString(const AsmMatcherInfo &Info) {
  StringRef String = AsmString;
  unsigned Prev = 0;
  bool InTok = true;
  for (unsigned i = 0, e = String.size(); i != e; ++i) {
    switch (String[i]) {
    case '[':
    case ']':
    case '*':
    case '!':
    case ' ':
    case '\t':
    case ',':
      if (InTok) {
        AsmOperands.push_back(Operand(String.slice(Prev, i)));
        InTok = false;
      }
      if (!isspace(String[i]) && String[i] != ',')
        AsmOperands.push_back(Operand(String.substr(i, 1)));
      Prev = i + 1;
      break;

    case '\\':
      if (InTok) {
        AsmOperands.push_back(Operand(String.slice(Prev, i)));
        InTok = false;
      }
      ++i;
      assert(i != String.size() && "Invalid quoted character");
      AsmOperands.push_back(Operand(String.substr(i, 1)));
      Prev = i + 1;
      break;

    case '$': {
      // If this isn't "${", treat like a normal token.
      if (i + 1 == String.size() || String[i + 1] != '{') {
        if (InTok) {
          AsmOperands.push_back(Operand(String.slice(Prev, i)));
          InTok = false;
        }
        Prev = i;
        break;
      }

      if (InTok) {
        AsmOperands.push_back(Operand(String.slice(Prev, i)));
        InTok = false;
      }

      StringRef::iterator End = std::find(String.begin() + i, String.end(),'}');
      assert(End != String.end() && "Missing brace in operand reference!");
      size_t EndPos = End - String.begin();
      AsmOperands.push_back(Operand(String.slice(i, EndPos+1)));
      Prev = EndPos + 1;
      i = EndPos;
      break;
    }

    case '.':
      if (InTok)
        AsmOperands.push_back(Operand(String.slice(Prev, i)));
      Prev = i;
      InTok = true;
      break;

    default:
      InTok = true;
    }
  }
  if (InTok && Prev != String.size())
    AsmOperands.push_back(Operand(String.substr(Prev)));
  
  // The first token of the instruction is the mnemonic, which must be a
  // simple string, not a $foo variable or a singleton register.
  assert(!AsmOperands.empty() && "Instruction has no tokens?");
  Mnemonic = AsmOperands[0].Token;
  if (Mnemonic[0] == '$' || getSingletonRegisterForAsmOperand(0, Info))
    throw TGError(TheDef->getLoc(),
                  "Invalid instruction mnemonic '" + Mnemonic.str() + "'!");
  
  // Remove the first operand, it is tracked in the mnemonic field.
  AsmOperands.erase(AsmOperands.begin());
}



bool MatchableInfo::Validate(StringRef CommentDelimiter, bool Hack) const {
  // Reject matchables with no .s string.
  if (AsmString.empty())
    throw TGError(TheDef->getLoc(), "instruction with empty asm string");
  
  // Reject any matchables with a newline in them, they should be marked
  // isCodeGenOnly if they are pseudo instructions.
  if (AsmString.find('\n') != std::string::npos)
    throw TGError(TheDef->getLoc(),
                  "multiline instruction is not valid for the asmparser, "
                  "mark it isCodeGenOnly");
  
  // Remove comments from the asm string.  We know that the asmstring only
  // has one line.
  if (!CommentDelimiter.empty() &&
      StringRef(AsmString).find(CommentDelimiter) != StringRef::npos)
    throw TGError(TheDef->getLoc(),
                  "asmstring for instruction has comment character in it, "
                  "mark it isCodeGenOnly");
  
  // Reject matchables with operand modifiers, these aren't something we can
  /// handle, the target should be refactored to use operands instead of
  /// modifiers.
  //
  // Also, check for instructions which reference the operand multiple times;
  // this implies a constraint we would not honor.
  std::set<std::string> OperandNames;
  for (unsigned i = 0, e = AsmOperands.size(); i != e; ++i) {
    StringRef Tok = AsmOperands[i].Token;
    if (Tok[0] == '$' && Tok.find(':') != StringRef::npos)
      throw TGError(TheDef->getLoc(),
                    "matchable with operand modifier '" + Tok.str() +
                    "' not supported by asm matcher.  Mark isCodeGenOnly!");
    
    // Verify that any operand is only mentioned once.
    if (Tok[0] == '$' && !OperandNames.insert(Tok).second) {
      if (!Hack)
        throw TGError(TheDef->getLoc(),
                      "ERROR: matchable with tied operand '" + Tok.str() +
                      "' can never be matched!");
      // FIXME: Should reject these.  The ARM backend hits this with $lane in a
      // bunch of instructions.  It is unclear what the right answer is.
      DEBUG({
        errs() << "warning: '" << InstrName << "': "
               << "ignoring instruction with tied operand '"
               << Tok.str() << "'\n";
      });
      return false;
    }
  }
  
  return true;
}


/// getSingletonRegisterForAsmOperand - If the specified token is a singleton
/// register, return the register name, otherwise return a null StringRef.
Record *MatchableInfo::
getSingletonRegisterForAsmOperand(unsigned i, const AsmMatcherInfo &Info) const{
  StringRef Tok = AsmOperands[i].Token;
  if (!Tok.startswith(Info.RegisterPrefix))
    return 0;
  
  StringRef RegName = Tok.substr(Info.RegisterPrefix.size());
  if (const CodeGenRegister *Reg = Info.Target.getRegisterByName(RegName))
    return Reg->TheDef;
  
  // If there is no register prefix (i.e. "%" in "%eax"), then this may
  // be some random non-register token, just ignore it.
  if (Info.RegisterPrefix.empty())
    return 0;
    
  // Otherwise, we have something invalid prefixed with the register prefix,
  // such as %foo.
  std::string Err = "unable to find register for '" + RegName.str() +
  "' (which matches register prefix)";
  throw TGError(TheDef->getLoc(), Err);
}


static std::string getEnumNameForToken(StringRef Str) {
  std::string Res;

  for (StringRef::iterator it = Str.begin(), ie = Str.end(); it != ie; ++it) {
    switch (*it) {
    case '*': Res += "_STAR_"; break;
    case '%': Res += "_PCT_"; break;
    case ':': Res += "_COLON_"; break;
    default:
      if (isalnum(*it))
        Res += *it;
      else
        Res += "_" + utostr((unsigned) *it) + "_";
    }
  }

  return Res;
}

ClassInfo *AsmMatcherInfo::getTokenClass(StringRef Token) {
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
AsmMatcherInfo::getOperandClass(StringRef Token,
                                const CGIOperandList::OperandInfo &OI) {
  if (OI.Rec->isSubClassOf("RegisterClass")) {
    if (ClassInfo *CI = RegisterClassClasses[OI.Rec])
      return CI;
    throw TGError(OI.Rec->getLoc(), "register class has no class info!");
  }

  assert(OI.Rec->isSubClassOf("Operand") && "Unexpected operand!");
  Record *MatchClass = OI.Rec->getValueAsDef("ParserMatchClass");
  if (ClassInfo *CI = AsmOperandClasses[MatchClass])
    return CI;

  throw TGError(OI.Rec->getLoc(), "operand has no match class!");
}

void AsmMatcherInfo::
BuildRegisterClasses(SmallPtrSet<Record*, 16> &SingletonRegisters) {
  const std::vector<CodeGenRegister> &Registers = Target.getRegisters();
  const std::vector<CodeGenRegisterClass> &RegClassList =
    Target.getRegisterClasses();

  // The register sets used for matching.
  std::set< std::set<Record*> > RegisterSets;

  // Gather the defined sets.
  for (std::vector<CodeGenRegisterClass>::const_iterator it =
       RegClassList.begin(), ie = RegClassList.end(); it != ie; ++it)
    RegisterSets.insert(std::set<Record*>(it->Elements.begin(),
                                          it->Elements.end()));

  // Add any required singleton sets.
  for (SmallPtrSet<Record*, 16>::iterator it = SingletonRegisters.begin(),
       ie = SingletonRegisters.end(); it != ie; ++it) {
    Record *Rec = *it;
    RegisterSets.insert(std::set<Record*>(&Rec, &Rec + 1));
  }

  // Introduce derived sets where necessary (when a register does not determine
  // a unique register set class), and build the mapping of registers to the set
  // they should classify to.
  std::map<Record*, std::set<Record*> > RegisterMap;
  for (std::vector<CodeGenRegister>::const_iterator it = Registers.begin(),
         ie = Registers.end(); it != ie; ++it) {
    const CodeGenRegister &CGR = *it;
    // Compute the intersection of all sets containing this register.
    std::set<Record*> ContainingSet;

    for (std::set< std::set<Record*> >::iterator it = RegisterSets.begin(),
           ie = RegisterSets.end(); it != ie; ++it) {
      if (!it->count(CGR.TheDef))
        continue;

      if (ContainingSet.empty()) {
        ContainingSet = *it;
        continue;
      }
      
      std::set<Record*> Tmp;
      std::swap(Tmp, ContainingSet);
      std::insert_iterator< std::set<Record*> > II(ContainingSet,
                                                   ContainingSet.begin());
      std::set_intersection(Tmp.begin(), Tmp.end(), it->begin(), it->end(), II);
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
  for (std::vector<CodeGenRegisterClass>::const_iterator
       it = RegClassList.begin(), ie = RegClassList.end(); it != ie; ++it) {
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
    RegisterClasses[it->first] = RegisterSetClasses[it->second];

  // Name the register classes which correspond to singleton registers.
  for (SmallPtrSet<Record*, 16>::iterator it = SingletonRegisters.begin(),
         ie = SingletonRegisters.end(); it != ie; ++it) {
    Record *Rec = *it;
    ClassInfo *CI = RegisterClasses[Rec];
    assert(CI && "Missing singleton register class info!");

    if (CI->ValueName.empty()) {
      CI->ClassName = Rec->getName();
      CI->Name = "MCK_" + Rec->getName();
      CI->ValueName = Rec->getName();
    } else
      CI->ValueName = CI->ValueName + "," + Rec->getName();
  }
}

void AsmMatcherInfo::BuildOperandClasses() {
  std::vector<Record*> AsmOperands =
    Records.getAllDerivedDefinitions("AsmOperandClass");

  // Pre-populate AsmOperandClasses map.
  for (std::vector<Record*>::iterator it = AsmOperands.begin(),
         ie = AsmOperands.end(); it != ie; ++it)
    AsmOperandClasses[*it] = new ClassInfo();

  unsigned Index = 0;
  for (std::vector<Record*>::iterator it = AsmOperands.begin(),
         ie = AsmOperands.end(); it != ie; ++it, ++Index) {
    ClassInfo *CI = AsmOperandClasses[*it];
    CI->Kind = ClassInfo::UserClass0 + Index;

    ListInit *Supers = (*it)->getValueAsListInit("SuperClasses");
    for (unsigned i = 0, e = Supers->getSize(); i != e; ++i) {
      DefInit *DI = dynamic_cast<DefInit*>(Supers->getElement(i));
      if (!DI) {
        PrintError((*it)->getLoc(), "Invalid super class reference!");
        continue;
      }

      ClassInfo *SC = AsmOperandClasses[DI->getDef()];
      if (!SC)
        PrintError((*it)->getLoc(), "Invalid super class reference!");
      else
        CI->SuperClasses.push_back(SC);
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

AsmMatcherInfo::AsmMatcherInfo(Record *asmParser, CodeGenTarget &target)
  : AsmParser(asmParser), Target(target),
    RegisterPrefix(AsmParser->getValueAsString("RegisterPrefix")) {
}


void AsmMatcherInfo::BuildInfo() {
  // Build information about all of the AssemblerPredicates.
  std::vector<Record*> AllPredicates =
    Records.getAllDerivedDefinitions("Predicate");
  for (unsigned i = 0, e = AllPredicates.size(); i != e; ++i) {
    Record *Pred = AllPredicates[i];
    // Ignore predicates that are not intended for the assembler.
    if (!Pred->getValueAsBit("AssemblerMatcherPredicate"))
      continue;
    
    if (Pred->getName().empty())
      throw TGError(Pred->getLoc(), "Predicate has no name!");
    
    unsigned FeatureNo = SubtargetFeatures.size();
    SubtargetFeatures[Pred] = new SubtargetFeatureInfo(Pred, FeatureNo);
    assert(FeatureNo < 32 && "Too many subtarget features!");
  }

  StringRef CommentDelimiter = AsmParser->getValueAsString("CommentDelimiter");
  
  // Parse the instructions; we need to do this first so that we can gather the
  // singleton register classes.
  SmallPtrSet<Record*, 16> SingletonRegisters;
  for (CodeGenTarget::inst_iterator I = Target.inst_begin(),
       E = Target.inst_end(); I != E; ++I) {
    const CodeGenInstruction &CGI = **I;

    // If the tblgen -match-prefix option is specified (for tblgen hackers),
    // filter the set of instructions we consider.
    if (!StringRef(CGI.TheDef->getName()).startswith(MatchPrefix))
      continue;

    // Ignore "codegen only" instructions.
    if (CGI.TheDef->getValueAsBit("isCodeGenOnly"))
      continue;
    
    OwningPtr<MatchableInfo> II(new MatchableInfo(CGI));

    II->Initialize(*this, SingletonRegisters);
    
    // Ignore instructions which shouldn't be matched and diagnose invalid
    // instruction definitions with an error.
    if (!II->Validate(CommentDelimiter, true))
      continue;
    
    // Ignore "Int_*" and "*_Int" instructions, which are internal aliases.
    //
    // FIXME: This is a total hack.
    if (StringRef(II->InstrName).startswith("Int_") ||
        StringRef(II->InstrName).endswith("_Int"))
      continue;
    
     Matchables.push_back(II.take());
  }
  
  // Parse all of the InstAlias definitions and stick them in the list of
  // matchables.
  std::vector<Record*> AllInstAliases =
    Records.getAllDerivedDefinitions("InstAlias");
  for (unsigned i = 0, e = AllInstAliases.size(); i != e; ++i) {
    CodeGenInstAlias *Alias = new CodeGenInstAlias(AllInstAliases[i]);

    OwningPtr<MatchableInfo> II(new MatchableInfo(Alias));
    
    II->Initialize(*this, SingletonRegisters);
    
    // Validate the alias definitions.
    II->Validate(CommentDelimiter, false);
    
    Matchables.push_back(II.take());
  }

  // Build info for the register classes.
  BuildRegisterClasses(SingletonRegisters);

  // Build info for the user defined assembly operand classes.
  BuildOperandClasses();

  // Build the information about matchables.
  for (std::vector<MatchableInfo*>::iterator it = Matchables.begin(),
         ie = Matchables.end(); it != ie; ++it) {
    MatchableInfo *II = *it;

    // Parse the tokens after the mnemonic.
    for (unsigned i = 0, e = II->AsmOperands.size(); i != e; ++i) {
      MatchableInfo::Operand &Op = II->AsmOperands[i];
      StringRef Token = Op.Token;

      // Check for singleton registers.
      if (Record *RegRecord = II->getSingletonRegisterForAsmOperand(i, *this)) {
        Op.Class = RegisterClasses[RegRecord];
        assert(Op.Class && Op.Class->Registers.size() == 1 &&
               "Unexpected class for singleton register");
        continue;
      }

      // Check for simple tokens.
      if (Token[0] != '$') {
        Op.Class = getTokenClass(Token);
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
      if (!II->OperandList.hasOperandNamed(OperandName, Idx))
        throw TGError(II->TheDef->getLoc(), "error: unable to find operand: '" +
                      OperandName.str() + "'");

      // FIXME: This is annoying, the named operand may be tied (e.g.,
      // XCHG8rm). What we want is the untied operand, which we now have to
      // grovel for. Only worry about this for single entry operands, we have to
      // clean this up anyway.
      const CGIOperandList::OperandInfo *OI = &II->OperandList[Idx];
      if (OI->Constraints[0].isTied()) {
        unsigned TiedOp = OI->Constraints[0].getTiedOperand();

        // The tied operand index is an MIOperand index, find the operand that
        // contains it.
        for (unsigned i = 0, e = II->OperandList.size(); i != e; ++i) {
          if (II->OperandList[i].MIOperandNo == TiedOp) {
            OI = &II->OperandList[i];
            break;
          }
        }

        assert(OI && "Unable to find tied operand target!");
      }

      Op.Class = getOperandClass(Token, *OI);
      Op.OperandInfo = OI;
    }
  }

  // Reorder classes so that classes preceed super classes.
  std::sort(Classes.begin(), Classes.end(), less_ptr<ClassInfo>());
}

static std::pair<unsigned, unsigned> *
GetTiedOperandAtIndex(SmallVectorImpl<std::pair<unsigned, unsigned> > &List,
                      unsigned Index) {
  for (unsigned i = 0, e = List.size(); i != e; ++i)
    if (Index == List[i].first)
      return &List[i];

  return 0;
}

static void EmitConvertToMCInst(CodeGenTarget &Target,
                                std::vector<MatchableInfo*> &Infos,
                                raw_ostream &OS) {
  // Write the convert function to a separate stream, so we can drop it after
  // the enum.
  std::string ConvertFnBody;
  raw_string_ostream CvtOS(ConvertFnBody);

  // Function we have already generated.
  std::set<std::string> GeneratedFns;

  // Start the unified conversion function.

  CvtOS << "static void ConvertToMCInst(ConversionKind Kind, MCInst &Inst, "
        << "unsigned Opcode,\n"
        << "                      const SmallVectorImpl<MCParsedAsmOperand*"
        << "> &Operands) {\n";
  CvtOS << "  Inst.setOpcode(Opcode);\n";
  CvtOS << "  switch (Kind) {\n";
  CvtOS << "  default:\n";

  // Start the enum, which we will generate inline.

  OS << "// Unified function for converting operants to MCInst instances.\n\n";
  OS << "enum ConversionKind {\n";

  // TargetOperandClass - This is the target's operand class, like X86Operand.
  std::string TargetOperandClass = Target.getName() + "Operand";

  for (std::vector<MatchableInfo*>::const_iterator it = Infos.begin(),
         ie = Infos.end(); it != ie; ++it) {
    MatchableInfo &II = **it;

    // Order the (class) operands by the order to convert them into an MCInst.
    SmallVector<std::pair<unsigned, unsigned>, 4> MIOperandList;
    for (unsigned i = 0, e = II.AsmOperands.size(); i != e; ++i) {
      MatchableInfo::Operand &Op = II.AsmOperands[i];
      if (Op.OperandInfo)
        MIOperandList.push_back(std::make_pair(Op.OperandInfo->MIOperandNo, i));
    }

    // Find any tied operands.
    SmallVector<std::pair<unsigned, unsigned>, 4> TiedOperands;
    for (unsigned i = 0, e = II.OperandList.size(); i != e; ++i) {
      const CGIOperandList::OperandInfo &OpInfo = II.OperandList[i];
      for (unsigned j = 0, e = OpInfo.Constraints.size(); j != e; ++j) {
        const CGIOperandList::ConstraintInfo &CI = OpInfo.Constraints[j];
        if (!CI.isTied()) continue;
        TiedOperands.push_back(std::make_pair(OpInfo.MIOperandNo,
                                              CI.getTiedOperand()));
      }
    }

    array_pod_sort(MIOperandList.begin(), MIOperandList.end());

    // Compute the total number of operands.
    unsigned NumMIOperands = 0;
    for (unsigned i = 0, e = II.OperandList.size(); i != e; ++i) {
      const CGIOperandList::OperandInfo &OI = II.OperandList[i];
      NumMIOperands = std::max(NumMIOperands, OI.MIOperandNo+OI.MINumOperands);
    }

    // Build the conversion function signature.
    std::string Signature = "Convert";
    unsigned CurIndex = 0;
    for (unsigned i = 0, e = MIOperandList.size(); i != e; ++i) {
      MatchableInfo::Operand &Op = II.AsmOperands[MIOperandList[i].second];
      assert(CurIndex <= Op.OperandInfo->MIOperandNo &&
             "Duplicate match for instruction operand!");

      // Skip operands which weren't matched by anything, this occurs when the
      // .td file encodes "implicit" operands as explicit ones.
      //
      // FIXME: This should be removed from the MCInst structure.
      for (; CurIndex != Op.OperandInfo->MIOperandNo; ++CurIndex) {
        std::pair<unsigned, unsigned> *Tie = GetTiedOperandAtIndex(TiedOperands,
                                                                   CurIndex);
        if (!Tie)
          Signature += "__Imp";
        else
          Signature += "__Tie" + utostr(Tie->second);
      }

      Signature += "__";

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
    for (; CurIndex != NumMIOperands; ++CurIndex) {
      std::pair<unsigned, unsigned> *Tie = GetTiedOperandAtIndex(TiedOperands,
                                                                 CurIndex);
      if (!Tie)
        Signature += "__Imp";
      else
        Signature += "__Tie" + utostr(Tie->second);
    }

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
      MatchableInfo::Operand &Op = II.AsmOperands[MIOperandList[i].second];

      // Add the implicit operands.
      for (; CurIndex != Op.OperandInfo->MIOperandNo; ++CurIndex) {
        // See if this is a tied operand.
        std::pair<unsigned, unsigned> *Tie = GetTiedOperandAtIndex(TiedOperands,
                                                                   CurIndex);

        if (!Tie) {
          // If not, this is some implicit operand. Just assume it is a register
          // for now.
          CvtOS << "    Inst.addOperand(MCOperand::CreateReg(0));\n";
        } else {
          // Copy the tied operand.
          assert(Tie->first>Tie->second && "Tied operand preceeds its target!");
          CvtOS << "    Inst.addOperand(Inst.getOperand("
                << Tie->second << "));\n";
        }
      }

      CvtOS << "    ((" << TargetOperandClass << "*)Operands["
         << MIOperandList[i].second
         << "+1])->" << Op.Class->RenderMethod
         << "(Inst, " << Op.OperandInfo->MINumOperands << ");\n";
      CurIndex += Op.OperandInfo->MINumOperands;
    }

    // And add trailing implicit operands.
    for (; CurIndex != NumMIOperands; ++CurIndex) {
      std::pair<unsigned, unsigned> *Tie = GetTiedOperandAtIndex(TiedOperands,
                                                                 CurIndex);

      if (!Tie) {
        // If not, this is some implicit operand. Just assume it is a register
        // for now.
        CvtOS << "    Inst.addOperand(MCOperand::CreateReg(0));\n";
      } else {
        // Copy the tied operand.
        assert(Tie->first>Tie->second && "Tied operand preceeds its target!");
        CvtOS << "    Inst.addOperand(Inst.getOperand("
              << Tie->second << "));\n";
      }
    }

    CvtOS << "    return;\n";
  }

  // Finish the convert function.

  CvtOS << "  }\n";
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
static void EmitClassifyOperand(AsmMatcherInfo &Info,
                                raw_ostream &OS) {
  OS << "static MatchClassKind ClassifyOperand(MCParsedAsmOperand *GOp) {\n"
     << "  " << Info.Target.getName() << "Operand &Operand = *("
     << Info.Target.getName() << "Operand*)GOp;\n";

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
    OS << "    case " << Info.Target.getName() << "::"
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



/// EmitMatchTokenString - Emit the function to match a token string to the
/// appropriate match class value.
static void EmitMatchTokenString(CodeGenTarget &Target,
                                 std::vector<ClassInfo*> &Infos,
                                 raw_ostream &OS) {
  // Construct the match list.
  std::vector<StringMatcher::StringPair> Matches;
  for (std::vector<ClassInfo*>::iterator it = Infos.begin(),
         ie = Infos.end(); it != ie; ++it) {
    ClassInfo &CI = **it;

    if (CI.Kind == ClassInfo::Token)
      Matches.push_back(StringMatcher::StringPair(CI.ValueName,
                                                  "return " + CI.Name + ";"));
  }

  OS << "static MatchClassKind MatchTokenString(StringRef Name) {\n";

  StringMatcher("Name", Matches, OS).Emit();

  OS << "  return InvalidMatchClass;\n";
  OS << "}\n\n";
}

/// EmitMatchRegisterName - Emit the function to match a string to the target
/// specific register enum.
static void EmitMatchRegisterName(CodeGenTarget &Target, Record *AsmParser,
                                  raw_ostream &OS) {
  // Construct the match list.
  std::vector<StringMatcher::StringPair> Matches;
  for (unsigned i = 0, e = Target.getRegisters().size(); i != e; ++i) {
    const CodeGenRegister &Reg = Target.getRegisters()[i];
    if (Reg.TheDef->getValueAsString("AsmName").empty())
      continue;

    Matches.push_back(StringMatcher::StringPair(
                                        Reg.TheDef->getValueAsString("AsmName"),
                                        "return " + utostr(i + 1) + ";"));
  }

  OS << "static unsigned MatchRegisterName(StringRef Name) {\n";

  StringMatcher("Name", Matches, OS).Emit();

  OS << "  return 0;\n";
  OS << "}\n\n";
}

/// EmitSubtargetFeatureFlagEnumeration - Emit the subtarget feature flag
/// definitions.
static void EmitSubtargetFeatureFlagEnumeration(AsmMatcherInfo &Info,
                                                raw_ostream &OS) {
  OS << "// Flags for subtarget features that participate in "
     << "instruction matching.\n";
  OS << "enum SubtargetFeatureFlag {\n";
  for (std::map<Record*, SubtargetFeatureInfo*>::const_iterator
         it = Info.SubtargetFeatures.begin(),
         ie = Info.SubtargetFeatures.end(); it != ie; ++it) {
    SubtargetFeatureInfo &SFI = *it->second;
    OS << "  " << SFI.getEnumName() << " = (1 << " << SFI.Index << "),\n";
  }
  OS << "  Feature_None = 0\n";
  OS << "};\n\n";
}

/// EmitComputeAvailableFeatures - Emit the function to compute the list of
/// available features given a subtarget.
static void EmitComputeAvailableFeatures(AsmMatcherInfo &Info,
                                         raw_ostream &OS) {
  std::string ClassName =
    Info.AsmParser->getValueAsString("AsmParserClassName");

  OS << "unsigned " << Info.Target.getName() << ClassName << "::\n"
     << "ComputeAvailableFeatures(const " << Info.Target.getName()
     << "Subtarget *Subtarget) const {\n";
  OS << "  unsigned Features = 0;\n";
  for (std::map<Record*, SubtargetFeatureInfo*>::const_iterator
         it = Info.SubtargetFeatures.begin(),
         ie = Info.SubtargetFeatures.end(); it != ie; ++it) {
    SubtargetFeatureInfo &SFI = *it->second;
    OS << "  if (" << SFI.TheDef->getValueAsString("CondString")
       << ")\n";
    OS << "    Features |= " << SFI.getEnumName() << ";\n";
  }
  OS << "  return Features;\n";
  OS << "}\n\n";
}

static std::string GetAliasRequiredFeatures(Record *R,
                                            const AsmMatcherInfo &Info) {
  std::vector<Record*> ReqFeatures = R->getValueAsListOfDefs("Predicates");
  std::string Result;
  unsigned NumFeatures = 0;
  for (unsigned i = 0, e = ReqFeatures.size(); i != e; ++i) {
    SubtargetFeatureInfo *F = Info.getSubtargetFeature(ReqFeatures[i]);
    
    if (F == 0)
      throw TGError(R->getLoc(), "Predicate '" + ReqFeatures[i]->getName() +
                    "' is not marked as an AssemblerPredicate!");
    
    if (NumFeatures)
      Result += '|';
  
    Result += F->getEnumName();
    ++NumFeatures;
  }
  
  if (NumFeatures > 1)
    Result = '(' + Result + ')';
  return Result;
}

/// EmitMnemonicAliases - If the target has any MnemonicAlias<> definitions,
/// emit a function for them and return true, otherwise return false.
static bool EmitMnemonicAliases(raw_ostream &OS, const AsmMatcherInfo &Info) {
  std::vector<Record*> Aliases =
    Records.getAllDerivedDefinitions("MnemonicAlias");
  if (Aliases.empty()) return false;

  OS << "static void ApplyMnemonicAliases(StringRef &Mnemonic, "
        "unsigned Features) {\n";
  
  // Keep track of all the aliases from a mnemonic.  Use an std::map so that the
  // iteration order of the map is stable.
  std::map<std::string, std::vector<Record*> > AliasesFromMnemonic;
  
  for (unsigned i = 0, e = Aliases.size(); i != e; ++i) {
    Record *R = Aliases[i];
    AliasesFromMnemonic[R->getValueAsString("FromMnemonic")].push_back(R);
  }

  // Process each alias a "from" mnemonic at a time, building the code executed
  // by the string remapper.
  std::vector<StringMatcher::StringPair> Cases;
  for (std::map<std::string, std::vector<Record*> >::iterator
       I = AliasesFromMnemonic.begin(), E = AliasesFromMnemonic.end();
       I != E; ++I) {
    const std::vector<Record*> &ToVec = I->second;

    // Loop through each alias and emit code that handles each case.  If there
    // are two instructions without predicates, emit an error.  If there is one,
    // emit it last.
    std::string MatchCode;
    int AliasWithNoPredicate = -1;
    
    for (unsigned i = 0, e = ToVec.size(); i != e; ++i) {
      Record *R = ToVec[i];
      std::string FeatureMask = GetAliasRequiredFeatures(R, Info);
    
      // If this unconditionally matches, remember it for later and diagnose
      // duplicates.
      if (FeatureMask.empty()) {
        if (AliasWithNoPredicate != -1) {
          // We can't have two aliases from the same mnemonic with no predicate.
          PrintError(ToVec[AliasWithNoPredicate]->getLoc(),
                     "two MnemonicAliases with the same 'from' mnemonic!");
          throw TGError(R->getLoc(), "this is the other MnemonicAlias.");
        }
        
        AliasWithNoPredicate = i;
        continue;
      }
     
      if (!MatchCode.empty())
        MatchCode += "else ";
      MatchCode += "if ((Features & " + FeatureMask + ") == "+FeatureMask+")\n";
      MatchCode += "  Mnemonic = \"" +R->getValueAsString("ToMnemonic")+"\";\n";
    }
    
    if (AliasWithNoPredicate != -1) {
      Record *R = ToVec[AliasWithNoPredicate];
      if (!MatchCode.empty())
        MatchCode += "else\n  ";
      MatchCode += "Mnemonic = \"" + R->getValueAsString("ToMnemonic")+"\";\n";
    }
    
    MatchCode += "return;";

    Cases.push_back(std::make_pair(I->first, MatchCode));
  }
  
  
  StringMatcher("Mnemonic", Cases, OS).Emit();
  OS << "}\n";
  
  return true;
}

void AsmMatcherEmitter::run(raw_ostream &OS) {
  CodeGenTarget Target;
  Record *AsmParser = Target.getAsmParser();
  std::string ClassName = AsmParser->getValueAsString("AsmParserClassName");

  // Compute the information on the instructions to match.
  AsmMatcherInfo Info(AsmParser, Target);
  Info.BuildInfo();

  // Sort the instruction table using the partial order on classes. We use
  // stable_sort to ensure that ambiguous instructions are still
  // deterministically ordered.
  std::stable_sort(Info.Matchables.begin(), Info.Matchables.end(),
                   less_ptr<MatchableInfo>());

  DEBUG_WITH_TYPE("instruction_info", {
      for (std::vector<MatchableInfo*>::iterator
             it = Info.Matchables.begin(), ie = Info.Matchables.end();
           it != ie; ++it)
        (*it)->dump();
    });

  // Check for ambiguous matchables.
  DEBUG_WITH_TYPE("ambiguous_instrs", {
    unsigned NumAmbiguous = 0;
    for (unsigned i = 0, e = Info.Matchables.size(); i != e; ++i) {
      for (unsigned j = i + 1; j != e; ++j) {
        MatchableInfo &A = *Info.Matchables[i];
        MatchableInfo &B = *Info.Matchables[j];

        if (A.CouldMatchAmiguouslyWith(B)) {
          errs() << "warning: ambiguous matchables:\n";
          A.dump();
          errs() << "\nis incomparable with:\n";
          B.dump();
          errs() << "\n\n";
          ++NumAmbiguous;
        }
      }
    }
    if (NumAmbiguous)
      errs() << "warning: " << NumAmbiguous
             << " ambiguous matchables!\n";
  });

  // Write the output.

  EmitSourceFileHeader("Assembly Matcher Source Fragment", OS);

  // Information for the class declaration.
  OS << "\n#ifdef GET_ASSEMBLER_HEADER\n";
  OS << "#undef GET_ASSEMBLER_HEADER\n";
  OS << "  // This should be included into the middle of the declaration of \n";
  OS << "  // your subclasses implementation of TargetAsmParser.\n";
  OS << "  unsigned ComputeAvailableFeatures(const " <<
           Target.getName() << "Subtarget *Subtarget) const;\n";
  OS << "  enum MatchResultTy {\n";
  OS << "    Match_Success, Match_MnemonicFail, Match_InvalidOperand,\n";
  OS << "    Match_MissingFeature\n";
  OS << "  };\n";
  OS << "  MatchResultTy MatchInstructionImpl(const "
     << "SmallVectorImpl<MCParsedAsmOperand*>"
     << " &Operands, MCInst &Inst, unsigned &ErrorInfo);\n\n";
  OS << "#endif // GET_ASSEMBLER_HEADER_INFO\n\n";




  OS << "\n#ifdef GET_REGISTER_MATCHER\n";
  OS << "#undef GET_REGISTER_MATCHER\n\n";

  // Emit the subtarget feature enumeration.
  EmitSubtargetFeatureFlagEnumeration(Info, OS);

  // Emit the function to match a register name to number.
  EmitMatchRegisterName(Target, AsmParser, OS);

  OS << "#endif // GET_REGISTER_MATCHER\n\n";


  OS << "\n#ifdef GET_MATCHER_IMPLEMENTATION\n";
  OS << "#undef GET_MATCHER_IMPLEMENTATION\n\n";

  // Generate the function that remaps for mnemonic aliases.
  bool HasMnemonicAliases = EmitMnemonicAliases(OS, Info);
  
  // Generate the unified function to convert operands into an MCInst.
  EmitConvertToMCInst(Target, Info.Matchables, OS);

  // Emit the enumeration for classes which participate in matching.
  EmitMatchClassEnumeration(Target, Info.Classes, OS);

  // Emit the routine to match token strings to their match class.
  EmitMatchTokenString(Target, Info.Classes, OS);

  // Emit the routine to classify an operand.
  EmitClassifyOperand(Info, OS);

  // Emit the subclass predicate routine.
  EmitIsSubclass(Target, Info.Classes, OS);

  // Emit the available features compute function.
  EmitComputeAvailableFeatures(Info, OS);


  size_t MaxNumOperands = 0;
  for (std::vector<MatchableInfo*>::const_iterator it =
         Info.Matchables.begin(), ie = Info.Matchables.end();
       it != ie; ++it)
    MaxNumOperands = std::max(MaxNumOperands, (*it)->AsmOperands.size());


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
  OS << "namespace {\n";
  OS << "  struct MatchEntry {\n";
  OS << "    unsigned Opcode;\n";
  OS << "    const char *Mnemonic;\n";
  OS << "    ConversionKind ConvertFn;\n";
  OS << "    MatchClassKind Classes[" << MaxNumOperands << "];\n";
  OS << "    unsigned RequiredFeatures;\n";
  OS << "  };\n\n";

  OS << "// Predicate for searching for an opcode.\n";
  OS << "  struct LessOpcode {\n";
  OS << "    bool operator()(const MatchEntry &LHS, StringRef RHS) {\n";
  OS << "      return StringRef(LHS.Mnemonic) < RHS;\n";
  OS << "    }\n";
  OS << "    bool operator()(StringRef LHS, const MatchEntry &RHS) {\n";
  OS << "      return LHS < StringRef(RHS.Mnemonic);\n";
  OS << "    }\n";
  OS << "    bool operator()(const MatchEntry &LHS, const MatchEntry &RHS) {\n";
  OS << "      return StringRef(LHS.Mnemonic) < StringRef(RHS.Mnemonic);\n";
  OS << "    }\n";
  OS << "  };\n";

  OS << "} // end anonymous namespace.\n\n";

  OS << "static const MatchEntry MatchTable["
     << Info.Matchables.size() << "] = {\n";

  for (std::vector<MatchableInfo*>::const_iterator it =
       Info.Matchables.begin(), ie = Info.Matchables.end();
       it != ie; ++it) {
    MatchableInfo &II = **it;

    OS << "  { " << Target.getName() << "::" << II.InstrName
    << ", \"" << II.Mnemonic << "\""
    << ", " << II.ConversionFnKind << ", { ";
    for (unsigned i = 0, e = II.AsmOperands.size(); i != e; ++i) {
      MatchableInfo::Operand &Op = II.AsmOperands[i];

      if (i) OS << ", ";
      OS << Op.Class->Name;
    }
    OS << " }, ";

    // Write the required features mask.
    if (!II.RequiredFeatures.empty()) {
      for (unsigned i = 0, e = II.RequiredFeatures.size(); i != e; ++i) {
        if (i) OS << "|";
        OS << II.RequiredFeatures[i]->getEnumName();
      }
    } else
      OS << "0";

    OS << "},\n";
  }

  OS << "};\n\n";

  // Finally, build the match function.
  OS << Target.getName() << ClassName << "::MatchResultTy "
     << Target.getName() << ClassName << "::\n"
     << "MatchInstructionImpl(const SmallVectorImpl<MCParsedAsmOperand*>"
     << " &Operands,\n";
  OS << "                     MCInst &Inst, unsigned &ErrorInfo) {\n";

  // Emit code to get the available features.
  OS << "  // Get the current feature set.\n";
  OS << "  unsigned AvailableFeatures = getAvailableFeatures();\n\n";

  OS << "  // Get the instruction mnemonic, which is the first token.\n";
  OS << "  StringRef Mnemonic = ((" << Target.getName()
     << "Operand*)Operands[0])->getToken();\n\n";

  if (HasMnemonicAliases) {
    OS << "  // Process all MnemonicAliases to remap the mnemonic.\n";
    OS << "  ApplyMnemonicAliases(Mnemonic, AvailableFeatures);\n\n";
  }
  
  // Emit code to compute the class list for this operand vector.
  OS << "  // Eliminate obvious mismatches.\n";
  OS << "  if (Operands.size() > " << (MaxNumOperands+1) << ") {\n";
  OS << "    ErrorInfo = " << (MaxNumOperands+1) << ";\n";
  OS << "    return Match_InvalidOperand;\n";
  OS << "  }\n\n";

  OS << "  // Compute the class list for this operand vector.\n";
  OS << "  MatchClassKind Classes[" << MaxNumOperands << "];\n";
  OS << "  for (unsigned i = 1, e = Operands.size(); i != e; ++i) {\n";
  OS << "    Classes[i-1] = ClassifyOperand(Operands[i]);\n\n";

  OS << "    // Check for invalid operands before matching.\n";
  OS << "    if (Classes[i-1] == InvalidMatchClass) {\n";
  OS << "      ErrorInfo = i;\n";
  OS << "      return Match_InvalidOperand;\n";
  OS << "    }\n";
  OS << "  }\n\n";

  OS << "  // Mark unused classes.\n";
  OS << "  for (unsigned i = Operands.size()-1, e = " << MaxNumOperands << "; "
     << "i != e; ++i)\n";
  OS << "    Classes[i] = InvalidMatchClass;\n\n";

  OS << "  // Some state to try to produce better error messages.\n";
  OS << "  bool HadMatchOtherThanFeatures = false;\n\n";
  OS << "  // Set ErrorInfo to the operand that mismatches if it is \n";
  OS << "  // wrong for all instances of the instruction.\n";
  OS << "  ErrorInfo = ~0U;\n";

  // Emit code to search the table.
  OS << "  // Search the table.\n";
  OS << "  std::pair<const MatchEntry*, const MatchEntry*> MnemonicRange =\n";
  OS << "    std::equal_range(MatchTable, MatchTable+"
     << Info.Matchables.size() << ", Mnemonic, LessOpcode());\n\n";

  OS << "  // Return a more specific error code if no mnemonics match.\n";
  OS << "  if (MnemonicRange.first == MnemonicRange.second)\n";
  OS << "    return Match_MnemonicFail;\n\n";

  OS << "  for (const MatchEntry *it = MnemonicRange.first, "
     << "*ie = MnemonicRange.second;\n";
  OS << "       it != ie; ++it) {\n";

  OS << "    // equal_range guarantees that instruction mnemonic matches.\n";
  OS << "    assert(Mnemonic == it->Mnemonic);\n";

  // Emit check that the subclasses match.
  OS << "    bool OperandsValid = true;\n";
  OS << "    for (unsigned i = 0; i != " << MaxNumOperands << "; ++i) {\n";
  OS << "      if (IsSubclass(Classes[i], it->Classes[i]))\n";
  OS << "        continue;\n";
  OS << "      // If this operand is broken for all of the instances of this\n";
  OS << "      // mnemonic, keep track of it so we can report loc info.\n";
  OS << "      if (it == MnemonicRange.first || ErrorInfo == i+1)\n";
  OS << "        ErrorInfo = i+1;\n";
  OS << "      else\n";
  OS << "        ErrorInfo = ~0U;";
  OS << "      // Otherwise, just reject this instance of the mnemonic.\n";
  OS << "      OperandsValid = false;\n";
  OS << "      break;\n";
  OS << "    }\n\n";

  OS << "    if (!OperandsValid) continue;\n";

  // Emit check that the required features are available.
  OS << "    if ((AvailableFeatures & it->RequiredFeatures) "
     << "!= it->RequiredFeatures) {\n";
  OS << "      HadMatchOtherThanFeatures = true;\n";
  OS << "      continue;\n";
  OS << "    }\n";

  OS << "\n";
  OS << "    ConvertToMCInst(it->ConvertFn, Inst, it->Opcode, Operands);\n";

  // Call the post-processing function, if used.
  std::string InsnCleanupFn =
    AsmParser->getValueAsString("AsmParserInstCleanup");
  if (!InsnCleanupFn.empty())
    OS << "    " << InsnCleanupFn << "(Inst);\n";

  OS << "    return Match_Success;\n";
  OS << "  }\n\n";

  OS << "  // Okay, we had no match.  Try to return a useful error code.\n";
  OS << "  if (HadMatchOtherThanFeatures) return Match_MissingFeature;\n";
  OS << "  return Match_InvalidOperand;\n";
  OS << "}\n\n";

  OS << "#endif // GET_MATCHER_IMPLEMENTATION\n\n";
}
