//===- ClangAttrEmitter.cpp - Generate Clang attribute handling =-*- C++ -*--=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// These tablegen backends emit Clang attribute processing code
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/StringMatcher.h"
#include "llvm/TableGen/TableGenBackend.h"
#include <algorithm>
#include <cctype>
#include <set>
#include <sstream>

using namespace llvm;

class FlattenedSpelling {
  std::string V, N, NS;
  bool K;

public:
  FlattenedSpelling(const std::string &Variety, const std::string &Name,
                    const std::string &Namespace, bool KnownToGCC) :
    V(Variety), N(Name), NS(Namespace), K(KnownToGCC) {}
  explicit FlattenedSpelling(const Record &Spelling) :
    V(Spelling.getValueAsString("Variety")),
    N(Spelling.getValueAsString("Name")) {

    assert(V != "GCC" && "Given a GCC spelling, which means this hasn't been"
           "flattened!");
    if (V == "CXX11")
      NS = Spelling.getValueAsString("Namespace");
    bool Unset;
    K = Spelling.getValueAsBitOrUnset("KnownToGCC", Unset);
  }

  const std::string &variety() const { return V; }
  const std::string &name() const { return N; }
  const std::string &nameSpace() const { return NS; }
  bool knownToGCC() const { return K; }
};

std::vector<FlattenedSpelling> GetFlattenedSpellings(const Record &Attr) {
  std::vector<Record *> Spellings = Attr.getValueAsListOfDefs("Spellings");
  std::vector<FlattenedSpelling> Ret;

  for (std::vector<Record *>::const_iterator I = Spellings.begin(),
       E = Spellings.end(); I != E; ++I) {
    const Record &Spelling = **I;

    if (Spelling.getValueAsString("Variety") == "GCC") {
      // Gin up two new spelling objects to add into the list.
      Ret.push_back(FlattenedSpelling("GNU", Spelling.getValueAsString("Name"),
                                      "", true));
      Ret.push_back(FlattenedSpelling("CXX11",
                                      Spelling.getValueAsString("Name"),
                                      "gnu", true));
    } else
      Ret.push_back(FlattenedSpelling(Spelling));
  }

  return Ret;
}

static std::string ReadPCHRecord(StringRef type) {
  return StringSwitch<std::string>(type)
    .EndsWith("Decl *", "GetLocalDeclAs<" 
              + std::string(type, 0, type.size()-1) + ">(F, Record[Idx++])")
    .Case("TypeSourceInfo *", "GetTypeSourceInfo(F, Record, Idx)")
    .Case("Expr *", "ReadExpr(F)")
    .Case("IdentifierInfo *", "GetIdentifierInfo(F, Record, Idx)")
    .Default("Record[Idx++]");
}

// Assumes that the way to get the value is SA->getname()
static std::string WritePCHRecord(StringRef type, StringRef name) {
  return StringSwitch<std::string>(type)
    .EndsWith("Decl *", "AddDeclRef(" + std::string(name) +
                        ", Record);\n")
    .Case("TypeSourceInfo *",
          "AddTypeSourceInfo(" + std::string(name) + ", Record);\n")
    .Case("Expr *", "AddStmt(" + std::string(name) + ");\n")
    .Case("IdentifierInfo *", 
          "AddIdentifierRef(" + std::string(name) + ", Record);\n")
    .Default("Record.push_back(" + std::string(name) + ");\n");
}

// Normalize attribute name by removing leading and trailing
// underscores. For example, __foo, foo__, __foo__ would
// become foo.
static StringRef NormalizeAttrName(StringRef AttrName) {
  if (AttrName.startswith("__"))
    AttrName = AttrName.substr(2, AttrName.size());

  if (AttrName.endswith("__"))
    AttrName = AttrName.substr(0, AttrName.size() - 2);

  return AttrName;
}

// Normalize the name by removing any and all leading and trailing underscores.
// This is different from NormalizeAttrName in that it also handles names like
// _pascal and __pascal.
static StringRef NormalizeNameForSpellingComparison(StringRef Name) {
  while (Name.startswith("_"))
    Name = Name.substr(1, Name.size());
  while (Name.endswith("_"))
    Name = Name.substr(0, Name.size() - 1);
  return Name;
}

// Normalize attribute spelling only if the spelling has both leading
// and trailing underscores. For example, __ms_struct__ will be 
// normalized to "ms_struct"; __cdecl will remain intact.
static StringRef NormalizeAttrSpelling(StringRef AttrSpelling) {
  if (AttrSpelling.startswith("__") && AttrSpelling.endswith("__")) {
    AttrSpelling = AttrSpelling.substr(2, AttrSpelling.size() - 4);
  }

  return AttrSpelling;
}

typedef std::vector<std::pair<std::string, Record *> > ParsedAttrMap;

static ParsedAttrMap getParsedAttrList(const RecordKeeper &Records,
                                       ParsedAttrMap *Dupes = 0) {
  std::vector<Record*> Attrs = Records.getAllDerivedDefinitions("Attr");
  std::set<std::string> Seen;
  ParsedAttrMap R;
  for (std::vector<Record*>::iterator I = Attrs.begin(), E = Attrs.end();
       I != E; ++I) {
    Record &Attr = **I;
    if (Attr.getValueAsBit("SemaHandler")) {
      std::string AN;
      if (Attr.isSubClassOf("TargetSpecificAttr") &&
          !Attr.isValueUnset("ParseKind")) {
        AN = Attr.getValueAsString("ParseKind");

        // If this attribute has already been handled, it does not need to be
        // handled again.
        if (Seen.find(AN) != Seen.end()) {
          if (Dupes)
            Dupes->push_back(std::make_pair(AN, *I));
          continue;
        }
        Seen.insert(AN);
      } else
        AN = NormalizeAttrName(Attr.getName()).str();

      R.push_back(std::make_pair(AN, *I));
    }
  }
  return R;
}

namespace {
  class Argument {
    std::string lowerName, upperName;
    StringRef attrName;
    bool isOpt;

  public:
    Argument(Record &Arg, StringRef Attr)
      : lowerName(Arg.getValueAsString("Name")), upperName(lowerName),
        attrName(Attr), isOpt(false) {
      if (!lowerName.empty()) {
        lowerName[0] = std::tolower(lowerName[0]);
        upperName[0] = std::toupper(upperName[0]);
      }
    }
    virtual ~Argument() {}

    StringRef getLowerName() const { return lowerName; }
    StringRef getUpperName() const { return upperName; }
    StringRef getAttrName() const { return attrName; }

    bool isOptional() const { return isOpt; }
    void setOptional(bool set) { isOpt = set; }

    // These functions print the argument contents formatted in different ways.
    virtual void writeAccessors(raw_ostream &OS) const = 0;
    virtual void writeAccessorDefinitions(raw_ostream &OS) const {}
    virtual void writeASTVisitorTraversal(raw_ostream &OS) const {}
    virtual void writeCloneArgs(raw_ostream &OS) const = 0;
    virtual void writeTemplateInstantiationArgs(raw_ostream &OS) const = 0;
    virtual void writeTemplateInstantiation(raw_ostream &OS) const {}
    virtual void writeCtorBody(raw_ostream &OS) const {}
    virtual void writeCtorInitializers(raw_ostream &OS) const = 0;
    virtual void writeCtorDefaultInitializers(raw_ostream &OS) const = 0;
    virtual void writeCtorParameters(raw_ostream &OS) const = 0;
    virtual void writeDeclarations(raw_ostream &OS) const = 0;
    virtual void writePCHReadArgs(raw_ostream &OS) const = 0;
    virtual void writePCHReadDecls(raw_ostream &OS) const = 0;
    virtual void writePCHWrite(raw_ostream &OS) const = 0;
    virtual void writeValue(raw_ostream &OS) const = 0;
    virtual void writeDump(raw_ostream &OS) const = 0;
    virtual void writeDumpChildren(raw_ostream &OS) const {}
    virtual void writeHasChildren(raw_ostream &OS) const { OS << "false"; }

    virtual bool isEnumArg() const { return false; }
    virtual bool isVariadicEnumArg() const { return false; }

    virtual void writeImplicitCtorArgs(raw_ostream &OS) const {
      OS << getUpperName();
    }
  };

  class SimpleArgument : public Argument {
    std::string type;

  public:
    SimpleArgument(Record &Arg, StringRef Attr, std::string T)
      : Argument(Arg, Attr), type(T)
    {}

    std::string getType() const { return type; }

    void writeAccessors(raw_ostream &OS) const {
      OS << "  " << type << " get" << getUpperName() << "() const {\n";
      OS << "    return " << getLowerName() << ";\n";
      OS << "  }";
    }
    void writeCloneArgs(raw_ostream &OS) const {
      OS << getLowerName();
    }
    void writeTemplateInstantiationArgs(raw_ostream &OS) const {
      OS << "A->get" << getUpperName() << "()";
    }
    void writeCtorInitializers(raw_ostream &OS) const {
      OS << getLowerName() << "(" << getUpperName() << ")";
    }
    void writeCtorDefaultInitializers(raw_ostream &OS) const {
      OS << getLowerName() << "()";
    }
    void writeCtorParameters(raw_ostream &OS) const {
      OS << type << " " << getUpperName();
    }
    void writeDeclarations(raw_ostream &OS) const {
      OS << type << " " << getLowerName() << ";";
    }
    void writePCHReadDecls(raw_ostream &OS) const {
      std::string read = ReadPCHRecord(type);
      OS << "    " << type << " " << getLowerName() << " = " << read << ";\n";
    }
    void writePCHReadArgs(raw_ostream &OS) const {
      OS << getLowerName();
    }
    void writePCHWrite(raw_ostream &OS) const {
      OS << "    " << WritePCHRecord(type, "SA->get" +
                                           std::string(getUpperName()) + "()");
    }
    void writeValue(raw_ostream &OS) const {
      if (type == "FunctionDecl *") {
        OS << "\" << get" << getUpperName()
           << "()->getNameInfo().getAsString() << \"";
      } else if (type == "IdentifierInfo *") {
        OS << "\" << get" << getUpperName() << "()->getName() << \"";
      } else if (type == "TypeSourceInfo *") {
        OS << "\" << get" << getUpperName() << "().getAsString() << \"";
      } else {
        OS << "\" << get" << getUpperName() << "() << \"";
      }
    }
    void writeDump(raw_ostream &OS) const {
      if (type == "FunctionDecl *") {
        OS << "    OS << \" \";\n";
        OS << "    dumpBareDeclRef(SA->get" << getUpperName() << "());\n"; 
      } else if (type == "IdentifierInfo *") {
        OS << "    OS << \" \" << SA->get" << getUpperName()
           << "()->getName();\n";
      } else if (type == "TypeSourceInfo *") {
        OS << "    OS << \" \" << SA->get" << getUpperName()
           << "().getAsString();\n";
      } else if (type == "bool") {
        OS << "    if (SA->get" << getUpperName() << "()) OS << \" "
           << getUpperName() << "\";\n";
      } else if (type == "int" || type == "unsigned") {
        OS << "    OS << \" \" << SA->get" << getUpperName() << "();\n";
      } else {
        llvm_unreachable("Unknown SimpleArgument type!");
      }
    }
  };

  class DefaultSimpleArgument : public SimpleArgument {
    int64_t Default;

  public:
    DefaultSimpleArgument(Record &Arg, StringRef Attr,
                          std::string T, int64_t Default)
      : SimpleArgument(Arg, Attr, T), Default(Default) {}

    void writeAccessors(raw_ostream &OS) const {
      SimpleArgument::writeAccessors(OS);

      OS << "\n\n  static const " << getType() << " Default" << getUpperName()
         << " = " << Default << ";";
    }
  };

  class StringArgument : public Argument {
  public:
    StringArgument(Record &Arg, StringRef Attr)
      : Argument(Arg, Attr)
    {}

    void writeAccessors(raw_ostream &OS) const {
      OS << "  llvm::StringRef get" << getUpperName() << "() const {\n";
      OS << "    return llvm::StringRef(" << getLowerName() << ", "
         << getLowerName() << "Length);\n";
      OS << "  }\n";
      OS << "  unsigned get" << getUpperName() << "Length() const {\n";
      OS << "    return " << getLowerName() << "Length;\n";
      OS << "  }\n";
      OS << "  void set" << getUpperName()
         << "(ASTContext &C, llvm::StringRef S) {\n";
      OS << "    " << getLowerName() << "Length = S.size();\n";
      OS << "    this->" << getLowerName() << " = new (C, 1) char ["
         << getLowerName() << "Length];\n";
      OS << "    std::memcpy(this->" << getLowerName() << ", S.data(), "
         << getLowerName() << "Length);\n";
      OS << "  }";
    }
    void writeCloneArgs(raw_ostream &OS) const {
      OS << "get" << getUpperName() << "()";
    }
    void writeTemplateInstantiationArgs(raw_ostream &OS) const {
      OS << "A->get" << getUpperName() << "()";
    }
    void writeCtorBody(raw_ostream &OS) const {
      OS << "      std::memcpy(" << getLowerName() << ", " << getUpperName()
         << ".data(), " << getLowerName() << "Length);";
    }
    void writeCtorInitializers(raw_ostream &OS) const {
      OS << getLowerName() << "Length(" << getUpperName() << ".size()),"
         << getLowerName() << "(new (Ctx, 1) char[" << getLowerName()
         << "Length])";
    }
    void writeCtorDefaultInitializers(raw_ostream &OS) const {
      OS << getLowerName() << "Length(0)," << getLowerName() << "(0)";
    }
    void writeCtorParameters(raw_ostream &OS) const {
      OS << "llvm::StringRef " << getUpperName();
    }
    void writeDeclarations(raw_ostream &OS) const {
      OS << "unsigned " << getLowerName() << "Length;\n";
      OS << "char *" << getLowerName() << ";";
    }
    void writePCHReadDecls(raw_ostream &OS) const {
      OS << "    std::string " << getLowerName()
         << "= ReadString(Record, Idx);\n";
    }
    void writePCHReadArgs(raw_ostream &OS) const {
      OS << getLowerName();
    }
    void writePCHWrite(raw_ostream &OS) const {
      OS << "    AddString(SA->get" << getUpperName() << "(), Record);\n";
    }
    void writeValue(raw_ostream &OS) const {
      OS << "\\\"\" << get" << getUpperName() << "() << \"\\\"";
    }
    void writeDump(raw_ostream &OS) const {
      OS << "    OS << \" \\\"\" << SA->get" << getUpperName()
         << "() << \"\\\"\";\n";
    }
  };

  class AlignedArgument : public Argument {
  public:
    AlignedArgument(Record &Arg, StringRef Attr)
      : Argument(Arg, Attr)
    {}

    void writeAccessors(raw_ostream &OS) const {
      OS << "  bool is" << getUpperName() << "Dependent() const;\n";

      OS << "  unsigned get" << getUpperName() << "(ASTContext &Ctx) const;\n";

      OS << "  bool is" << getUpperName() << "Expr() const {\n";
      OS << "    return is" << getLowerName() << "Expr;\n";
      OS << "  }\n";

      OS << "  Expr *get" << getUpperName() << "Expr() const {\n";
      OS << "    assert(is" << getLowerName() << "Expr);\n";
      OS << "    return " << getLowerName() << "Expr;\n";
      OS << "  }\n";

      OS << "  TypeSourceInfo *get" << getUpperName() << "Type() const {\n";
      OS << "    assert(!is" << getLowerName() << "Expr);\n";
      OS << "    return " << getLowerName() << "Type;\n";
      OS << "  }";
    }
    void writeAccessorDefinitions(raw_ostream &OS) const {
      OS << "bool " << getAttrName() << "Attr::is" << getUpperName()
         << "Dependent() const {\n";
      OS << "  if (is" << getLowerName() << "Expr)\n";
      OS << "    return " << getLowerName() << "Expr && (" << getLowerName()
         << "Expr->isValueDependent() || " << getLowerName()
         << "Expr->isTypeDependent());\n"; 
      OS << "  else\n";
      OS << "    return " << getLowerName()
         << "Type->getType()->isDependentType();\n";
      OS << "}\n";

      // FIXME: Do not do the calculation here
      // FIXME: Handle types correctly
      // A null pointer means maximum alignment
      // FIXME: Load the platform-specific maximum alignment, rather than
      //        16, the x86 max.
      OS << "unsigned " << getAttrName() << "Attr::get" << getUpperName()
         << "(ASTContext &Ctx) const {\n";
      OS << "  assert(!is" << getUpperName() << "Dependent());\n";
      OS << "  if (is" << getLowerName() << "Expr)\n";
      OS << "    return (" << getLowerName() << "Expr ? " << getLowerName()
         << "Expr->EvaluateKnownConstInt(Ctx).getZExtValue() : 16)"
         << "* Ctx.getCharWidth();\n";
      OS << "  else\n";
      OS << "    return 0; // FIXME\n";
      OS << "}\n";
    }
    void writeCloneArgs(raw_ostream &OS) const {
      OS << "is" << getLowerName() << "Expr, is" << getLowerName()
         << "Expr ? static_cast<void*>(" << getLowerName()
         << "Expr) : " << getLowerName()
         << "Type";
    }
    void writeTemplateInstantiationArgs(raw_ostream &OS) const {
      // FIXME: move the definition in Sema::InstantiateAttrs to here.
      // In the meantime, aligned attributes are cloned.
    }
    void writeCtorBody(raw_ostream &OS) const {
      OS << "    if (is" << getLowerName() << "Expr)\n";
      OS << "       " << getLowerName() << "Expr = reinterpret_cast<Expr *>("
         << getUpperName() << ");\n";
      OS << "    else\n";
      OS << "       " << getLowerName()
         << "Type = reinterpret_cast<TypeSourceInfo *>(" << getUpperName()
         << ");";
    }
    void writeCtorInitializers(raw_ostream &OS) const {
      OS << "is" << getLowerName() << "Expr(Is" << getUpperName() << "Expr)";
    }
    void writeCtorDefaultInitializers(raw_ostream &OS) const {
      OS << "is" << getLowerName() << "Expr(false)";
    }
    void writeCtorParameters(raw_ostream &OS) const {
      OS << "bool Is" << getUpperName() << "Expr, void *" << getUpperName();
    }
    void writeImplicitCtorArgs(raw_ostream &OS) const {
      OS << "Is" << getUpperName() << "Expr, " << getUpperName();
    }
    void writeDeclarations(raw_ostream &OS) const {
      OS << "bool is" << getLowerName() << "Expr;\n";
      OS << "union {\n";
      OS << "Expr *" << getLowerName() << "Expr;\n";
      OS << "TypeSourceInfo *" << getLowerName() << "Type;\n";
      OS << "};";
    }
    void writePCHReadArgs(raw_ostream &OS) const {
      OS << "is" << getLowerName() << "Expr, " << getLowerName() << "Ptr";
    }
    void writePCHReadDecls(raw_ostream &OS) const {
      OS << "    bool is" << getLowerName() << "Expr = Record[Idx++];\n";
      OS << "    void *" << getLowerName() << "Ptr;\n";
      OS << "    if (is" << getLowerName() << "Expr)\n";
      OS << "      " << getLowerName() << "Ptr = ReadExpr(F);\n";
      OS << "    else\n";
      OS << "      " << getLowerName()
         << "Ptr = GetTypeSourceInfo(F, Record, Idx);\n";
    }
    void writePCHWrite(raw_ostream &OS) const {
      OS << "    Record.push_back(SA->is" << getUpperName() << "Expr());\n";
      OS << "    if (SA->is" << getUpperName() << "Expr())\n";
      OS << "      AddStmt(SA->get" << getUpperName() << "Expr());\n";
      OS << "    else\n";
      OS << "      AddTypeSourceInfo(SA->get" << getUpperName()
         << "Type(), Record);\n";
    }
    void writeValue(raw_ostream &OS) const {
      OS << "\";\n"
         << "  " << getLowerName() << "Expr->printPretty(OS, 0, Policy);\n"
         << "  OS << \"";
    }
    void writeDump(raw_ostream &OS) const {
    }
    void writeDumpChildren(raw_ostream &OS) const {
      OS << "    if (SA->is" << getUpperName() << "Expr()) {\n";
      OS << "      lastChild();\n";
      OS << "      dumpStmt(SA->get" << getUpperName() << "Expr());\n";
      OS << "    } else\n";
      OS << "      dumpType(SA->get" << getUpperName()
         << "Type()->getType());\n";
    }
    void writeHasChildren(raw_ostream &OS) const {
      OS << "SA->is" << getUpperName() << "Expr()";
    }
  };

  class VariadicArgument : public Argument {
    std::string type;

  public:
    VariadicArgument(Record &Arg, StringRef Attr, std::string T)
      : Argument(Arg, Attr), type(T)
    {}

    std::string getType() const { return type; }

    void writeAccessors(raw_ostream &OS) const {
      OS << "  typedef " << type << "* " << getLowerName() << "_iterator;\n";
      OS << "  " << getLowerName() << "_iterator " << getLowerName()
         << "_begin() const {\n";
      OS << "    return " << getLowerName() << ";\n";
      OS << "  }\n";
      OS << "  " << getLowerName() << "_iterator " << getLowerName()
         << "_end() const {\n";
      OS << "    return " << getLowerName() << " + " << getLowerName()
         << "Size;\n";
      OS << "  }\n";
      OS << "  unsigned " << getLowerName() << "_size() const {\n"
         << "    return " << getLowerName() << "Size;\n";
      OS << "  }";
    }
    void writeCloneArgs(raw_ostream &OS) const {
      OS << getLowerName() << ", " << getLowerName() << "Size";
    }
    void writeTemplateInstantiationArgs(raw_ostream &OS) const {
      // This isn't elegant, but we have to go through public methods...
      OS << "A->" << getLowerName() << "_begin(), "
         << "A->" << getLowerName() << "_size()";
    }
    void writeCtorBody(raw_ostream &OS) const {
      // FIXME: memcpy is not safe on non-trivial types.
      OS << "    std::memcpy(" << getLowerName() << ", " << getUpperName()
         << ", " << getLowerName() << "Size * sizeof(" << getType() << "));\n";
    }
    void writeCtorInitializers(raw_ostream &OS) const {
      OS << getLowerName() << "Size(" << getUpperName() << "Size), "
         << getLowerName() << "(new (Ctx, 16) " << getType() << "["
         << getLowerName() << "Size])";
    }
    void writeCtorDefaultInitializers(raw_ostream &OS) const {
      OS << getLowerName() << "Size(0), " << getLowerName() << "(0)";
    }
    void writeCtorParameters(raw_ostream &OS) const {
      OS << getType() << " *" << getUpperName() << ", unsigned "
         << getUpperName() << "Size";
    }
    void writeImplicitCtorArgs(raw_ostream &OS) const {
      OS << getUpperName() << ", " << getUpperName() << "Size";
    }
    void writeDeclarations(raw_ostream &OS) const {
      OS << "  unsigned " << getLowerName() << "Size;\n";
      OS << "  " << getType() << " *" << getLowerName() << ";";
    }
    void writePCHReadDecls(raw_ostream &OS) const {
      OS << "  unsigned " << getLowerName() << "Size = Record[Idx++];\n";
      OS << "  SmallVector<" << type << ", 4> " << getLowerName()
         << ";\n";
      OS << "  " << getLowerName() << ".reserve(" << getLowerName()
         << "Size);\n";
      OS << "    for (unsigned i = " << getLowerName() << "Size; i; --i)\n";
      
      std::string read = ReadPCHRecord(type);
      OS << "    " << getLowerName() << ".push_back(" << read << ");\n";
    }
    void writePCHReadArgs(raw_ostream &OS) const {
      OS << getLowerName() << ".data(), " << getLowerName() << "Size";
    }
    void writePCHWrite(raw_ostream &OS) const{
      OS << "    Record.push_back(SA->" << getLowerName() << "_size());\n";
      OS << "    for (" << getAttrName() << "Attr::" << getLowerName()
         << "_iterator i = SA->" << getLowerName() << "_begin(), e = SA->"
         << getLowerName() << "_end(); i != e; ++i)\n";
      OS << "      " << WritePCHRecord(type, "(*i)");
    }
    void writeValue(raw_ostream &OS) const {
      OS << "\";\n";
      OS << "  bool isFirst = true;\n"
         << "  for (" << getAttrName() << "Attr::" << getLowerName()
         << "_iterator i = " << getLowerName() << "_begin(), e = "
         << getLowerName() << "_end(); i != e; ++i) {\n"
         << "    if (isFirst) isFirst = false;\n"
         << "    else OS << \", \";\n"
         << "    OS << *i;\n"
         << "  }\n";
      OS << "  OS << \"";
    }
    void writeDump(raw_ostream &OS) const {
      OS << "    for (" << getAttrName() << "Attr::" << getLowerName()
         << "_iterator I = SA->" << getLowerName() << "_begin(), E = SA->"
         << getLowerName() << "_end(); I != E; ++I)\n";
      OS << "      OS << \" \" << *I;\n";
    }
  };

  class EnumArgument : public Argument {
    std::string type;
    std::vector<std::string> values, enums, uniques;
  public:
    EnumArgument(Record &Arg, StringRef Attr)
      : Argument(Arg, Attr), type(Arg.getValueAsString("Type")),
        values(Arg.getValueAsListOfStrings("Values")),
        enums(Arg.getValueAsListOfStrings("Enums")),
        uniques(enums)
    {
      // Calculate the various enum values
      std::sort(uniques.begin(), uniques.end());
      uniques.erase(std::unique(uniques.begin(), uniques.end()), uniques.end());
      // FIXME: Emit a proper error
      assert(!uniques.empty());
    }

    bool isEnumArg() const { return true; }

    void writeAccessors(raw_ostream &OS) const {
      OS << "  " << type << " get" << getUpperName() << "() const {\n";
      OS << "    return " << getLowerName() << ";\n";
      OS << "  }";
    }
    void writeCloneArgs(raw_ostream &OS) const {
      OS << getLowerName();
    }
    void writeTemplateInstantiationArgs(raw_ostream &OS) const {
      OS << "A->get" << getUpperName() << "()";
    }
    void writeCtorInitializers(raw_ostream &OS) const {
      OS << getLowerName() << "(" << getUpperName() << ")";
    }
    void writeCtorDefaultInitializers(raw_ostream &OS) const {
      OS << getLowerName() << "(" << type << "(0))";
    }
    void writeCtorParameters(raw_ostream &OS) const {
      OS << type << " " << getUpperName();
    }
    void writeDeclarations(raw_ostream &OS) const {
      std::vector<std::string>::const_iterator i = uniques.begin(),
                                               e = uniques.end();
      // The last one needs to not have a comma.
      --e;

      OS << "public:\n";
      OS << "  enum " << type << " {\n";
      for (; i != e; ++i)
        OS << "    " << *i << ",\n";
      OS << "    " << *e << "\n";
      OS << "  };\n";
      OS << "private:\n";
      OS << "  " << type << " " << getLowerName() << ";";
    }
    void writePCHReadDecls(raw_ostream &OS) const {
      OS << "    " << getAttrName() << "Attr::" << type << " " << getLowerName()
         << "(static_cast<" << getAttrName() << "Attr::" << type
         << ">(Record[Idx++]));\n";
    }
    void writePCHReadArgs(raw_ostream &OS) const {
      OS << getLowerName();
    }
    void writePCHWrite(raw_ostream &OS) const {
      OS << "Record.push_back(SA->get" << getUpperName() << "());\n";
    }
    void writeValue(raw_ostream &OS) const {
      OS << "\" << get" << getUpperName() << "() << \"";
    }
    void writeDump(raw_ostream &OS) const {
      OS << "    switch(SA->get" << getUpperName() << "()) {\n";
      for (std::vector<std::string>::const_iterator I = uniques.begin(),
           E = uniques.end(); I != E; ++I) {
        OS << "    case " << getAttrName() << "Attr::" << *I << ":\n";
        OS << "      OS << \" " << *I << "\";\n";
        OS << "      break;\n";
      }
      OS << "    }\n";
    }

    void writeConversion(raw_ostream &OS) const {
      OS << "  static bool ConvertStrTo" << type << "(StringRef Val, ";
      OS << type << " &Out) {\n";
      OS << "    Optional<" << type << "> R = llvm::StringSwitch<Optional<";
      OS << type << "> >(Val)\n";
      for (size_t I = 0; I < enums.size(); ++I) {
        OS << "      .Case(\"" << values[I] << "\", ";
        OS << getAttrName() << "Attr::" << enums[I] << ")\n";
      }
      OS << "      .Default(Optional<" << type << ">());\n";
      OS << "    if (R) {\n";
      OS << "      Out = *R;\n      return true;\n    }\n";
      OS << "    return false;\n";
      OS << "  }\n";
    }
  };
  
  class VariadicEnumArgument: public VariadicArgument {
    std::string type, QualifiedTypeName;
    std::vector<std::string> values, enums, uniques;
  public:
    VariadicEnumArgument(Record &Arg, StringRef Attr)
      : VariadicArgument(Arg, Attr, Arg.getValueAsString("Type")),
        type(Arg.getValueAsString("Type")),
        values(Arg.getValueAsListOfStrings("Values")),
        enums(Arg.getValueAsListOfStrings("Enums")),
        uniques(enums)
    {
      // Calculate the various enum values
      std::sort(uniques.begin(), uniques.end());
      uniques.erase(std::unique(uniques.begin(), uniques.end()), uniques.end());
      
      QualifiedTypeName = getAttrName().str() + "Attr::" + type;
      
      // FIXME: Emit a proper error
      assert(!uniques.empty());
    }

    bool isVariadicEnumArg() const { return true; }
    
    void writeDeclarations(raw_ostream &OS) const {
      std::vector<std::string>::const_iterator i = uniques.begin(),
                                               e = uniques.end();
      // The last one needs to not have a comma.
      --e;

      OS << "public:\n";
      OS << "  enum " << type << " {\n";
      for (; i != e; ++i)
        OS << "    " << *i << ",\n";
      OS << "    " << *e << "\n";
      OS << "  };\n";
      OS << "private:\n";
      
      VariadicArgument::writeDeclarations(OS);
    }
    void writeDump(raw_ostream &OS) const {
      OS << "    for (" << getAttrName() << "Attr::" << getLowerName()
         << "_iterator I = SA->" << getLowerName() << "_begin(), E = SA->"
         << getLowerName() << "_end(); I != E; ++I) {\n";
      OS << "      switch(*I) {\n";
      for (std::vector<std::string>::const_iterator UI = uniques.begin(),
           UE = uniques.end(); UI != UE; ++UI) {
        OS << "    case " << getAttrName() << "Attr::" << *UI << ":\n";
        OS << "      OS << \" " << *UI << "\";\n";
        OS << "      break;\n";
      }
      OS << "      }\n";
      OS << "    }\n";
    }
    void writePCHReadDecls(raw_ostream &OS) const {
      OS << "    unsigned " << getLowerName() << "Size = Record[Idx++];\n";
      OS << "    SmallVector<" << QualifiedTypeName << ", 4> " << getLowerName()
         << ";\n";
      OS << "    " << getLowerName() << ".reserve(" << getLowerName()
         << "Size);\n";
      OS << "    for (unsigned i = " << getLowerName() << "Size; i; --i)\n";
      OS << "      " << getLowerName() << ".push_back(" << "static_cast<"
         << QualifiedTypeName << ">(Record[Idx++]));\n";
    }
    void writePCHWrite(raw_ostream &OS) const{
      OS << "    Record.push_back(SA->" << getLowerName() << "_size());\n";
      OS << "    for (" << getAttrName() << "Attr::" << getLowerName()
         << "_iterator i = SA->" << getLowerName() << "_begin(), e = SA->"
         << getLowerName() << "_end(); i != e; ++i)\n";
      OS << "      " << WritePCHRecord(QualifiedTypeName, "(*i)");
    }
    void writeConversion(raw_ostream &OS) const {
      OS << "  static bool ConvertStrTo" << type << "(StringRef Val, ";
      OS << type << " &Out) {\n";
      OS << "    Optional<" << type << "> R = llvm::StringSwitch<Optional<";
      OS << type << "> >(Val)\n";
      for (size_t I = 0; I < enums.size(); ++I) {
        OS << "      .Case(\"" << values[I] << "\", ";
        OS << getAttrName() << "Attr::" << enums[I] << ")\n";
      }
      OS << "      .Default(Optional<" << type << ">());\n";
      OS << "    if (R) {\n";
      OS << "      Out = *R;\n      return true;\n    }\n";
      OS << "    return false;\n";
      OS << "  }\n";
    }
  };

  class VersionArgument : public Argument {
  public:
    VersionArgument(Record &Arg, StringRef Attr)
      : Argument(Arg, Attr)
    {}

    void writeAccessors(raw_ostream &OS) const {
      OS << "  VersionTuple get" << getUpperName() << "() const {\n";
      OS << "    return " << getLowerName() << ";\n";
      OS << "  }\n";
      OS << "  void set" << getUpperName() 
         << "(ASTContext &C, VersionTuple V) {\n";
      OS << "    " << getLowerName() << " = V;\n";
      OS << "  }";
    }
    void writeCloneArgs(raw_ostream &OS) const {
      OS << "get" << getUpperName() << "()";
    }
    void writeTemplateInstantiationArgs(raw_ostream &OS) const {
      OS << "A->get" << getUpperName() << "()";
    }
    void writeCtorBody(raw_ostream &OS) const {
    }
    void writeCtorInitializers(raw_ostream &OS) const {
      OS << getLowerName() << "(" << getUpperName() << ")";
    }
    void writeCtorDefaultInitializers(raw_ostream &OS) const {
      OS << getLowerName() << "()";
    }
    void writeCtorParameters(raw_ostream &OS) const {
      OS << "VersionTuple " << getUpperName();
    }
    void writeDeclarations(raw_ostream &OS) const {
      OS << "VersionTuple " << getLowerName() << ";\n";
    }
    void writePCHReadDecls(raw_ostream &OS) const {
      OS << "    VersionTuple " << getLowerName()
         << "= ReadVersionTuple(Record, Idx);\n";
    }
    void writePCHReadArgs(raw_ostream &OS) const {
      OS << getLowerName();
    }
    void writePCHWrite(raw_ostream &OS) const {
      OS << "    AddVersionTuple(SA->get" << getUpperName() << "(), Record);\n";
    }
    void writeValue(raw_ostream &OS) const {
      OS << getLowerName() << "=\" << get" << getUpperName() << "() << \"";
    }
    void writeDump(raw_ostream &OS) const {
      OS << "    OS << \" \" << SA->get" << getUpperName() << "();\n";
    }
  };

  class ExprArgument : public SimpleArgument {
  public:
    ExprArgument(Record &Arg, StringRef Attr)
      : SimpleArgument(Arg, Attr, "Expr *")
    {}

    virtual void writeASTVisitorTraversal(raw_ostream &OS) const {
      OS << "  if (!"
         << "getDerived().TraverseStmt(A->get" << getUpperName() << "()))\n";
      OS << "    return false;\n";
    }

    void writeTemplateInstantiationArgs(raw_ostream &OS) const {
      OS << "tempInst" << getUpperName();
    }

    void writeTemplateInstantiation(raw_ostream &OS) const {
      OS << "      " << getType() << " tempInst" << getUpperName() << ";\n";
      OS << "      {\n";
      OS << "        EnterExpressionEvaluationContext "
         << "Unevaluated(S, Sema::Unevaluated);\n";
      OS << "        ExprResult " << "Result = S.SubstExpr("
         << "A->get" << getUpperName() << "(), TemplateArgs);\n";
      OS << "        tempInst" << getUpperName() << " = "
         << "Result.takeAs<Expr>();\n";
      OS << "      }\n";
    }

    void writeDump(raw_ostream &OS) const {
    }

    void writeDumpChildren(raw_ostream &OS) const {
      OS << "    lastChild();\n";
      OS << "    dumpStmt(SA->get" << getUpperName() << "());\n";
    }
    void writeHasChildren(raw_ostream &OS) const { OS << "true"; }
  };

  class VariadicExprArgument : public VariadicArgument {
  public:
    VariadicExprArgument(Record &Arg, StringRef Attr)
      : VariadicArgument(Arg, Attr, "Expr *")
    {}

    virtual void writeASTVisitorTraversal(raw_ostream &OS) const {
      OS << "  {\n";
      OS << "    " << getType() << " *I = A->" << getLowerName()
         << "_begin();\n";
      OS << "    " << getType() << " *E = A->" << getLowerName()
         << "_end();\n";
      OS << "    for (; I != E; ++I) {\n";
      OS << "      if (!getDerived().TraverseStmt(*I))\n";
      OS << "        return false;\n";
      OS << "    }\n";
      OS << "  }\n";
    }

    void writeTemplateInstantiationArgs(raw_ostream &OS) const {
      OS << "tempInst" << getUpperName() << ", "
         << "A->" << getLowerName() << "_size()";
    }

    void writeTemplateInstantiation(raw_ostream &OS) const {
      OS << "      " << getType() << " *tempInst" << getUpperName()
         << " = new (C, 16) " << getType()
         << "[A->" << getLowerName() << "_size()];\n";
      OS << "      {\n";
      OS << "        EnterExpressionEvaluationContext "
         << "Unevaluated(S, Sema::Unevaluated);\n";
      OS << "        " << getType() << " *TI = tempInst" << getUpperName()
         << ";\n";
      OS << "        " << getType() << " *I = A->" << getLowerName()
         << "_begin();\n";
      OS << "        " << getType() << " *E = A->" << getLowerName()
         << "_end();\n";
      OS << "        for (; I != E; ++I, ++TI) {\n";
      OS << "          ExprResult Result = S.SubstExpr(*I, TemplateArgs);\n";
      OS << "          *TI = Result.takeAs<Expr>();\n";
      OS << "        }\n";
      OS << "      }\n";
    }

    void writeDump(raw_ostream &OS) const {
    }

    void writeDumpChildren(raw_ostream &OS) const {
      OS << "    for (" << getAttrName() << "Attr::" << getLowerName()
         << "_iterator I = SA->" << getLowerName() << "_begin(), E = SA->"
         << getLowerName() << "_end(); I != E; ++I) {\n";
      OS << "      if (I + 1 == E)\n";
      OS << "        lastChild();\n";
      OS << "      dumpStmt(*I);\n";
      OS << "    }\n";
    }

    void writeHasChildren(raw_ostream &OS) const {
      OS << "SA->" << getLowerName() << "_begin() != "
         << "SA->" << getLowerName() << "_end()";
    }
  };

  class TypeArgument : public SimpleArgument {
  public:
    TypeArgument(Record &Arg, StringRef Attr)
      : SimpleArgument(Arg, Attr, "TypeSourceInfo *")
    {}

    void writeAccessors(raw_ostream &OS) const {
      OS << "  QualType get" << getUpperName() << "() const {\n";
      OS << "    return " << getLowerName() << "->getType();\n";
      OS << "  }";
      OS << "  " << getType() << " get" << getUpperName() << "Loc() const {\n";
      OS << "    return " << getLowerName() << ";\n";
      OS << "  }";
    }
    void writeTemplateInstantiationArgs(raw_ostream &OS) const {
      OS << "A->get" << getUpperName() << "Loc()";
    }
    void writePCHWrite(raw_ostream &OS) const {
      OS << "    " << WritePCHRecord(
          getType(), "SA->get" + std::string(getUpperName()) + "Loc()");
    }
  };
}

static Argument *createArgument(Record &Arg, StringRef Attr,
                                Record *Search = 0) {
  if (!Search)
    Search = &Arg;

  Argument *Ptr = 0;
  llvm::StringRef ArgName = Search->getName();

  if (ArgName == "AlignedArgument") Ptr = new AlignedArgument(Arg, Attr);
  else if (ArgName == "EnumArgument") Ptr = new EnumArgument(Arg, Attr);
  else if (ArgName == "ExprArgument") Ptr = new ExprArgument(Arg, Attr);
  else if (ArgName == "FunctionArgument")
    Ptr = new SimpleArgument(Arg, Attr, "FunctionDecl *");
  else if (ArgName == "IdentifierArgument")
    Ptr = new SimpleArgument(Arg, Attr, "IdentifierInfo *");
  else if (ArgName == "DefaultBoolArgument")
    Ptr = new DefaultSimpleArgument(Arg, Attr, "bool",
                                    Arg.getValueAsBit("Default"));
  else if (ArgName == "BoolArgument") Ptr = new SimpleArgument(Arg, Attr, 
                                                               "bool");
  else if (ArgName == "DefaultIntArgument")
    Ptr = new DefaultSimpleArgument(Arg, Attr, "int",
                                    Arg.getValueAsInt("Default"));
  else if (ArgName == "IntArgument") Ptr = new SimpleArgument(Arg, Attr, "int");
  else if (ArgName == "StringArgument") Ptr = new StringArgument(Arg, Attr);
  else if (ArgName == "TypeArgument") Ptr = new TypeArgument(Arg, Attr);
  else if (ArgName == "UnsignedArgument")
    Ptr = new SimpleArgument(Arg, Attr, "unsigned");
  else if (ArgName == "VariadicUnsignedArgument")
    Ptr = new VariadicArgument(Arg, Attr, "unsigned");
  else if (ArgName == "VariadicEnumArgument")
    Ptr = new VariadicEnumArgument(Arg, Attr);
  else if (ArgName == "VariadicExprArgument")
    Ptr = new VariadicExprArgument(Arg, Attr);
  else if (ArgName == "VersionArgument")
    Ptr = new VersionArgument(Arg, Attr);

  if (!Ptr) {
    // Search in reverse order so that the most-derived type is handled first.
    std::vector<Record*> Bases = Search->getSuperClasses();
    for (std::vector<Record*>::reverse_iterator i = Bases.rbegin(),
         e = Bases.rend(); i != e; ++i) {
      Ptr = createArgument(Arg, Attr, *i);
      if (Ptr)
        break;
    }
  }

  if (Ptr && Arg.getValueAsBit("Optional"))
    Ptr->setOptional(true);

  return Ptr;
}

static void writeAvailabilityValue(raw_ostream &OS) {
  OS << "\" << getPlatform()->getName();\n"
     << "  if (!getIntroduced().empty()) OS << \", introduced=\" << getIntroduced();\n"
     << "  if (!getDeprecated().empty()) OS << \", deprecated=\" << getDeprecated();\n"
     << "  if (!getObsoleted().empty()) OS << \", obsoleted=\" << getObsoleted();\n"
     << "  if (getUnavailable()) OS << \", unavailable\";\n"
     << "  OS << \"";
}

static void writeGetSpellingFunction(Record &R, raw_ostream &OS) {
  std::vector<FlattenedSpelling> Spellings = GetFlattenedSpellings(R);

  OS << "const char *" << R.getName() << "Attr::getSpelling() const {\n";
  if (Spellings.empty()) {
    OS << "  return \"(No spelling)\";\n}\n\n";
    return;
  }

  OS << "  switch (SpellingListIndex) {\n"
        "  default:\n"
        "    llvm_unreachable(\"Unknown attribute spelling!\");\n"
        "    return \"(No spelling)\";\n";

  for (unsigned I = 0; I < Spellings.size(); ++I)
    OS << "  case " << I << ":\n"
          "    return \"" << Spellings[I].name() << "\";\n";
  // End of the switch statement.
  OS << "  }\n";
  // End of the getSpelling function.
  OS << "}\n\n";
}

static void writePrettyPrintFunction(Record &R, std::vector<Argument*> &Args,
                                     raw_ostream &OS) {
  std::vector<FlattenedSpelling> Spellings = GetFlattenedSpellings(R);

  OS << "void " << R.getName() << "Attr::printPretty("
    << "raw_ostream &OS, const PrintingPolicy &Policy) const {\n";

  if (Spellings.size() == 0) {
    OS << "}\n\n";
    return;
  }

  OS <<
    "  switch (SpellingListIndex) {\n"
    "  default:\n"
    "    llvm_unreachable(\"Unknown attribute spelling!\");\n"
    "    break;\n";

  for (unsigned I = 0; I < Spellings.size(); ++ I) {
    llvm::SmallString<16> Prefix;
    llvm::SmallString<8> Suffix;
    // The actual spelling of the name and namespace (if applicable)
    // of an attribute without considering prefix and suffix.
    llvm::SmallString<64> Spelling;
    std::string Name = Spellings[I].name();
    std::string Variety = Spellings[I].variety();

    if (Variety == "GNU") {
      Prefix = " __attribute__((";
      Suffix = "))";
    } else if (Variety == "CXX11") {
      Prefix = " [[";
      Suffix = "]]";
      std::string Namespace = Spellings[I].nameSpace();
      if (Namespace != "") {
        Spelling += Namespace;
        Spelling += "::";
      }
    } else if (Variety == "Declspec") {
      Prefix = " __declspec(";
      Suffix = ")";
    } else if (Variety == "Keyword") {
      Prefix = " ";
      Suffix = "";
    } else {
      llvm_unreachable("Unknown attribute syntax variety!");
    }

    Spelling += Name;

    OS <<
      "  case " << I << " : {\n"
      "    OS << \"" + Prefix.str() + Spelling.str();

    if (Args.size()) OS << "(";
    if (Spelling == "availability") {
      writeAvailabilityValue(OS);
    } else {
      for (std::vector<Argument*>::const_iterator I = Args.begin(),
           E = Args.end(); I != E; ++ I) {
        if (I != Args.begin()) OS << ", ";
        (*I)->writeValue(OS);
      }
    }

    if (Args.size()) OS << ")";
    OS << Suffix.str() + "\";\n";

    OS <<
      "    break;\n"
      "  }\n";
  }

  // End of the switch statement.
  OS << "}\n";
  // End of the print function.
  OS << "}\n\n";
}

/// \brief Return the index of a spelling in a spelling list.
static unsigned
getSpellingListIndex(const std::vector<FlattenedSpelling> &SpellingList,
                     const FlattenedSpelling &Spelling) {
  assert(SpellingList.size() && "Spelling list is empty!");

  for (unsigned Index = 0; Index < SpellingList.size(); ++Index) {
    const FlattenedSpelling &S = SpellingList[Index];
    if (S.variety() != Spelling.variety())
      continue;
    if (S.nameSpace() != Spelling.nameSpace())
      continue;
    if (S.name() != Spelling.name())
      continue;

    return Index;
  }

  llvm_unreachable("Unknown spelling!");
}

static void writeAttrAccessorDefinition(Record &R, raw_ostream &OS) {
  std::vector<Record*> Accessors = R.getValueAsListOfDefs("Accessors");
  for (std::vector<Record*>::const_iterator I = Accessors.begin(),
       E = Accessors.end(); I != E; ++I) {
    Record *Accessor = *I;
    std::string Name = Accessor->getValueAsString("Name");
    std::vector<FlattenedSpelling> Spellings = 
      GetFlattenedSpellings(*Accessor);
    std::vector<FlattenedSpelling> SpellingList = GetFlattenedSpellings(R);
    assert(SpellingList.size() &&
           "Attribute with empty spelling list can't have accessors!");

    OS << "  bool " << Name << "() const { return SpellingListIndex == ";
    for (unsigned Index = 0; Index < Spellings.size(); ++Index) {
      OS << getSpellingListIndex(SpellingList, Spellings[Index]);
      if (Index != Spellings.size() -1)
        OS << " ||\n    SpellingListIndex == ";
      else
        OS << "; }\n";
    }
  }
}

static bool
SpellingNamesAreCommon(const std::vector<FlattenedSpelling>& Spellings) {
  assert(!Spellings.empty() && "An empty list of spellings was provided");
  std::string FirstName = NormalizeNameForSpellingComparison(
    Spellings.front().name());
  for (std::vector<FlattenedSpelling>::const_iterator
       I = llvm::next(Spellings.begin()), E = Spellings.end(); I != E; ++I) {
    std::string Name = NormalizeNameForSpellingComparison(I->name());
    if (Name != FirstName)
      return false;
  }
  return true;
}

typedef std::map<unsigned, std::string> SemanticSpellingMap;
static std::string
CreateSemanticSpellings(const std::vector<FlattenedSpelling> &Spellings,
                        SemanticSpellingMap &Map) {
  // The enumerants are automatically generated based on the variety,
  // namespace (if present) and name for each attribute spelling. However,
  // care is taken to avoid trampling on the reserved namespace due to
  // underscores.
  std::string Ret("  enum Spelling {\n");
  std::set<std::string> Uniques;
  unsigned Idx = 0;
  for (std::vector<FlattenedSpelling>::const_iterator I = Spellings.begin(),
        E = Spellings.end(); I != E; ++I, ++Idx) {
    const FlattenedSpelling &S = *I;
    std::string Variety = S.variety();
    std::string Spelling = S.name();
    std::string Namespace = S.nameSpace();
    std::string EnumName = "";

    EnumName += (Variety + "_");
    if (!Namespace.empty())
      EnumName += (NormalizeNameForSpellingComparison(Namespace).str() +
      "_");
    EnumName += NormalizeNameForSpellingComparison(Spelling);

    // Even if the name is not unique, this spelling index corresponds to a
    // particular enumerant name that we've calculated.
    Map[Idx] = EnumName;

    // Since we have been stripping underscores to avoid trampling on the
    // reserved namespace, we may have inadvertently created duplicate
    // enumerant names. These duplicates are not considered part of the
    // semantic spelling, and can be elided.
    if (Uniques.find(EnumName) != Uniques.end())
      continue;

    Uniques.insert(EnumName);
    if (I != Spellings.begin())
      Ret += ",\n";
    Ret += "    " + EnumName;
  }
  Ret += "\n  };\n\n";
  return Ret;
}

void WriteSemanticSpellingSwitch(const std::string &VarName,
                                 const SemanticSpellingMap &Map,
                                 raw_ostream &OS) {
  OS << "  switch (" << VarName << ") {\n    default: "
    << "llvm_unreachable(\"Unknown spelling list index\");\n";
  for (SemanticSpellingMap::const_iterator I = Map.begin(), E = Map.end();
       I != E; ++I)
       OS << "    case " << I->first << ": return " << I->second << ";\n";
  OS << "  }\n";
}

// Emits the LateParsed property for attributes.
static void emitClangAttrLateParsedList(RecordKeeper &Records, raw_ostream &OS) {
  OS << "#if defined(CLANG_ATTR_LATE_PARSED_LIST)\n";
  std::vector<Record*> Attrs = Records.getAllDerivedDefinitions("Attr");

  for (std::vector<Record*>::iterator I = Attrs.begin(), E = Attrs.end();
       I != E; ++I) {
    Record &Attr = **I;

    bool LateParsed = Attr.getValueAsBit("LateParsed");

    if (LateParsed) {
      std::vector<FlattenedSpelling> Spellings = GetFlattenedSpellings(Attr);

      // FIXME: Handle non-GNU attributes
      for (std::vector<FlattenedSpelling>::const_iterator
           I = Spellings.begin(), E = Spellings.end(); I != E; ++I) {
        if (I->variety() != "GNU")
          continue;
        OS << ".Case(\"" << I->name() << "\", " << LateParsed << ")\n";
      }
    }
  }
  OS << "#endif // CLANG_ATTR_LATE_PARSED_LIST\n\n";
}

/// \brief Emits the first-argument-is-type property for attributes.
static void emitClangAttrTypeArgList(RecordKeeper &Records, raw_ostream &OS) {
  OS << "#if defined(CLANG_ATTR_TYPE_ARG_LIST)\n";
  std::vector<Record *> Attrs = Records.getAllDerivedDefinitions("Attr");

  for (std::vector<Record *>::iterator I = Attrs.begin(), E = Attrs.end();
       I != E; ++I) {
    Record &Attr = **I;

    // Determine whether the first argument is a type.
    std::vector<Record *> Args = Attr.getValueAsListOfDefs("Args");
    if (Args.empty())
      continue;

    if (Args[0]->getSuperClasses().back()->getName() != "TypeArgument")
      continue;

    // All these spellings take a single type argument.
    std::vector<FlattenedSpelling> Spellings = GetFlattenedSpellings(Attr);
    std::set<std::string> Emitted;
    for (std::vector<FlattenedSpelling>::const_iterator I = Spellings.begin(),
         E = Spellings.end(); I != E; ++I) {
      if (Emitted.insert(I->name()).second)
        OS << ".Case(\"" << I->name() << "\", " << "true" << ")\n";
    }
  }
  OS << "#endif // CLANG_ATTR_TYPE_ARG_LIST\n\n";
}

/// \brief Emits the parse-arguments-in-unevaluated-context property for
/// attributes.
static void emitClangAttrArgContextList(RecordKeeper &Records, raw_ostream &OS) {
  OS << "#if defined(CLANG_ATTR_ARG_CONTEXT_LIST)\n";
  ParsedAttrMap Attrs = getParsedAttrList(Records);
  for (ParsedAttrMap::const_iterator I = Attrs.begin(), E = Attrs.end();
       I != E; ++I) {
    const Record &Attr = *I->second;

    if (!Attr.getValueAsBit("ParseArgumentsAsUnevaluated"))
      continue;

    // All these spellings take are parsed unevaluated.
    std::vector<FlattenedSpelling> Spellings = GetFlattenedSpellings(Attr);
    std::set<std::string> Emitted;
    for (std::vector<FlattenedSpelling>::const_iterator I = Spellings.begin(),
         E = Spellings.end(); I != E; ++I) {
      if (Emitted.insert(I->name()).second)
        OS << ".Case(\"" << I->name() << "\", " << "true" << ")\n";
    }
  }
  OS << "#endif // CLANG_ATTR_ARG_CONTEXT_LIST\n\n";
}

static bool isIdentifierArgument(Record *Arg) {
  return !Arg->getSuperClasses().empty() &&
    llvm::StringSwitch<bool>(Arg->getSuperClasses().back()->getName())
    .Case("IdentifierArgument", true)
    .Case("EnumArgument", true)
    .Default(false);
}

// Emits the first-argument-is-identifier property for attributes.
static void emitClangAttrIdentifierArgList(RecordKeeper &Records, raw_ostream &OS) {
  OS << "#if defined(CLANG_ATTR_IDENTIFIER_ARG_LIST)\n";
  std::vector<Record*> Attrs = Records.getAllDerivedDefinitions("Attr");

  for (std::vector<Record*>::iterator I = Attrs.begin(), E = Attrs.end();
       I != E; ++I) {
    Record &Attr = **I;

    // Determine whether the first argument is an identifier.
    std::vector<Record *> Args = Attr.getValueAsListOfDefs("Args");
    if (Args.empty() || !isIdentifierArgument(Args[0]))
      continue;

    // All these spellings take an identifier argument.
    std::vector<FlattenedSpelling> Spellings = GetFlattenedSpellings(Attr);
    std::set<std::string> Emitted;
    for (std::vector<FlattenedSpelling>::const_iterator I = Spellings.begin(),
         E = Spellings.end(); I != E; ++I) {
      if (Emitted.insert(I->name()).second)
        OS << ".Case(\"" << I->name() << "\", " << "true" << ")\n";
    }
  }
  OS << "#endif // CLANG_ATTR_IDENTIFIER_ARG_LIST\n\n";
}

namespace clang {

// Emits the class definitions for attributes.
void EmitClangAttrClass(RecordKeeper &Records, raw_ostream &OS) {
  emitSourceFileHeader("Attribute classes' definitions", OS);

  OS << "#ifndef LLVM_CLANG_ATTR_CLASSES_INC\n";
  OS << "#define LLVM_CLANG_ATTR_CLASSES_INC\n\n";

  std::vector<Record*> Attrs = Records.getAllDerivedDefinitions("Attr");

  for (std::vector<Record*>::iterator i = Attrs.begin(), e = Attrs.end();
       i != e; ++i) {
    Record &R = **i;
    
    if (!R.getValueAsBit("ASTNode"))
      continue;
    
    const std::vector<Record *> Supers = R.getSuperClasses();
    assert(!Supers.empty() && "Forgot to specify a superclass for the attr");
    std::string SuperName;
    for (std::vector<Record *>::const_reverse_iterator I = Supers.rbegin(),
         E = Supers.rend(); I != E; ++I) {
      const Record &R = **I;
      if (R.getName() != "TargetSpecificAttr" && SuperName.empty())
        SuperName = R.getName();
    }

    OS << "class " << R.getName() << "Attr : public " << SuperName << " {\n";

    std::vector<Record*> ArgRecords = R.getValueAsListOfDefs("Args");
    std::vector<Argument*> Args;
    std::vector<Argument*>::iterator ai, ae;
    Args.reserve(ArgRecords.size());

    for (std::vector<Record*>::iterator ri = ArgRecords.begin(),
                                        re = ArgRecords.end();
         ri != re; ++ri) {
      Record &ArgRecord = **ri;
      Argument *Arg = createArgument(ArgRecord, R.getName());
      assert(Arg);
      Args.push_back(Arg);

      Arg->writeDeclarations(OS);
      OS << "\n\n";
    }

    ae = Args.end();

    OS << "\npublic:\n";

    std::vector<FlattenedSpelling> Spellings = GetFlattenedSpellings(R);

    // If there are zero or one spellings, all spelling-related functionality
    // can be elided. If all of the spellings share the same name, the spelling
    // functionality can also be elided.
    bool ElideSpelling = (Spellings.size() <= 1) ||
                         SpellingNamesAreCommon(Spellings);

    // This maps spelling index values to semantic Spelling enumerants.
    SemanticSpellingMap SemanticToSyntacticMap;

    if (!ElideSpelling)
      OS << CreateSemanticSpellings(Spellings, SemanticToSyntacticMap);

    OS << "  static " << R.getName() << "Attr *CreateImplicit(";
    OS << "ASTContext &Ctx";
    if (!ElideSpelling)
      OS << ", Spelling S";
    for (ai = Args.begin(); ai != ae; ++ai) {
      OS << ", ";
      (*ai)->writeCtorParameters(OS);
    }
    OS << ", SourceRange Loc = SourceRange()";
    OS << ") {\n";
    OS << "    " << R.getName() << "Attr *A = new (Ctx) " << R.getName();
    OS << "Attr(Loc, Ctx, ";
    for (ai = Args.begin(); ai != ae; ++ai) {
      (*ai)->writeImplicitCtorArgs(OS);
      OS << ", ";
    }
    OS << (ElideSpelling ? "0" : "S") << ");\n";
    OS << "    A->setImplicit(true);\n";
    OS << "    return A;\n  }\n\n";

    OS << "  " << R.getName() << "Attr(SourceRange R, ASTContext &Ctx\n";
    
    bool HasOpt = false;
    for (ai = Args.begin(); ai != ae; ++ai) {
      OS << "              , ";
      (*ai)->writeCtorParameters(OS);
      OS << "\n";
      if ((*ai)->isOptional())
        HasOpt = true;
    }

    OS << "              , ";
    OS << "unsigned SI\n";

    OS << "             )\n";
    OS << "    : " << SuperName << "(attr::" << R.getName() << ", R, SI)\n";

    for (ai = Args.begin(); ai != ae; ++ai) {
      OS << "              , ";
      (*ai)->writeCtorInitializers(OS);
      OS << "\n";
    }

    OS << "  {\n";
  
    for (ai = Args.begin(); ai != ae; ++ai) {
      (*ai)->writeCtorBody(OS);
      OS << "\n";
    }
    OS << "  }\n\n";

    // If there are optional arguments, write out a constructor that elides the
    // optional arguments as well.
    if (HasOpt) {
      OS << "  " << R.getName() << "Attr(SourceRange R, ASTContext &Ctx\n";
      for (ai = Args.begin(); ai != ae; ++ai) {
        if (!(*ai)->isOptional()) {
          OS << "              , ";
          (*ai)->writeCtorParameters(OS);
          OS << "\n";
        }
      }

      OS << "              , ";
      OS << "unsigned SI\n";

      OS << "             )\n";
      OS << "    : " << SuperName << "(attr::" << R.getName() << ", R, SI)\n";

      for (ai = Args.begin(); ai != ae; ++ai) {
        OS << "              , ";
        (*ai)->writeCtorDefaultInitializers(OS);
        OS << "\n";
      }

      OS << "  {\n";
  
      for (ai = Args.begin(); ai != ae; ++ai) {
        if (!(*ai)->isOptional()) {
          (*ai)->writeCtorBody(OS);
          OS << "\n";
        }
      }
      OS << "  }\n\n";
    }

    OS << "  virtual " << R.getName() << "Attr *clone (ASTContext &C) const;\n";
    OS << "  virtual void printPretty(raw_ostream &OS,\n"
       << "                           const PrintingPolicy &Policy) const;\n";
    OS << "  virtual const char *getSpelling() const;\n";
    
    if (!ElideSpelling) {
      assert(!SemanticToSyntacticMap.empty() && "Empty semantic mapping list");
      OS << "  Spelling getSemanticSpelling() const {\n";
      WriteSemanticSpellingSwitch("SpellingListIndex", SemanticToSyntacticMap,
                                  OS);
      OS << "  }\n";
    }

    writeAttrAccessorDefinition(R, OS);

    for (ai = Args.begin(); ai != ae; ++ai) {
      (*ai)->writeAccessors(OS);
      OS << "\n\n";

      if ((*ai)->isEnumArg()) {
        EnumArgument *EA = (EnumArgument *)*ai;
        EA->writeConversion(OS);
      } else if ((*ai)->isVariadicEnumArg()) {
        VariadicEnumArgument *VEA = (VariadicEnumArgument *)*ai;
        VEA->writeConversion(OS);
      }
    }

    OS << R.getValueAsString("AdditionalMembers");
    OS << "\n\n";

    OS << "  static bool classof(const Attr *A) { return A->getKind() == "
       << "attr::" << R.getName() << "; }\n";

    bool LateParsed = R.getValueAsBit("LateParsed");
    OS << "  virtual bool isLateParsed() const { return "
       << LateParsed << "; }\n";

    if (R.getValueAsBit("DuplicatesAllowedWhileMerging"))
      OS << "  virtual bool duplicatesAllowed() const { return true; }\n\n";

    OS << "};\n\n";
  }

  OS << "#endif\n";
}

// Emits the class method definitions for attributes.
void EmitClangAttrImpl(RecordKeeper &Records, raw_ostream &OS) {
  emitSourceFileHeader("Attribute classes' member function definitions", OS);

  std::vector<Record*> Attrs = Records.getAllDerivedDefinitions("Attr");
  std::vector<Record*>::iterator i = Attrs.begin(), e = Attrs.end(), ri, re;
  std::vector<Argument*>::iterator ai, ae;

  for (; i != e; ++i) {
    Record &R = **i;
    
    if (!R.getValueAsBit("ASTNode"))
      continue;
    
    std::vector<Record*> ArgRecords = R.getValueAsListOfDefs("Args");
    std::vector<Argument*> Args;
    for (ri = ArgRecords.begin(), re = ArgRecords.end(); ri != re; ++ri)
      Args.push_back(createArgument(**ri, R.getName()));

    for (ai = Args.begin(), ae = Args.end(); ai != ae; ++ai)
      (*ai)->writeAccessorDefinitions(OS);

    OS << R.getName() << "Attr *" << R.getName()
       << "Attr::clone(ASTContext &C) const {\n";
    OS << "  return new (C) " << R.getName() << "Attr(getLocation(), C";
    for (ai = Args.begin(); ai != ae; ++ai) {
      OS << ", ";
      (*ai)->writeCloneArgs(OS);
    }
    OS << ", getSpellingListIndex());\n}\n\n";

    writePrettyPrintFunction(R, Args, OS);
    writeGetSpellingFunction(R, OS);
  }
}

} // end namespace clang

static void EmitAttrList(raw_ostream &OS, StringRef Class,
                         const std::vector<Record*> &AttrList) {
  std::vector<Record*>::const_iterator i = AttrList.begin(), e = AttrList.end();

  if (i != e) {
    // Move the end iterator back to emit the last attribute.
    for(--e; i != e; ++i) {
      if (!(*i)->getValueAsBit("ASTNode"))
        continue;
      
      OS << Class << "(" << (*i)->getName() << ")\n";
    }
    
    OS << "LAST_" << Class << "(" << (*i)->getName() << ")\n\n";
  }
}

namespace clang {

// Emits the enumeration list for attributes.
void EmitClangAttrList(RecordKeeper &Records, raw_ostream &OS) {
  emitSourceFileHeader("List of all attributes that Clang recognizes", OS);

  OS << "#ifndef LAST_ATTR\n";
  OS << "#define LAST_ATTR(NAME) ATTR(NAME)\n";
  OS << "#endif\n\n";

  OS << "#ifndef INHERITABLE_ATTR\n";
  OS << "#define INHERITABLE_ATTR(NAME) ATTR(NAME)\n";
  OS << "#endif\n\n";

  OS << "#ifndef LAST_INHERITABLE_ATTR\n";
  OS << "#define LAST_INHERITABLE_ATTR(NAME) INHERITABLE_ATTR(NAME)\n";
  OS << "#endif\n\n";

  OS << "#ifndef INHERITABLE_PARAM_ATTR\n";
  OS << "#define INHERITABLE_PARAM_ATTR(NAME) ATTR(NAME)\n";
  OS << "#endif\n\n";

  OS << "#ifndef LAST_INHERITABLE_PARAM_ATTR\n";
  OS << "#define LAST_INHERITABLE_PARAM_ATTR(NAME)"
        " INHERITABLE_PARAM_ATTR(NAME)\n";
  OS << "#endif\n\n";

  Record *InhClass = Records.getClass("InheritableAttr");
  Record *InhParamClass = Records.getClass("InheritableParamAttr");
  std::vector<Record*> Attrs = Records.getAllDerivedDefinitions("Attr"),
                       NonInhAttrs, InhAttrs, InhParamAttrs;
  for (std::vector<Record*>::iterator i = Attrs.begin(), e = Attrs.end();
       i != e; ++i) {
    if (!(*i)->getValueAsBit("ASTNode"))
      continue;
    
    if ((*i)->isSubClassOf(InhParamClass))
      InhParamAttrs.push_back(*i);
    else if ((*i)->isSubClassOf(InhClass))
      InhAttrs.push_back(*i);
    else
      NonInhAttrs.push_back(*i);
  }

  EmitAttrList(OS, "INHERITABLE_PARAM_ATTR", InhParamAttrs);
  EmitAttrList(OS, "INHERITABLE_ATTR", InhAttrs);
  EmitAttrList(OS, "ATTR", NonInhAttrs);

  OS << "#undef LAST_ATTR\n";
  OS << "#undef INHERITABLE_ATTR\n";
  OS << "#undef LAST_INHERITABLE_ATTR\n";
  OS << "#undef LAST_INHERITABLE_PARAM_ATTR\n";
  OS << "#undef ATTR\n";
}

// Emits the code to read an attribute from a precompiled header.
void EmitClangAttrPCHRead(RecordKeeper &Records, raw_ostream &OS) {
  emitSourceFileHeader("Attribute deserialization code", OS);

  Record *InhClass = Records.getClass("InheritableAttr");
  std::vector<Record*> Attrs = Records.getAllDerivedDefinitions("Attr"),
                       ArgRecords;
  std::vector<Record*>::iterator i = Attrs.begin(), e = Attrs.end(), ai, ae;
  std::vector<Argument*> Args;
  std::vector<Argument*>::iterator ri, re;

  OS << "  switch (Kind) {\n";
  OS << "  default:\n";
  OS << "    assert(0 && \"Unknown attribute!\");\n";
  OS << "    break;\n";
  for (; i != e; ++i) {
    Record &R = **i;
    if (!R.getValueAsBit("ASTNode"))
      continue;
    
    OS << "  case attr::" << R.getName() << ": {\n";
    if (R.isSubClassOf(InhClass))
      OS << "    bool isInherited = Record[Idx++];\n";
    OS << "    bool isImplicit = Record[Idx++];\n";
    OS << "    unsigned Spelling = Record[Idx++];\n";
    ArgRecords = R.getValueAsListOfDefs("Args");
    Args.clear();
    for (ai = ArgRecords.begin(), ae = ArgRecords.end(); ai != ae; ++ai) {
      Argument *A = createArgument(**ai, R.getName());
      Args.push_back(A);
      A->writePCHReadDecls(OS);
    }
    OS << "    New = new (Context) " << R.getName() << "Attr(Range, Context";
    for (ri = Args.begin(), re = Args.end(); ri != re; ++ri) {
      OS << ", ";
      (*ri)->writePCHReadArgs(OS);
    }
    OS << ", Spelling);\n";
    if (R.isSubClassOf(InhClass))
      OS << "    cast<InheritableAttr>(New)->setInherited(isInherited);\n";
    OS << "    New->setImplicit(isImplicit);\n";
    OS << "    break;\n";
    OS << "  }\n";
  }
  OS << "  }\n";
}

// Emits the code to write an attribute to a precompiled header.
void EmitClangAttrPCHWrite(RecordKeeper &Records, raw_ostream &OS) {
  emitSourceFileHeader("Attribute serialization code", OS);

  Record *InhClass = Records.getClass("InheritableAttr");
  std::vector<Record*> Attrs = Records.getAllDerivedDefinitions("Attr"), Args;
  std::vector<Record*>::iterator i = Attrs.begin(), e = Attrs.end(), ai, ae;

  OS << "  switch (A->getKind()) {\n";
  OS << "  default:\n";
  OS << "    llvm_unreachable(\"Unknown attribute kind!\");\n";
  OS << "    break;\n";
  for (; i != e; ++i) {
    Record &R = **i;
    if (!R.getValueAsBit("ASTNode"))
      continue;
    OS << "  case attr::" << R.getName() << ": {\n";
    Args = R.getValueAsListOfDefs("Args");
    if (R.isSubClassOf(InhClass) || !Args.empty())
      OS << "    const " << R.getName() << "Attr *SA = cast<" << R.getName()
         << "Attr>(A);\n";
    if (R.isSubClassOf(InhClass))
      OS << "    Record.push_back(SA->isInherited());\n";
    OS << "    Record.push_back(A->isImplicit());\n";
    OS << "    Record.push_back(A->getSpellingListIndex());\n";

    for (ai = Args.begin(), ae = Args.end(); ai != ae; ++ai)
      createArgument(**ai, R.getName())->writePCHWrite(OS);
    OS << "    break;\n";
    OS << "  }\n";
  }
  OS << "  }\n";
}

// Emits the list of spellings for attributes.
void EmitClangAttrSpellingList(RecordKeeper &Records, raw_ostream &OS) {
  emitSourceFileHeader("llvm::StringSwitch code to match attributes based on "
                       "the target triple, T", OS);

  std::vector<Record*> Attrs = Records.getAllDerivedDefinitions("Attr");
  
  for (std::vector<Record*>::iterator I = Attrs.begin(), E = Attrs.end();
       I != E; ++I) {
    Record &Attr = **I;

    // It is assumed that there will be an llvm::Triple object named T within
    // scope that can be used to determine whether the attribute exists in
    // a given target.
    std::string Test;
    if (Attr.isSubClassOf("TargetSpecificAttr")) {
      const Record *R = Attr.getValueAsDef("Target");
      std::vector<std::string> Arches = R->getValueAsListOfStrings("Arches");

      Test += "(";
      for (std::vector<std::string>::const_iterator AI = Arches.begin(),
           AE = Arches.end(); AI != AE; ++AI) {
        std::string Part = *AI;
        Test += "T.getArch() == llvm::Triple::" + Part;
        if (AI + 1 != AE)
          Test += " || ";
      }
      Test += ")";

      std::vector<std::string> OSes;
      if (!R->isValueUnset("OSes")) {
        Test += " && (";
        std::vector<std::string> OSes = R->getValueAsListOfStrings("OSes");
        for (std::vector<std::string>::const_iterator AI = OSes.begin(),
             AE = OSes.end(); AI != AE; ++AI) {
          std::string Part = *AI;

          Test += "T.getOS() == llvm::Triple::" + Part;
          if (AI + 1 != AE)
            Test += " || ";
        }
        Test += ")";
      }
    } else
      Test = "true";

    std::vector<FlattenedSpelling> Spellings = GetFlattenedSpellings(Attr);
    for (std::vector<FlattenedSpelling>::const_iterator I = Spellings.begin(),
         E = Spellings.end(); I != E; ++I)
      OS << ".Case(\"" << I->name() << "\", " << Test << ")\n";
  }

}

void EmitClangAttrSpellingListIndex(RecordKeeper &Records, raw_ostream &OS) {
  emitSourceFileHeader("Code to translate different attribute spellings "
                       "into internal identifiers", OS);

  OS <<
    "  switch (AttrKind) {\n"
    "  default:\n"
    "    llvm_unreachable(\"Unknown attribute kind!\");\n"
    "    break;\n";

  ParsedAttrMap Attrs = getParsedAttrList(Records);
  for (ParsedAttrMap::const_iterator I = Attrs.begin(), E = Attrs.end();
       I != E; ++I) {
    Record &R = *I->second;
    std::vector<FlattenedSpelling> Spellings = GetFlattenedSpellings(R);
    OS << "  case AT_" << I->first << ": {\n";
    for (unsigned I = 0; I < Spellings.size(); ++ I) {
      OS << "    if (Name == \""
        << Spellings[I].name() << "\" && "
        << "SyntaxUsed == "
        << StringSwitch<unsigned>(Spellings[I].variety())
          .Case("GNU", 0)
          .Case("CXX11", 1)
          .Case("Declspec", 2)
          .Case("Keyword", 3)
          .Default(0)
        << " && Scope == \"" << Spellings[I].nameSpace() << "\")\n"
        << "        return " << I << ";\n";
    }

    OS << "    break;\n";
    OS << "  }\n";
  }

  OS << "  }\n";
  OS << "  return 0;\n";
}

// Emits code used by RecursiveASTVisitor to visit attributes
void EmitClangAttrASTVisitor(RecordKeeper &Records, raw_ostream &OS) {
  emitSourceFileHeader("Used by RecursiveASTVisitor to visit attributes.", OS);

  std::vector<Record*> Attrs = Records.getAllDerivedDefinitions("Attr");

  // Write method declarations for Traverse* methods.
  // We emit this here because we only generate methods for attributes that
  // are declared as ASTNodes.
  OS << "#ifdef ATTR_VISITOR_DECLS_ONLY\n\n";
  for (std::vector<Record*>::iterator I = Attrs.begin(), E = Attrs.end();
       I != E; ++I) {
    Record &R = **I;
    if (!R.getValueAsBit("ASTNode"))
      continue;
    OS << "  bool Traverse"
       << R.getName() << "Attr(" << R.getName() << "Attr *A);\n";
    OS << "  bool Visit"
       << R.getName() << "Attr(" << R.getName() << "Attr *A) {\n"
       << "    return true; \n"
       << "  };\n";
  }
  OS << "\n#else // ATTR_VISITOR_DECLS_ONLY\n\n";

  // Write individual Traverse* methods for each attribute class.
  for (std::vector<Record*>::iterator I = Attrs.begin(), E = Attrs.end();
       I != E; ++I) {
    Record &R = **I;
    if (!R.getValueAsBit("ASTNode"))
      continue;

    OS << "template <typename Derived>\n"
       << "bool VISITORCLASS<Derived>::Traverse"
       << R.getName() << "Attr(" << R.getName() << "Attr *A) {\n"
       << "  if (!getDerived().VisitAttr(A))\n"
       << "    return false;\n"
       << "  if (!getDerived().Visit" << R.getName() << "Attr(A))\n"
       << "    return false;\n";

    std::vector<Record*> ArgRecords = R.getValueAsListOfDefs("Args");
    for (std::vector<Record*>::iterator ri = ArgRecords.begin(),
                                        re = ArgRecords.end();
         ri != re; ++ri) {
      Record &ArgRecord = **ri;
      Argument *Arg = createArgument(ArgRecord, R.getName());
      assert(Arg);
      Arg->writeASTVisitorTraversal(OS);
    }

    OS << "  return true;\n";
    OS << "}\n\n";
  }

  // Write generic Traverse routine
  OS << "template <typename Derived>\n"
     << "bool VISITORCLASS<Derived>::TraverseAttr(Attr *A) {\n"
     << "  if (!A)\n"
     << "    return true;\n"
     << "\n"
     << "  switch (A->getKind()) {\n"
     << "    default:\n"
     << "      return true;\n";

  for (std::vector<Record*>::iterator I = Attrs.begin(), E = Attrs.end();
       I != E; ++I) {
    Record &R = **I;
    if (!R.getValueAsBit("ASTNode"))
      continue;

    OS << "    case attr::" << R.getName() << ":\n"
       << "      return getDerived().Traverse" << R.getName() << "Attr("
       << "cast<" << R.getName() << "Attr>(A));\n";
  }
  OS << "  }\n";  // end case
  OS << "}\n";  // end function
  OS << "#endif  // ATTR_VISITOR_DECLS_ONLY\n";
}

// Emits code to instantiate dependent attributes on templates.
void EmitClangAttrTemplateInstantiate(RecordKeeper &Records, raw_ostream &OS) {
  emitSourceFileHeader("Template instantiation code for attributes", OS);

  std::vector<Record*> Attrs = Records.getAllDerivedDefinitions("Attr");

  OS << "namespace clang {\n"
     << "namespace sema {\n\n"
     << "Attr *instantiateTemplateAttribute(const Attr *At, ASTContext &C, "
     << "Sema &S,\n"
     << "        const MultiLevelTemplateArgumentList &TemplateArgs) {\n"
     << "  switch (At->getKind()) {\n"
     << "    default:\n"
     << "      break;\n";

  for (std::vector<Record*>::iterator I = Attrs.begin(), E = Attrs.end();
       I != E; ++I) {
    Record &R = **I;
    if (!R.getValueAsBit("ASTNode"))
      continue;

    OS << "    case attr::" << R.getName() << ": {\n";
    bool ShouldClone = R.getValueAsBit("Clone");

    if (!ShouldClone) {
      OS << "      return NULL;\n";
      OS << "    }\n";
      continue;
    }

    OS << "      const " << R.getName() << "Attr *A = cast<"
       << R.getName() << "Attr>(At);\n";
    bool TDependent = R.getValueAsBit("TemplateDependent");

    if (!TDependent) {
      OS << "      return A->clone(C);\n";
      OS << "    }\n";
      continue;
    }

    std::vector<Record*> ArgRecords = R.getValueAsListOfDefs("Args");
    std::vector<Argument*> Args;
    std::vector<Argument*>::iterator ai, ae;
    Args.reserve(ArgRecords.size());

    for (std::vector<Record*>::iterator ri = ArgRecords.begin(),
                                        re = ArgRecords.end();
         ri != re; ++ri) {
      Record &ArgRecord = **ri;
      Argument *Arg = createArgument(ArgRecord, R.getName());
      assert(Arg);
      Args.push_back(Arg);
    }
    ae = Args.end();

    for (ai = Args.begin(); ai != ae; ++ai) {
      (*ai)->writeTemplateInstantiation(OS);
    }
    OS << "      return new (C) " << R.getName() << "Attr(A->getLocation(), C";
    for (ai = Args.begin(); ai != ae; ++ai) {
      OS << ", ";
      (*ai)->writeTemplateInstantiationArgs(OS);
    }
    OS << ", A->getSpellingListIndex());\n    }\n";
  }
  OS << "  } // end switch\n"
     << "  llvm_unreachable(\"Unknown attribute!\");\n"
     << "  return 0;\n"
     << "}\n\n"
     << "} // end namespace sema\n"
     << "} // end namespace clang\n";
}

// Emits the list of parsed attributes.
void EmitClangAttrParsedAttrList(RecordKeeper &Records, raw_ostream &OS) {
  emitSourceFileHeader("List of all attributes that Clang recognizes", OS);

  OS << "#ifndef PARSED_ATTR\n";
  OS << "#define PARSED_ATTR(NAME) NAME\n";
  OS << "#endif\n\n";
  
  ParsedAttrMap Names = getParsedAttrList(Records);
  for (ParsedAttrMap::iterator I = Names.begin(), E = Names.end(); I != E;
       ++I) {
    OS << "PARSED_ATTR(" << I->first << ")\n";
  }
}

static void emitArgInfo(const Record &R, std::stringstream &OS) {
  // This function will count the number of arguments specified for the
  // attribute and emit the number of required arguments followed by the
  // number of optional arguments.
  std::vector<Record *> Args = R.getValueAsListOfDefs("Args");
  unsigned ArgCount = 0, OptCount = 0;
  for (std::vector<Record *>::const_iterator I = Args.begin(), E = Args.end();
       I != E; ++I) {
    const Record &Arg = **I;
    Arg.getValueAsBit("Optional") ? ++OptCount : ++ArgCount;
  }
  OS << ArgCount << ", " << OptCount;
}

static void GenerateDefaultAppertainsTo(raw_ostream &OS) {
  OS << "static bool defaultAppertainsTo(Sema &, const AttributeList &,";
  OS << "const Decl *) {\n";
  OS << "  return true;\n";
  OS << "}\n\n";
}

static std::string CalculateDiagnostic(const Record &S) {
  // If the SubjectList object has a custom diagnostic associated with it,
  // return that directly.
  std::string CustomDiag = S.getValueAsString("CustomDiag");
  if (!CustomDiag.empty())
    return CustomDiag;

  // Given the list of subjects, determine what diagnostic best fits.
  enum {
    Func = 1U << 0,
    Var = 1U << 1,
    ObjCMethod = 1U << 2,
    Param = 1U << 3,
    Class = 1U << 4,
    GenericRecord = 1U << 5,
    Type = 1U << 6,
    ObjCIVar = 1U << 7,
    ObjCProp = 1U << 8,
    ObjCInterface = 1U << 9,
    Block = 1U << 10,
    Namespace = 1U << 11,
    FuncTemplate = 1U << 12,
    Field = 1U << 13,
    CXXMethod = 1U << 14,
    ObjCProtocol = 1U << 15
  };
  uint32_t SubMask = 0;

  std::vector<Record *> Subjects = S.getValueAsListOfDefs("Subjects");
  for (std::vector<Record *>::const_iterator I = Subjects.begin(),
       E = Subjects.end(); I != E; ++I) {
    const Record &R = (**I);
    std::string Name;

    if (R.isSubClassOf("SubsetSubject")) {
      PrintError(R.getLoc(), "SubsetSubjects should use a custom diagnostic");
      // As a fallback, look through the SubsetSubject to see what its base
      // type is, and use that. This needs to be updated if SubsetSubjects
      // are allowed within other SubsetSubjects.
      Name = R.getValueAsDef("Base")->getName();
    } else
      Name = R.getName();

    uint32_t V = StringSwitch<uint32_t>(Name)
                   .Case("Function", Func)
                   .Case("Var", Var)
                   .Case("ObjCMethod", ObjCMethod)
                   .Case("ParmVar", Param)
                   .Case("TypedefName", Type)
                   .Case("ObjCIvar", ObjCIVar)
                   .Case("ObjCProperty", ObjCProp)
                   .Case("Record", GenericRecord)
                   .Case("ObjCInterface", ObjCInterface)
                   .Case("ObjCProtocol", ObjCProtocol)
                   .Case("Block", Block)
                   .Case("CXXRecord", Class)
                   .Case("Namespace", Namespace)
                   .Case("FunctionTemplate", FuncTemplate)
                   .Case("Field", Field)
                   .Case("CXXMethod", CXXMethod)
                   .Default(0);
    if (!V) {
      // Something wasn't in our mapping, so be helpful and let the developer
      // know about it.
      PrintFatalError((*I)->getLoc(), "Unknown subject type: " +
                      (*I)->getName());
      return "";
    }

    SubMask |= V;
  }

  switch (SubMask) {
    // For the simple cases where there's only a single entry in the mask, we
    // don't have to resort to bit fiddling.
    case Func:  return "ExpectedFunction";
    case Var:   return "ExpectedVariable";
    case Param: return "ExpectedParameter";
    case Class: return "ExpectedClass";
    case CXXMethod:
      // FIXME: Currently, this maps to ExpectedMethod based on existing code,
      // but should map to something a bit more accurate at some point.
    case ObjCMethod:  return "ExpectedMethod";
    case Type:  return "ExpectedType";
    case ObjCInterface: return "ExpectedObjectiveCInterface";
    case ObjCProtocol: return "ExpectedObjectiveCProtocol";
    
    // "GenericRecord" means struct, union or class; check the language options
    // and if not compiling for C++, strip off the class part. Note that this
    // relies on the fact that the context for this declares "Sema &S".
    case GenericRecord:
      return "(S.getLangOpts().CPlusPlus ? ExpectedStructOrUnionOrClass : "
                                           "ExpectedStructOrUnion)";
    case Func | ObjCMethod | Block: return "ExpectedFunctionMethodOrBlock";
    case Func | ObjCMethod | Class: return "ExpectedFunctionMethodOrClass";
    case Func | Param:
    case Func | ObjCMethod | Param: return "ExpectedFunctionMethodOrParameter";
    case Func | FuncTemplate:
    case Func | ObjCMethod: return "ExpectedFunctionOrMethod";
    case Func | Var: return "ExpectedVariableOrFunction";

    // If not compiling for C++, the class portion does not apply.
    case Func | Var | Class:
      return "(S.getLangOpts().CPlusPlus ? ExpectedFunctionVariableOrClass : "
                                           "ExpectedVariableOrFunction)";

    case ObjCMethod | ObjCProp: return "ExpectedMethodOrProperty";
    case Field | Var: return "ExpectedFieldOrGlobalVar";
  }

  PrintFatalError(S.getLoc(),
                  "Could not deduce diagnostic argument for Attr subjects");

  return "";
}

static std::string GetSubjectWithSuffix(const Record *R) {
  std::string B = R->getName();
  if (B == "DeclBase")
    return "Decl";
  return B + "Decl";
}
static std::string GenerateCustomAppertainsTo(const Record &Subject,
                                              raw_ostream &OS) {
  std::string FnName = "is" + Subject.getName();

  // If this code has already been generated, simply return the previous
  // instance of it.
  static std::set<std::string> CustomSubjectSet;
  std::set<std::string>::iterator I = CustomSubjectSet.find(FnName);
  if (I != CustomSubjectSet.end())
    return *I;

  Record *Base = Subject.getValueAsDef("Base");

  // Not currently support custom subjects within custom subjects.
  if (Base->isSubClassOf("SubsetSubject")) {
    PrintFatalError(Subject.getLoc(),
                    "SubsetSubjects within SubsetSubjects is not supported");
    return "";
  }

  OS << "static bool " << FnName << "(const Decl *D) {\n";
  OS << "  if (const " << GetSubjectWithSuffix(Base) << " *S = dyn_cast<";
  OS << GetSubjectWithSuffix(Base);
  OS << ">(D))\n";
  OS << "    return " << Subject.getValueAsString("CheckCode") << ";\n";
  OS << "  return false;\n";
  OS << "}\n\n";

  CustomSubjectSet.insert(FnName);
  return FnName;
}

static std::string GenerateAppertainsTo(const Record &Attr, raw_ostream &OS) {
  // If the attribute does not contain a Subjects definition, then use the
  // default appertainsTo logic.
  if (Attr.isValueUnset("Subjects"))
    return "defaultAppertainsTo";

  const Record *SubjectObj = Attr.getValueAsDef("Subjects");
  std::vector<Record*> Subjects = SubjectObj->getValueAsListOfDefs("Subjects");

  // If the list of subjects is empty, it is assumed that the attribute
  // appertains to everything.
  if (Subjects.empty())
    return "defaultAppertainsTo";

  bool Warn = SubjectObj->getValueAsDef("Diag")->getValueAsBit("Warn");

  // Otherwise, generate an appertainsTo check specific to this attribute which
  // checks all of the given subjects against the Decl passed in. Return the
  // name of that check to the caller.
  std::string FnName = "check" + Attr.getName() + "AppertainsTo";
  std::stringstream SS;
  SS << "static bool " << FnName << "(Sema &S, const AttributeList &Attr, ";
  SS << "const Decl *D) {\n";
  SS << "  if (";
  for (std::vector<Record *>::const_iterator I = Subjects.begin(),
       E = Subjects.end(); I != E; ++I) {
    // If the subject has custom code associated with it, generate a function
    // for it. The function cannot be inlined into this check (yet) because it
    // requires the subject to be of a specific type, and were that information
    // inlined here, it would not support an attribute with multiple custom
    // subjects.
    if ((*I)->isSubClassOf("SubsetSubject")) {
      SS << "!" << GenerateCustomAppertainsTo(**I, OS) << "(D)";
    } else {
      SS << "!isa<" << GetSubjectWithSuffix(*I) << ">(D)";
    }

    if (I + 1 != E)
      SS << " && ";
  }
  SS << ") {\n";
  SS << "    S.Diag(Attr.getLoc(), diag::";
  SS << (Warn ? "warn_attribute_wrong_decl_type" :
               "err_attribute_wrong_decl_type");
  SS << ")\n";
  SS << "      << Attr.getName() << ";
  SS << CalculateDiagnostic(*SubjectObj) << ";\n";
  SS << "    return false;\n";
  SS << "  }\n";
  SS << "  return true;\n";
  SS << "}\n\n";

  OS << SS.str();
  return FnName;
}

static void GenerateDefaultLangOptRequirements(raw_ostream &OS) {
  OS << "static bool defaultDiagnoseLangOpts(Sema &, ";
  OS << "const AttributeList &) {\n";
  OS << "  return true;\n";
  OS << "}\n\n";
}

static std::string GenerateLangOptRequirements(const Record &R,
                                               raw_ostream &OS) {
  // If the attribute has an empty or unset list of language requirements,
  // return the default handler.
  std::vector<Record *> LangOpts = R.getValueAsListOfDefs("LangOpts");
  if (LangOpts.empty())
    return "defaultDiagnoseLangOpts";

  // Generate the test condition, as well as a unique function name for the
  // diagnostic test. The list of options should usually be short (one or two
  // options), and the uniqueness isn't strictly necessary (it is just for
  // codegen efficiency).
  std::string FnName = "check", Test;
  for (std::vector<Record *>::const_iterator I = LangOpts.begin(),
       E = LangOpts.end(); I != E; ++I) {
    std::string Part = (*I)->getValueAsString("Name");
    Test += "S.LangOpts." + Part;
    if (I + 1 != E)
      Test += " || ";
    FnName += Part;
  }
  FnName += "LangOpts";

  // If this code has already been generated, simply return the previous
  // instance of it.
  static std::set<std::string> CustomLangOptsSet;
  std::set<std::string>::iterator I = CustomLangOptsSet.find(FnName);
  if (I != CustomLangOptsSet.end())
    return *I;

  OS << "static bool " << FnName << "(Sema &S, const AttributeList &Attr) {\n";
  OS << "  if (" << Test << ")\n";
  OS << "    return true;\n\n";
  OS << "  S.Diag(Attr.getLoc(), diag::warn_attribute_ignored) ";
  OS << "<< Attr.getName();\n";
  OS << "  return false;\n";
  OS << "}\n\n";

  CustomLangOptsSet.insert(FnName);
  return FnName;
}

static void GenerateDefaultTargetRequirements(raw_ostream &OS) {
  OS << "static bool defaultTargetRequirements(llvm::Triple) {\n";
  OS << "  return true;\n";
  OS << "}\n\n";
}

static std::string GenerateTargetRequirements(const Record &Attr,
                                              const ParsedAttrMap &Dupes,
                                              raw_ostream &OS) {
  // If the attribute is not a target specific attribute, return the default
  // target handler.
  if (!Attr.isSubClassOf("TargetSpecificAttr"))
    return "defaultTargetRequirements";

  // Get the list of architectures to be tested for.
  const Record *R = Attr.getValueAsDef("Target");
  std::vector<std::string> Arches = R->getValueAsListOfStrings("Arches");
  if (Arches.empty()) {
    PrintError(Attr.getLoc(), "Empty list of target architectures for a "
                              "target-specific attr");
    return "defaultTargetRequirements";
  }

  // If there are other attributes which share the same parsed attribute kind,
  // such as target-specific attributes with a shared spelling, collapse the
  // duplicate architectures. This is required because a shared target-specific
  // attribute has only one AttributeList::Kind enumeration value, but it
  // applies to multiple target architectures. In order for the attribute to be
  // considered valid, all of its architectures need to be included.
  if (!Attr.isValueUnset("ParseKind")) {
    std::string APK = Attr.getValueAsString("ParseKind");
    for (ParsedAttrMap::const_iterator I = Dupes.begin(), E = Dupes.end();
         I != E; ++I) {
      if (I->first == APK) {
        std::vector<std::string> DA = I->second->getValueAsDef("Target")->
                                            getValueAsListOfStrings("Arches");
        std::copy(DA.begin(), DA.end(), std::back_inserter(Arches));
      }
    }
  }

  std::string FnName = "isTarget", Test = "(";
  for (std::vector<std::string>::const_iterator I = Arches.begin(),
       E = Arches.end(); I != E; ++I) {
    std::string Part = *I;
    Test += "Arch == llvm::Triple::" + Part;
    if (I + 1 != E)
      Test += " || ";
    FnName += Part;
  }
  Test += ")";

  // If the target also requires OS testing, generate those tests as well.
  bool UsesOS = false;
  if (!R->isValueUnset("OSes")) {
    UsesOS = true;
    
    // We know that there was at least one arch test, so we need to and in the
    // OS tests.
    Test += " && (";
    std::vector<std::string> OSes = R->getValueAsListOfStrings("OSes");
    for (std::vector<std::string>::const_iterator I = OSes.begin(),
         E = OSes.end(); I != E; ++I) {
      std::string Part = *I;

      Test += "OS == llvm::Triple::" + Part;
      if (I + 1 != E)
        Test += " || ";
      FnName += Part;
    }
    Test += ")";
  }

  // If this code has already been generated, simply return the previous
  // instance of it.
  static std::set<std::string> CustomTargetSet;
  std::set<std::string>::iterator I = CustomTargetSet.find(FnName);
  if (I != CustomTargetSet.end())
    return *I;

  OS << "static bool " << FnName << "(llvm::Triple T) {\n";
  OS << "  llvm::Triple::ArchType Arch = T.getArch();\n";
  if (UsesOS)
    OS << "  llvm::Triple::OSType OS = T.getOS();\n";
  OS << "  return " << Test << ";\n";
  OS << "}\n\n";

  CustomTargetSet.insert(FnName);
  return FnName;
}

static void GenerateDefaultSpellingIndexToSemanticSpelling(raw_ostream &OS) {
  OS << "static unsigned defaultSpellingIndexToSemanticSpelling("
     << "const AttributeList &Attr) {\n";
  OS << "  return UINT_MAX;\n";
  OS << "}\n\n";
}

static std::string GenerateSpellingIndexToSemanticSpelling(const Record &Attr,
                                                           raw_ostream &OS) {
  // If the attribute does not have a semantic form, we can bail out early.
  if (!Attr.getValueAsBit("ASTNode"))
    return "defaultSpellingIndexToSemanticSpelling";

  std::vector<FlattenedSpelling> Spellings = GetFlattenedSpellings(Attr);

  // If there are zero or one spellings, or all of the spellings share the same
  // name, we can also bail out early.
  if (Spellings.size() <= 1 || SpellingNamesAreCommon(Spellings))
    return "defaultSpellingIndexToSemanticSpelling";

  // Generate the enumeration we will use for the mapping.
  SemanticSpellingMap SemanticToSyntacticMap;
  std::string Enum = CreateSemanticSpellings(Spellings, SemanticToSyntacticMap);
  std::string Name = Attr.getName() + "AttrSpellingMap";

  OS << "static unsigned " << Name << "(const AttributeList &Attr) {\n";
  OS << Enum;
  OS << "  unsigned Idx = Attr.getAttributeSpellingListIndex();\n";
  WriteSemanticSpellingSwitch("Idx", SemanticToSyntacticMap, OS);
  OS << "}\n\n";

  return Name;
}

static bool IsKnownToGCC(const Record &Attr) {
  // Look at the spellings for this subject; if there are any spellings which
  // claim to be known to GCC, the attribute is known to GCC.
  std::vector<FlattenedSpelling> Spellings = GetFlattenedSpellings(Attr);
  for (std::vector<FlattenedSpelling>::const_iterator I = Spellings.begin(),
       E = Spellings.end(); I != E; ++I) {
    if (I->knownToGCC())
      return true;
  }
  return false;
}

/// Emits the parsed attribute helpers
void EmitClangAttrParsedAttrImpl(RecordKeeper &Records, raw_ostream &OS) {
  emitSourceFileHeader("Parsed attribute helpers", OS);

  // Get the list of parsed attributes, and accept the optional list of
  // duplicates due to the ParseKind.
  ParsedAttrMap Dupes;
  ParsedAttrMap Attrs = getParsedAttrList(Records, &Dupes);

  // Generate the default appertainsTo, target and language option diagnostic,
  // and spelling list index mapping methods.
  GenerateDefaultAppertainsTo(OS);
  GenerateDefaultLangOptRequirements(OS);
  GenerateDefaultTargetRequirements(OS);
  GenerateDefaultSpellingIndexToSemanticSpelling(OS);

  // Generate the appertainsTo diagnostic methods and write their names into
  // another mapping. At the same time, generate the AttrInfoMap object
  // contents. Due to the reliance on generated code, use separate streams so
  // that code will not be interleaved.
  std::stringstream SS;
  for (ParsedAttrMap::iterator I = Attrs.begin(), E = Attrs.end(); I != E;
       ++I) {
    // TODO: If the attribute's kind appears in the list of duplicates, that is
    // because it is a target-specific attribute that appears multiple times.
    // It would be beneficial to test whether the duplicates are "similar
    // enough" to each other to not cause problems. For instance, check that
    // the spellings are identical, and custom parsing rules match, etc.

    // We need to generate struct instances based off ParsedAttrInfo from
    // AttributeList.cpp.
    SS << "  { ";
    emitArgInfo(*I->second, SS);
    SS << ", " << I->second->getValueAsBit("HasCustomParsing");
    SS << ", " << I->second->isSubClassOf("TargetSpecificAttr");
    SS << ", " << I->second->isSubClassOf("TypeAttr");
    SS << ", " << IsKnownToGCC(*I->second);
    SS << ", " << GenerateAppertainsTo(*I->second, OS);
    SS << ", " << GenerateLangOptRequirements(*I->second, OS);
    SS << ", " << GenerateTargetRequirements(*I->second, Dupes, OS);
    SS << ", " << GenerateSpellingIndexToSemanticSpelling(*I->second, OS);
    SS << " }";

    if (I + 1 != E)
      SS << ",";

    SS << "  // AT_" << I->first << "\n";
  }

  OS << "static const ParsedAttrInfo AttrInfoMap[AttributeList::UnknownAttribute + 1] = {\n";
  OS << SS.str();
  OS << "};\n\n";
}

// Emits the kind list of parsed attributes
void EmitClangAttrParsedAttrKinds(RecordKeeper &Records, raw_ostream &OS) {
  emitSourceFileHeader("Attribute name matcher", OS);

  std::vector<Record *> Attrs = Records.getAllDerivedDefinitions("Attr");
  std::vector<StringMatcher::StringPair> GNU, Declspec, CXX11, Keywords;
  std::set<std::string> Seen;
  for (std::vector<Record*>::iterator I = Attrs.begin(), E = Attrs.end();
       I != E; ++I) {
    Record &Attr = **I;

    bool SemaHandler = Attr.getValueAsBit("SemaHandler");
    bool Ignored = Attr.getValueAsBit("Ignored");
    if (SemaHandler || Ignored) {
      // Attribute spellings can be shared between target-specific attributes,
      // and can be shared between syntaxes for the same attribute. For
      // instance, an attribute can be spelled GNU<"interrupt"> for an ARM-
      // specific attribute, or MSP430-specific attribute. Additionally, an
      // attribute can be spelled GNU<"dllexport"> and Declspec<"dllexport">
      // for the same semantic attribute. Ultimately, we need to map each of
      // these to a single AttributeList::Kind value, but the StringMatcher
      // class cannot handle duplicate match strings. So we generate a list of
      // string to match based on the syntax, and emit multiple string matchers
      // depending on the syntax used.
      std::string AttrName;
      if (Attr.isSubClassOf("TargetSpecificAttr") &&
          !Attr.isValueUnset("ParseKind")) {
        AttrName = Attr.getValueAsString("ParseKind");
        if (Seen.find(AttrName) != Seen.end())
          continue;
        Seen.insert(AttrName);
      } else
        AttrName = NormalizeAttrName(StringRef(Attr.getName())).str();

      std::vector<FlattenedSpelling> Spellings = GetFlattenedSpellings(Attr);
      for (std::vector<FlattenedSpelling>::const_iterator
           I = Spellings.begin(), E = Spellings.end(); I != E; ++I) {
        std::string RawSpelling = I->name();
        std::vector<StringMatcher::StringPair> *Matches = 0;
        std::string Spelling, Variety = I->variety();
        if (Variety == "CXX11") {
          Matches = &CXX11;
          Spelling += I->nameSpace();
          Spelling += "::";
        } else if (Variety == "GNU")
          Matches = &GNU;
        else if (Variety == "Declspec")
          Matches = &Declspec;
        else if (Variety == "Keyword")
          Matches = &Keywords;

        assert(Matches && "Unsupported spelling variety found");

        Spelling += NormalizeAttrSpelling(RawSpelling);
        if (SemaHandler)
          Matches->push_back(StringMatcher::StringPair(Spelling,
                              "return AttributeList::AT_" + AttrName + ";"));
        else
          Matches->push_back(StringMatcher::StringPair(Spelling,
                              "return AttributeList::IgnoredAttribute;"));
      }
    }
  }
  
  OS << "static AttributeList::Kind getAttrKind(StringRef Name, ";
  OS << "AttributeList::Syntax Syntax) {\n";
  OS << "  if (AttributeList::AS_GNU == Syntax) {\n";
  StringMatcher("Name", GNU, OS).Emit();
  OS << "  } else if (AttributeList::AS_Declspec == Syntax) {\n";
  StringMatcher("Name", Declspec, OS).Emit();
  OS << "  } else if (AttributeList::AS_CXX11 == Syntax) {\n";
  StringMatcher("Name", CXX11, OS).Emit();
  OS << "  } else if (AttributeList::AS_Keyword == Syntax) {\n";
  StringMatcher("Name", Keywords, OS).Emit();
  OS << "  }\n";
  OS << "  return AttributeList::UnknownAttribute;\n"
     << "}\n";
}

// Emits the code to dump an attribute.
void EmitClangAttrDump(RecordKeeper &Records, raw_ostream &OS) {
  emitSourceFileHeader("Attribute dumper", OS);

  OS <<
    "  switch (A->getKind()) {\n"
    "  default:\n"
    "    llvm_unreachable(\"Unknown attribute kind!\");\n"
    "    break;\n";
  std::vector<Record*> Attrs = Records.getAllDerivedDefinitions("Attr"), Args;
  for (std::vector<Record*>::iterator I = Attrs.begin(), E = Attrs.end();
       I != E; ++I) {
    Record &R = **I;
    if (!R.getValueAsBit("ASTNode"))
      continue;
    OS << "  case attr::" << R.getName() << ": {\n";

    // If the attribute has a semantically-meaningful name (which is determined
    // by whether there is a Spelling enumeration for it), then write out the
    // spelling used for the attribute.
    std::vector<FlattenedSpelling> Spellings = GetFlattenedSpellings(R);
    if (Spellings.size() > 1 && !SpellingNamesAreCommon(Spellings))
      OS << "    OS << \" \" << A->getSpelling();\n";

    Args = R.getValueAsListOfDefs("Args");
    if (!Args.empty()) {
      OS << "    const " << R.getName() << "Attr *SA = cast<" << R.getName()
         << "Attr>(A);\n";
      for (std::vector<Record*>::iterator I = Args.begin(), E = Args.end();
           I != E; ++I)
        createArgument(**I, R.getName())->writeDump(OS);

      // Code for detecting the last child.
      OS << "    bool OldMoreChildren = hasMoreChildren();\n";
      OS << "    bool MoreChildren = OldMoreChildren;\n";     

      for (std::vector<Record*>::iterator I = Args.begin(), E = Args.end();
           I != E; ++I) {
        // More code for detecting the last child.
        OS << "    MoreChildren = OldMoreChildren";
        for (std::vector<Record*>::iterator Next = I + 1; Next != E; ++Next) {
          OS << " || ";
          createArgument(**Next, R.getName())->writeHasChildren(OS);
        }
        OS << ";\n";
        OS << "    setMoreChildren(MoreChildren);\n";

        createArgument(**I, R.getName())->writeDumpChildren(OS);
      }

      // Reset the last child.
      OS << "    setMoreChildren(OldMoreChildren);\n";
    }
    OS <<
      "    break;\n"
      "  }\n";
  }
  OS << "  }\n";
}

void EmitClangAttrParserStringSwitches(RecordKeeper &Records,
                                       raw_ostream &OS) {
  emitSourceFileHeader("Parser-related llvm::StringSwitch cases", OS);
  emitClangAttrArgContextList(Records, OS);
  emitClangAttrIdentifierArgList(Records, OS);
  emitClangAttrTypeArgList(Records, OS);
  emitClangAttrLateParsedList(Records, OS);
}

} // end namespace clang
