//===--- TypePrinter.cpp - Pretty-Print Clang Types -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code to print types from Clang's type system.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/Decl.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Type.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/Basic/LangOptions.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"
using namespace clang;

namespace {
  class TypePrinter {
    PrintingPolicy Policy;

  public:
    explicit TypePrinter(const PrintingPolicy &Policy) : Policy(Policy) { }
    
    void Print(QualType T, std::string &S);
    void PrintTag(const TagType *T, std::string &S);
#define ABSTRACT_TYPE(CLASS, PARENT)
#define TYPE(CLASS, PARENT) \
  void Print##CLASS(const CLASS##Type *T, std::string &S);
#include "clang/AST/TypeNodes.def"
  };
}

static void AppendTypeQualList(std::string &S, unsigned TypeQuals) {
  if (TypeQuals & Qualifiers::Const) {
    if (!S.empty()) S += ' ';
    S += "const";
  }
  if (TypeQuals & Qualifiers::Volatile) {
    if (!S.empty()) S += ' ';
    S += "volatile";
  }
  if (TypeQuals & Qualifiers::Restrict) {
    if (!S.empty()) S += ' ';
    S += "restrict";
  }
}

void TypePrinter::Print(QualType T, std::string &S) {
  if (T.isNull()) {
    S += "NULL TYPE";
    return;
  }
  
  if (Policy.SuppressSpecifiers && T->isSpecifierType())
    return;
  
  // Print qualifiers as appropriate.
  Qualifiers Quals = T.getQualifiers();
  if (!Quals.empty()) {
    std::string TQS;
    Quals.getAsStringInternal(TQS, Policy);
    
    if (!S.empty()) {
      TQS += ' ';
      TQS += S;
    }
    std::swap(S, TQS);
  }
  
  switch (T->getTypeClass()) {
#define ABSTRACT_TYPE(CLASS, PARENT)
#define TYPE(CLASS, PARENT) case Type::CLASS:                \
    Print##CLASS(cast<CLASS##Type>(T.getTypePtr()), S);      \
    break;
#include "clang/AST/TypeNodes.def"
  }
}

void TypePrinter::PrintBuiltin(const BuiltinType *T, std::string &S) {
  if (S.empty()) {
    S = T->getName(Policy.LangOpts);
  } else {
    // Prefix the basic type, e.g. 'int X'.
    S = ' ' + S;
    S = T->getName(Policy.LangOpts) + S;
  }
}

void TypePrinter::PrintFixedWidthInt(const FixedWidthIntType *T, 
                                     std::string &S) {
  // FIXME: Once we get bitwidth attribute, write as
  // "int __attribute__((bitwidth(x)))".
  std::string prefix = "__clang_fixedwidth";
  prefix += llvm::utostr_32(T->getWidth());
  prefix += (char)(T->isSigned() ? 'S' : 'U');
  if (S.empty()) {
    S = prefix;
  } else {
    // Prefix the basic type, e.g. 'int X'.
    S = prefix + S;
  }  
}

void TypePrinter::PrintComplex(const ComplexType *T, std::string &S) {
  Print(T->getElementType(), S);
  S = "_Complex " + S;
}

void TypePrinter::PrintPointer(const PointerType *T, std::string &S) { 
  S = '*' + S;
  
  // Handle things like 'int (*A)[4];' correctly.
  // FIXME: this should include vectors, but vectors use attributes I guess.
  if (isa<ArrayType>(T->getPointeeType()))
    S = '(' + S + ')';
  
  Print(T->getPointeeType(), S);
}

void TypePrinter::PrintBlockPointer(const BlockPointerType *T, std::string &S) {
  S = '^' + S;
  Print(T->getPointeeType(), S);
}

void TypePrinter::PrintLValueReference(const LValueReferenceType *T, 
                                       std::string &S) { 
  S = '&' + S;
  
  // Handle things like 'int (&A)[4];' correctly.
  // FIXME: this should include vectors, but vectors use attributes I guess.
  if (isa<ArrayType>(T->getPointeeTypeAsWritten()))
    S = '(' + S + ')';
  
  Print(T->getPointeeTypeAsWritten(), S);
}

void TypePrinter::PrintRValueReference(const RValueReferenceType *T, 
                                       std::string &S) { 
  S = "&&" + S;
  
  // Handle things like 'int (&&A)[4];' correctly.
  // FIXME: this should include vectors, but vectors use attributes I guess.
  if (isa<ArrayType>(T->getPointeeTypeAsWritten()))
    S = '(' + S + ')';
  
  Print(T->getPointeeTypeAsWritten(), S);
}

void TypePrinter::PrintMemberPointer(const MemberPointerType *T, 
                                     std::string &S) { 
  std::string C;
  Print(QualType(T->getClass(), 0), C);
  C += "::*";
  S = C + S;
  
  // Handle things like 'int (Cls::*A)[4];' correctly.
  // FIXME: this should include vectors, but vectors use attributes I guess.
  if (isa<ArrayType>(T->getPointeeType()))
    S = '(' + S + ')';
  
  Print(T->getPointeeType(), S);
}

void TypePrinter::PrintConstantArray(const ConstantArrayType *T, 
                                     std::string &S) {
  S += '[';
  S += llvm::utostr(T->getSize().getZExtValue());
  S += ']';
  
  Print(T->getElementType(), S);
}

void TypePrinter::PrintIncompleteArray(const IncompleteArrayType *T, 
                                       std::string &S) {
  S += "[]";
  Print(T->getElementType(), S);
}

void TypePrinter::PrintVariableArray(const VariableArrayType *T, 
                                     std::string &S) { 
  S += '[';
  
  if (T->getIndexTypeQualifiers().hasQualifiers()) {
    AppendTypeQualList(S, T->getIndexTypeCVRQualifiers());
    S += ' ';
  }
  
  if (T->getSizeModifier() == VariableArrayType::Static)
    S += "static";
  else if (T->getSizeModifier() == VariableArrayType::Star)
    S += '*';
  
  if (T->getSizeExpr()) {
    std::string SStr;
    llvm::raw_string_ostream s(SStr);
    T->getSizeExpr()->printPretty(s, 0, Policy);
    S += s.str();
  }
  S += ']';
  
  Print(T->getElementType(), S);
}

void TypePrinter::PrintDependentSizedArray(const DependentSizedArrayType *T, 
                                           std::string &S) {  
  S += '[';
  
  if (T->getSizeExpr()) {
    std::string SStr;
    llvm::raw_string_ostream s(SStr);
    T->getSizeExpr()->printPretty(s, 0, Policy);
    S += s.str();
  }
  S += ']';
  
  Print(T->getElementType(), S);
}

void TypePrinter::PrintDependentSizedExtVector(
                                          const DependentSizedExtVectorType *T, 
                                               std::string &S) { 
  Print(T->getElementType(), S);
  
  S += " __attribute__((ext_vector_type(";
  if (T->getSizeExpr()) {
    std::string SStr;
    llvm::raw_string_ostream s(SStr);
    T->getSizeExpr()->printPretty(s, 0, Policy);
    S += s.str();
  }
  S += ")))";  
}

void TypePrinter::PrintVector(const VectorType *T, std::string &S) { 
  // FIXME: We prefer to print the size directly here, but have no way
  // to get the size of the type.
  S += " __attribute__((__vector_size__(";
  S += llvm::utostr_32(T->getNumElements()); // convert back to bytes.
  std::string ET;
  Print(T->getElementType(), ET);
  S += " * sizeof(" + ET + "))))";
  Print(T->getElementType(), S);
}

void TypePrinter::PrintExtVector(const ExtVectorType *T, std::string &S) { 
  S += " __attribute__((ext_vector_type(";
  S += llvm::utostr_32(T->getNumElements());
  S += ")))";
  Print(T->getElementType(), S);
}

void TypePrinter::PrintFunctionProto(const FunctionProtoType *T, 
                                     std::string &S) { 
  // If needed for precedence reasons, wrap the inner part in grouping parens.
  if (!S.empty())
    S = "(" + S + ")";
  
  S += "(";
  std::string Tmp;
  PrintingPolicy ParamPolicy(Policy);
  ParamPolicy.SuppressSpecifiers = false;
  for (unsigned i = 0, e = T->getNumArgs(); i != e; ++i) {
    if (i) S += ", ";
    Print(T->getArgType(i), Tmp);
    S += Tmp;
    Tmp.clear();
  }
  
  if (T->isVariadic()) {
    if (T->getNumArgs())
      S += ", ";
    S += "...";
  } else if (T->getNumArgs() == 0 && !Policy.LangOpts.CPlusPlus) {
    // Do not emit int() if we have a proto, emit 'int(void)'.
    S += "void";
  }
  
  S += ")";
  if (T->getNoReturnAttr())
    S += " __attribute__((noreturn))";
  Print(T->getResultType(), S);
  
}

void TypePrinter::PrintFunctionNoProto(const FunctionNoProtoType *T, 
                                       std::string &S) { 
  // If needed for precedence reasons, wrap the inner part in grouping parens.
  if (!S.empty())
    S = "(" + S + ")";
  
  S += "()";
  if (T->getNoReturnAttr())
    S += " __attribute__((noreturn))";
  Print(T->getResultType(), S);
}

void TypePrinter::PrintTypedef(const TypedefType *T, std::string &S) { 
  if (!S.empty())    // Prefix the basic type, e.g. 'typedefname X'.
    S = ' ' + S;
  S = T->getDecl()->getIdentifier()->getName().str() + S;  
}

void TypePrinter::PrintTypeOfExpr(const TypeOfExprType *T, std::string &S) {
  if (!S.empty())    // Prefix the basic type, e.g. 'typeof(e) X'.
    S = ' ' + S;
  std::string Str;
  llvm::raw_string_ostream s(Str);
  T->getUnderlyingExpr()->printPretty(s, 0, Policy);
  S = "typeof " + s.str() + S;
}

void TypePrinter::PrintTypeOf(const TypeOfType *T, std::string &S) { 
  if (!S.empty())    // Prefix the basic type, e.g. 'typeof(t) X'.
    S = ' ' + S;
  std::string Tmp;
  Print(T->getUnderlyingType(), Tmp);
  S = "typeof(" + Tmp + ")" + S;
}

void TypePrinter::PrintDecltype(const DecltypeType *T, std::string &S) { 
  if (!S.empty())    // Prefix the basic type, e.g. 'decltype(t) X'.
    S = ' ' + S;
  std::string Str;
  llvm::raw_string_ostream s(Str);
  T->getUnderlyingExpr()->printPretty(s, 0, Policy);
  S = "decltype(" + s.str() + ")" + S;
}

void TypePrinter::PrintTag(const TagType *T, std::string &InnerString) {
  if (Policy.SuppressTag)
    return;
  
  if (!InnerString.empty())    // Prefix the basic type, e.g. 'typedefname X'.
    InnerString = ' ' + InnerString;
  
  const char *Kind = Policy.SuppressTagKind? 0 : T->getDecl()->getKindName();
  const char *ID;
  if (const IdentifierInfo *II = T->getDecl()->getIdentifier())
    ID = II->getNameStart();
  else if (TypedefDecl *Typedef = T->getDecl()->getTypedefForAnonDecl()) {
    Kind = 0;
    assert(Typedef->getIdentifier() && "Typedef without identifier?");
    ID = Typedef->getIdentifier()->getNameStart();
  } else
    ID = "<anonymous>";
  
  // If this is a class template specialization, print the template
  // arguments.
  if (ClassTemplateSpecializationDecl *Spec
      = dyn_cast<ClassTemplateSpecializationDecl>(T->getDecl())) {
    const TemplateArgumentList &TemplateArgs = Spec->getTemplateArgs();
    std::string TemplateArgsStr
      = TemplateSpecializationType::PrintTemplateArgumentList(
                                            TemplateArgs.getFlatArgumentList(),
                                            TemplateArgs.flat_size(),
                                            Policy);
    InnerString = TemplateArgsStr + InnerString;
  }
  
  if (!Policy.SuppressScope) {
    // Compute the full nested-name-specifier for this type. In C,
    // this will always be empty.
    std::string ContextStr;
    for (DeclContext *DC = T->getDecl()->getDeclContext();
         !DC->isTranslationUnit(); DC = DC->getParent()) {
      std::string MyPart;
      if (NamespaceDecl *NS = dyn_cast<NamespaceDecl>(DC)) {
        if (NS->getIdentifier())
          MyPart = NS->getNameAsString();
      } else if (ClassTemplateSpecializationDecl *Spec
                  = dyn_cast<ClassTemplateSpecializationDecl>(DC)) {
        const TemplateArgumentList &TemplateArgs = Spec->getTemplateArgs();
        std::string TemplateArgsStr
          = TemplateSpecializationType::PrintTemplateArgumentList(
                                            TemplateArgs.getFlatArgumentList(),
                                            TemplateArgs.flat_size(),
                                            Policy);
        MyPart = Spec->getIdentifier()->getName().str() + TemplateArgsStr;
      } else if (TagDecl *Tag = dyn_cast<TagDecl>(DC)) {
        if (TypedefDecl *Typedef = Tag->getTypedefForAnonDecl())
          MyPart = Typedef->getIdentifier()->getName();
        else if (Tag->getIdentifier())
          MyPart = Tag->getIdentifier()->getName();
      }
      
      if (!MyPart.empty())
        ContextStr = MyPart + "::" + ContextStr;
    }
    
    if (Kind)
      InnerString = std::string(Kind) + ' ' + ContextStr + ID + InnerString;
    else
      InnerString = ContextStr + ID + InnerString;
  } else
    InnerString = ID + InnerString;  
}

void TypePrinter::PrintRecord(const RecordType *T, std::string &S) {
  PrintTag(T, S);
}

void TypePrinter::PrintEnum(const EnumType *T, std::string &S) { 
  PrintTag(T, S);
}

void TypePrinter::PrintElaborated(const ElaboratedType *T, std::string &S) { 
  std::string TypeStr;
  PrintingPolicy InnerPolicy(Policy);
  InnerPolicy.SuppressTagKind = true;
  TypePrinter(InnerPolicy).Print(T->getUnderlyingType(), S);
  
  S = std::string(T->getNameForTagKind(T->getTagKind())) + ' ' + S;  
}

void TypePrinter::PrintTemplateTypeParm(const TemplateTypeParmType *T, 
                                        std::string &S) { 
  if (!S.empty())    // Prefix the basic type, e.g. 'parmname X'.
    S = ' ' + S;
  
  if (!T->getName())
    S = "type-parameter-" + llvm::utostr_32(T->getDepth()) + '-' +
        llvm::utostr_32(T->getIndex()) + S;
  else
    S = T->getName()->getName().str() + S;  
}

void TypePrinter::PrintSubstTemplateTypeParm(const SubstTemplateTypeParmType *T, 
                                             std::string &S) { 
  Print(T->getReplacementType(), S);
}

void TypePrinter::PrintTemplateSpecialization(
                                            const TemplateSpecializationType *T, 
                                              std::string &S) { 
  std::string SpecString;
  
  {
    llvm::raw_string_ostream OS(SpecString);
    T->getTemplateName().print(OS, Policy);
  }
  
  SpecString += TemplateSpecializationType::PrintTemplateArgumentList(
                                                                  T->getArgs(), 
                                                                T->getNumArgs(), 
                                                                      Policy);
  if (S.empty())
    S.swap(SpecString);
  else
    S = SpecString + ' ' + S;
}

void TypePrinter::PrintQualifiedName(const QualifiedNameType *T, 
                                     std::string &S) { 
  std::string MyString;
  
  {
    llvm::raw_string_ostream OS(MyString);
    T->getQualifier()->print(OS, Policy);
  }
  
  std::string TypeStr;
  PrintingPolicy InnerPolicy(Policy);
  InnerPolicy.SuppressTagKind = true;
  InnerPolicy.SuppressScope = true;
  TypePrinter(InnerPolicy).Print(T->getNamedType(), TypeStr);
  
  MyString += TypeStr;
  if (S.empty())
    S.swap(MyString);
  else
    S = MyString + ' ' + S;  
}

void TypePrinter::PrintTypename(const TypenameType *T, std::string &S) { 
  std::string MyString;
  
  {
    llvm::raw_string_ostream OS(MyString);
    OS << "typename ";
    T->getQualifier()->print(OS, Policy);
    
    if (const IdentifierInfo *Ident = T->getIdentifier())
      OS << Ident->getName();
    else if (const TemplateSpecializationType *Spec = T->getTemplateId()) {
      Spec->getTemplateName().print(OS, Policy, true);
      OS << TemplateSpecializationType::PrintTemplateArgumentList(
                                                            Spec->getArgs(),
                                                            Spec->getNumArgs(),
                                                            Policy);
    }
  }
  
  if (S.empty())
    S.swap(MyString);
  else
    S = MyString + ' ' + S;
}

void TypePrinter::PrintObjCInterface(const ObjCInterfaceType *T, 
                                     std::string &S) { 
  if (!S.empty())    // Prefix the basic type, e.g. 'typedefname X'.
    S = ' ' + S;
  
  std::string ObjCQIString = T->getDecl()->getNameAsString();
  if (T->getNumProtocols()) {
    ObjCQIString += '<';
    bool isFirst = true;
    for (ObjCInterfaceType::qual_iterator I = T->qual_begin(), 
                                          E = T->qual_end(); 
         I != E; ++I) {
      if (isFirst)
        isFirst = false;
      else
        ObjCQIString += ',';
      ObjCQIString += (*I)->getNameAsString();
    }
    ObjCQIString += '>';
  }
  S = ObjCQIString + S;
}

void TypePrinter::PrintObjCObjectPointer(const ObjCObjectPointerType *T, 
                                         std::string &S) { 
  std::string ObjCQIString;
  
  if (T->isObjCIdType() || T->isObjCQualifiedIdType())
    ObjCQIString = "id";
  else if (T->isObjCClassType() || T->isObjCQualifiedClassType())
    ObjCQIString = "Class";
  else
    ObjCQIString = T->getInterfaceDecl()->getNameAsString();
  
  if (!T->qual_empty()) {
    ObjCQIString += '<';
    for (ObjCObjectPointerType::qual_iterator I = T->qual_begin(), 
                                              E = T->qual_end();
         I != E; ++I) {
      ObjCQIString += (*I)->getNameAsString();
      if (I+1 != E)
        ObjCQIString += ',';
    }
    ObjCQIString += '>';
  }
  
  T->getPointeeType().getQualifiers().getAsStringInternal(ObjCQIString, Policy);
  
  if (!T->isObjCIdType() && !T->isObjCQualifiedIdType())
    ObjCQIString += " *"; // Don't forget the implicit pointer.
  else if (!S.empty()) // Prefix the basic type, e.g. 'typedefname X'.
    S = ' ' + S;
  
  S = ObjCQIString + S;  
}

static void PrintTemplateArgument(std::string &Buffer,
                                  const TemplateArgument &Arg,
                                  const PrintingPolicy &Policy) {
  switch (Arg.getKind()) {
    case TemplateArgument::Null:
      assert(false && "Null template argument");
      break;
      
    case TemplateArgument::Type:
      Arg.getAsType().getAsStringInternal(Buffer, Policy);
      break;
      
    case TemplateArgument::Declaration:
      Buffer = cast<NamedDecl>(Arg.getAsDecl())->getNameAsString();
      break;
      
    case TemplateArgument::Integral:
      Buffer = Arg.getAsIntegral()->toString(10, true);
      break;
      
    case TemplateArgument::Expression: {
      llvm::raw_string_ostream s(Buffer);
      Arg.getAsExpr()->printPretty(s, 0, Policy);
      break;
    }
      
    case TemplateArgument::Pack:
      assert(0 && "FIXME: Implement!");
      break;
  }
}

std::string
TemplateSpecializationType::PrintTemplateArgumentList(
                                                const TemplateArgument *Args,
                                                unsigned NumArgs,
                                                const PrintingPolicy &Policy) {
  std::string SpecString;
  SpecString += '<';
  for (unsigned Arg = 0; Arg < NumArgs; ++Arg) {
    if (Arg)
      SpecString += ", ";
    
    // Print the argument into a string.
    std::string ArgString;
    PrintTemplateArgument(ArgString, Args[Arg], Policy);
    
    // If this is the first argument and its string representation
    // begins with the global scope specifier ('::foo'), add a space
    // to avoid printing the diagraph '<:'.
    if (!Arg && !ArgString.empty() && ArgString[0] == ':')
      SpecString += ' ';
    
    SpecString += ArgString;
  }
  
  // If the last character of our string is '>', add another space to
  // keep the two '>''s separate tokens. We don't *have* to do this in
  // C++0x, but it's still good hygiene.
  if (SpecString[SpecString.size() - 1] == '>')
    SpecString += ' ';
  
  SpecString += '>';
  
  return SpecString;
}

// Sadly, repeat all that with TemplateArgLoc.
std::string TemplateSpecializationType::
PrintTemplateArgumentList(const TemplateArgumentLoc *Args, unsigned NumArgs,
                          const PrintingPolicy &Policy) {
  std::string SpecString;
  SpecString += '<';
  for (unsigned Arg = 0; Arg < NumArgs; ++Arg) {
    if (Arg)
      SpecString += ", ";
    
    // Print the argument into a string.
    std::string ArgString;
    PrintTemplateArgument(ArgString, Args[Arg].getArgument(), Policy);
    
    // If this is the first argument and its string representation
    // begins with the global scope specifier ('::foo'), add a space
    // to avoid printing the diagraph '<:'.
    if (!Arg && !ArgString.empty() && ArgString[0] == ':')
      SpecString += ' ';
    
    SpecString += ArgString;
  }
  
  // If the last character of our string is '>', add another space to
  // keep the two '>''s separate tokens. We don't *have* to do this in
  // C++0x, but it's still good hygiene.
  if (SpecString[SpecString.size() - 1] == '>')
    SpecString += ' ';
  
  SpecString += '>';
  
  return SpecString;
}

void QualType::dump(const char *msg) const {
  std::string R = "identifier";
  LangOptions LO;
  getAsStringInternal(R, PrintingPolicy(LO));
  if (msg)
    fprintf(stderr, "%s: %s\n", msg, R.c_str());
  else
    fprintf(stderr, "%s\n", R.c_str());
}
void QualType::dump() const {
  dump("");
}

void Type::dump() const {
  QualType(this, 0).dump();
}

std::string Qualifiers::getAsString() const {
  LangOptions LO;
  return getAsString(PrintingPolicy(LO));
}

// Appends qualifiers to the given string, separated by spaces.  Will
// prefix a space if the string is non-empty.  Will not append a final
// space.
void Qualifiers::getAsStringInternal(std::string &S,
                                     const PrintingPolicy&) const {
  AppendTypeQualList(S, getCVRQualifiers());
  if (unsigned AddressSpace = getAddressSpace()) {
    if (!S.empty()) S += ' ';
    S += "__attribute__((address_space(";
    S += llvm::utostr_32(AddressSpace);
    S += ")))";
  }
  if (Qualifiers::GC GCAttrType = getObjCGCAttr()) {
    if (!S.empty()) S += ' ';
    S += "__attribute__((objc_gc(";
    if (GCAttrType == Qualifiers::Weak)
      S += "weak";
    else
      S += "strong";
    S += ")))";
  }
}

std::string QualType::getAsString() const {
  std::string S;
  LangOptions LO;
  getAsStringInternal(S, PrintingPolicy(LO));
  return S;
}

void QualType::getAsStringInternal(std::string &S,
                                   const PrintingPolicy &Policy) const {
  TypePrinter Printer(Policy);
  Printer.Print(*this, S);
}

