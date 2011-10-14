//===--- CodeCompleteConsumer.cpp - Code Completion Interface ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the CodeCompleteConsumer class.
//
//===----------------------------------------------------------------------===//
#include "clang/Sema/CodeCompleteConsumer.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/Sema.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/Lex/Preprocessor.h"
#include "clang-c/Index.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstring>
#include <functional>

using namespace clang;

//===----------------------------------------------------------------------===//
// Code completion context implementation
//===----------------------------------------------------------------------===//

bool CodeCompletionContext::wantConstructorResults() const {
  switch (Kind) {
  case CCC_Recovery:
  case CCC_Statement:
  case CCC_Expression:
  case CCC_ObjCMessageReceiver:
  case CCC_ParenthesizedExpression:
    return true;
    
  case CCC_TopLevel:
  case CCC_ObjCInterface:
  case CCC_ObjCImplementation:
  case CCC_ObjCIvarList:
  case CCC_ClassStructUnion:
  case CCC_DotMemberAccess:
  case CCC_ArrowMemberAccess:
  case CCC_ObjCPropertyAccess:
  case CCC_EnumTag:
  case CCC_UnionTag:
  case CCC_ClassOrStructTag:
  case CCC_ObjCProtocolName:
  case CCC_Namespace:
  case CCC_Type:
  case CCC_Name:
  case CCC_PotentiallyQualifiedName:
  case CCC_MacroName:
  case CCC_MacroNameUse:
  case CCC_PreprocessorExpression:
  case CCC_PreprocessorDirective:
  case CCC_NaturalLanguage:
  case CCC_SelectorName:
  case CCC_TypeQualifiers:
  case CCC_Other:
  case CCC_OtherWithMacros:
  case CCC_ObjCInstanceMessage:
  case CCC_ObjCClassMessage:
  case CCC_ObjCInterfaceName:
  case CCC_ObjCCategoryName:
    return false;
  }
  
  return false;
}

//===----------------------------------------------------------------------===//
// Code completion string implementation
//===----------------------------------------------------------------------===//
CodeCompletionString::Chunk::Chunk(ChunkKind Kind, const char *Text) 
  : Kind(Kind), Text("")
{
  switch (Kind) {
  case CK_TypedText:
  case CK_Text:
  case CK_Placeholder:
  case CK_Informative:
  case CK_ResultType:
  case CK_CurrentParameter:
    this->Text = Text;
    break;

  case CK_Optional:
    llvm_unreachable("Optional strings cannot be created from text");
    break;
      
  case CK_LeftParen:
    this->Text = "(";
    break;

  case CK_RightParen:
    this->Text = ")";
    break;

  case CK_LeftBracket:
    this->Text = "[";
    break;
    
  case CK_RightBracket:
    this->Text = "]";
    break;
    
  case CK_LeftBrace:
    this->Text = "{";
    break;

  case CK_RightBrace:
    this->Text = "}";
    break;

  case CK_LeftAngle:
    this->Text = "<";
    break;
    
  case CK_RightAngle:
    this->Text = ">";
    break;
      
  case CK_Comma:
    this->Text = ", ";
    break;

  case CK_Colon:
    this->Text = ":";
    break;

  case CK_SemiColon:
    this->Text = ";";
    break;

  case CK_Equal:
    this->Text = " = ";
    break;

  case CK_HorizontalSpace:
    this->Text = " ";
    break;

  case CK_VerticalSpace:
    this->Text = "\n";
    break;
  }
}

CodeCompletionString::Chunk
CodeCompletionString::Chunk::CreateText(const char *Text) {
  return Chunk(CK_Text, Text);
}

CodeCompletionString::Chunk 
CodeCompletionString::Chunk::CreateOptional(CodeCompletionString *Optional) {
  Chunk Result;
  Result.Kind = CK_Optional;
  Result.Optional = Optional;
  return Result;
}

CodeCompletionString::Chunk 
CodeCompletionString::Chunk::CreatePlaceholder(const char *Placeholder) {
  return Chunk(CK_Placeholder, Placeholder);
}

CodeCompletionString::Chunk 
CodeCompletionString::Chunk::CreateInformative(const char *Informative) {
  return Chunk(CK_Informative, Informative);
}

CodeCompletionString::Chunk 
CodeCompletionString::Chunk::CreateResultType(const char *ResultType) {
  return Chunk(CK_ResultType, ResultType);
}

CodeCompletionString::Chunk 
CodeCompletionString::Chunk::CreateCurrentParameter(
                                                const char *CurrentParameter) {
  return Chunk(CK_CurrentParameter, CurrentParameter);
}

CodeCompletionString::CodeCompletionString(const Chunk *Chunks, 
                                           unsigned NumChunks,
                                           unsigned Priority, 
                                           CXAvailabilityKind Availability,
                                           const char **Annotations,
                                           unsigned NumAnnotations)
  : NumChunks(NumChunks), NumAnnotations(NumAnnotations)
  , Priority(Priority), Availability(Availability)
{ 
  assert(NumChunks <= 0xffff);
  assert(NumAnnotations <= 0xffff);

  Chunk *StoredChunks = reinterpret_cast<Chunk *>(this + 1);
  for (unsigned I = 0; I != NumChunks; ++I)
    StoredChunks[I] = Chunks[I];

  const char **StoredAnnotations = reinterpret_cast<const char **>(StoredChunks + NumChunks);
  for (unsigned I = 0; I != NumAnnotations; ++I)
    StoredAnnotations[I] = Annotations[I];
}

unsigned CodeCompletionString::getAnnotationCount() const {
  return NumAnnotations;
}

const char *CodeCompletionString::getAnnotation(unsigned AnnotationNr) const {
  if (AnnotationNr < NumAnnotations)
    return reinterpret_cast<const char * const*>(end())[AnnotationNr];
  else
    return 0;
}


std::string CodeCompletionString::getAsString() const {
  std::string Result;
  llvm::raw_string_ostream OS(Result);
                          
  for (iterator C = begin(), CEnd = end(); C != CEnd; ++C) {
    switch (C->Kind) {
    case CK_Optional: OS << "{#" << C->Optional->getAsString() << "#}"; break;
    case CK_Placeholder: OS << "<#" << C->Text << "#>"; break;
        
    case CK_Informative: 
    case CK_ResultType:
      OS << "[#" << C->Text << "#]"; 
      break;
        
    case CK_CurrentParameter: OS << "<#" << C->Text << "#>"; break;
    default: OS << C->Text; break;
    }
  }
  return OS.str();
}

const char *CodeCompletionString::getTypedText() const {
  for (iterator C = begin(), CEnd = end(); C != CEnd; ++C)
    if (C->Kind == CK_TypedText)
      return C->Text;
  
  return 0;
}

const char *CodeCompletionAllocator::CopyString(StringRef String) {
  char *Mem = (char *)Allocate(String.size() + 1, 1);
  std::copy(String.begin(), String.end(), Mem);
  Mem[String.size()] = 0;
  return Mem;
}

const char *CodeCompletionAllocator::CopyString(Twine String) {
  // FIXME: It would be more efficient to teach Twine to tell us its size and
  // then add a routine there to fill in an allocated char* with the contents
  // of the string.
  llvm::SmallString<128> Data;
  return CopyString(String.toStringRef(Data));
}

CodeCompletionString *CodeCompletionBuilder::TakeString() {
  void *Mem = Allocator.Allocate(
                  sizeof(CodeCompletionString) + sizeof(Chunk) * Chunks.size()
                                    + sizeof(const char *) * Annotations.size(),
                                 llvm::alignOf<CodeCompletionString>());
  CodeCompletionString *Result 
    = new (Mem) CodeCompletionString(Chunks.data(), Chunks.size(),
                                     Priority, Availability,
                                     Annotations.data(), Annotations.size());
  Chunks.clear();
  return Result;
}

unsigned CodeCompletionResult::getPriorityFromDecl(NamedDecl *ND) {
  if (!ND)
    return CCP_Unlikely;
  
  // Context-based decisions.
  DeclContext *DC = ND->getDeclContext()->getRedeclContext();
  if (DC->isFunctionOrMethod() || isa<BlockDecl>(DC)) {
    // _cmd is relatively rare
    if (ImplicitParamDecl *ImplicitParam = dyn_cast<ImplicitParamDecl>(ND))
      if (ImplicitParam->getIdentifier() &&
          ImplicitParam->getIdentifier()->isStr("_cmd"))
        return CCP_ObjC_cmd;
    
    return CCP_LocalDeclaration;
  }
  if (DC->isRecord() || isa<ObjCContainerDecl>(DC))
    return CCP_MemberDeclaration;
  
  // Content-based decisions.
  if (isa<EnumConstantDecl>(ND))
    return CCP_Constant;
  if (isa<TypeDecl>(ND) || isa<ObjCInterfaceDecl>(ND))
    return CCP_Type;
  
  return CCP_Declaration;
}

//===----------------------------------------------------------------------===//
// Code completion overload candidate implementation
//===----------------------------------------------------------------------===//
FunctionDecl *
CodeCompleteConsumer::OverloadCandidate::getFunction() const {
  if (getKind() == CK_Function)
    return Function;
  else if (getKind() == CK_FunctionTemplate)
    return FunctionTemplate->getTemplatedDecl();
  else
    return 0;
}

const FunctionType *
CodeCompleteConsumer::OverloadCandidate::getFunctionType() const {
  switch (Kind) {
  case CK_Function:
    return Function->getType()->getAs<FunctionType>();
      
  case CK_FunctionTemplate:
    return FunctionTemplate->getTemplatedDecl()->getType()
             ->getAs<FunctionType>();
      
  case CK_FunctionType:
    return Type;
  }
  
  return 0;
}

//===----------------------------------------------------------------------===//
// Code completion consumer implementation
//===----------------------------------------------------------------------===//

CodeCompleteConsumer::~CodeCompleteConsumer() { }

void 
PrintingCodeCompleteConsumer::ProcessCodeCompleteResults(Sema &SemaRef,
                                                 CodeCompletionContext Context,
                                                 CodeCompletionResult *Results,
                                                         unsigned NumResults) {
  std::stable_sort(Results, Results + NumResults);
  
  // Print the results.
  for (unsigned I = 0; I != NumResults; ++I) {
    OS << "COMPLETION: ";
    switch (Results[I].Kind) {
    case CodeCompletionResult::RK_Declaration:
      OS << *Results[I].Declaration;
      if (Results[I].Hidden)
        OS << " (Hidden)";
      if (CodeCompletionString *CCS 
            = Results[I].CreateCodeCompletionString(SemaRef, Allocator)) {
        OS << " : " << CCS->getAsString();
      }
        
      OS << '\n';
      break;
      
    case CodeCompletionResult::RK_Keyword:
      OS << Results[I].Keyword << '\n';
      break;
        
    case CodeCompletionResult::RK_Macro: {
      OS << Results[I].Macro->getName();
      if (CodeCompletionString *CCS 
            = Results[I].CreateCodeCompletionString(SemaRef, Allocator)) {
        OS << " : " << CCS->getAsString();
      }
      OS << '\n';
      break;
    }
        
    case CodeCompletionResult::RK_Pattern: {
      OS << "Pattern : " 
         << Results[I].Pattern->getAsString() << '\n';
      break;
    }
    }
  }
}

void 
PrintingCodeCompleteConsumer::ProcessOverloadCandidates(Sema &SemaRef,
                                                        unsigned CurrentArg,
                                              OverloadCandidate *Candidates,
                                                     unsigned NumCandidates) {
  for (unsigned I = 0; I != NumCandidates; ++I) {
    if (CodeCompletionString *CCS
          = Candidates[I].CreateSignatureString(CurrentArg, SemaRef,
                                                Allocator)) {
      OS << "OVERLOAD: " << CCS->getAsString() << "\n";
    }
  }
}

void CodeCompletionResult::computeCursorKindAndAvailability(bool Accessible) {
  switch (Kind) {
  case RK_Declaration:
    // Set the availability based on attributes.
    switch (Declaration->getAvailability()) {
    case AR_Available:
    case AR_NotYetIntroduced:
      Availability = CXAvailability_Available;      
      break;
      
    case AR_Deprecated:
      Availability = CXAvailability_Deprecated;
      break;
      
    case AR_Unavailable:
      Availability = CXAvailability_NotAvailable;
      break;
    }

    if (FunctionDecl *Function = dyn_cast<FunctionDecl>(Declaration))
      if (Function->isDeleted())
        Availability = CXAvailability_NotAvailable;
      
    CursorKind = getCursorKindForDecl(Declaration);
    if (CursorKind == CXCursor_UnexposedDecl)
      CursorKind = CXCursor_NotImplemented;
    break;

  case RK_Macro:
    Availability = CXAvailability_Available;      
    CursorKind = CXCursor_MacroDefinition;
    break;
      
  case RK_Keyword:
    Availability = CXAvailability_Available;      
    CursorKind = CXCursor_NotImplemented;
    break;
      
  case RK_Pattern:
    // Do nothing: Patterns can come with cursor kinds!
    break;
  }

  if (!Accessible)
    Availability = CXAvailability_NotAccessible;
}

/// \brief Retrieve the name that should be used to order a result.
///
/// If the name needs to be constructed as a string, that string will be
/// saved into Saved and the returned StringRef will refer to it.
static StringRef getOrderedName(const CodeCompletionResult &R,
                                    std::string &Saved) {
  switch (R.Kind) {
    case CodeCompletionResult::RK_Keyword:
      return R.Keyword;
      
    case CodeCompletionResult::RK_Pattern:
      return R.Pattern->getTypedText();
      
    case CodeCompletionResult::RK_Macro:
      return R.Macro->getName();
      
    case CodeCompletionResult::RK_Declaration:
      // Handle declarations below.
      break;
  }
  
  DeclarationName Name = R.Declaration->getDeclName();
  
  // If the name is a simple identifier (by far the common case), or a
  // zero-argument selector, just return a reference to that identifier.
  if (IdentifierInfo *Id = Name.getAsIdentifierInfo())
    return Id->getName();
  if (Name.isObjCZeroArgSelector())
    if (IdentifierInfo *Id
        = Name.getObjCSelector().getIdentifierInfoForSlot(0))
      return Id->getName();
  
  Saved = Name.getAsString();
  return Saved;
}
    
bool clang::operator<(const CodeCompletionResult &X, 
                      const CodeCompletionResult &Y) {
  std::string XSaved, YSaved;
  StringRef XStr = getOrderedName(X, XSaved);
  StringRef YStr = getOrderedName(Y, YSaved);
  int cmp = XStr.compare_lower(YStr);
  if (cmp)
    return cmp < 0;
  
  // If case-insensitive comparison fails, try case-sensitive comparison.
  cmp = XStr.compare(YStr);
  if (cmp)
    return cmp < 0;
  
  return false;
}
