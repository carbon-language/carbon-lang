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
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstring>
#include <functional>

using namespace clang;
using llvm::StringRef;

//===----------------------------------------------------------------------===//
// Code completion string implementation
//===----------------------------------------------------------------------===//
CodeCompletionString::Chunk::Chunk(ChunkKind Kind, llvm::StringRef Text) 
  : Kind(Kind), Text("")
{
  switch (Kind) {
  case CK_TypedText:
  case CK_Text:
  case CK_Placeholder:
  case CK_Informative:
  case CK_ResultType:
  case CK_CurrentParameter: {
    char *New = new char [Text.size() + 1];
    std::memcpy(New, Text.data(), Text.size());
    New[Text.size()] = '\0';
    this->Text = New;
    break;
  }

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
CodeCompletionString::Chunk::CreateText(StringRef Text) {
  return Chunk(CK_Text, Text);
}

CodeCompletionString::Chunk 
CodeCompletionString::Chunk::CreateOptional(
                                 std::auto_ptr<CodeCompletionString> Optional) {
  Chunk Result;
  Result.Kind = CK_Optional;
  Result.Optional = Optional.release();
  return Result;
}

CodeCompletionString::Chunk 
CodeCompletionString::Chunk::CreatePlaceholder(StringRef Placeholder) {
  return Chunk(CK_Placeholder, Placeholder);
}

CodeCompletionString::Chunk 
CodeCompletionString::Chunk::CreateInformative(StringRef Informative) {
  return Chunk(CK_Informative, Informative);
}

CodeCompletionString::Chunk 
CodeCompletionString::Chunk::CreateResultType(StringRef ResultType) {
  return Chunk(CK_ResultType, ResultType);
}

CodeCompletionString::Chunk 
CodeCompletionString::Chunk::CreateCurrentParameter(
                                                StringRef CurrentParameter) {
  return Chunk(CK_CurrentParameter, CurrentParameter);
}

CodeCompletionString::Chunk CodeCompletionString::Chunk::Clone() const {
  switch (Kind) {
  case CK_TypedText:
  case CK_Text:
  case CK_Placeholder:
  case CK_Informative:
  case CK_ResultType:
  case CK_CurrentParameter:
  case CK_LeftParen:
  case CK_RightParen:
  case CK_LeftBracket:
  case CK_RightBracket:
  case CK_LeftBrace:
  case CK_RightBrace:
  case CK_LeftAngle:
  case CK_RightAngle:
  case CK_Comma:
  case CK_Colon:
  case CK_SemiColon:
  case CK_Equal:
  case CK_HorizontalSpace:
  case CK_VerticalSpace:
    return Chunk(Kind, Text);
      
  case CK_Optional: {
    std::auto_ptr<CodeCompletionString> Opt(Optional->Clone());
    return CreateOptional(Opt);
  }
  }

  // Silence GCC warning.
  return Chunk();
}

void
CodeCompletionString::Chunk::Destroy() {
  switch (Kind) {
  case CK_Optional: 
    delete Optional; 
    break;
      
  case CK_TypedText:
  case CK_Text: 
  case CK_Placeholder:
  case CK_Informative:
  case CK_ResultType:
  case CK_CurrentParameter:
    delete [] Text;
    break;

  case CK_LeftParen:
  case CK_RightParen:
  case CK_LeftBracket:
  case CK_RightBracket:
  case CK_LeftBrace:
  case CK_RightBrace:
  case CK_LeftAngle:
  case CK_RightAngle:
  case CK_Comma:
  case CK_Colon:
  case CK_SemiColon:
  case CK_Equal:
  case CK_HorizontalSpace:
  case CK_VerticalSpace:
    break;
  }
}

void CodeCompletionString::clear() {
  std::for_each(Chunks.begin(), Chunks.end(), 
                std::mem_fun_ref(&Chunk::Destroy));
  Chunks.clear();
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

CodeCompletionString *
CodeCompletionString::Clone(CodeCompletionString *Result) const {
  if (!Result)
    Result = new CodeCompletionString;
  for (iterator C = begin(), CEnd = end(); C != CEnd; ++C)
    Result->AddChunk(C->Clone());
  return Result;
}

static void WriteUnsigned(llvm::raw_ostream &OS, unsigned Value) {
  OS.write((const char *)&Value, sizeof(unsigned));
}

static bool ReadUnsigned(const char *&Memory, const char *MemoryEnd,
                         unsigned &Value) {
  if (Memory + sizeof(unsigned) > MemoryEnd)
    return true;

  memmove(&Value, Memory, sizeof(unsigned));
  Memory += sizeof(unsigned);
  return false;
}

void CodeCompletionString::Serialize(llvm::raw_ostream &OS) const {
  // Write the number of chunks.
  WriteUnsigned(OS, size());

  for (iterator C = begin(), CEnd = end(); C != CEnd; ++C) {
    WriteUnsigned(OS, C->Kind);

    switch (C->Kind) {
    case CK_TypedText:
    case CK_Text:
    case CK_Placeholder:
    case CK_Informative:
    case CK_ResultType:
    case CK_CurrentParameter: {
      const char *Text = C->Text;
      unsigned StrLen = strlen(Text);
      WriteUnsigned(OS, StrLen);
      OS.write(Text, StrLen);
      break;
    }

    case CK_Optional:
      C->Optional->Serialize(OS);
      break;

    case CK_LeftParen:
    case CK_RightParen:
    case CK_LeftBracket:
    case CK_RightBracket:
    case CK_LeftBrace:
    case CK_RightBrace:
    case CK_LeftAngle:
    case CK_RightAngle:
    case CK_Comma:
    case CK_Colon:
    case CK_SemiColon:
    case CK_Equal:
    case CK_HorizontalSpace:
    case CK_VerticalSpace:
      break;
    }
  }
}

bool CodeCompletionString::Deserialize(const char *&Str, const char *StrEnd) {
  if (Str == StrEnd || *Str == 0)
    return false;

  unsigned NumBlocks;
  if (ReadUnsigned(Str, StrEnd, NumBlocks))
    return false;

  for (unsigned I = 0; I != NumBlocks; ++I) {
    if (Str + 1 >= StrEnd)
      break;

    // Parse the next kind.
    unsigned KindValue;
    if (ReadUnsigned(Str, StrEnd, KindValue))
      return false;

    switch (ChunkKind Kind = (ChunkKind)KindValue) {
    case CK_TypedText:
    case CK_Text:
    case CK_Placeholder:
    case CK_Informative:
    case CK_ResultType:
    case CK_CurrentParameter: {
      unsigned StrLen;
      if (ReadUnsigned(Str, StrEnd, StrLen) || (Str + StrLen > StrEnd))
        return false;

      AddChunk(Chunk(Kind, StringRef(Str, StrLen)));
      Str += StrLen;
      break;
    }

    case CK_Optional: {
      std::auto_ptr<CodeCompletionString> Optional(new CodeCompletionString());
      if (Optional->Deserialize(Str, StrEnd))
        AddOptionalChunk(Optional);
      break;
    }

    case CK_LeftParen:
    case CK_RightParen:
    case CK_LeftBracket:
    case CK_RightBracket:
    case CK_LeftBrace:
    case CK_RightBrace:
    case CK_LeftAngle:
    case CK_RightAngle:
    case CK_Comma:
    case CK_Colon:
    case CK_SemiColon:
    case CK_Equal:
    case CK_HorizontalSpace:
    case CK_VerticalSpace:
      AddChunk(Chunk(Kind));
      break;      
    }
  };
  
  return true;
}

void CodeCompletionResult::Destroy() {
  if (Kind == RK_Pattern) {
    delete Pattern;
    Pattern = 0;
  }
}

unsigned CodeCompletionResult::getPriorityFromDecl(NamedDecl *ND) {
  if (!ND)
    return CCP_Unlikely;
  
  // Context-based decisions.
  DeclContext *DC = ND->getDeclContext()->getRedeclContext();
  if (DC->isFunctionOrMethod() || isa<BlockDecl>(DC))
    return CCP_LocalDeclaration;
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
      OS << Results[I].Declaration;
      if (Results[I].Hidden)
        OS << " (Hidden)";
      if (CodeCompletionString *CCS 
            = Results[I].CreateCodeCompletionString(SemaRef)) {
        OS << " : " << CCS->getAsString();
        delete CCS;
      }
        
      OS << '\n';
      break;
      
    case CodeCompletionResult::RK_Keyword:
      OS << Results[I].Keyword << '\n';
      break;
        
    case CodeCompletionResult::RK_Macro: {
      OS << Results[I].Macro->getName();
      if (CodeCompletionString *CCS 
            = Results[I].CreateCodeCompletionString(SemaRef)) {
        OS << " : " << CCS->getAsString();
        delete CCS;
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
          = Candidates[I].CreateSignatureString(CurrentArg, SemaRef)) {
      OS << "OVERLOAD: " << CCS->getAsString() << "\n";
      delete CCS;
    }
  }
}

void CodeCompletionResult::computeCursorKindAndAvailability() {
  switch (Kind) {
  case RK_Declaration:
    // Set the availability based on attributes.
    Availability = CXAvailability_Available;      
    if (Declaration->getAttr<UnavailableAttr>())
      Availability = CXAvailability_NotAvailable;
    else if (Declaration->getAttr<DeprecatedAttr>())
      Availability = CXAvailability_Deprecated;
      
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
}

/// \brief Retrieve the name that should be used to order a result.
///
/// If the name needs to be constructed as a string, that string will be
/// saved into Saved and the returned StringRef will refer to it.
static llvm::StringRef getOrderedName(const CodeCompletionResult &R,
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
  llvm::StringRef XStr = getOrderedName(X, XSaved);
  llvm::StringRef YStr = getOrderedName(Y, YSaved);
  int cmp = XStr.compare_lower(YStr);
  if (cmp)
    return cmp < 0;
  
  // If case-insensitive comparison fails, try case-sensitive comparison.
  cmp = XStr.compare(YStr);
  if (cmp)
    return cmp < 0;
  
  return false;
}

void 
CIndexCodeCompleteConsumer::ProcessCodeCompleteResults(Sema &SemaRef,
                                                 CodeCompletionContext Context,
                                                 CodeCompletionResult *Results,
                                                       unsigned NumResults) {
  // Print the results.
  for (unsigned I = 0; I != NumResults; ++I) {
    WriteUnsigned(OS, Results[I].CursorKind);
    WriteUnsigned(OS, Results[I].Priority);
    WriteUnsigned(OS, Results[I].Availability);
    CodeCompletionString *CCS = Results[I].CreateCodeCompletionString(SemaRef);
    assert(CCS && "No code-completion string?");
    CCS->Serialize(OS);
    delete CCS;
  }
}

void 
CIndexCodeCompleteConsumer::ProcessOverloadCandidates(Sema &SemaRef,
                                                      unsigned CurrentArg,
                                                OverloadCandidate *Candidates,
                                                       unsigned NumCandidates) {
  for (unsigned I = 0; I != NumCandidates; ++I) {
    WriteUnsigned(OS, CXCursor_NotImplemented);
    WriteUnsigned(OS, /*Priority=*/I);
    WriteUnsigned(OS, /*Availability=*/CXAvailability_Available);
    CodeCompletionString *CCS
      = Candidates[I].CreateSignatureString(CurrentArg, SemaRef);
    assert(CCS && "No code-completion string?");
    CCS->Serialize(OS);
    delete CCS;
  }
}
