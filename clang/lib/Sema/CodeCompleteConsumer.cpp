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
#include "clang/AST/DeclCXX.h"
#include "clang/Parse/Scope.h"
#include "clang/Lex/Preprocessor.h"
#include "Sema.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstring>
#include <functional>
using namespace clang;

//===----------------------------------------------------------------------===//
// Code completion string implementation
//===----------------------------------------------------------------------===//
CodeCompletionString::Chunk::Chunk(ChunkKind Kind, const char *Text) 
  : Kind(Kind), Text(0)
{
  assert((Kind == CK_Text || Kind == CK_Placeholder || Kind == CK_Informative)
         && "Invalid text chunk kind");
  char *New = new char [std::strlen(Text) + 1];
  std::strcpy(New, Text);
  this->Text = New;
}

CodeCompletionString::Chunk
CodeCompletionString::Chunk::CreateText(const char *Text) {
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
CodeCompletionString::Chunk::CreatePlaceholder(const char *Placeholder) {
  return Chunk(CK_Placeholder, Placeholder);
}

CodeCompletionString::Chunk 
CodeCompletionString::Chunk::CreateInformative(const char *Informative) {
  return Chunk(CK_Informative, Informative);
}

void
CodeCompletionString::Chunk::Destroy() {
  switch (Kind) {
  case CK_Optional: 
    delete Optional; 
    break;
      
  case CK_Text: 
  case CK_Placeholder:
  case CK_Informative:
    delete [] Text; 
    break;
  }
}

CodeCompletionString::~CodeCompletionString() {
  std::for_each(Chunks.begin(), Chunks.end(), 
                std::mem_fun_ref(&Chunk::Destroy));
}

std::string CodeCompletionString::getAsString() const {
  std::string Result;
  llvm::raw_string_ostream OS(Result);
                          
  for (iterator C = begin(), CEnd = end(); C != CEnd; ++C) {
    switch (C->Kind) {
    case CK_Text: OS << C->Text; break;
    case CK_Optional: OS << "{#" << C->Optional->getAsString() << "#}"; break;
    case CK_Placeholder: OS << "<#" << C->Text << "#>"; break;
    case CK_Informative: OS << "[#" << C->Text << "#]"; break;
    }
  }
  OS.flush();
  return Result;
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
PrintingCodeCompleteConsumer::ProcessCodeCompleteResults(Result *Results, 
                                                         unsigned NumResults) {
  // Print the results.
  for (unsigned I = 0; I != NumResults; ++I) {
    switch (Results[I].Kind) {
    case Result::RK_Declaration:
      OS << Results[I].Declaration->getNameAsString() << " : " 
         << Results[I].Rank;
      if (Results[I].Hidden)
        OS << " (Hidden)";
      if (CodeCompletionString *CCS 
            = Results[I].CreateCodeCompletionString(SemaRef)) {
        OS << " : " << CCS->getAsString();
        delete CCS;
      }
        
      OS << '\n';
      break;
      
    case Result::RK_Keyword:
      OS << Results[I].Keyword << " : " << Results[I].Rank << '\n';
      break;
    }
  }
  
  // Once we've printed the code-completion results, suppress remaining
  // diagnostics.
  // FIXME: Move this somewhere else!
  SemaRef.PP.getDiagnostics().setSuppressAllDiagnostics();
}

void 
PrintingCodeCompleteConsumer::ProcessOverloadCandidates(unsigned CurrentArg,
                                              OverloadCandidate *Candidates,
                                                     unsigned NumCandidates) {
  for (unsigned I = 0; I != NumCandidates; ++I) {
    if (CodeCompletionString *CCS
          = Candidates[I].CreateSignatureString(CurrentArg, SemaRef)) {
      OS << CCS->getAsString() << "\n";
      delete CCS;
    }
  }

  // Once we've printed the code-completion results, suppress remaining
  // diagnostics.
  // FIXME: Move this somewhere else!
  SemaRef.PP.getDiagnostics().setSuppressAllDiagnostics();
}
