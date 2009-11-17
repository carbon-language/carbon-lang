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
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Compiler.h"
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
  case CK_CurrentParameter: {
    char *New = new char [Text.size() + 1];
    std::memcpy(New, Text.data(), Text.size());
    New[Text.size()] = '\0';
    this->Text = New;
    break;
  }

  case CK_Optional:
    llvm::llvm_unreachable("Optional strings cannot be created from text");
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
CodeCompletionString::Chunk::CreateCurrentParameter(
                                                StringRef CurrentParameter) {
  return Chunk(CK_CurrentParameter, CurrentParameter);
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
    case CK_Optional: OS << "{#" << C->Optional->getAsString() << "#}"; break;
    case CK_Placeholder: OS << "<#" << C->Text << "#>"; break;
    case CK_Informative: OS << "[#" << C->Text << "#]"; break;
    case CK_CurrentParameter: OS << "<#" << C->Text << "#>"; break;
    default: OS << C->Text; break;
    }
  }
  OS.flush();
  return Result;
}


namespace {
  // Escape a string for XML-like formatting.
  struct EscapedString {
    EscapedString(llvm::StringRef Str) : Str(Str) { }
    
    llvm::StringRef Str;
  };
  
  llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, EscapedString EStr) {
    llvm::StringRef Str = EStr.Str;
    while (!Str.empty()) {
      // Find the next escaped character.
      llvm::StringRef::size_type Pos = Str.find_first_of("<>&\"'");
      
      // Print everything before that escaped character.
      OS << Str.substr(0, Pos);

      // If we didn't find any escaped characters, we're done.
      if (Pos == llvm::StringRef::npos)
        break;
      
      // Print the appropriate escape sequence.
      switch (Str[Pos]) {
        case '<': OS << "&lt;"; break;
        case '>': OS << "&gt;"; break;
        case '&': OS << "&amp;"; break;
        case '"': OS << "&quot;"; break;
        case '\'': OS << "&apos;"; break;
      }
      
      // Remove everything up to and including that escaped character.
      Str = Str.substr(Pos + 1);
    }
    
    return OS;
  }
  
  /// \brief Remove XML-like escaping from a string.
  std::string UnescapeString(llvm::StringRef Str) {
    using llvm::StringRef;
    
    std::string Result;
    llvm::raw_string_ostream OS(Result);
    
    while (!Str.empty()) {
      StringRef::size_type Amp = Str.find('&');
      OS << Str.substr(0, Amp);
      
      if (Amp == StringRef::npos)
        break;
      
      StringRef::size_type Semi = Str.substr(Amp).find(';');
      if (Semi == StringRef::npos) {
        // Malformed input; do the best we can.
        OS << '&';
        Str = Str.substr(Amp + 1);
        continue;
      }
      
      char Unescaped = llvm::StringSwitch<char>(Str.substr(Amp + 1, Semi - 1))
        .Case("lt", '<')
        .Case("gt", '>')
        .Case("amp", '&')
        .Case("quot", '"')
        .Case("apos", '\'')
        .Default('\0');
      
      if (Unescaped)
        OS << Unescaped;
      else
        OS << Str.substr(Amp, Semi + 1);
      Str = Str.substr(Amp + Semi + 1);
    }
    
    return OS.str();
  }
}

void CodeCompletionString::Serialize(llvm::raw_ostream &OS) const {
  for (iterator C = begin(), CEnd = end(); C != CEnd; ++C) {
    switch (C->Kind) {
    case CK_TypedText:
      OS << "<typed-text>" << EscapedString(C->Text) << "</>";
      break;
    case CK_Text:
      OS << "<text>" << EscapedString(C->Text) << "</>";
      break;
    case CK_Optional:
      OS << "<optional>";
      C->Optional->Serialize(OS);
      OS << "</>";
      break;
    case CK_Placeholder:
      OS << "<placeholder>" << EscapedString(C->Text) << "</>";
      break;
    case CK_Informative:
      OS << "<informative>" << EscapedString(C->Text) << "</>";
      break;
    case CK_CurrentParameter:
      OS << "<current-parameter>" << EscapedString(C->Text) << "</>";
      break;
    case CK_LeftParen:
      OS << "<lparen/>";
      break;
    case CK_RightParen:
      OS << "<rparen/>";
      break;
    case CK_LeftBracket:
      OS << "<lbracket/>";
      break;
    case CK_RightBracket:
      OS << "<rbracket/>";
      break;
    case CK_LeftBrace:
      OS << "<lbrace/>";
      break;
    case CK_RightBrace:
      OS << "<rbrace/>";
      break;
    case CK_LeftAngle:
      OS << "<langle/>";
      break;
    case CK_RightAngle:
      OS << "<rangle/>";
      break;
    case CK_Comma:
      OS << "<comma/>";
      break;
    }  
  }
}

/// \brief Parse the next XML-ish tag of the form <blah>.
///
/// \param Str the string in which we're looking for the next tag.
///
/// \param TagPos if successful, will be set to the start of the tag we found.
///
/// \param Standalone will indicate whether this is a "standalone" tag that
/// has no associated data, e.g., <comma/>.
///
/// \param Terminator will indicate whether this is a terminating tag (that is
/// or starts with '/').
///
/// \returns the tag itself, without the angle brackets.
static llvm::StringRef ParseNextTag(llvm::StringRef Str, 
                                    llvm::StringRef::size_type &StartTag,
                                    llvm::StringRef::size_type &AfterTag,
                                    bool &Standalone, bool &Terminator) {
  using llvm::StringRef;
  
  Standalone = false;
  Terminator = false;
  AfterTag = StringRef::npos;
  
  // Find the starting '<'. 
  StartTag = Str.find('<');
  if (StartTag == StringRef::npos)
    return llvm::StringRef();
  
  // Find the corresponding '>'.
  llvm::StringRef::size_type EndTag = Str.substr(StartTag).find('>');
  if (EndTag == StringRef::npos)
    return llvm::StringRef();
  AfterTag = StartTag + EndTag + 1;
  
  // Determine whether this is a terminating tag.
  if (Str[StartTag + 1] == '/') {
    Terminator = true;
    Str = Str.substr(1);
    --EndTag;
  }
  
  // Determine whether this is a standalone tag.
  if (!Terminator && Str[StartTag + EndTag - 1] == '/') {
    Standalone = true;
    if (EndTag > 1)
      --EndTag;
  }

  return Str.substr(StartTag + 1, EndTag - 1);
}

CodeCompletionString *CodeCompletionString::Deserialize(llvm::StringRef &Str) {
  using llvm::StringRef;
  
  CodeCompletionString *Result = new CodeCompletionString;
  
  do {
    // Parse the next tag.
    StringRef::size_type StartTag, AfterTag;
    bool Standalone, Terminator;
    StringRef Tag = ParseNextTag(Str, StartTag, AfterTag, Standalone, 
                                 Terminator);
    
    if (StartTag == StringRef::npos)
      break;
    
    // Figure out what kind of chunk we have.
    const unsigned UnknownKind = 10000;
    unsigned Kind = llvm::StringSwitch<unsigned>(Tag)
      .Case("typed-text", CK_TypedText)
      .Case("text", CK_Text)
      .Case("optional", CK_Optional)
      .Case("placeholder", CK_Placeholder)
      .Case("informative", CK_Informative)
      .Case("current-parameter", CK_CurrentParameter)
      .Case("lparen", CK_LeftParen)
      .Case("rparen", CK_RightParen)
      .Case("lbracket", CK_LeftBracket)
      .Case("rbracket", CK_RightBracket)
      .Case("lbrace", CK_LeftBrace)
      .Case("rbrace", CK_RightBrace)
      .Case("langle", CK_LeftAngle)
      .Case("rangle", CK_RightAngle)
      .Case("comma", CK_Comma)
      .Default(UnknownKind);
    
    // If we've hit a terminator tag, we're done.
    if (Terminator)
      break;
    
    // Consume the tag.
    Str = Str.substr(AfterTag);

    // Handle standalone tags now, since they don't need to be matched to
    // anything.
    if (Standalone) {
      // Ignore anything we don't know about.
      if (Kind == UnknownKind)
        continue;
      
      switch ((ChunkKind)Kind) {
      case CK_TypedText:
      case CK_Text:
      case CK_Optional:
      case CK_Placeholder:
      case CK_Informative:
      case CK_CurrentParameter:
        // There is no point in creating empty chunks of these kinds.
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
        Result->AddChunk(Chunk((ChunkKind)Kind));
        break;
      }
      
      continue;
    }
    
    if (Kind == CK_Optional) {
      // Deserialize the optional code-completion string.
      std::auto_ptr<CodeCompletionString> Optional(Deserialize(Str));
      Result->AddOptionalChunk(Optional);
    }
    
    StringRef EndTag = ParseNextTag(Str, StartTag, AfterTag, Standalone, 
                                    Terminator);
    if (StartTag == StringRef::npos || !Terminator || Standalone)
      break; // Parsing failed; just give up.
    
    if (EndTag.empty() || Tag == EndTag) {
      // Found the matching end tag. Add this chunk based on the text
      // between the tags, then consume that input.
      StringRef Text = Str.substr(0, StartTag);
      switch ((ChunkKind)Kind) {
      case CK_TypedText:
      case CK_Text:
      case CK_Placeholder:
      case CK_Informative:
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
        Result->AddChunk(Chunk((ChunkKind)Kind, UnescapeString(Text)));
        break;
          
      case CK_Optional:
        // We've already added the optional chunk.
        break;
      }
    }
    
    // Remove this tag.
    Str = Str.substr(AfterTag);
  } while (!Str.empty());
  
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
PrintingCodeCompleteConsumer::ProcessCodeCompleteResults(Sema &SemaRef,
                                                         Result *Results, 
                                                         unsigned NumResults) {
  // Print the results.
  for (unsigned I = 0; I != NumResults; ++I) {
    OS << "COMPLETION: ";
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
        
    case Result::RK_Macro: {
      OS << Results[I].Macro->getName() << " : " << Results[I].Rank;
      if (CodeCompletionString *CCS 
          = Results[I].CreateCodeCompletionString(SemaRef)) {
        OS << " : " << CCS->getAsString();
        delete CCS;
      }
      OS << '\n';
      break;
    }
    }
  }
  
  // Once we've printed the code-completion results, suppress remaining
  // diagnostics.
  // FIXME: Move this somewhere else!
  SemaRef.PP.getDiagnostics().setSuppressAllDiagnostics();
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

  // Once we've printed the code-completion results, suppress remaining
  // diagnostics.
  // FIXME: Move this somewhere else!
  SemaRef.PP.getDiagnostics().setSuppressAllDiagnostics();
}

void 
CIndexCodeCompleteConsumer::ProcessCodeCompleteResults(Sema &SemaRef,
                                                       Result *Results, 
                                                       unsigned NumResults) {
  // Print the results.
  for (unsigned I = 0; I != NumResults; ++I) {
    OS << "COMPLETION:" << Results[I].Rank << ":";
    switch (Results[I].Kind) {
      case Result::RK_Declaration:
        if (RecordDecl *Record = dyn_cast<RecordDecl>(Results[I].Declaration)) {
          if (Record->isStruct())
            OS << "Struct:";
          else if (Record->isUnion())
            OS << "Union:";
          else
            OS << "Class:";
        } else if (ObjCMethodDecl *Method
                     = dyn_cast<ObjCMethodDecl>(Results[I].Declaration)) {
          if (Method->isInstanceMethod())
            OS << "ObjCInstanceMethod:";
          else
            OS << "ObjCClassMethod:";
        } else {
          OS << Results[I].Declaration->getDeclKindName() << ":";
        }
        if (CodeCompletionString *CCS 
              = Results[I].CreateCodeCompletionString(SemaRef)) {
          CCS->Serialize(OS);
          delete CCS;
        } else {
          OS << "<typed-text>" 
             << Results[I].Declaration->getNameAsString() 
             << "</>";
        }
        
        OS << '\n';
        break;
        
      case Result::RK_Keyword:
        OS << "Keyword:<typed-text>" << Results[I].Keyword << "</>\n";
        break;
        
      case Result::RK_Macro: {
        OS << "Macro:";
        if (CodeCompletionString *CCS 
              = Results[I].CreateCodeCompletionString(SemaRef)) {
          CCS->Serialize(OS);
          delete CCS;
        } else {
          OS << "<typed-text>" << Results[I].Macro->getName() << "</>";
        }
        OS << '\n';
        break;
      }
    }
  }
  
  // Once we've printed the code-completion results, suppress remaining
  // diagnostics.
  // FIXME: Move this somewhere else!
  SemaRef.PP.getDiagnostics().setSuppressAllDiagnostics();
}

void 
CIndexCodeCompleteConsumer::ProcessOverloadCandidates(Sema &SemaRef,
                                                      unsigned CurrentArg,
                                                OverloadCandidate *Candidates,
                                                       unsigned NumCandidates) {
  for (unsigned I = 0; I != NumCandidates; ++I) {
    if (CodeCompletionString *CCS
        = Candidates[I].CreateSignatureString(CurrentArg, SemaRef)) {
      OS << "OVERLOAD:";
      CCS->Serialize(OS);
      OS << '\n';
      delete CCS;
    }
  }
  
  // Once we've printed the code-completion results, suppress remaining
  // diagnostics.
  // FIXME: Move this somewhere else!
  SemaRef.PP.getDiagnostics().setSuppressAllDiagnostics();
}
