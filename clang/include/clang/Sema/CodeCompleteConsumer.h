//===---- CodeCompleteConsumer.h - Code Completion Interface ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the CodeCompleteConsumer class.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_SEMA_CODECOMPLETECONSUMER_H
#define LLVM_CLANG_SEMA_CODECOMPLETECONSUMER_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <memory>
#include <string>

namespace llvm {
class raw_ostream;
}

namespace clang {

class FunctionDecl;
class FunctionType;
class FunctionTemplateDecl;
class IdentifierInfo;
class NamedDecl;
class NestedNameSpecifier;
class Sema;

/// \brief A "string" used to describe how code completion can
/// be performed for an entity.
///
/// A code completion string typically shows how a particular entity can be 
/// used. For example, the code completion string for a function would show
/// the syntax to call it, including the parentheses, placeholders for the 
/// arguments, etc.  
class CodeCompletionString {
public:
  /// \brief The different kinds of "chunks" that can occur within a code
  /// completion string.
  enum ChunkKind {
    /// \brief The piece of text that the user is expected to type to
    /// match the code-completion string, typically a keyword or the name of a
    /// declarator or macro.
    CK_TypedText,
    /// \brief A piece of text that should be placed in the buffer, e.g.,
    /// parentheses or a comma in a function call.
    CK_Text,
    /// \brief A code completion string that is entirely optional. For example,
    /// an optional code completion string that describes the default arguments
    /// in a function call.
    CK_Optional,
    /// \brief A string that acts as a placeholder for, e.g., a function 
    /// call argument.
    CK_Placeholder,
    /// \brief A piece of text that describes something about the result but
    /// should not be inserted into the buffer.
    CK_Informative,
    /// \brief A piece of text that describes the parameter that corresponds
    /// to the code-completion location within a function call, message send,
    /// macro invocation, etc.
    CK_CurrentParameter,
    /// \brief A left parenthesis ('(').
    CK_LeftParen,
    /// \brief A right parenthesis (')').
    CK_RightParen,
    /// \brief A left bracket ('[').
    CK_LeftBracket,
    /// \brief A right bracket (']').
    CK_RightBracket,
    /// \brief A left brace ('{').
    CK_LeftBrace,
    /// \brief A right brace ('}').
    CK_RightBrace,
    /// \brief A left angle bracket ('<').
    CK_LeftAngle,
    /// \brief A right angle bracket ('>').
    CK_RightAngle,
    /// \brief A comma separator (',').
    CK_Comma
  };
  
  /// \brief One piece of the code completion string.
  struct Chunk {
    /// \brief The kind of data stored in this piece of the code completion 
    /// string.
    ChunkKind Kind;
    
    union {
      /// \brief The text string associated with a CK_Text, CK_Placeholder,
      /// CK_Informative, or CK_Comma chunk.
      /// The string is owned by the chunk and will be deallocated 
      /// (with delete[]) when the chunk is destroyed.
      const char *Text;
      
      /// \brief The code completion string associated with a CK_Optional chunk.
      /// The optional code completion string is owned by the chunk, and will
      /// be deallocated (with delete) when the chunk is destroyed.
      CodeCompletionString *Optional;
    };
    
    Chunk() : Kind(CK_Text), Text(0) { }
    
    Chunk(ChunkKind Kind, llvm::StringRef Text = "");
    
    /// \brief Create a new text chunk.
    static Chunk CreateText(llvm::StringRef Text);

    /// \brief Create a new optional chunk.
    static Chunk CreateOptional(std::auto_ptr<CodeCompletionString> Optional);

    /// \brief Create a new placeholder chunk.
    static Chunk CreatePlaceholder(llvm::StringRef Placeholder);

    /// \brief Create a new informative chunk.
    static Chunk CreateInformative(llvm::StringRef Informative);

    /// \brief Create a new current-parameter chunk.
    static Chunk CreateCurrentParameter(llvm::StringRef CurrentParameter);

    /// \brief Clone the given chunk.
    Chunk Clone() const;
    
    /// \brief Destroy this chunk, deallocating any memory it owns.
    void Destroy();
  };
  
private:
  /// \brief The chunks stored in this string.
  llvm::SmallVector<Chunk, 4> Chunks;
  
  CodeCompletionString(const CodeCompletionString &); // DO NOT IMPLEMENT
  CodeCompletionString &operator=(const CodeCompletionString &); // DITTO
  
public:
  CodeCompletionString() { }
  ~CodeCompletionString();
  
  typedef llvm::SmallVector<Chunk, 4>::const_iterator iterator;
  iterator begin() const { return Chunks.begin(); }
  iterator end() const { return Chunks.end(); }
  bool empty() const { return Chunks.empty(); }
  unsigned size() const { return Chunks.size(); }
  
  Chunk &operator[](unsigned I) {
    assert(I < size() && "Chunk index out-of-range");
    return Chunks[I];
  }

  const Chunk &operator[](unsigned I) const {
    assert(I < size() && "Chunk index out-of-range");
    return Chunks[I];
  }
  
  /// \brief Add a new typed-text chunk.
  /// The text string will be copied.
  void AddTypedTextChunk(llvm::StringRef Text) { 
    Chunks.push_back(Chunk(CK_TypedText, Text));
  }
  
  /// \brief Add a new text chunk.
  /// The text string will be copied.
  void AddTextChunk(llvm::StringRef Text) { 
    Chunks.push_back(Chunk::CreateText(Text)); 
  }
  
  /// \brief Add a new optional chunk.
  void AddOptionalChunk(std::auto_ptr<CodeCompletionString> Optional) {
    Chunks.push_back(Chunk::CreateOptional(Optional));
  }
  
  /// \brief Add a new placeholder chunk.
  /// The placeholder text will be copied.
  void AddPlaceholderChunk(llvm::StringRef Placeholder) {
    Chunks.push_back(Chunk::CreatePlaceholder(Placeholder));
  }

  /// \brief Add a new informative chunk.
  /// The text will be copied.
  void AddInformativeChunk(llvm::StringRef Text) {
    Chunks.push_back(Chunk::CreateInformative(Text));
  }

  /// \brief Add a new current-parameter chunk.
  /// The text will be copied.
  void AddCurrentParameterChunk(llvm::StringRef CurrentParameter) {
    Chunks.push_back(Chunk::CreateCurrentParameter(CurrentParameter));
  }
  
  /// \brief Add a new chunk.
  void AddChunk(Chunk C) { Chunks.push_back(C); }
  
  /// \brief Returns the text in the TypedText chunk.
  const char *getTypedText() const;

  /// \brief Retrieve a string representation of the code completion string,
  /// which is mainly useful for debugging.
  std::string getAsString() const; 
  
  /// \brief Clone this code-completion string.
  CodeCompletionString *Clone() const;
  
  /// \brief Serialize this code-completion string to the given stream.
  void Serialize(llvm::raw_ostream &OS) const;
  
  /// \brief Deserialize a code-completion string from the given string.
  static CodeCompletionString *Deserialize(llvm::StringRef &Str);  
};
  
llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, 
                              const CodeCompletionString &CCS);

/// \brief Abstract interface for a consumer of code-completion 
/// information.
class CodeCompleteConsumer {
protected:
  /// \brief Whether to include macros in the code-completion results.
  bool IncludeMacros;
  
public:
  /// \brief Captures a result of code completion.
  struct Result {
    /// \brief Describes the kind of result generated.
    enum ResultKind {
      RK_Declaration = 0, //< Refers to a declaration
      RK_Keyword,         //< Refers to a keyword or symbol.
      RK_Macro,           //< Refers to a macro
      RK_Pattern          //< Refers to a precomputed pattern.
    };
    
    /// \brief The kind of result stored here.
    ResultKind Kind;
    
    union {
      /// \brief When Kind == RK_Declaration, the declaration we are referring
      /// to.
      NamedDecl *Declaration;
      
      /// \brief When Kind == RK_Keyword, the string representing the keyword 
      /// or symbol's spelling.
      const char *Keyword;
      
      /// \brief When Kind == RK_Pattern, the code-completion string that
      /// describes the completion text to insert.
      CodeCompletionString *Pattern;
      
      /// \brief When Kind == RK_Macro, the identifier that refers to a macro.
      IdentifierInfo *Macro;
    };
    
    /// \brief Describes how good this result is, with zero being the best
    /// result and progressively higher numbers representing poorer results.
    unsigned Rank;
    
    /// \brief Specifiers which parameter (of a function, Objective-C method,
    /// macro, etc.) we should start with when formatting the result.
    unsigned StartParameter;
    
    /// \brief Whether this result is hidden by another name.
    bool Hidden : 1;
    
    /// \brief Whether this result was found via lookup into a base class.
    bool QualifierIsInformative : 1;
    
    /// \brief Whether this declaration is the beginning of a 
    /// nested-name-specifier and, therefore, should be followed by '::'.
    bool StartsNestedNameSpecifier : 1;

    /// \brief Whether all parameters (of a function, Objective-C
    /// method, etc.) should be considered "informative".
    bool AllParametersAreInformative : 1;

    /// \brief If the result should have a nested-name-specifier, this is it.
    /// When \c QualifierIsInformative, the nested-name-specifier is 
    /// informative rather than required.
    NestedNameSpecifier *Qualifier;
    
    /// \brief Build a result that refers to a declaration.
    Result(NamedDecl *Declaration, unsigned Rank, 
           NestedNameSpecifier *Qualifier = 0,
           bool QualifierIsInformative = false)
      : Kind(RK_Declaration), Declaration(Declaration), Rank(Rank), 
        StartParameter(0), Hidden(false), 
        QualifierIsInformative(QualifierIsInformative),
        StartsNestedNameSpecifier(false), AllParametersAreInformative(false),
        Qualifier(Qualifier) { }
    
    /// \brief Build a result that refers to a keyword or symbol.
    Result(const char *Keyword, unsigned Rank)
      : Kind(RK_Keyword), Keyword(Keyword), Rank(Rank), StartParameter(0),
        Hidden(false), QualifierIsInformative(0), 
        StartsNestedNameSpecifier(false), AllParametersAreInformative(false),
        Qualifier(0) { }
    
    /// \brief Build a result that refers to a macro.
    Result(IdentifierInfo *Macro, unsigned Rank)
     : Kind(RK_Macro), Macro(Macro), Rank(Rank), StartParameter(0), 
       Hidden(false), QualifierIsInformative(0), 
       StartsNestedNameSpecifier(false), AllParametersAreInformative(false),
       Qualifier(0) { }

    /// \brief Build a result that refers to a pattern.
    Result(CodeCompletionString *Pattern, unsigned Rank)
      : Kind(RK_Pattern), Pattern(Pattern), Rank(Rank), StartParameter(0), 
        Hidden(false), QualifierIsInformative(0), 
        StartsNestedNameSpecifier(false), AllParametersAreInformative(false),
        Qualifier(0) { }
    
    /// \brief Retrieve the declaration stored in this result.
    NamedDecl *getDeclaration() const {
      assert(Kind == RK_Declaration && "Not a declaration result");
      return Declaration;
    }
    
    /// \brief Retrieve the keyword stored in this result.
    const char *getKeyword() const {
      assert(Kind == RK_Keyword && "Not a keyword result");
      return Keyword;
    }
    
    /// \brief Create a new code-completion string that describes how to insert
    /// this result into a program.
    CodeCompletionString *CreateCodeCompletionString(Sema &S);
    
    void Destroy();
  };
    
  class OverloadCandidate {
  public:
    /// \brief Describes the type of overload candidate.
    enum CandidateKind {
      /// \brief The candidate is a function declaration.
      CK_Function,
      /// \brief The candidate is a function template.
      CK_FunctionTemplate,
      /// \brief The "candidate" is actually a variable, expression, or block
      /// for which we only have a function prototype.
      CK_FunctionType
    };
    
  private:
    /// \brief The kind of overload candidate.
    CandidateKind Kind;
    
    union {
      /// \brief The function overload candidate, available when 
      /// Kind == CK_Function.
      FunctionDecl *Function;
      
      /// \brief The function template overload candidate, available when
      /// Kind == CK_FunctionTemplate.
      FunctionTemplateDecl *FunctionTemplate;
      
      /// \brief The function type that describes the entity being called,
      /// when Kind == CK_FunctionType.
      const FunctionType *Type;
    };
    
  public:
    OverloadCandidate(FunctionDecl *Function)
      : Kind(CK_Function), Function(Function) { }

    OverloadCandidate(FunctionTemplateDecl *FunctionTemplateDecl)
      : Kind(CK_FunctionTemplate), FunctionTemplate(FunctionTemplate) { }

    OverloadCandidate(const FunctionType *Type)
      : Kind(CK_FunctionType), Type(Type) { }

    /// \brief Determine the kind of overload candidate.
    CandidateKind getKind() const { return Kind; }
    
    /// \brief Retrieve the function overload candidate or the templated 
    /// function declaration for a function template.
    FunctionDecl *getFunction() const;
    
    /// \brief Retrieve the function template overload candidate.
    FunctionTemplateDecl *getFunctionTemplate() const {
      assert(getKind() == CK_FunctionTemplate && "Not a function template");
      return FunctionTemplate;
    }
    
    /// \brief Retrieve the function type of the entity, regardless of how the
    /// function is stored.
    const FunctionType *getFunctionType() const;
    
    /// \brief Create a new code-completion string that describes the function
    /// signature of this overload candidate.
    CodeCompletionString *CreateSignatureString(unsigned CurrentArg, 
                                                Sema &S) const;    
  };
  
  CodeCompleteConsumer() : IncludeMacros(false) { }
  
  explicit CodeCompleteConsumer(bool IncludeMacros)
    : IncludeMacros(IncludeMacros) { }
  
  /// \brief Whether the code-completion consumer wants to see macros.
  bool includeMacros() const { return IncludeMacros; }
  
  /// \brief Deregisters and destroys this code-completion consumer.
  virtual ~CodeCompleteConsumer();
    
  /// \name Code-completion callbacks
  //@{
  /// \brief Process the finalized code-completion results.
  virtual void ProcessCodeCompleteResults(Sema &S, Result *Results,
                                          unsigned NumResults) { }

  /// \param S the semantic-analyzer object for which code-completion is being
  /// done.
  ///
  /// \param CurrentArg the index of the current argument.
  ///
  /// \param Candidates an array of overload candidates.
  ///
  /// \param NumCandidates the number of overload candidates
  virtual void ProcessOverloadCandidates(Sema &S, unsigned CurrentArg,
                                         OverloadCandidate *Candidates,
                                         unsigned NumCandidates) { }
  //@}
};
  
/// \brief A simple code-completion consumer that prints the results it 
/// receives in a simple format.
class PrintingCodeCompleteConsumer : public CodeCompleteConsumer {
  /// \brief The raw output stream.
  llvm::raw_ostream &OS;
    
public:
  /// \brief Create a new printing code-completion consumer that prints its
  /// results to the given raw output stream.
  PrintingCodeCompleteConsumer(bool IncludeMacros,
                               llvm::raw_ostream &OS)
    : CodeCompleteConsumer(IncludeMacros), OS(OS) { }
  
  /// \brief Prints the finalized code-completion results.
  virtual void ProcessCodeCompleteResults(Sema &S, Result *Results,
                                          unsigned NumResults);
  
  virtual void ProcessOverloadCandidates(Sema &S, unsigned CurrentArg,
                                         OverloadCandidate *Candidates,
                                         unsigned NumCandidates);  
};
  
/// \brief A code-completion consumer that prints the results it receives
/// in a format that is parsable by the CIndex library.
class CIndexCodeCompleteConsumer : public CodeCompleteConsumer {
  /// \brief The raw output stream.
  llvm::raw_ostream &OS;
  
public:
  /// \brief Create a new CIndex code-completion consumer that prints its
  /// results to the given raw output stream in a format readable to the CIndex
  /// library.
  CIndexCodeCompleteConsumer(bool IncludeMacros, llvm::raw_ostream &OS)
    : CodeCompleteConsumer(IncludeMacros), OS(OS) { }
  
  /// \brief Prints the finalized code-completion results.
  virtual void ProcessCodeCompleteResults(Sema &S, Result *Results, 
                                          unsigned NumResults);
  
  virtual void ProcessOverloadCandidates(Sema &S, unsigned CurrentArg,
                                         OverloadCandidate *Candidates,
                                         unsigned NumCandidates);  
};
  
} // end namespace clang

#endif // LLVM_CLANG_SEMA_CODECOMPLETECONSUMER_H
