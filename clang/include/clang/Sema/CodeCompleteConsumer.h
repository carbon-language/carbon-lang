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
#include <memory>
#include <string>

namespace llvm {
class raw_ostream;
}

namespace clang {

class FunctionDecl;
class FunctionType;
class FunctionTemplateDecl;
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
    CK_Informative
  };
  
  /// \brief One piece of the code completion string.
  struct Chunk {
    /// \brief The kind of data stored in this piece of the code completion 
    /// string.
    ChunkKind Kind;
    
    union {
      /// \brief The text string associated with a CK_Text, CK_Placeholder,
      /// or CK_Informative chunk.
      /// The string is owned by the chunk and will be deallocated 
      /// (with delete[]) when the chunk is destroyed.
      const char *Text;
      
      /// \brief The code completion string associated with a CK_Optional chunk.
      /// The optional code completion string is owned by the chunk, and will
      /// be deallocated (with delete) when the chunk is destroyed.
      CodeCompletionString *Optional;
    };
    
    Chunk() : Kind(CK_Text), Text(0) { }
    
  private:
    Chunk(ChunkKind Kind, const char *Text);
          
  public:
    /// \brief Create a new text chunk.
    static Chunk CreateText(const char *Text);

    /// \brief Create a new optional chunk.
    static Chunk CreateOptional(std::auto_ptr<CodeCompletionString> Optional);

    /// \brief Create a new placeholder chunk.
    static Chunk CreatePlaceholder(const char *Placeholder);

    /// \brief Create a new informative chunk.
    static Chunk CreateInformative(const char *Informative);

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
  
  /// \brief Add a new text chunk.
  /// The text string will be copied.
  void AddTextChunk(const char *Text) { 
    Chunks.push_back(Chunk::CreateText(Text)); 
  }
  
  /// \brief Add a new optional chunk.
  void AddOptionalChunk(std::auto_ptr<CodeCompletionString> Optional) {
    Chunks.push_back(Chunk::CreateOptional(Optional));
  }
  
  /// \brief Add a new placeholder chunk.
  /// The placeholder text will be copied.
  void AddPlaceholderChunk(const char *Placeholder) {
    Chunks.push_back(Chunk::CreatePlaceholder(Placeholder));
  }

  /// \brief Add a new informative chunk.
  /// The text will be copied.
  void AddInformativeChunk(const char *Text) {
    Chunks.push_back(Chunk::CreateInformative(Text));
  }
  
  /// \brief Retrieve a string representation of the code completion string,
  /// which is mainly useful for debugging.
  std::string getAsString() const;
};
  
/// \brief Abstract interface for a consumer of code-completion 
/// information.
class CodeCompleteConsumer {
public:
  /// \brief Captures a result of code completion.
  struct Result {
    /// \brief Describes the kind of result generated.
    enum ResultKind {
      RK_Declaration = 0, //< Refers to a declaration
      RK_Keyword          //< Refers to a keyword or symbol.
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
    };
    
    /// \brief Describes how good this result is, with zero being the best
    /// result and progressively higher numbers representing poorer results.
    unsigned Rank;
    
    /// \brief Whether this result is hidden by another name.
    bool Hidden : 1;
    
    /// \brief Whether this result was found via lookup into a base class.
    bool QualifierIsInformative : 1;
    
    /// \brief Whether this declaration is the beginning of a 
    /// nested-name-specifier and, therefore, should be followed by '::'.
    bool StartsNestedNameSpecifier : 1;
    
    /// \brief If the result should have a nested-name-specifier, this is it.
    /// When \c QualifierIsInformative, the nested-name-specifier is 
    /// informative rather than required.
    NestedNameSpecifier *Qualifier;
    
    /// \brief Build a result that refers to a declaration.
    Result(NamedDecl *Declaration, unsigned Rank, 
           NestedNameSpecifier *Qualifier = 0,
           bool QualifierIsInformative = false)
      : Kind(RK_Declaration), Declaration(Declaration), Rank(Rank), 
        Hidden(false), QualifierIsInformative(QualifierIsInformative),
        StartsNestedNameSpecifier(false), Qualifier(Qualifier) { }
    
    /// \brief Build a result that refers to a keyword or symbol.
    Result(const char *Keyword, unsigned Rank)
      : Kind(RK_Keyword), Keyword(Keyword), Rank(Rank), Hidden(false),
        QualifierIsInformative(0), StartsNestedNameSpecifier(false), 
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
  
  /// \brief Deregisters and destroys this code-completion consumer.
  virtual ~CodeCompleteConsumer();
    
  /// \name Code-completion callbacks
  //@{
  /// \brief Process the finalized code-completion results.
  virtual void ProcessCodeCompleteResults(Result *Results, 
                                          unsigned NumResults) { }
  
  /// \brief Process the set of overload candidates.
  ///
  /// \param CurrentArg the index of the current argument.
  ///
  /// \param Candidates an array of overload candidates.
  ///
  /// \param NumCandidates the number of overload candidates
  virtual void ProcessOverloadCandidates(unsigned CurrentArg,
                                         OverloadCandidate *Candidates,
                                         unsigned NumCandidates) { }
  //@}
};
  
/// \brief A simple code-completion consumer that prints the results it 
/// receives in a simple format.
class PrintingCodeCompleteConsumer : public CodeCompleteConsumer {
  /// \brief The semantic-analysis object to which this code-completion
  /// consumer is attached.
  Sema &SemaRef;
  
  /// \brief The raw output stream.
  llvm::raw_ostream &OS;
    
public:
  /// \brief Create a new printing code-completion consumer that prints its
  /// results to the given raw output stream.
  PrintingCodeCompleteConsumer(Sema &S, llvm::raw_ostream &OS)
    : SemaRef(S), OS(OS) { }
  
  /// \brief Prints the finalized code-completion results.
  virtual void ProcessCodeCompleteResults(Result *Results, 
                                          unsigned NumResults);
  
  virtual void ProcessOverloadCandidates(unsigned CurrentArg,
                                         OverloadCandidate *Candidates,
                                         unsigned NumCandidates);  
};
  
} // end namespace clang

#endif // LLVM_CLANG_SEMA_CODECOMPLETECONSUMER_H
