//=- PreprocessorTracker.cpp - Preprocessor tracking -*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===--------------------------------------------------------------------===//

#include "clang/Lex/LexDiagnostic.h"
#include "clang/Lex/MacroArgs.h"
#include "clang/Lex/PPCallbacks.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/StringPool.h"
#include "PreprocessorTracker.h"

namespace Modularize {

// Forwards.
class PreprocessorTrackerImpl;

// Some handle types

// String handle.
typedef llvm::PooledStringPtr StringHandle;

// Header handle.
typedef int HeaderHandle;
const HeaderHandle HeaderHandleInvalid = -1;

// Header inclusion path handle.
typedef int InclusionPathHandle;
const InclusionPathHandle InclusionPathHandleInvalid = -1;

// Some utility functions.

// Get a "file:line:column" source location string.
static std::string getSourceLocationString(clang::Preprocessor &PP,
                                           clang::SourceLocation Loc) {
  if (Loc.isInvalid())
    return std::string("(none)");
  else
    return Loc.printToString(PP.getSourceManager());
}

// Get just the file name from a source location.
static std::string getSourceLocationFile(clang::Preprocessor &PP,
                                         clang::SourceLocation Loc) {
  std::string Source(getSourceLocationString(PP, Loc));
  size_t Offset = Source.find(':', 2);
  if (Offset == std::string::npos)
    return Source;
  return Source.substr(0, Offset);
}

// Get just the line and column from a source location.
static void getSourceLocationLineAndColumn(clang::Preprocessor &PP,
                                           clang::SourceLocation Loc, int &Line,
                                           int &Column) {
  clang::PresumedLoc PLoc = PP.getSourceManager().getPresumedLoc(Loc);
  if (PLoc.isInvalid()) {
    Line = 0;
    Column = 0;
    return;
  }
  Line = PLoc.getLine();
  Column = PLoc.getColumn();
}

// Retrieve source snippet from file image.
std::string getSourceString(clang::Preprocessor &PP, clang::SourceRange Range) {
  clang::SourceLocation BeginLoc = Range.getBegin();
  clang::SourceLocation EndLoc = Range.getEnd();
  const char *BeginPtr = PP.getSourceManager().getCharacterData(BeginLoc);
  const char *EndPtr = PP.getSourceManager().getCharacterData(EndLoc);
  size_t Length = EndPtr - BeginPtr;
  return llvm::StringRef(BeginPtr, Length).trim().str();
}

// Retrieve source line from file image.
std::string getSourceLine(clang::Preprocessor &PP, clang::SourceLocation Loc) {
  const llvm::MemoryBuffer *MemBuffer =
      PP.getSourceManager().getBuffer(PP.getSourceManager().getFileID(Loc));
  const char *Buffer = MemBuffer->getBufferStart();
  const char *BufferEnd = MemBuffer->getBufferEnd();
  const char *BeginPtr = PP.getSourceManager().getCharacterData(Loc);
  const char *EndPtr = BeginPtr;
  while (BeginPtr > Buffer) {
    if (*BeginPtr == '\n') {
      BeginPtr++;
      break;
    }
    BeginPtr--;
  }
  while (EndPtr < BufferEnd) {
    if (*EndPtr == '\n') {
      break;
    }
    EndPtr++;
  }
  size_t Length = EndPtr - BeginPtr;
  return llvm::StringRef(BeginPtr, Length).str();
}

// Get the string for the Unexpanded macro instance.
// The soureRange is expected to end at the last token
// for the macro instance, which in the case of a function-style
// macro will be a ')', but for an object-style macro, it
// will be the macro name itself.
std::string getMacroUnexpandedString(clang::SourceRange Range,
                                     clang::Preprocessor &PP,
                                     llvm::StringRef MacroName,
                                     const clang::MacroInfo *MI) {
  clang::SourceLocation BeginLoc(Range.getBegin());
  const char *BeginPtr = PP.getSourceManager().getCharacterData(BeginLoc);
  size_t Length;
  std::string Unexpanded;
  if (MI->isFunctionLike()) {
    clang::SourceLocation EndLoc(Range.getEnd());
    const char *EndPtr = PP.getSourceManager().getCharacterData(EndLoc) + 1;
    Length = (EndPtr - BeginPtr) + 1; // +1 is ')' width.
  } else
    Length = MacroName.size();
  return llvm::StringRef(BeginPtr, Length).trim().str();
}

// Get the expansion for a macro instance, given the information
// provided by PPCallbacks.
std::string getMacroExpandedString(clang::Preprocessor &PP,
                                   llvm::StringRef MacroName,
                                   const clang::MacroInfo *MI,
                                   const clang::MacroArgs *Args) {
  std::string Expanded;
  // Walk over the macro Tokens.
  typedef clang::MacroInfo::tokens_iterator Iter;
  for (Iter I = MI->tokens_begin(), E = MI->tokens_end(); I != E; ++I) {
    clang::IdentifierInfo *II = I->getIdentifierInfo();
    int ArgNo = (II && Args ? MI->getArgumentNum(II) : -1);
    if (ArgNo == -1) {
      // This isn't an argument, just add it.
      if (II == NULL)
        Expanded += PP.getSpelling((*I)); // Not an identifier.
      else {
        // Token is for an identifier.
        std::string Name = II->getName().str();
        // Check for nexted macro references.
        clang::MacroInfo *MacroInfo = PP.getMacroInfo(II);
        if (MacroInfo != NULL)
          Expanded += getMacroExpandedString(PP, Name, MacroInfo, NULL);
        else
          Expanded += Name;
      }
      continue;
    }
    // We get here if it's a function-style macro with arguments.
    const clang::Token *ResultArgToks;
    const clang::Token *ArgTok = Args->getUnexpArgument(ArgNo);
    if (Args->ArgNeedsPreexpansion(ArgTok, PP))
      ResultArgToks = &(const_cast<clang::MacroArgs *>(Args))
          ->getPreExpArgument(ArgNo, MI, PP)[0];
    else
      ResultArgToks = ArgTok; // Use non-preexpanded Tokens.
    // If the arg token didn't expand into anything, ignore it.
    if (ResultArgToks->is(clang::tok::eof))
      continue;
    unsigned NumToks = clang::MacroArgs::getArgLength(ResultArgToks);
    // Append the resulting argument expansions.
    for (unsigned ArgumentIndex = 0; ArgumentIndex < NumToks; ++ArgumentIndex) {
      const clang::Token &AT = ResultArgToks[ArgumentIndex];
      clang::IdentifierInfo *II = AT.getIdentifierInfo();
      if (II == NULL)
        Expanded += PP.getSpelling(AT); // Not an identifier.
      else {
        // It's an identifier.  Check for further expansion.
        std::string Name = II->getName().str();
        clang::MacroInfo *MacroInfo = PP.getMacroInfo(II);
        if (MacroInfo != NULL)
          Expanded += getMacroExpandedString(PP, Name, MacroInfo, NULL);
        else
          Expanded += Name;
      }
    }
  }
  return Expanded;
}

// Get the string representing a vector of Tokens.
std::string
getTokensSpellingString(clang::Preprocessor &PP,
                        llvm::SmallVectorImpl<clang::Token> &Tokens) {
  std::string Expanded;
  // Walk over the macro Tokens.
  typedef llvm::SmallVectorImpl<clang::Token>::iterator Iter;
  for (Iter I = Tokens.begin(), E = Tokens.end(); I != E; ++I)
    Expanded += PP.getSpelling(*I); // Not an identifier.
  return llvm::StringRef(Expanded).trim().str();
}

// Get the expansion for a macro instance, given the information
// provided by PPCallbacks.
std::string getExpandedString(clang::Preprocessor &PP,
                              llvm::StringRef MacroName,
                              const clang::MacroInfo *MI,
                              const clang::MacroArgs *Args) {
  std::string Expanded;
  // Walk over the macro Tokens.
  typedef clang::MacroInfo::tokens_iterator Iter;
  for (Iter I = MI->tokens_begin(), E = MI->tokens_end(); I != E; ++I) {
    clang::IdentifierInfo *II = I->getIdentifierInfo();
    int ArgNo = (II && Args ? MI->getArgumentNum(II) : -1);
    if (ArgNo == -1) {
      // This isn't an argument, just add it.
      if (II == NULL)
        Expanded += PP.getSpelling((*I)); // Not an identifier.
      else {
        // Token is for an identifier.
        std::string Name = II->getName().str();
        // Check for nexted macro references.
        clang::MacroInfo *MacroInfo = PP.getMacroInfo(II);
        if (MacroInfo != NULL)
          Expanded += getMacroExpandedString(PP, Name, MacroInfo, NULL);
        else
          Expanded += Name;
      }
      continue;
    }
    // We get here if it's a function-style macro with arguments.
    const clang::Token *ResultArgToks;
    const clang::Token *ArgTok = Args->getUnexpArgument(ArgNo);
    if (Args->ArgNeedsPreexpansion(ArgTok, PP))
      ResultArgToks = &(const_cast<clang::MacroArgs *>(Args))
          ->getPreExpArgument(ArgNo, MI, PP)[0];
    else
      ResultArgToks = ArgTok; // Use non-preexpanded Tokens.
    // If the arg token didn't expand into anything, ignore it.
    if (ResultArgToks->is(clang::tok::eof))
      continue;
    unsigned NumToks = clang::MacroArgs::getArgLength(ResultArgToks);
    // Append the resulting argument expansions.
    for (unsigned ArgumentIndex = 0; ArgumentIndex < NumToks; ++ArgumentIndex) {
      const clang::Token &AT = ResultArgToks[ArgumentIndex];
      clang::IdentifierInfo *II = AT.getIdentifierInfo();
      if (II == NULL)
        Expanded += PP.getSpelling(AT); // Not an identifier.
      else {
        // It's an identifier.  Check for further expansion.
        std::string Name = II->getName().str();
        clang::MacroInfo *MacroInfo = PP.getMacroInfo(II);
        if (MacroInfo != NULL)
          Expanded += getMacroExpandedString(PP, Name, MacroInfo, NULL);
        else
          Expanded += Name;
      }
    }
  }
  return Expanded;
}

// We need some operator overloads for string handles.
bool operator==(const StringHandle &H1, const StringHandle &H2) {
  const char *S1 = (H1 ? *H1 : "");
  const char *S2 = (H2 ? *H2 : "");
  int Diff = strcmp(S1, S2);
  return Diff == 0;
}
bool operator!=(const StringHandle &H1, const StringHandle &H2) {
  const char *S1 = (H1 ? *H1 : "");
  const char *S2 = (H2 ? *H2 : "");
  int Diff = strcmp(S1, S2);
  return Diff != 0;
}
bool operator<(const StringHandle &H1, const StringHandle &H2) {
  const char *S1 = (H1 ? *H1 : "");
  const char *S2 = (H2 ? *H2 : "");
  int Diff = strcmp(S1, S2);
  return Diff < 0;
}
bool operator>(const StringHandle &H1, const StringHandle &H2) {
  const char *S1 = (H1 ? *H1 : "");
  const char *S2 = (H2 ? *H2 : "");
  int Diff = strcmp(S1, S2);
  return Diff > 0;
}

// Preprocessor item key.
//
// This class represents a location in a source file, for use
// as a key representing a unique name/file/line/column quadruplet,
// which in this case is used to identify a macro expansion instance,
// but could be used for other things as well.
// The file is a header file handle, the line is a line number,
// and the column is a column number.
class PPItemKey {
public:
  PPItemKey(clang::Preprocessor &PP, StringHandle Name, HeaderHandle File,
            clang::SourceLocation Loc)
      : Name(Name), File(File) {
    getSourceLocationLineAndColumn(PP, Loc, Line, Column);
  }
  PPItemKey(StringHandle Name, HeaderHandle File, int Line, int Column)
      : Name(Name), File(File), Line(Line), Column(Column) {}
  PPItemKey(const PPItemKey &Other)
      : Name(Other.Name), File(Other.File), Line(Other.Line),
        Column(Other.Column) {}
  PPItemKey() : File(HeaderHandleInvalid), Line(0), Column(0) {}
  bool operator==(const PPItemKey &Other) const {
    if (Name != Other.Name)
      return false;
    if (File != Other.File)
      return false;
    if (Line != Other.Line)
      return false;
    return Column == Other.Column;
  }
  bool operator<(const PPItemKey &Other) const {
    if (Name < Other.Name)
      return true;
    else if (Name > Other.Name)
      return false;
    if (File < Other.File)
      return true;
    else if (File > Other.File)
      return false;
    if (Line < Other.Line)
      return true;
    else if (Line > Other.Line)
      return false;
    return Column < Other.Column;
  }
  StringHandle Name;
  HeaderHandle File;
  int Line;
  int Column;
};

// Header inclusion path.
class HeaderInclusionPath {
public:
  HeaderInclusionPath(std::vector<HeaderHandle> HeaderInclusionPath)
      : Path(HeaderInclusionPath) {}
  HeaderInclusionPath(const HeaderInclusionPath &Other) : Path(Other.Path) {}
  HeaderInclusionPath() {}
  std::vector<HeaderHandle> Path;
};

// Macro expansion instance.
//
// This class represents an instance of a macro expansion with a
// unique value.  It also stores the unique header inclusion paths
// for use in telling the user the nested include path f
class MacroExpansionInstance {
public:
  MacroExpansionInstance(StringHandle MacroExpanded,
                         PPItemKey &DefinitionLocation,
                         StringHandle DefinitionSourceLine,
                         InclusionPathHandle H)
      : MacroExpanded(MacroExpanded), DefinitionLocation(DefinitionLocation),
        DefinitionSourceLine(DefinitionSourceLine) {
    InclusionPathHandles.push_back(H);
  }
  MacroExpansionInstance() {}

  // Check for the presence of a header inclusion path handle entry.
  // Return false if not found.
  bool haveInclusionPathHandle(InclusionPathHandle H) {
    for (std::vector<InclusionPathHandle>::iterator
             I = InclusionPathHandles.begin(),
             E = InclusionPathHandles.end();
         I != E; ++I) {
      if (*I == H)
        return true;
    }
    return InclusionPathHandleInvalid;
  }
  // Add a new header inclusion path entry, if not already present.
  void addInclusionPathHandle(InclusionPathHandle H) {
    if (!haveInclusionPathHandle(H))
      InclusionPathHandles.push_back(H);
  }

  // A string representing the macro instance after preprocessing.
  StringHandle MacroExpanded;
  // A file/line/column triplet representing the macro definition location.
  PPItemKey DefinitionLocation;
  // A place to save the macro definition line string.
  StringHandle DefinitionSourceLine;
  // The header inclusion path handles for all the instances.
  std::vector<InclusionPathHandle> InclusionPathHandles;
};

// Macro expansion instance tracker.
//
// This class represents one macro expansion, keyed by a PPItemKey.
// It stores a string representing the macro reference in the source,
// and a list of ConditionalExpansionInstances objects representing
// the unique value the condition expands to in instances of the header.
class MacroExpansionTracker {
public:
  MacroExpansionTracker(StringHandle MacroUnexpanded,
                        StringHandle MacroExpanded,
                        StringHandle InstanceSourceLine,
                        PPItemKey &DefinitionLocation,
                        StringHandle DefinitionSourceLine,
                        InclusionPathHandle InclusionPathHandle)
      : MacroUnexpanded(MacroUnexpanded),
        InstanceSourceLine(InstanceSourceLine) {
    addMacroExpansionInstance(MacroExpanded, DefinitionLocation,
                              DefinitionSourceLine, InclusionPathHandle);
  }
  MacroExpansionTracker() {}

  // Find a matching macro expansion instance.
  MacroExpansionInstance *
  findMacroExpansionInstance(StringHandle MacroExpanded,
                             PPItemKey &DefinitionLocation) {
    for (std::vector<MacroExpansionInstance>::iterator
             I = MacroExpansionInstances.begin(),
             E = MacroExpansionInstances.end();
         I != E; ++I) {
      if ((I->MacroExpanded == MacroExpanded) &&
          (I->DefinitionLocation == DefinitionLocation)) {
        return &*I; // Found.
      }
    }
    return NULL; // Not found.
  }

  // Add a macro expansion instance.
  void addMacroExpansionInstance(StringHandle MacroExpanded,
                                 PPItemKey &DefinitionLocation,
                                 StringHandle DefinitionSourceLine,
                                 InclusionPathHandle InclusionPathHandle) {
    MacroExpansionInstances.push_back(
        MacroExpansionInstance(MacroExpanded, DefinitionLocation,
                               DefinitionSourceLine, InclusionPathHandle));
  }

  // Return true if there is a mismatch.
  bool hasMismatch() { return MacroExpansionInstances.size() > 1; }

  // A string representing the macro instance without expansion.
  StringHandle MacroUnexpanded;
  // A place to save the macro instance source line string.
  StringHandle InstanceSourceLine;
  // The macro expansion instances.
  // If all instances of the macro expansion expand to the same value,
  // This vector will only have one instance.
  std::vector<MacroExpansionInstance> MacroExpansionInstances;
};

// Conditional expansion instance.
//
// This class represents an instance of a macro expansion with a
// unique value.  It also stores the unique header inclusion paths
// for use in telling the user the nested include path f
class ConditionalExpansionInstance {
public:
  ConditionalExpansionInstance(bool ConditionValue, InclusionPathHandle H)
      : ConditionValue(ConditionValue) {
    InclusionPathHandles.push_back(H);
  }
  ConditionalExpansionInstance() {}

  // Check for the presence of a header inclusion path handle entry.
  // Return false if not found.
  bool haveInclusionPathHandle(InclusionPathHandle H) {
    for (std::vector<InclusionPathHandle>::iterator
             I = InclusionPathHandles.begin(),
             E = InclusionPathHandles.end();
         I != E; ++I) {
      if (*I == H)
        return true;
    }
    return InclusionPathHandleInvalid;
  }
  // Add a new header inclusion path entry, if not already present.
  void addInclusionPathHandle(InclusionPathHandle H) {
    if (!haveInclusionPathHandle(H))
      InclusionPathHandles.push_back(H);
  }

  // A flag representing the evaluated condition value.
  bool ConditionValue;
  // The header inclusion path handles for all the instances.
  std::vector<InclusionPathHandle> InclusionPathHandles;
};

// Conditional directive instance tracker.
//
// This class represents one conditional directive, keyed by a PPItemKey.
// It stores a string representing the macro reference in the source,
// and a list of MacroExpansionInstance objects representing
// the unique value the macro expands to in instances of the header.
class ConditionalTracker {
public:
  ConditionalTracker(clang::tok::PPKeywordKind DirectiveKind,
                     bool ConditionValue, StringHandle ConditionUnexpanded,
                     InclusionPathHandle InclusionPathHandle)
      : DirectiveKind(DirectiveKind), ConditionUnexpanded(ConditionUnexpanded) {
    addConditionalExpansionInstance(ConditionValue, InclusionPathHandle);
  }
  ConditionalTracker() {}

  // Find a matching condition expansion instance.
  ConditionalExpansionInstance *
  findConditionalExpansionInstance(bool ConditionValue) {
    for (std::vector<ConditionalExpansionInstance>::iterator
             I = ConditionalExpansionInstances.begin(),
             E = ConditionalExpansionInstances.end();
         I != E; ++I) {
      if (I->ConditionValue == ConditionValue) {
        return &*I; // Found.
      }
    }
    return NULL; // Not found.
  }

  // Add a conditional expansion instance.
  void
  addConditionalExpansionInstance(bool ConditionValue,
                                  InclusionPathHandle InclusionPathHandle) {
    ConditionalExpansionInstances.push_back(
        ConditionalExpansionInstance(ConditionValue, InclusionPathHandle));
  }

  // Return true if there is a mismatch.
  bool hasMismatch() { return ConditionalExpansionInstances.size() > 1; }

  // The kind of directive.
  clang::tok::PPKeywordKind DirectiveKind;
  // A string representing the macro instance without expansion.
  StringHandle ConditionUnexpanded;
  // The condition expansion instances.
  // If all instances of the conditional expression expand to the same value,
  // This vector will only have one instance.
  std::vector<ConditionalExpansionInstance> ConditionalExpansionInstances;
};

// Preprocessor callbacks for modularize.
//
// This class derives from the Clang PPCallbacks class to track preprocessor
// actions, such as changing files and handling preprocessor directives and
// macro expansions.  It has to figure out when a new header file is entered
// and left, as the provided handler is not particularly clear about it.
class PreprocessorCallbacks : public clang::PPCallbacks {
public:
  PreprocessorCallbacks(PreprocessorTrackerImpl &ppTracker,
                        clang::Preprocessor &PP, llvm::StringRef rootHeaderFile)
      : PPTracker(ppTracker), PP(PP), RootHeaderFile(rootHeaderFile) {}
  ~PreprocessorCallbacks() {}

  // Overidden handlers.
  void FileChanged(clang::SourceLocation Loc,
                   clang::PPCallbacks::FileChangeReason Reason,
                   clang::SrcMgr::CharacteristicKind FileType,
                   clang::FileID PrevFID = clang::FileID());
  void MacroExpands(const clang::Token &MacroNameTok,
                    const clang::MacroDirective *MD, clang::SourceRange Range,
                    const clang::MacroArgs *Args);
  void Defined(const clang::Token &MacroNameTok,
               const clang::MacroDirective *MD, clang::SourceRange Range);
  void If(clang::SourceLocation Loc, clang::SourceRange ConditionRange,
          bool ConditionResult);
  void Elif(clang::SourceLocation Loc, clang::SourceRange ConditionRange,
            bool ConditionResult, clang::SourceLocation IfLoc);
  void Ifdef(clang::SourceLocation Loc, const clang::Token &MacroNameTok,
             const clang::MacroDirective *MD);
  void Ifndef(clang::SourceLocation Loc, const clang::Token &MacroNameTok,
              const clang::MacroDirective *MD);

private:
  PreprocessorTrackerImpl &PPTracker;
  clang::Preprocessor &PP;
  std::string RootHeaderFile;
};

// Preprocessor macro expansion item map types.
typedef std::map<PPItemKey, MacroExpansionTracker> MacroExpansionMap;
typedef std::map<PPItemKey, MacroExpansionTracker>::iterator
MacroExpansionMapIter;

// Preprocessor conditional expansion item map types.
typedef std::map<PPItemKey, ConditionalTracker> ConditionalExpansionMap;
typedef std::map<PPItemKey, ConditionalTracker>::iterator
ConditionalExpansionMapIter;

// Preprocessor tracker for modularize.
//
// This class stores information about all the headers processed in the
// course of running modularize.
class PreprocessorTrackerImpl : public PreprocessorTracker {
public:
  PreprocessorTrackerImpl()
      : CurrentInclusionPathHandle(InclusionPathHandleInvalid) {}
  ~PreprocessorTrackerImpl() {}

  // Handle entering a preprocessing session.
  void handlePreprocessorEntry(clang::Preprocessor &PP,
                               llvm::StringRef rootHeaderFile) {
    assert((HeaderStack.size() == 0) && "Header stack should be empty.");
    pushHeaderHandle(addHeader(rootHeaderFile));
    PP.addPPCallbacks(new PreprocessorCallbacks(*this, PP, rootHeaderFile));
  }
  // Handle exiting a preprocessing session.
  void handlePreprocessorExit() { HeaderStack.clear(); }

  // Handle entering a header source file.
  void handleHeaderEntry(clang::Preprocessor &PP, llvm::StringRef HeaderPath) {
    // Ignore <built-in> and <command-line> to reduce message clutter.
    if (HeaderPath.startswith("<"))
      return;
    HeaderHandle H = addHeader(HeaderPath);
    if (H != getCurrentHeaderHandle())
      pushHeaderHandle(H);
  }
  // Handle exiting a header source file.
  void handleHeaderExit(llvm::StringRef HeaderPath) {
    // Ignore <built-in> and <command-line> to reduce message clutter.
    if (HeaderPath.startswith("<"))
      return;
    HeaderHandle H = findHeaderHandle(HeaderPath);
    if (isHeaderHandleInStack(H)) {
      while ((H != getCurrentHeaderHandle()) && (HeaderStack.size() != 0))
        popHeaderHandle();
    }
  }

  // Lookup/add string.
  StringHandle addString(llvm::StringRef Str) { return Strings.intern(Str); }

  // Get the handle of a header file entry.
  // Return HeaderHandleInvalid if not found.
  HeaderHandle findHeaderHandle(llvm::StringRef HeaderPath) const {
    HeaderHandle H = 0;
    for (std::vector<StringHandle>::const_iterator I = HeaderPaths.begin(),
                                                   E = HeaderPaths.end();
         I != E; ++I, ++H) {
      if (**I == HeaderPath)
        return H;
    }
    return HeaderHandleInvalid;
  }

  // Add a new header file entry, or return existing handle.
  // Return the header handle.
  HeaderHandle addHeader(llvm::StringRef HeaderPath) {
    std::string canonicalPath(HeaderPath);
    std::replace(canonicalPath.begin(), canonicalPath.end(), '\\', '/');
    HeaderHandle H = findHeaderHandle(canonicalPath);
    if (H == HeaderHandleInvalid) {
      H = HeaderPaths.size();
      HeaderPaths.push_back(addString(canonicalPath));
    }
    return H;
  }

  // Return a header file path string given its handle.
  StringHandle getHeaderFilePath(HeaderHandle H) const {
    if ((H >= 0) && (H < (HeaderHandle)HeaderPaths.size()))
      return HeaderPaths[H];
    return StringHandle();
  }

  // Returns a handle to the inclusion path.
  InclusionPathHandle pushHeaderHandle(HeaderHandle H) {
    HeaderStack.push_back(H);
    return CurrentInclusionPathHandle = addInclusionPathHandle(HeaderStack);
  }
  // Pops the last header handle from the stack;
  void popHeaderHandle() {
    // assert((HeaderStack.size() != 0) && "Header stack already empty.");
    if (HeaderStack.size() != 0) {
      HeaderStack.pop_back();
      CurrentInclusionPathHandle = addInclusionPathHandle(HeaderStack);
    }
  }
  // Get the top handle on the header stack.
  HeaderHandle getCurrentHeaderHandle() const {
    if (HeaderStack.size() != 0)
      return HeaderStack.back();
    return HeaderHandleInvalid;
  }

  // Check for presence of header handle in the header stack.
  bool isHeaderHandleInStack(HeaderHandle H) const {
    for (std::vector<HeaderHandle>::const_iterator I = HeaderStack.begin(),
                                                   E = HeaderStack.end();
         I != E; ++I) {
      if (*I == H)
        return true;
    }
    return false;
  }

  // Get the handle of a header inclusion path entry.
  // Return InclusionPathHandleInvalid if not found.
  InclusionPathHandle
  findInclusionPathHandle(const std::vector<HeaderHandle> &Path) const {
    InclusionPathHandle H = 0;
    for (std::vector<HeaderInclusionPath>::const_iterator
             I = InclusionPaths.begin(),
             E = InclusionPaths.end();
         I != E; ++I, ++H) {
      if (I->Path == Path)
        return H;
    }
    return HeaderHandleInvalid;
  }
  // Add a new header inclusion path entry, or return existing handle.
  // Return the header inclusion path entry handle.
  InclusionPathHandle
  addInclusionPathHandle(const std::vector<HeaderHandle> &Path) {
    InclusionPathHandle H = findInclusionPathHandle(Path);
    if (H == HeaderHandleInvalid) {
      H = InclusionPaths.size();
      InclusionPaths.push_back(HeaderInclusionPath(Path));
    }
    return H;
  }
  // Return the current inclusion path handle.
  InclusionPathHandle getCurrentInclusionPathHandle() const {
    return CurrentInclusionPathHandle;
  }

  // Return an inclusion path given its handle.
  const std::vector<HeaderHandle> &
  getInclusionPath(InclusionPathHandle H) const {
    if ((H >= 0) && (H <= (InclusionPathHandle)InclusionPaths.size()))
      return InclusionPaths[H].Path;
    static std::vector<HeaderHandle> Empty;
    return Empty;
  }

  // Add a macro expansion instance.
  void addMacroExpansionInstance(clang::Preprocessor &PP, HeaderHandle H,
                                 clang::SourceLocation InstanceLoc,
                                 clang::SourceLocation DefinitionLoc,
                                 clang::IdentifierInfo *II,
                                 llvm::StringRef MacroUnexpanded,
                                 llvm::StringRef MacroExpanded,
                                 InclusionPathHandle InclusionPathHandle) {
    StringHandle MacroName = addString(II->getName());
    PPItemKey instanceKey(PP, MacroName, H, InstanceLoc);
    PPItemKey definitionKey(PP, MacroName, H, DefinitionLoc);
    MacroExpansionMapIter I = MacroExpansions.find(instanceKey);
    if (I == MacroExpansions.end()) {
      std::string instanceSourceLine =
          getSourceLocationString(PP, InstanceLoc) + ":\n" +
          getSourceLine(PP, InstanceLoc) + "\n";
      std::string definitionSourceLine =
          getSourceLocationString(PP, DefinitionLoc) + ":\n" +
          getSourceLine(PP, DefinitionLoc) + "\n";
      MacroExpansions[instanceKey] = MacroExpansionTracker(
          addString(MacroUnexpanded), addString(MacroExpanded),
          addString(instanceSourceLine), definitionKey,
          addString(definitionSourceLine), InclusionPathHandle);
    } else {
      MacroExpansionTracker &CondTracker = I->second;
      MacroExpansionInstance *MacroInfo =
          CondTracker.findMacroExpansionInstance(addString(MacroExpanded),
                                                 definitionKey);
      if (MacroInfo != NULL)
        MacroInfo->addInclusionPathHandle(InclusionPathHandle);
      else {
        std::string definitionSourceLine =
            getSourceLocationString(PP, DefinitionLoc) + ":\n" +
            getSourceLine(PP, DefinitionLoc) + "\n";
        CondTracker.addMacroExpansionInstance(
            addString(MacroExpanded), definitionKey,
            addString(definitionSourceLine), InclusionPathHandle);
      }
    }
  }

  // Add a conditional expansion instance.
  void
  addConditionalExpansionInstance(clang::Preprocessor &PP, HeaderHandle H,
                                  clang::SourceLocation InstanceLoc,
                                  clang::tok::PPKeywordKind DirectiveKind,
                                  bool ConditionValue,
                                  llvm::StringRef ConditionUnexpanded,
                                  InclusionPathHandle InclusionPathHandle) {
    StringHandle conditionUnexpanded(addString(ConditionUnexpanded));
    PPItemKey instanceKey(PP, conditionUnexpanded, H, InstanceLoc);
    ConditionalExpansionMapIter I = ConditionalExpansions.find(instanceKey);
    if (I == ConditionalExpansions.end()) {
      std::string instanceSourceLine =
          getSourceLocationString(PP, InstanceLoc) + ":\n" +
          getSourceLine(PP, InstanceLoc) + "\n";
      ConditionalExpansions[instanceKey] =
          ConditionalTracker(DirectiveKind, ConditionValue, conditionUnexpanded,
                             InclusionPathHandle);
    } else {
      ConditionalTracker &CondTracker = I->second;
      ConditionalExpansionInstance *MacroInfo =
          CondTracker.findConditionalExpansionInstance(ConditionValue);
      if (MacroInfo != NULL)
        MacroInfo->addInclusionPathHandle(InclusionPathHandle);
      else {
        CondTracker.addConditionalExpansionInstance(ConditionValue,
                                                    InclusionPathHandle);
      }
    }
  }

  // Report on inconsistent macro instances.
  // Returns true if any mismatches.
  bool reportInconsistentMacros(llvm::raw_ostream &OS) {
    bool returnValue = false;
    for (MacroExpansionMapIter I = MacroExpansions.begin(),
                               E = MacroExpansions.end();
         I != E; ++I) {
      const PPItemKey &ItemKey = I->first;
      MacroExpansionTracker &MacroExpTracker = I->second;
      if (!MacroExpTracker.hasMismatch())
        continue;
      returnValue = true;
      OS << *MacroExpTracker.InstanceSourceLine;
      if (ItemKey.Column > 0)
        OS << std::string(ItemKey.Column - 1, ' ') << "^\n";
      OS << "error: Macro instance '" << *MacroExpTracker.MacroUnexpanded
         << "' has different values in this header, depending on how it was "
            "included.\n";
      for (std::vector<MacroExpansionInstance>::iterator
               IMT = MacroExpTracker.MacroExpansionInstances.begin(),
               EMT = MacroExpTracker.MacroExpansionInstances.end();
           IMT != EMT; ++IMT) {
        MacroExpansionInstance &MacroInfo = *IMT;
        OS << "  '" << *MacroExpTracker.MacroUnexpanded << "' Expanded to: '"
           << *MacroInfo.MacroExpanded
           << "' with respect to these inclusion paths:\n";
        for (std::vector<InclusionPathHandle>::iterator
                 IIP = MacroInfo.InclusionPathHandles.begin(),
                 EIP = MacroInfo.InclusionPathHandles.end();
             IIP != EIP; ++IIP) {
          const std::vector<HeaderHandle> &ip = getInclusionPath(*IIP);
          int Count = (int)ip.size();
          for (int Index = 0; Index < Count; ++Index) {
            HeaderHandle H = ip[Index];
            OS << std::string((Index * 2) + 4, ' ') << *getHeaderFilePath(H)
               << "\n";
          }
        }
        // For a macro that wasn't defined, we flag it by using the
        // instance location.
        // If there is a definition...
        if (MacroInfo.DefinitionLocation.Line != ItemKey.Line) {
          OS << *MacroInfo.DefinitionSourceLine;
          if (MacroInfo.DefinitionLocation.Column > 0)
            OS << std::string(MacroInfo.DefinitionLocation.Column - 1, ' ')
               << "^\n";
          OS << "Macro defined here.\n";
        } else
          OS << "(no macro definition)"
             << "\n";
      }
    }
    return returnValue;
  }

  // Report on inconsistent conditional instances.
  // Returns true if any mismatches.
  bool reportInconsistentConditionals(llvm::raw_ostream &OS) {
    bool returnValue = false;
    for (ConditionalExpansionMapIter I = ConditionalExpansions.begin(),
                                     E = ConditionalExpansions.end();
         I != E; ++I) {
      const PPItemKey &ItemKey = I->first;
      ConditionalTracker &CondTracker = I->second;
      if (!CondTracker.hasMismatch())
        continue;
      returnValue = true;
      OS << *HeaderPaths[ItemKey.File] << ":" << ItemKey.Line << ":"
         << ItemKey.Column << "\n";
      OS << "#" << getDirectiveSpelling(CondTracker.DirectiveKind) << " "
         << *CondTracker.ConditionUnexpanded << "\n";
      OS << "^\n";
      OS << "error: Conditional expression instance '"
         << *CondTracker.ConditionUnexpanded
         << "' has different values in this header, depending on how it was "
            "included.\n";
      for (std::vector<ConditionalExpansionInstance>::iterator
               IMT = CondTracker.ConditionalExpansionInstances.begin(),
               EMT = CondTracker.ConditionalExpansionInstances.end();
           IMT != EMT; ++IMT) {
        ConditionalExpansionInstance &MacroInfo = *IMT;
        OS << "  '" << *CondTracker.ConditionUnexpanded << "' Expanded to: '"
           << (MacroInfo.ConditionValue ? "true" : "false")
           << "' with respect to these inclusion paths:\n";
        for (std::vector<InclusionPathHandle>::iterator
                 IIP = MacroInfo.InclusionPathHandles.begin(),
                 EIP = MacroInfo.InclusionPathHandles.end();
             IIP != EIP; ++IIP) {
          const std::vector<HeaderHandle> &ip = getInclusionPath(*IIP);
          int Count = (int)ip.size();
          for (int Index = 0; Index < Count; ++Index) {
            HeaderHandle H = ip[Index];
            OS << std::string((Index * 2) + 4, ' ') << *getHeaderFilePath(H)
               << "\n";
          }
        }
      }
    }
    return returnValue;
  }

  // Get directive spelling.
  static const char *getDirectiveSpelling(clang::tok::PPKeywordKind kind) {
    switch (kind) {
    case clang::tok::pp_if:
      return "if";
    case clang::tok::pp_elif:
      return "elif";
    case clang::tok::pp_ifdef:
      return "ifdef";
    case clang::tok::pp_ifndef:
      return "ifndef";
    default:
      return "(unknown)";
    }
  }

private:
  llvm::StringPool Strings;
  std::vector<StringHandle> HeaderPaths;
  std::vector<HeaderHandle> HeaderStack;
  std::vector<HeaderInclusionPath> InclusionPaths;
  InclusionPathHandle CurrentInclusionPathHandle;
  MacroExpansionMap MacroExpansions;
  ConditionalExpansionMap ConditionalExpansions;
};

// PreprocessorTracker functions.

// PreprocessorTracker desctructor.
PreprocessorTracker::~PreprocessorTracker() {}

// Create instance of PreprocessorTracker.
PreprocessorTracker *PreprocessorTracker::create() {
  return new PreprocessorTrackerImpl();
}

// Preprocessor callbacks for modularize.

// Handle file entry/exit.
void PreprocessorCallbacks::FileChanged(
    clang::SourceLocation Loc, clang::PPCallbacks::FileChangeReason Reason,
    clang::SrcMgr::CharacteristicKind FileType, clang::FileID PrevFID) {
  switch (Reason) {
  case EnterFile:
    PPTracker.handleHeaderEntry(PP, getSourceLocationFile(PP, Loc));
    break;
  case ExitFile:
    if (PrevFID.isInvalid())
      PPTracker.handleHeaderExit(RootHeaderFile);
    else
      PPTracker.handleHeaderExit(getSourceLocationFile(PP, Loc));
    break;
  case SystemHeaderPragma:
    return;
  case RenameFile:
    return;
  default:
    return;
  }
}

// Handle macro expansion.
void PreprocessorCallbacks::MacroExpands(const clang::Token &MacroNameTok,
                                         const clang::MacroDirective *MD,
                                         clang::SourceRange Range,
                                         const clang::MacroArgs *Args) {
  clang::SourceLocation Loc = Range.getBegin();
  clang::IdentifierInfo *II = MacroNameTok.getIdentifierInfo();
  const clang::MacroInfo *MI = PP.getMacroInfo(II);
  std::string MacroName = II->getName().str();
  std::string Unexpanded(getMacroUnexpandedString(Range, PP, MacroName, MI));
  std::string Expanded(getMacroExpandedString(PP, MacroName, MI, Args));
  PPTracker.addMacroExpansionInstance(
      PP, PPTracker.getCurrentHeaderHandle(), Loc, MI->getDefinitionLoc(), II,
      Unexpanded, Expanded, PPTracker.getCurrentInclusionPathHandle());
}

void PreprocessorCallbacks::Defined(const clang::Token &MacroNameTok,
                                    const clang::MacroDirective *MD,
                                    clang::SourceRange Range) {
  clang::SourceLocation Loc(Range.getBegin());
  clang::IdentifierInfo *II = MacroNameTok.getIdentifierInfo();
  const clang::MacroInfo *MI = PP.getMacroInfo(II);
  std::string MacroName = II->getName().str();
  std::string Unexpanded(getSourceString(PP, Range));
  PPTracker.addMacroExpansionInstance(
      PP, PPTracker.getCurrentHeaderHandle(), Loc,
      (MI ? MI->getDefinitionLoc() : Loc), II, Unexpanded,
      (MI ? "true" : "false"), PPTracker.getCurrentInclusionPathHandle());
}

void PreprocessorCallbacks::If(clang::SourceLocation Loc,
                               clang::SourceRange ConditionRange,
                               bool ConditionResult) {
  std::string Unexpanded(getSourceString(PP, ConditionRange));
  PPTracker.addConditionalExpansionInstance(
      PP, PPTracker.getCurrentHeaderHandle(), Loc, clang::tok::pp_if,
      ConditionResult, Unexpanded, PPTracker.getCurrentInclusionPathHandle());
}

void PreprocessorCallbacks::Elif(clang::SourceLocation Loc,
                                 clang::SourceRange ConditionRange,
                                 bool ConditionResult,
                                 clang::SourceLocation IfLoc) {
  std::string Unexpanded(getSourceString(PP, ConditionRange));
  PPTracker.addConditionalExpansionInstance(
      PP, PPTracker.getCurrentHeaderHandle(), Loc, clang::tok::pp_elif,
      ConditionResult, Unexpanded, PPTracker.getCurrentInclusionPathHandle());
}

void PreprocessorCallbacks::Ifdef(clang::SourceLocation Loc,
                                  const clang::Token &MacroNameTok,
                                  const clang::MacroDirective *MD) {
  bool IsDefined = (MD != 0);
  PPTracker.addConditionalExpansionInstance(
      PP, PPTracker.getCurrentHeaderHandle(), Loc, clang::tok::pp_ifdef,
      IsDefined, PP.getSpelling(MacroNameTok),
      PPTracker.getCurrentInclusionPathHandle());
}

void PreprocessorCallbacks::Ifndef(clang::SourceLocation Loc,
                                   const clang::Token &MacroNameTok,
                                   const clang::MacroDirective *MD) {
  bool IsNotDefined = (MD == 0);
  PPTracker.addConditionalExpansionInstance(
      PP, PPTracker.getCurrentHeaderHandle(), Loc, clang::tok::pp_ifndef,
      IsNotDefined, PP.getSpelling(MacroNameTok),
      PPTracker.getCurrentInclusionPathHandle());
}
} // end namespace Modularize
