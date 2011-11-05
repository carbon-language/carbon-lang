//===--- SerializedDiagnosticPrinter.cpp - Serializer for diagnostics -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <vector>
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/DenseSet.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/Version.h"
#include "clang/Frontend/SerializedDiagnosticPrinter.h"

using namespace clang;
using namespace clang::serialized_diags;

namespace {

/// \brief A utility class for entering and exiting bitstream blocks.
class BlockEnterExit {
  llvm::BitstreamWriter &Stream;
public:
  BlockEnterExit(llvm::BitstreamWriter &stream, unsigned blockID,
                 unsigned codelen = 3)
    : Stream(stream) {
      Stream.EnterSubblock(blockID, codelen);
  }  
  ~BlockEnterExit() {
    Stream.ExitBlock();
  }
};
  
class AbbreviationMap {
  llvm::DenseMap<unsigned, unsigned> Abbrevs;
public:
  AbbreviationMap() {}
  
  void set(unsigned recordID, unsigned abbrevID) {
    assert(Abbrevs.find(recordID) == Abbrevs.end() 
           && "Abbreviation already set.");
    Abbrevs[recordID] = abbrevID;
  }
  
  unsigned get(unsigned recordID) {
    assert(Abbrevs.find(recordID) != Abbrevs.end() &&
           "Abbreviation not set.");
    return Abbrevs[recordID];
  }
};
 
typedef llvm::SmallVector<uint64_t, 64> RecordData;
typedef llvm::SmallVectorImpl<uint64_t> RecordDataImpl;
  
class SDiagsWriter : public DiagnosticConsumer {
public:  
  SDiagsWriter(DiagnosticsEngine &diags, llvm::raw_ostream *os) 
    : Stream(Buffer), OS(os), Diags(diags), inNonNoteDiagnostic(false)
  { 
    EmitPreamble();
  };
  
  ~SDiagsWriter() {}
  
  void HandleDiagnostic(DiagnosticsEngine::Level DiagLevel,
                        const Diagnostic &Info);
  
  void EndSourceFile();
  
  DiagnosticConsumer *clone(DiagnosticsEngine &Diags) const {
    // It makes no sense to clone this.
    return 0;
  }

private:
  /// \brief Emit the preamble for the serialized diagnostics.
  void EmitPreamble();
  
  /// \brief Emit the BLOCKINFO block.
  void EmitBlockInfoBlock();
  
  /// \brief Emit the raw characters of the provided string.
  void EmitRawStringContents(StringRef str);
  
  /// \brief Emit the block containing categories and file names.
  void EmitCategoriesAndFileNames();
  
  /// \brief Emit a record for a CharSourceRange.
  void EmitCharSourceRange(CharSourceRange R);
  
  /// \brief The version of the diagnostics file.
  enum { Version = 1 };

  /// \brief The byte buffer for the serialized content.
  std::vector<unsigned char> Buffer;

  /// \brief The BitStreamWriter for the serialized diagnostics.
  llvm::BitstreamWriter Stream;

  /// \brief The name of the diagnostics file.
  llvm::OwningPtr<llvm::raw_ostream> OS;
  
  /// \brief The DiagnosticsEngine tied to all diagnostic locations.
  DiagnosticsEngine &Diags;
  
  /// \brief The set of constructed record abbreviations.
  AbbreviationMap Abbrevs;

  /// \brief A utility buffer for constructing record content.
  RecordData Record;

  /// \brief A text buffer for rendering diagnostic text.
  llvm::SmallString<256> diagBuf;
  
  /// \brief The collection of diagnostic categories used.
  llvm::DenseSet<unsigned> Categories;
  
  /// \brief The collection of files used.
  llvm::DenseSet<FileID> Files;

  typedef llvm::DenseMap<const void *, std::pair<unsigned, llvm::StringRef> > 
          DiagFlagsTy;

  /// \brief Map for uniquing strings.
  DiagFlagsTy DiagFlags;
  
  /// \brief Flag indicating whether or not we are in the process of
  /// emitting a non-note diagnostic.
  bool inNonNoteDiagnostic;
};
} // end anonymous namespace

namespace clang {
namespace serialized_diags {
DiagnosticConsumer *create(llvm::raw_ostream *OS, DiagnosticsEngine &Diags) {
  return new SDiagsWriter(Diags, OS);
}
} // end namespace serialized_diags
} // end namespace clang

//===----------------------------------------------------------------------===//
// Serialization methods.
//===----------------------------------------------------------------------===//

/// \brief Emits a block ID in the BLOCKINFO block.
static void EmitBlockID(unsigned ID, const char *Name,
                        llvm::BitstreamWriter &Stream,
                        RecordDataImpl &Record) {
  Record.clear();
  Record.push_back(ID);
  Stream.EmitRecord(llvm::bitc::BLOCKINFO_CODE_SETBID, Record);
  
  // Emit the block name if present.
  if (Name == 0 || Name[0] == 0)
    return;

  Record.clear();

  while (*Name)
    Record.push_back(*Name++);

  Stream.EmitRecord(llvm::bitc::BLOCKINFO_CODE_BLOCKNAME, Record);
}

/// \brief Emits a record ID in the BLOCKINFO block.
static void EmitRecordID(unsigned ID, const char *Name,
                         llvm::BitstreamWriter &Stream,
                         RecordDataImpl &Record){
  Record.clear();
  Record.push_back(ID);

  while (*Name)
    Record.push_back(*Name++);

  Stream.EmitRecord(llvm::bitc::BLOCKINFO_CODE_SETRECORDNAME, Record);
}

static void AddLocToRecord(SourceManager &SM,
                           SourceLocation Loc,
                           RecordDataImpl &Record) {
  if (Loc.isInvalid()) {
    // Emit a "sentinel" location.
    Record.push_back(~(unsigned)0);  // Line.
    Record.push_back(~(unsigned)0);  // Column.
    Record.push_back(~(unsigned)0);  // Offset.
    return;
  }

  Loc = SM.getSpellingLoc(Loc);
  Record.push_back(SM.getSpellingLineNumber(Loc));
  Record.push_back(SM.getSpellingColumnNumber(Loc));
  
  std::pair<FileID, unsigned> LocInfo = SM.getDecomposedLoc(Loc);
  FileID FID = LocInfo.first;
  unsigned FileOffset = LocInfo.second;
  Record.push_back(FileOffset);
}

void SDiagsWriter::EmitCharSourceRange(CharSourceRange R) {
  Record.clear();
  Record.push_back(RECORD_SOURCE_RANGE);
  AddLocToRecord(Diags.getSourceManager(), R.getBegin(), Record);
  AddLocToRecord(Diags.getSourceManager(), R.getEnd(), Record);
  Stream.EmitRecordWithAbbrev(Abbrevs.get(RECORD_SOURCE_RANGE), Record);
}

/// \brief Emits the preamble of the diagnostics file.
void SDiagsWriter::EmitPreamble() {
 // EmitRawStringContents("CLANG_DIAGS");
 // Stream.Emit(Version, 32);
  
  // Emit the file header.
  Stream.Emit((unsigned)'D', 8);
  Stream.Emit((unsigned) Version, 32 - 8);

  EmitBlockInfoBlock();
}

static void AddSourceLocationAbbrev(llvm::BitCodeAbbrev *Abbrev) {
  using namespace llvm;
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 32)); // Line.
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 32)); // Column.
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 32)); // Offset;
}
void SDiagsWriter::EmitBlockInfoBlock() {
  Stream.EnterBlockInfoBlock(3);
  
  // ==---------------------------------------------------------------------==//
  // The subsequent records and Abbrevs are for the "Diagnostic" block.
  // ==---------------------------------------------------------------------==//

  EmitBlockID(BLOCK_DIAG, "Diag", Stream, Record);
  EmitRecordID(RECORD_DIAG, "DiagInfo", Stream, Record);
  EmitRecordID(RECORD_SOURCE_RANGE, "SrcRange", Stream, Record);
  
  // Emit Abbrevs.
  using namespace llvm;

  // Emit abbreviation for RECORD_DIAG.
  BitCodeAbbrev *Abbrev = new BitCodeAbbrev();
  Abbrev->Add(BitCodeAbbrevOp(RECORD_DIAG));
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 3));  // Diag level.
  AddSourceLocationAbbrev(Abbrev);
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 10)); // Category.  
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 10)); // Mapped Diag ID.
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 16)); // Text size.
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob)); // Diagnostc text.
  Abbrevs.set(RECORD_DIAG, Stream.EmitBlockInfoAbbrev(BLOCK_DIAG, Abbrev));
  
  // Emit abbrevation for RECORD_SOURCE_RANGE.
  Abbrev = new BitCodeAbbrev();
  Abbrev->Add(BitCodeAbbrevOp(RECORD_SOURCE_RANGE));
  AddSourceLocationAbbrev(Abbrev);
  AddSourceLocationAbbrev(Abbrev);
  Abbrevs.set(RECORD_SOURCE_RANGE,
              Stream.EmitBlockInfoAbbrev(BLOCK_DIAG, Abbrev));

  // ==---------------------------------------------------------------------==//
  // The subsequent records and Abbrevs are for the "Strings" block.
  // ==---------------------------------------------------------------------==//

  EmitBlockID(BLOCK_STRINGS, "Strings", Stream, Record);
  EmitRecordID(RECORD_CATEGORY, "CatName", Stream, Record);
  EmitRecordID(RECORD_FILENAME, "FileName", Stream, Record);
  EmitRecordID(RECORD_DIAG_FLAG, "DiagFlag", Stream, Record);

  Abbrev = new BitCodeAbbrev();
  Abbrev->Add(BitCodeAbbrevOp(RECORD_CATEGORY));
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 8)); // Text size.
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob)); // Category text.
  Abbrevs.set(RECORD_CATEGORY, Stream.EmitBlockInfoAbbrev(BLOCK_STRINGS,
                                                          Abbrev));
  
  Abbrev = new BitCodeAbbrev();
  Abbrev->Add(BitCodeAbbrevOp(RECORD_FILENAME));
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 64)); // Size.
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 64)); // Modifcation time.  
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 16)); // Text size.
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob)); // File name text.
  Abbrevs.set(RECORD_FILENAME, Stream.EmitBlockInfoAbbrev(BLOCK_STRINGS,
                                                          Abbrev));
  
  // Emit the abbreviation for RECORD_DIAG_FLAG.
  Abbrev = new BitCodeAbbrev();
  Abbrev->Add(BitCodeAbbrevOp(RECORD_DIAG_FLAG));
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 10)); // Mapped Diag ID.
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 16)); // Text size.
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob)); // Flag name text.
  Abbrevs.set(RECORD_DIAG_FLAG, Stream.EmitBlockInfoAbbrev(BLOCK_STRINGS,
                                                           Abbrev));

  Stream.ExitBlock();
}

void SDiagsWriter::EmitRawStringContents(llvm::StringRef str) {
  for (StringRef::const_iterator I = str.begin(), E = str.end(); I!=E; ++I)
    Stream.Emit(*I, 8);
}

void SDiagsWriter::HandleDiagnostic(DiagnosticsEngine::Level DiagLevel,
                                    const Diagnostic &Info) {

  if (DiagLevel != DiagnosticsEngine::Note) {
    if (inNonNoteDiagnostic) {
      // We have encountered a non-note diagnostic.  Finish up the previous
      // diagnostic block before starting a new one.
      Stream.ExitBlock();
    }
    inNonNoteDiagnostic = true;
  }
  
  Stream.EnterSubblock(BLOCK_DIAG, 3);
  
  // Emit the RECORD_DIAG record.
  Record.clear();
  Record.push_back(RECORD_DIAG);
  Record.push_back(DiagLevel);
  AddLocToRecord(Diags.getSourceManager(), Info.getLocation(), Record);  
  unsigned category = DiagnosticIDs::getCategoryNumberForDiag(Info.getID());
  Record.push_back(category);
  Categories.insert(category);
  if (DiagLevel == DiagnosticsEngine::Note)
    Record.push_back(0); // No flag for notes.
  else {
    StringRef FlagName = DiagnosticIDs::getWarningOptionForDiag(Info.getID());
    if (FlagName.empty())
      Record.push_back(0);
    else {
      // Here we assume that FlagName points to static data whose pointer
      // value is fixed.
      const void *data = FlagName.data();
      std::pair<unsigned, StringRef> &entry = DiagFlags[data];
      if (entry.first == 0) {
        entry.first = DiagFlags.size();
        entry.second = FlagName;
      }
      Record.push_back(entry.first);
    }
  }
                     
  diagBuf.clear();   
  Info.FormatDiagnostic(diagBuf); // Compute the diagnostic text.
  Record.push_back(diagBuf.str().size());
  Stream.EmitRecordWithBlob(Abbrevs.get(RECORD_DIAG), Record, diagBuf.str());
  
  ArrayRef<CharSourceRange> Ranges = Info.getRanges();
  for (ArrayRef<CharSourceRange>::iterator it=Ranges.begin(), ei=Ranges.end();
       it != ei; ++it) {
    EmitCharSourceRange(*it);    
  }

  // FIXME: emit fixits
  
  if (DiagLevel == DiagnosticsEngine::Note) {
    // Notes currently cannot have child diagnostics.  Complete the
    // diagnostic now.
    Stream.ExitBlock();
  }
}

template <typename T>
static void populateAndSort(std::vector<T> &scribble,
                            llvm::DenseSet<T> &set) {
  scribble.clear();

  for (typename llvm::DenseSet<T>::iterator it = set.begin(), ei = set.end();
       it != ei; ++it)
    scribble.push_back(*it);
  
  // Sort 'scribble' so we always have a deterministic ordering in the
  // serialized file.
  std::sort(scribble.begin(), scribble.end());
}

void SDiagsWriter::EmitCategoriesAndFileNames() {

  if (Categories.empty() && Files.empty())
    return;
  
  BlockEnterExit BlockEnter(Stream, BLOCK_STRINGS);
  
  // Emit the category names.
  {
    std::vector<unsigned> scribble;
    populateAndSort(scribble, Categories);
    for (std::vector<unsigned>::iterator it = scribble.begin(), 
          ei = scribble.end(); it != ei ; ++it) {
      Record.clear();
      Record.push_back(RECORD_CATEGORY);
      StringRef catName = DiagnosticIDs::getCategoryNameFromID(*it);
      Record.push_back(catName.size());
      Stream.EmitRecordWithBlob(Abbrevs.get(RECORD_CATEGORY), Record, catName);
    }
  }

  // Emit the file names.
  {
    std::vector<FileID> scribble;
    populateAndSort(scribble, Files);
    for (std::vector<FileID>::iterator it = scribble.begin(), 
         ei = scribble.end(); it != ei; ++it) {
      SourceManager &SM = Diags.getSourceManager();
      const FileEntry *FE = SM.getFileEntryForID(*it);
      StringRef Name = FE->getName();
      
      Record.clear();
      Record.push_back(FE->getSize());
      Record.push_back(FE->getModificationTime());
      Record.push_back(Name.size());
      Stream.EmitRecordWithBlob(Abbrevs.get(RECORD_FILENAME), Record, Name);
    }
  }

  // Emit the flag strings.
  {
    std::vector<StringRef> scribble;
    scribble.resize(DiagFlags.size());

    for (DiagFlagsTy::iterator it = DiagFlags.begin(), ei = DiagFlags.end();
         it != ei; ++it) {
      scribble[it->second.first - 1] = it->second.second;
    }
    for (unsigned i = 0, n = scribble.size(); i != n; ++i) {
      Record.clear();
      Record.push_back(RECORD_DIAG_FLAG);
      Record.push_back(i+1);
      StringRef FlagName = scribble[i];
      Record.push_back(FlagName.size());
      Stream.EmitRecordWithBlob(Abbrevs.get(RECORD_DIAG_FLAG),
                                Record, FlagName);
    }
  }
}

void SDiagsWriter::EndSourceFile() {
  if (inNonNoteDiagnostic) {
    // Finish off any diagnostics we were in the process of emitting.
    Stream.ExitBlock();
    inNonNoteDiagnostic = false;
  }
  
  EmitCategoriesAndFileNames();
  
  // Write the generated bitstream to "Out".
  OS->write((char *)&Buffer.front(), Buffer.size());
  OS->flush();
  
  OS.reset(0);
}

