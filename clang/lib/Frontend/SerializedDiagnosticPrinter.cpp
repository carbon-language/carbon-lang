//===--- SerializedDiagnosticPrinter.cpp - Serializer for diagnostics -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <vector>
#include "llvm/Bitcode/BitstreamWriter.h"
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
    : Stream(Buffer), OS(os), Diags(diags)
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
  
  enum BlockIDs {
    /// \brief The DIAG block, which acts as a container around a diagnostic.
    BLOCK_DIAG = llvm::bitc::FIRST_APPLICATION_BLOCKID,
    /// \brief The STRINGS block, which contains strings 
    /// from multiple diagnostics.
    BLOCK_STRINGS
  };
  
  enum RecordIDs {
    RECORD_DIAG = 1,
    RECORD_DIAG_FLAG,
    RECORD_CATEGORY,
    RECORD_FILENAME
  };

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

/// \brief Emits the preamble of the diagnostics file.
void SDiagsWriter::EmitPreamble() {
 // EmitRawStringContents("CLANG_DIAGS");
 // Stream.Emit(Version, 32);
  
  // Emit the file header.
  Stream.Emit((unsigned)'D', 8);
  Stream.Emit((unsigned)'I', 8);
  Stream.Emit((unsigned)'A', 8);
  Stream.Emit((unsigned)'G', 8);
  
  EmitBlockInfoBlock();
}

void SDiagsWriter::EmitBlockInfoBlock() {
  Stream.EnterBlockInfoBlock(3);
  
  // ==---------------------------------------------------------------------==//
  // The subsequent records and Abbrevs are for the "Diagnostic" block.
  // ==---------------------------------------------------------------------==//

  EmitBlockID(BLOCK_DIAG, "Diagnostic", Stream, Record);
  EmitRecordID(RECORD_DIAG, "Diagnostic Info", Stream, Record);
  EmitRecordID(RECORD_DIAG_FLAG, "Diagnostic Flag", Stream, Record);
  
  // Emit Abbrevs.
  using namespace llvm;

  // Emit abbreviation for RECORD_DIAG.
  BitCodeAbbrev *Abbrev = new BitCodeAbbrev();
  Abbrev->Add(BitCodeAbbrevOp(RECORD_DIAG));
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 3)); // Diag level.
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 16-3)); // Category.  
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 16)); // Text size.
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob)); // Diagnostc text.
  Abbrevs.set(RECORD_DIAG, Stream.EmitBlockInfoAbbrev(BLOCK_DIAG, Abbrev));

  
  // Emit the abbreviation for RECORD_DIAG_FLAG.
  Abbrev = new BitCodeAbbrev();
  Abbrev->Add(BitCodeAbbrevOp(RECORD_DIAG_FLAG));
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 16)); // Text size.
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob)); // Flag name text.
  Abbrevs.set(RECORD_DIAG_FLAG, Stream.EmitBlockInfoAbbrev(BLOCK_DIAG, Abbrev));

  // ==---------------------------------------------------------------------==//
  // The subsequent records and Abbrevs are for the "Strings" block.
  // ==---------------------------------------------------------------------==//

  EmitBlockID(BLOCK_STRINGS, "Strings", Stream, Record);
  EmitRecordID(RECORD_CATEGORY, "Category Name", Stream, Record);
  EmitRecordID(RECORD_FILENAME, "File Name", Stream, Record);

  Abbrev = new BitCodeAbbrev();
  Abbrev->Add(BitCodeAbbrevOp(RECORD_CATEGORY));
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 8)); // Text size.
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob)); // Category text.
  Abbrevs.set(RECORD_CATEGORY, Stream.EmitBlockInfoAbbrev(BLOCK_STRINGS,
                                                          Abbrev));
  
  Abbrev = new BitCodeAbbrev();
  Abbrev->Add(BitCodeAbbrevOp(RECORD_CATEGORY));
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 16)); // Text size.
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob)); // File name text.
  Abbrevs.set(RECORD_FILENAME, Stream.EmitBlockInfoAbbrev(BLOCK_STRINGS,
                                                          Abbrev));

  Stream.ExitBlock();
}

void SDiagsWriter::EmitRawStringContents(llvm::StringRef str) {
  for (StringRef::const_iterator I = str.begin(), E = str.end(); I!=E; ++I)
    Stream.Emit(*I, 8);
}

void SDiagsWriter::HandleDiagnostic(DiagnosticsEngine::Level DiagLevel,
                                    const Diagnostic &Info) {

  BlockEnterExit DiagBlock(Stream, BLOCK_DIAG);
  
  // Emit the RECORD_DIAG record.
  Record.clear();
  Record.push_back(RECORD_DIAG);
  Record.push_back(DiagLevel);
  unsigned category = DiagnosticIDs::getCategoryNumberForDiag(Info.getID());
  Record.push_back(category);
  Categories.insert(category);
  diagBuf.clear();   
  Info.FormatDiagnostic(diagBuf); // Compute the diagnostic text.
  Record.push_back(diagBuf.str().size());
  Stream.EmitRecordWithBlob(Abbrevs.get(RECORD_DIAG), Record, diagBuf.str());

  // Emit the RECORD_DIAG_FLAG record.
  StringRef FlagName = DiagnosticIDs::getWarningOptionForDiag(Info.getID());
  if (!FlagName.empty()) {
    Record.clear();
    Record.push_back(RECORD_DIAG_FLAG);
    Record.push_back(FlagName.size());
    Stream.EmitRecordWithBlob(Abbrevs.get(RECORD_DIAG_FLAG),
                              Record, FlagName.str());
  }
  
  // FIXME: emit location
  // FIXME: emit ranges
  // FIXME: emit notes
  // FIXME: emit fixits
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
      Record.push_back(Name.size());
      Stream.EmitRecordWithBlob(Abbrevs.get(RECORD_FILENAME), Record, Name);
    }
  }

}

void SDiagsWriter::EndSourceFile() {
  EmitCategoriesAndFileNames();
  
  // Write the generated bitstream to "Out".
  OS->write((char *)&Buffer.front(), Buffer.size());
  OS->flush();
  
  OS.reset(0);
}

