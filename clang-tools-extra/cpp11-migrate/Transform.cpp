#include "Transform.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;

void collectResults(clang::Rewriter &Rewrite,
                    FileContentsByPath &Results) {
  for (Rewriter::buffer_iterator I = Rewrite.buffer_begin(),
                                 E = Rewrite.buffer_end();
       I != E; ++I) {
    const FileEntry *Entry = Rewrite.getSourceMgr().getFileEntryForID(I->first);
    assert(Entry != 0 && "Expected a FileEntry");
    assert(Entry->getName() != 0 &&
           "Unexpected NULL return from FileEntry::getName()");

    FileContentsByPath::value_type ResultEntry;

    ResultEntry.first = Entry->getName();

    // Get a copy of the rewritten buffer from the Rewriter.
    llvm::raw_string_ostream StringStream(ResultEntry.second);
    I->second.write(StringStream);

    // Cause results to be written to ResultEntry.second.
    StringStream.str();

    // FIXME: Use move semantics to avoid copies of the buffer contents if
    // benchmarking shows the copies are expensive, especially for large source
    // files.
    Results.push_back(ResultEntry);
  }
}
