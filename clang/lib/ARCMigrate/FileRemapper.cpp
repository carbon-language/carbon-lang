//===--- FileRemapper.cpp - File Remapping Helper -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/ARCMigrate/FileRemapper.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Basic/FileManager.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>

using namespace clang;
using namespace arcmt;

FileRemapper::FileRemapper() {
  FileMgr.reset(new FileManager(FileSystemOptions()));
}

FileRemapper::~FileRemapper() {
  clear();
}

void FileRemapper::clear(llvm::StringRef outputDir) {
  for (MappingsTy::iterator
         I = FromToMappings.begin(), E = FromToMappings.end(); I != E; ++I)
    resetTarget(I->second);
  FromToMappings.clear();
  assert(ToFromMappings.empty());
  if (!outputDir.empty()) {
    std::string infoFile = getRemapInfoFile(outputDir);
    bool existed;
    llvm::sys::fs::remove(infoFile, existed);
  }
}

std::string FileRemapper::getRemapInfoFile(llvm::StringRef outputDir) {
  assert(!outputDir.empty());
  llvm::sys::Path dir(outputDir);
  llvm::sys::Path infoFile = dir;
  infoFile.appendComponent("remap");
  return infoFile.str();
}

bool FileRemapper::initFromDisk(llvm::StringRef outputDir, Diagnostic &Diag,
                                bool ignoreIfFilesChanged) {
  assert(FromToMappings.empty() &&
         "initFromDisk should be called before any remap calls");
  std::string infoFile = getRemapInfoFile(outputDir);
  bool fileExists = false;
  llvm::sys::fs::exists(infoFile, fileExists);
  if (!fileExists)
    return false;

  std::vector<std::pair<const FileEntry *, const FileEntry *> > pairs;

  std::ifstream fin(infoFile.c_str());
  if (!fin.good())
    return report(std::string("Error opening file: ") + infoFile, Diag);

  while (true) {
    std::string fromFilename, toFilename;
    uint64_t timeModified;

    fin >> fromFilename >> timeModified >> toFilename;
    if (fin.eof())
      break;
    if (!fin.good())
      return report(std::string("Error in format of file: ") + infoFile, Diag);

    const FileEntry *origFE = FileMgr->getFile(fromFilename);
    if (!origFE) {
      if (ignoreIfFilesChanged)
        continue;
      return report(std::string("File does not exist: ") + fromFilename, Diag);
    }
    const FileEntry *newFE = FileMgr->getFile(toFilename);
    if (!newFE) {
      if (ignoreIfFilesChanged)
        continue;
      return report(std::string("File does not exist: ") + toFilename, Diag);
    }

    if ((uint64_t)origFE->getModificationTime() != timeModified) {
      if (ignoreIfFilesChanged)
        continue;
      return report(std::string("File was modified: ") + fromFilename, Diag);
    }

    pairs.push_back(std::make_pair(origFE, newFE));
  }

  for (unsigned i = 0, e = pairs.size(); i != e; ++i)
    remap(pairs[i].first, pairs[i].second);

  return false;
}

bool FileRemapper::flushToDisk(llvm::StringRef outputDir, Diagnostic &Diag) {
  using namespace llvm::sys;

  bool existed;
  if (fs::create_directory(outputDir, existed) != llvm::errc::success)
    return report(std::string("Could not create directory: ") + outputDir.str(),
                  Diag);

  std::string errMsg;
  std::string infoFile = getRemapInfoFile(outputDir);
  llvm::raw_fd_ostream infoOut(infoFile.c_str(), errMsg);
  if (!errMsg.empty() || infoOut.has_error())
    return report(errMsg, Diag);

  for (MappingsTy::iterator
         I = FromToMappings.begin(), E = FromToMappings.end(); I != E; ++I) {

    const FileEntry *origFE = I->first;
    llvm::SmallString<200> origPath = llvm::StringRef(origFE->getName());
    fs::make_absolute(origPath);
    infoOut << origPath << '\n';
    infoOut << (uint64_t)origFE->getModificationTime() << '\n';

    if (const FileEntry *FE = I->second.dyn_cast<const FileEntry *>()) {
      llvm::SmallString<200> newPath = llvm::StringRef(FE->getName());
      fs::make_absolute(newPath);
      infoOut << newPath << '\n';
    } else {

      llvm::SmallString<64> tempPath;
      tempPath = path::filename(origFE->getName());
      tempPath += "-%%%%%%%%";
      tempPath += path::extension(origFE->getName());
      int fd;
      if (fs::unique_file(tempPath.str(), fd, tempPath) != llvm::errc::success)
        return report(std::string("Could not create file: ") + tempPath.c_str(),
                      Diag);

      llvm::raw_fd_ostream newOut(fd, /*shouldClose=*/true);
      llvm::MemoryBuffer *mem = I->second.get<llvm::MemoryBuffer *>();
      newOut.write(mem->getBufferStart(), mem->getBufferSize());
      newOut.close();
      
      const FileEntry *newE = FileMgr->getFile(tempPath);
      remap(origFE, newE);
      infoOut << newE->getName() << '\n';
    }
  }

  infoOut.close();
  return false;
}

bool FileRemapper::overwriteOriginal(Diagnostic &Diag,
                                     llvm::StringRef outputDir) {
  using namespace llvm::sys;

  for (MappingsTy::iterator
         I = FromToMappings.begin(), E = FromToMappings.end(); I != E; ++I) {
    const FileEntry *origFE = I->first;
    if (const FileEntry *newFE = I->second.dyn_cast<const FileEntry *>()) {
      if (fs::copy_file(newFE->getName(), origFE->getName(),
                 fs::copy_option::overwrite_if_exists) != llvm::errc::success) {
        std::string err = "Could not copy file '";
        llvm::raw_string_ostream os(err);
        os << "Could not copy file '" << newFE->getName() << "' to file '"
           << origFE->getName() << "'";
        os.flush();
        return report(err, Diag);
      }
    } else {

      bool fileExists = false;
      fs::exists(origFE->getName(), fileExists);
      if (!fileExists)
        return report(std::string("File does not exist: ") + origFE->getName(),
                      Diag);

      std::string errMsg;
      llvm::raw_fd_ostream Out(origFE->getName(), errMsg,
                               llvm::raw_fd_ostream::F_Binary);
      if (!errMsg.empty() || Out.has_error())
        return report(errMsg, Diag);

      llvm::MemoryBuffer *mem = I->second.get<llvm::MemoryBuffer *>();
      Out.write(mem->getBufferStart(), mem->getBufferSize());
      Out.close();
    }
  }

  clear(outputDir);
  return false;
}

void FileRemapper::applyMappings(CompilerInvocation &CI) const {
  PreprocessorOptions &PPOpts = CI.getPreprocessorOpts();
  for (MappingsTy::const_iterator
         I = FromToMappings.begin(), E = FromToMappings.end(); I != E; ++I) {
    if (const FileEntry *FE = I->second.dyn_cast<const FileEntry *>()) {
      PPOpts.addRemappedFile(I->first->getName(), FE->getName());
    } else {
      llvm::MemoryBuffer *mem = I->second.get<llvm::MemoryBuffer *>();
      PPOpts.addRemappedFile(I->first->getName(), mem);
    }
  }

  PPOpts.RetainRemappedFileBuffers = true;
}

void FileRemapper::transferMappingsAndClear(CompilerInvocation &CI) {
  PreprocessorOptions &PPOpts = CI.getPreprocessorOpts();
  for (MappingsTy::iterator
         I = FromToMappings.begin(), E = FromToMappings.end(); I != E; ++I) {
    if (const FileEntry *FE = I->second.dyn_cast<const FileEntry *>()) {
      PPOpts.addRemappedFile(I->first->getName(), FE->getName());
    } else {
      llvm::MemoryBuffer *mem = I->second.get<llvm::MemoryBuffer *>();
      PPOpts.addRemappedFile(I->first->getName(), mem);
    }
    I->second = Target();
  }

  PPOpts.RetainRemappedFileBuffers = false;
  clear();
}

void FileRemapper::remap(llvm::StringRef filePath, llvm::MemoryBuffer *memBuf) {
  remap(getOriginalFile(filePath), memBuf);
}

void FileRemapper::remap(llvm::StringRef filePath, llvm::StringRef newPath) {
  const FileEntry *file = getOriginalFile(filePath);
  const FileEntry *newfile = FileMgr->getFile(newPath);
  remap(file, newfile);
}

void FileRemapper::remap(const FileEntry *file, llvm::MemoryBuffer *memBuf) {
  assert(file);
  Target &targ = FromToMappings[file];
  resetTarget(targ);
  targ = memBuf;
}

void FileRemapper::remap(const FileEntry *file, const FileEntry *newfile) {
  assert(file && newfile);
  Target &targ = FromToMappings[file];
  resetTarget(targ);
  targ = newfile;
  ToFromMappings[newfile] = file;
}

const FileEntry *FileRemapper::getOriginalFile(llvm::StringRef filePath) {
  const FileEntry *file = FileMgr->getFile(filePath);
  // If we are updating a file that overriden an original file,
  // actually update the original file.
  llvm::DenseMap<const FileEntry *, const FileEntry *>::iterator
    I = ToFromMappings.find(file);
  if (I != ToFromMappings.end()) {
    file = I->second;
    assert(FromToMappings.find(file) != FromToMappings.end() &&
           "Original file not in mappings!");
  }
  return file;
}

void FileRemapper::resetTarget(Target &targ) {
  if (!targ)
    return;

  if (llvm::MemoryBuffer *oldmem = targ.dyn_cast<llvm::MemoryBuffer *>()) {
    delete oldmem;
  } else {
    const FileEntry *toFE = targ.get<const FileEntry *>();
    llvm::DenseMap<const FileEntry *, const FileEntry *>::iterator
      I = ToFromMappings.find(toFE);
    if (I != ToFromMappings.end())
      ToFromMappings.erase(I);
  }
}

bool FileRemapper::report(const std::string &err, Diagnostic &Diag) {
  unsigned ID = Diag.getDiagnosticIDs()->getCustomDiagID(DiagnosticIDs::Error,
                                                         err);
  Diag.Report(ID);
  return true;
}
