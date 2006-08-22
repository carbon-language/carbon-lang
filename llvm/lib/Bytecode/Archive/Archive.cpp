//===-- Archive.cpp - Generic LLVM archive functions ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of the Archive and ArchiveMember
// classes that is common to both reading and writing archives..
//
//===----------------------------------------------------------------------===//

#include "ArchiveInternals.h"
#include "llvm/ModuleProvider.h"
#include "llvm/System/Process.h"

using namespace llvm;

// getMemberSize - compute the actual physical size of the file member as seen
// on disk. This isn't the size of member's payload. Use getSize() for that.
unsigned
ArchiveMember::getMemberSize() const {
  // Basically its the file size plus the header size
  unsigned result =  info.fileSize + sizeof(ArchiveMemberHeader);

  // If it has a long filename, include the name length
  if (hasLongFilename())
    result += path.toString().length() + 1;

  // If its now odd lengthed, include the padding byte
  if (result % 2 != 0 )
    result++;

  return result;
}

// This default constructor is only use by the ilist when it creates its
// sentry node. We give it specific static values to make it stand out a bit.
ArchiveMember::ArchiveMember()
  : next(0), prev(0), parent(0), path("--invalid--"), flags(0), data(0)
{
  info.user = sys::Process::GetCurrentUserId();
  info.group = sys::Process::GetCurrentGroupId();
  info.mode = 0777;
  info.fileSize = 0;
  info.modTime = sys::TimeValue::now();
}

// This is the constructor that the Archive class uses when it is building or
// reading an archive. It just defaults a few things and ensures the parent is
// set for the iplist. The Archive class fills in the ArchiveMember's data.
// This is required because correctly setting the data may depend on other
// things in the Archive.
ArchiveMember::ArchiveMember(Archive* PAR)
  : next(0), prev(0), parent(PAR), path(), flags(0), data(0)
{
}

// This method allows an ArchiveMember to be replaced with the data for a
// different file, presumably as an update to the member. It also makes sure
// the flags are reset correctly.
void ArchiveMember::replaceWith(const sys::Path& newFile) {
  assert(newFile.exists() && "Can't replace with a non-existent file");
  data = 0;
  path = newFile;

  // SVR4 symbol tables have an empty name
  if (path.toString() == ARFILE_SVR4_SYMTAB_NAME)
    flags |= SVR4SymbolTableFlag;
  else
    flags &= ~SVR4SymbolTableFlag;

  // BSD4.4 symbol tables have a special name
  if (path.toString() == ARFILE_BSD4_SYMTAB_NAME)
    flags |= BSD4SymbolTableFlag;
  else
    flags &= ~BSD4SymbolTableFlag;

  // LLVM symbol tables have a very specific name
  if (path.toString() == ARFILE_LLVM_SYMTAB_NAME)
    flags |= LLVMSymbolTableFlag;
  else
    flags &= ~LLVMSymbolTableFlag;

  // String table name
  if (path.toString() == ARFILE_STRTAB_NAME)
    flags |= StringTableFlag;
  else
    flags &= ~StringTableFlag;

  // If it has a slash then it has a path
  bool hasSlash = path.toString().find('/') != std::string::npos;
  if (hasSlash)
    flags |= HasPathFlag;
  else
    flags &= ~HasPathFlag;

  // If it has a slash or its over 15 chars then its a long filename format
  if (hasSlash || path.toString().length() > 15)
    flags |= HasLongFilenameFlag;
  else
    flags &= ~HasLongFilenameFlag;

  // Get the signature and status info
  const char* signature = (const char*) data;
  std::string magic;
  if (!signature) {
    path.getMagicNumber(magic,4);
    signature = magic.c_str();
    std::string err;
    if (path.getFileStatus(info, &err))
      throw err;
  }

  // Determine what kind of file it is
  switch (sys::IdentifyFileType(signature,4)) {
    case sys::BytecodeFileType:
      flags |= BytecodeFlag;
      break;
    case sys::CompressedBytecodeFileType:
      flags |= CompressedBytecodeFlag;
      flags &= ~CompressedFlag;
      break;
    default:
      flags &= ~(BytecodeFlag|CompressedBytecodeFlag);
      break;
  }
}

// Archive constructor - this is the only constructor that gets used for the
// Archive class. Everything else (default,copy) is deprecated. This just
// initializes and maps the file into memory, if requested.
Archive::Archive(const sys::Path& filename, bool map )
  : archPath(filename), members(), mapfile(0), base(0), symTab(), strtab(),
    symTabSize(0), firstFileOffset(0), modules(), foreignST(0)
{
  if (map) {
    std::string ErrMsg;
    mapfile = new sys::MappedFile();
    if (mapfile->open(filename, sys::MappedFile::READ_ACCESS, &ErrMsg))
      throw ErrMsg;
    if (!(base = (char*) mapfile->map(&ErrMsg)))
      throw ErrMsg;
  }
}

void Archive::cleanUpMemory() {
  // Shutdown the file mapping
  if (mapfile) {
    mapfile->close();
    delete mapfile;
    
    mapfile = 0;
    base = 0;
  }
  
  // Forget the entire symbol table
  symTab.clear();
  symTabSize = 0;
  
  firstFileOffset = 0;
  
  // Free the foreign symbol table member
  if (foreignST) {
    delete foreignST;
    foreignST = 0;
  }
  
  // Delete any ModuleProviders and ArchiveMember's we've allocated as a result
  // of symbol table searches.
  for (ModuleMap::iterator I=modules.begin(), E=modules.end(); I != E; ++I ) {
    delete I->second.first;
    delete I->second.second;
  }
}

// Archive destructor - just clean up memory
Archive::~Archive() {
  cleanUpMemory();
}

