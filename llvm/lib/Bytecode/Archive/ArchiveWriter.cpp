//===-- ArchiveWriter.cpp - LLVM archive writing --------------------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencerand is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// Builds up standard unix archive files (.a) containing LLVM bytecode.
//
//===----------------------------------------------------------------------===//

#include "ArchiveInternals.h"
#include "llvm/Module.h"
#include "llvm/Bytecode/Reader.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/System/MappedFile.h"
#include <fstream>
#include <iostream>

using namespace llvm;

Archive* 
Archive::CreateEmpty(const sys::Path& Filename) {
  Archive* result = new Archive;
  Archive::ArchiveInternals* impl = result->impl = new Archive::ArchiveInternals;
  impl->fname = Filename;
  return result;
}

Archive*
Archive::CreateFromFiles(
  const sys::Path& Filename,
  const PathList& Files,
  const std::string& StripName
) {
  Archive* result = new Archive;
  Archive::ArchiveInternals* impl = result->impl = new Archive::ArchiveInternals;
  impl->fname = Filename;

  try {
    size_t strip_len = StripName.length();
    for (PathList::const_iterator P = Files.begin(), E = Files.end(); P != E ;++P)
    {
      if (P->readable()) {
        std::string name(P->get());
        if (strip_len > 0 && StripName == name.substr(0,strip_len)) {
          name.erase(0,strip_len);
        }
        if (P->isBytecodeFile()) {
          std::vector<std::string> syms;
          if (!GetBytecodeSymbols(*P, syms))
            throw std::string("Can not get symbols from: ") + P->get();
          impl->addFileMember(*P, name, &syms);
        } else {
          impl->addFileMember(*P, name);
        }
      }
      else
        throw std::string("Can not read: ") + P->get();
    }

    // Now that we've collected everything, write the archive
    impl->writeArchive();

  } catch(...) {
    delete impl;
    result->impl = 0;
    delete result;
    throw;
  }

  return result;
}

void
Archive::ArchiveInternals::addFileMember(
  const sys::Path& filePath,
  const std::string& memberName,
  const StrTab* symbols
) {
  MemberInfo info;
  info.path = filePath;
  info.name = memberName;
  filePath.getStatusInfo(info.status);
  if (symbols)
    info.symbols = *symbols;
  info.offset = 0;
  members.push_back(info);
}

void
Archive::ArchiveInternals::writeInteger(int num, std::ofstream& ARFile) {
  char buff[4];
  buff[0] = (num >> 24) & 255;
  buff[1] = (num >> 16) & 255;
  buff[2] = (num >> 8) & 255;
  buff[3] = num & 255;
  ARFile.write(buff, sizeof(buff));
}

void
Archive::ArchiveInternals::writeSymbolTable( std::ofstream& ARFile ) {
 
  // Compute the number of symbols in the symbol table and the
  // total byte size of the string pool. While we're traversing,
  // build the string pool for supporting long file names. Also,
  // build the table of file offsets for the symbol table and 
  // the 
  typedef std::map<std::string,unsigned> SymbolMap;
  StrTab stringPool;
  SymbolMap symbolTable;
  std::vector<unsigned> fileOffsets;
  std::string symTabStrings;
  unsigned fileOffset = 0;
  unsigned spOffset = 0;
  unsigned numSymbols = 0;
  unsigned numSymBytes = 0;
  for (unsigned i = 0; i < members.size(); i++ ) {
    MemberInfo& mi = members[i];
    StrTab& syms = mi.symbols;
    size_t numSym = syms.size();
    numSymbols += numSym;
    for (unsigned j = 0; j < numSym; j++ ) {
      numSymBytes += syms[j].size() + 1;
      symbolTable[syms[i]] = i;
    }
    if (mi.name.length() > 15 || std::string::npos != mi.name.find('/')) {
      stringPool.push_back(mi.name + "/\n");
      mi.name = std::string("/") + utostr(spOffset);
      spOffset += mi.name.length() + 2;
    } else if (mi.name[mi.name.length()-1] != '/') {
      mi.name += "/";
    }
    fileOffsets.push_back(fileOffset);
    fileOffset += sizeof(ArchiveMemberHeader) + mi.status.fileSize;
  }


  // Compute the size of the symbol table file member
  unsigned symTabSize = 0;
  if (numSymbols != 0) 
    symTabSize = 
      sizeof(ArchiveMemberHeader) + // Size of the file header
      4 +                           // Size of "number of entries"
      (4 * numSymbols) +            // Size of member file indices
      numSymBytes;                  // Size of the string table

  // Compute the size of the string pool
  unsigned strPoolSize = 0;
  if (spOffset != 0 )
    strPoolSize = 
      sizeof(ArchiveMemberHeader) + // Size of the file header
      spOffset;                     // Number of bytes in the string pool

  // Compute the byte index offset created by symbol table and string pool
  unsigned firstFileOffset = symTabSize + strPoolSize;

  // Create header for symbol table. This must be first if there is
  // a symbol table and must have a special name.
  if ( symTabSize > 0 ) {
    ArchiveMemberHeader Hdr;
    Hdr.init();

    // Name of symbol table is '/               ' but "" is passed in
    // because the setName method always terminates with a /
    Hdr.setName(ARFILE_SYMTAB_NAME);
    Hdr.setDate();
    Hdr.setSize(symTabSize - sizeof(ArchiveMemberHeader));
    Hdr.setMode(0);
    Hdr.setUid(0);
    Hdr.setGid(0);
    
    // Write header to archive file
    ARFile.write((char*)&Hdr, sizeof(Hdr));
    
    // Write the number of entries in the symbol table
    this->writeInteger(numSymbols, ARFile);

    // Write the file offset indices for each symbol and build the
    // symbol table string pool
    std::string symTabStrPool;
    symTabStrPool.reserve(256 * 1024); // Reserve 256KBytes for symbols
    for (SymbolMap::iterator I = symbolTable.begin(), E = symbolTable.end();
         I != E; ++I ) {
      this->writeInteger(firstFileOffset + fileOffsets[I->second], ARFile);
      symTabStrPool += I->first;
      symTabStrPool += "\0";
    }

    // Write the symbol table's string pool
    ARFile.write(symTabStrPool.data(), symTabStrPool.size());
  }

  //============== DONE WITH SYMBOL TABLE 

  if (strPoolSize > 0) {
    // Initialize the header for the string pool
    ArchiveMemberHeader Hdr;
    Hdr.init();
    Hdr.setName(ARFILE_STRTAB_NAME); 
    Hdr.setDate();
    Hdr.setSize(spOffset);
    Hdr.setMode(0);
    Hdr.setUid(0);
    Hdr.setGid(0);

    // Write the string pool header
    ARFile.write((char*)&Hdr, sizeof(Hdr));

    // Write the string pool
    for (unsigned i = 0; i < stringPool.size(); i++) {
      ARFile.write(stringPool[i].data(), stringPool[i].size());
    }
  }
}

void
Archive::ArchiveInternals::writeMember(
  const MemberInfo& member,
  std::ofstream& ARFile
) {

  // Map the file into memory. We do this early for two reasons. First,
  // if there's any kind of error, we want to know about it. Second, we
  // want to ensure we're using the most recent size for this file.
  sys::MappedFile mFile(member.path);
  mFile.map();

  // Header for the archive member
  ArchiveMemberHeader Hdr;
  Hdr.init();

  // Set the name. If its longer than 15 chars, it will have already
  // been reduced by the writeSymbolTable.
  Hdr.setName(member.name);

  // Set the other header members
  Hdr.setSize( mFile.size() );
  Hdr.setMode( member.status.mode);
  Hdr.setUid ( member.status.user);
  Hdr.setGid ( member.status.group);
  Hdr.setDate( member.status.modTime.ToPosixTime() );

  // Write header to archive file
  ARFile.write((char*)&Hdr, sizeof(Hdr));
  
  //write to archive file
  ARFile.write(mFile.charBase(),mFile.size());

  mFile.unmap();
}

void
Archive::ArchiveInternals::writeArchive() {
  
  // Create archive file for output.
  std::ofstream ArchiveFile(fname.get().c_str());
  
  // Check for errors opening or creating archive file.
  if ( !ArchiveFile.is_open() || ArchiveFile.bad() ) {
    throw std::string("Error opening archive file: ") + fname.get();
  }

  // Write magic string to archive.
  ArchiveFile << ARFILE_MAGIC;

  // Write the symbol table and string pool
  writeSymbolTable(ArchiveFile);

  //Loop over all member files, and add to the archive.
  for ( unsigned i = 0; i < members.size(); ++i) {
    if(ArchiveFile.tellp() % 2 != 0)
      ArchiveFile << ARFILE_PAD;
    writeMember(members[i],ArchiveFile);
  }

  //Close archive file.
  ArchiveFile.close();
}

// vim: sw=2 ai
