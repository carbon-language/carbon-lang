//===- ReadArchive.cpp - Code to read LLVM bytecode from .a files ---------===//
//
// This file implements the ReadArchiveFile interface, which allows a linker to
// read all of the LLVM bytecode files contained in a .a file.  This file
// understands the standard system .a file format.  This can only handle the .a
// variant prevelant on linux systems so far, but may be extended.  See
// information in this source file for more information:
//   http://sources.redhat.com/cgi-bin/cvsweb.cgi/src/bfd/archive.c?cvsroot=src
//
//===----------------------------------------------------------------------===//

#include "llvm/Bytecode/Reader.h"
#include "llvm/Module.h"
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>

namespace {
  struct ar_hdr {
    char name[16];
    char date[12];
    char uid[6];
    char gid[6];
    char mode[8];
    char size[10];
    char fmag[2];          // Always equal to '`\n'
  };

  enum ObjectType {
    UserObject,            // A user .o/.bc file
    Unknown,               // Unknown file, just ignore it
    SVR4LongFilename,      // a "//" section used for long file names
  };
}


// getObjectType - Determine the type of object that this header represents.
// This is capable of parsing the variety of special sections used for various
// purposes.
static enum ObjectType getObjectType(ar_hdr *H, unsigned Size) {
  // Check for sections with special names...
  if (!memcmp(H->name, "//              ", 16))
    return SVR4LongFilename;

  // Check to see if it looks like an llvm object file...
  if (Size >= 4 && !memcmp(H+1, "llvm", 4))
    return UserObject;

  return Unknown;
}


static inline bool Error(std::string *ErrorStr, const char *Message) {
  if (ErrorStr) *ErrorStr = Message;
  return true;
}

static bool ParseLongFilenameSection(unsigned char *Buffer, unsigned Size,
                                     std::vector<std::string> &LongFilenames,
                                     std::string *S) {
  if (!LongFilenames.empty())
    return Error(S, "archive file contains multiple long filename entries");
                 
  while (Size) {
    // Long filename entries are newline delimited to keep the archive readable.
    unsigned char *Ptr = (unsigned char*)memchr(Buffer, '\n', Size);
    if (Ptr == 0)
      return Error(S, "archive long filename entry doesn't end with newline!");
    assert(*Ptr == '\n');

    if (Ptr == Buffer) break;  // Last entry contains just a newline.

    unsigned char *End = Ptr;
    if (End[-1] == '/') --End; // Remove trailing / from name
    
    LongFilenames.push_back(std::string(Buffer, End));
    Size -= Ptr-Buffer+1;
    Buffer = Ptr+1;
  }
  
  return false;
}


static bool ReadArchiveBuffer(unsigned char *Buffer, unsigned Length,
                              std::vector<Module*> &Objects,
                              std::string *ErrorStr) {
  if (Length < 8 || memcmp(Buffer, "!<arch>\n", 8))
    return Error(ErrorStr, "signature incorrect for an archive file!");
  Buffer += 8;  Length -= 8; // Skip the magic string.

  std::vector<std::string> LongFilenames;

  while (Length >= sizeof(ar_hdr)) {
    ar_hdr *Hdr = (ar_hdr*)Buffer;
    unsigned Size = atoi(Hdr->size);
    if (Size+sizeof(ar_hdr) > Length)
      return Error(ErrorStr, "invalid record length in archive file!");

    switch (getObjectType(Hdr, Size)) {
    case SVR4LongFilename:
      // If this is a long filename section, read all of the file names into the
      // LongFilenames vector.
      //
      if (ParseLongFilenameSection(Buffer+sizeof(ar_hdr), Size,
                                   LongFilenames, ErrorStr))
        return true;
      break;
    case UserObject: {
      Module *M = ParseBytecodeBuffer(Buffer+sizeof(ar_hdr), Size, ErrorStr);
      if (!M) return true;
      Objects.push_back(M);
      break;
    }
    case Unknown:
      std::cerr << "ReadArchiveBuffer: WARNING: Skipping unknown file: ";
      std::cerr << std::string(Hdr->name, Hdr->name+sizeof(Hdr->name+1)) <<"\n";
      break;   // Just ignore unknown files.
    }

    // Round Size up to an even number...
    Size = (Size+1)/2*2;
    Buffer += sizeof(ar_hdr)+Size;   // Move to the next entry
    Length -= sizeof(ar_hdr)+Size;
  }

  return Length != 0;
}


// ReadArchiveFile - Read bytecode files from the specfied .a file, returning
// true on error, or false on success.  This does not support reading files from
// standard input.
//
bool ReadArchiveFile(const std::string &Filename, std::vector<Module*> &Objects,
                     std::string *ErrorStr) {
  int FD = open(Filename.c_str(), O_RDONLY);
  if (FD == -1)
    return Error(ErrorStr, "Error opening file!");
  
  // Stat the file to get its length...
  struct stat StatBuf;
  if (fstat(FD, &StatBuf) == -1 || StatBuf.st_size == 0)
    return Error(ErrorStr, "Error stat'ing file!");
  
    // mmap in the file all at once...
  int Length = StatBuf.st_size;
  unsigned char *Buffer = (unsigned char*)mmap(0, Length, PROT_READ, 
                                               MAP_PRIVATE, FD, 0);
  if (Buffer == (unsigned char*)MAP_FAILED)
    return Error(ErrorStr, "Error mmapping file!");
  
  // Parse the archive files we mmap'ped in
  bool Result = ReadArchiveBuffer(Buffer, Length, Objects, ErrorStr);
  
  // Unmmap the archive...
  munmap((char*)Buffer, Length);

  if (Result)    // Free any loaded objects
    while (!Objects.empty()) {
      delete Objects.back();
      Objects.pop_back();
    }
  
  return Result;
}
