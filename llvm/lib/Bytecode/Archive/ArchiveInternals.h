//===-- lib/Bytecode/ArchiveInternals.h -------------------------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// Internal implementation header for LLVM Archive files.
//
//===----------------------------------------------------------------------===//

#ifndef LIB_BYTECODE_ARCHIVEINTERNALS_H
#define LIB_BYTECODE_ARCHIVEINTERNALS_H

#include "llvm/Bytecode/Archive.h"
#include "llvm/System/TimeValue.h"

#define ARFILE_MAGIC "!<arch>\n"                   ///< magic string 
#define ARFILE_MAGIC_LEN (sizeof(ARFILE_MAGIC)-1)  ///< length of magic string 
#define ARFILE_SYMTAB_NAME "/"                     ///< name of symtab entry
#define ARFILE_STRTAB_NAME "//"                    ///< name of strtab entry
#define ARFILE_PAD '\n'                            ///< inter-file align padding

namespace llvm {

  /// The ArchiveMemberHeader structure is used internally for bytecode archives. 
  /// The header precedes each file member in the archive. This structure is 
  /// defined using character arrays for direct and correct interpretation
  /// regardless of the endianess of the machine that produced it.
  /// @brief Archive File Member Header
  class ArchiveMemberHeader {
    public:
    void init() {
      memset(name,' ',16);
      memset(date,' ',12);
      memset(uid,' ',6);
      memset(gid,' ',6);
      memset(mode,' ',8);
      memset(size,' ',10);
      fmag[0] = '`';
      fmag[1] = '\n';
    }
    void setDate( int secondsSinceEpoch = 0 ) {
      if (secondsSinceEpoch == 0) {
        sys::TimeValue tv = sys::TimeValue::now();
        uint64_t secs; uint32_t nanos;
        tv.GetTimespecTime(secs,nanos);
        secondsSinceEpoch = (int) secs;
      }
      char buffer[20];
      sprintf(buffer,"%d", secondsSinceEpoch);
      memcpy(date,buffer,strlen(buffer));
    }

    void setSize(size_t sz) {
      char buffer[20];
      sprintf(buffer, "%u", (unsigned)sz);
      memcpy(size,buffer,strlen(buffer));
    }

    void setMode(int m) {
      char buffer[20];
      sprintf(buffer, "%o", m);
      memcpy(mode,buffer,strlen(buffer));
    }

    void setUid(unsigned u) {
      char buffer[20];
      sprintf(buffer, "%u", u);
      memcpy(uid,buffer,strlen(buffer));
    }

    void setGid(unsigned g) {
      char buffer[20];
      sprintf(buffer, "%u", g);
      memcpy(gid,buffer,strlen(buffer));
    }

    bool setName(const std::string& nm) {
      if (nm.length() > 0 && nm.length() <= 16) {
        memcpy(name,nm.c_str(),nm.length());
        for (int i = nm.length()+1; i < 16; i++ ) name[i] = ' ';
        return true;
      }
      return false;
    }

    private:
    char name[16];  ///< Name of the file member. The filename is terminated with '/'
                    ///< and blanks. The empty name (/ and 15 blanks) is for the 
                    ///< symbol table. The special name "//" and 15 blanks is for
                    ///< the string table, used for long file names. It must be
                    ///< first in the archive.
    char date[12];  ///< File date, decimal seconds since Epoch
    char uid[6];    ///< user id in ASCII decimal
    char gid[6];    ///< group id in ASCII decimal
    char mode[8];   ///< file mode in ASCII octal
    char size[10];  ///< file size in ASCII decimal
    char fmag[2];   ///< Always contains ARFILE_MAGIC_TERMINATOR

  };

  /// The ArchiveInternals class is used to hold the content of the archive
  /// while it is in memory. It also provides the bulk of the implementation for
  /// the llvm:Archive class's interface.
  class Archive::ArchiveInternals {
    /// @name Types
    /// @{
    public:
      typedef std::vector<std::string> StrTab;

      /// This structure holds information for one member in the archive. It is
      /// used temporarily while the contents of the archive are being
      /// determined.
      struct MemberInfo {
        MemberInfo() {}
        sys::Path path;
        std::string name;
        sys::Path::StatusInfo status;
        StrTab symbols;
        unsigned offset;
      };

    /// @}
    /// @name Methods
    /// @{
    public:
      /// @brief Add a file member to the archive.
      void addFileMember(
        const sys::Path& path,         ///< The path to the file to be added
        const std::string& name,       ///< The name for the member
        const StrTab* syms = 0         ///< The symbol table of the member
      );

      /// @brief Write the accumulated archive information to an archive file
      void writeArchive();
      void writeMember(const MemberInfo& member,std::ofstream& ARFile);
      void writeSymbolTable(std::ofstream& ARFile);
      void writeInteger(int num, std::ofstream& ARFile);

    /// @}
    /// @name  Data
    /// @{
    private:
      friend class Archive;            ///< Parent class is a friend
      sys::Path       fname;           ///< Path to the archive file
      std::vector<MemberInfo> members; ///< Info about member files
      Archive::SymTab* symtab;         ///< User's symbol table

    /// @}
  };
}

#endif

// vim: sw=2 ai
