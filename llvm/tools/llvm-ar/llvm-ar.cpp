//===-- tools/llvm-ar/llvm-ar.cpp - LLVM archive librarian utility --------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
// 
//
// Builds up standard unix archive files (.a) containing LLVM bytecode.
//
//===----------------------------------------------------------------------===//

#include "Support/CommandLine.h"
#include "llvm/Bytecode/Reader.h"
#include "llvm/Module.h"
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <sys/stat.h>
#include <cstdio>
#include <sys/types.h> 
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

using std::string;
using std::vector;
using std::cout;


#define  ARFMAG    "\n"      /* header trailer string */ 
#define  ARMAG   "!<arch>\n"  /* magic string */ 
#define  SARMAG  8            /* length of magic string */ 

namespace {

  // Each file member is preceded by a file member header. Which is
  // of the following format:
  //
  // char ar_name[16]  - '/' terminated file member name. 
  //                     If the file name does not fit, a dummy name is used.
  // char ar_date[12]  - file date in decimal
  // char ar_uid[6]    - User id of file owner in decimal.
  // char ar_gid[6]    - Group ID file belongs to in decimal.
  // char ar_mode[8]   - File mode in octal.
  // char ar_size[10]  - Size of file in decimal.
  // char ar_fmag[2]   - Trailer of header file, a newline.
  struct ar_hdr {
    char name[16];
    char date[12];
    char uid[6];
    char gid[6];
    char mode[8];
    char size[10];
    char fmag[2]; 
    void init() {
      memset(name,' ',16);
      memset(date,' ',12);
      memset(uid,' ',6);
      memset(gid,' ',6);
      memset(mode,' ',8);
      memset(size,' ',10);
      memset(fmag,' ',2);
    }
  };
}

//Option to generate symbol table or not
//running llvm-ar -s is the same as ranlib
cl::opt<bool> SymbolTable ("s", cl::desc("Generate an archive symbol table"));

//Archive name
cl::opt<string> Archive (cl::Positional, cl::desc("<archive file>"), 
			 cl::Required);

//For now we require one or more member files, this should change so
//we can just run llvm-ar -s on an archive to generate the symbol
//table
cl::list<string> Members(cl::ConsumeAfter, cl::desc("<archive members>..."));


static inline bool Error(std::string *ErrorStr, const char *Message) {
  if (ErrorStr) *ErrorStr = Message;
  return true;
}


// WriteSymbolTable - Writes symbol table to ArchiveFile, return false
// on errors. Also returns by reference size of symbol table.
//
// Overview of method:
// 1) Generate the header for the symbol table. This is a normal
//    archive member header, but it has a zero length name.
// 2) For each archive member file, stat the file and parse the bytecode
//    Store cumulative offset (file size + header size).
// 3) Loop over all the symbols for the current member file, 
//    add offset entry to offset vector, and add symbol name to its vector.
//    Note: The symbol name vector is a vector of chars to speed up calculating
//    the total size of the symbol table.
// 4) Update offset vector once we know the total size of symbol table. This is
//    because the symbol table appears before all archive member file contents.
//    We add the size of magic string, and size of symbol table to each offset.
// 5) If the new updated offset it not even, we add 1 byte to offset because
//    a newline will be inserted when writing member files. This adjustment is
//    cummulative (ie. each time we have an odd offset we add 1 to total adjustment).
// 6) Lastly, write symbol table to file.
//
bool WriteSymbolTable(std::ofstream &ArchiveFile) {
 
  //Create header for symbol table. This is essentially an empty header with the
  //name set to a '/' to indicate its a symbol table.
  ar_hdr Hdr;
  Hdr.init();

  //Name of symbol table is '/'
  Hdr.name[0] = '/';
  Hdr.name[1] = '\0';
  
  //Set the header trailer to a newline
  memcpy(Hdr.fmag,ARFMAG,sizeof(ARFMAG));

  
  //Write header to archive file
  ArchiveFile.write((char*)&Hdr, sizeof(Hdr));
  

  unsigned memoff = 0;  //Keep Track of total size of files added to archive
  vector<unsigned> offsets; //Vector of offsets into archive file
  vector<char> names; //Vector of characters that are the symbol names. 

  //Loop over archive member files, parse bytecode, and generate symbol table.
  for(unsigned i=0; i<Members.size(); ++i) { 
    
    //Open Member file for reading and copy to buffer
    int FD = open(Members[i].c_str(),O_RDONLY);
    
    //Check for errors opening the file.
    if (FD == -1) {
      std::cerr << "Error opening file!\n";
      return false;
    }

    //Stat the file to get its size.
    struct stat StatBuf;
    if (stat(Members[i].c_str(), &StatBuf) == -1 || StatBuf.st_size == 0) {
      std::cerr << "Error stating file\n";
      return false;
    }

    //Size of file
    unsigned Length = StatBuf.st_size;
    
    //Read in file into a buffer.
    unsigned char *buf = (unsigned char*)mmap(0, Length,PROT_READ,
					      MAP_PRIVATE, FD, 0);
  
    //Check if mmap failed.
    if (buf == (unsigned char*)MAP_FAILED) {
      std::cerr << "Error mmapping file!\n";
      return false;
    }
    
    //Parse the bytecode file and get all the symbols.
    string ErrorStr;
    Module *M = ParseBytecodeBuffer(buf,Length,Members[i],&ErrorStr);
    
    //Check for errors parsing bytecode.
    //if(ErrorStr) {
    //std::cerr << "Error Parsing Bytecode\n";
    //return false;
    //}

    //Loop over function names and global vars and add to symbol maps
    for(Module::iterator I = M->begin(), E=M->end(); I != E; ++I) {
      
      //get function name
      string NM = ((Function*)I)->getName();
            
      //Loop over the characters in the name and add to symbol name vector
      for(unsigned i=0; i<NM.size(); ++i)
	names.push_back(NM[i]);

      //Each symbol is null terminated.
      names.push_back('\0');

      //Add offset to vector of offsets
      offsets.push_back(memoff);
    }

    memoff += Length + sizeof(Hdr);
  }

  //Determine how large our symbol table is.
  unsigned symbolTableSize = sizeof(Hdr) + 4 + 4*(offsets.size()) + names.size();
  cout << "Symbol Table Size: " << symbolTableSize << "\n";

  //Number of symbols should be in network byte order as well
  char num[4];
  unsigned temp = offsets.size();
  num[0] = (temp >> 24) & 255;
  num[1] = (temp >> 16) & 255;
  num[2] = (temp >> 8) & 255;
  num[3] = temp & 255;

  //Write number of symbols to archive file
  ArchiveFile.write(num,4);

  //Adjustment to offset to start files on even byte boundaries
  unsigned adjust = 0;
  
  //Update offsets write symbol table to archive.
  for(unsigned i=0; i<offsets.size(); ++i) {
    char output[4];
    offsets[i] = offsets[i] + symbolTableSize + SARMAG;
    offsets[i] += adjust;
    if((offsets[i] % 2 != 0)) {
      adjust++;
      offsets[i] += adjust;
    }
    
    cout << "Offset: " << offsets[i] << "\n";
    output[0] = (offsets[i] >> 24) & 255;
    output[1] = (offsets[i] >> 16) & 255;
    output[2] = (offsets[i] >> 8) & 255;
    output[3] = offsets[i] & 255;
    ArchiveFile.write(output,4);
  }


  //Write out symbol name vector.
  for(unsigned i=0; i<names.size(); ++i)
    ArchiveFile << names[i];

  return true;
}

// AddMemberToArchive - Writes member file to archive. Returns false on errors.
// 
// Overview of method: 
// 1) Open file, and stat it.  
// 2) Fill out header using stat information. If name is longer then 15 
//    characters, use "dummy" name.
// 3) Write header and file contents to disk.
// 4) Keep track of total offset into file, and insert a newline if it is odd.
//
bool AddMemberToArchive(string Member, std::ofstream &ArchiveFile) {
  
  ar_hdr Hdr; //Header for archive member file.
  
  //stat the file to get info
  struct stat StatBuf;
  if (stat(Member.c_str(), &StatBuf) == -1 || StatBuf.st_size == 0)
    cout << "ERROR\n";

  //fill in header
  
  //set name to white spaces
  memset(Hdr.name,' ', sizeof(Hdr.name));

  //check the size of the name, if less than 15, we can copy it directly
  //otherwise we give it a dummy name for now
  if(Member.length() < 16)
    memcpy(Hdr.name,Member.c_str(),Member.length());
  else
    memcpy(Hdr.name, "Dummy", 5);

  //terminate name with forward slash
  Hdr.name[15] = '/';

  //file member size in decimal
  unsigned Length = StatBuf.st_size;
  sprintf(Hdr.size,"%d", Length);
  cout << "Size: " << Length << "\n";

  //file member user id in decimal
  sprintf(Hdr.uid, "%d", StatBuf.st_uid);

  //file member group id in decimal
  sprintf(Hdr.gid, "%d", StatBuf.st_gid);

  //file member date in decimal
  sprintf(Hdr.date,"%d", (int)StatBuf.st_mtime);
  
  //file member mode in OCTAL
  sprintf(Hdr.mode,"%d", StatBuf.st_mode);
 
  //add our header trailer
  memcpy(Hdr.fmag,ARFMAG,sizeof(ARFMAG));

  //write header to archive file
  ArchiveFile.write((char*)&Hdr, sizeof(Hdr));
  
  //open Member file for reading and copy to buffer
  int FD = open(Member.c_str(),O_RDONLY);
  if (FD == -1) {
    std::cerr << "Error opening file!\n";
    return false;
  }

  unsigned char *buf = (unsigned char*)mmap(0, Length,PROT_READ,
			  MAP_PRIVATE, FD, 0);
  
  //check if mmap failed
  if (buf == (unsigned char*)MAP_FAILED) {
    std::cerr << "Error mmapping file!\n";
    return false;
  }

  //write to archive file
  ArchiveFile.write((char*)buf,Length);
    
  // Unmmap the memberfile
  munmap((char*)buf, Length);

  return true;
}


// CreateArchive - Generates archive with or without symbol table.
//
void CreateArchive() {
  
  //Create archive file for output.
  std::ofstream ArchiveFile(Archive.c_str());
  
  //Check for errors opening or creating archive file.
  if(!ArchiveFile.is_open() || ArchiveFile.bad() ) {
    std::cerr << "Error opening Archive File\n";
    exit(1);
  }

  //Write magic string to archive.
  ArchiveFile << ARMAG;

  //If the '-s' option was specified, generate symbol table.
  if(SymbolTable) {
    cout << "Symbol Table Start: " << ArchiveFile.tellp() << "\n";
    if(!WriteSymbolTable(ArchiveFile)) {
      std::cerr << "Error creating symbol table. Exiting program.";
      exit(1);
    }
    cout << "Symbol Table End: " << ArchiveFile.tellp() << "\n";
  }
  //Loop over all member files, and add to the archive.
  for(unsigned i=0; i<Members.size(); ++i) {
    if(ArchiveFile.tellp() % 2 != 0)
      ArchiveFile << ARFMAG;

    cout << "Member File Start: " << ArchiveFile.tellp() << "\n";

    if(AddMemberToArchive(Members[i],ArchiveFile) != true) {
      std::cerr << "Error adding file to archive. Exiting program.";
      exit(1);
    }
    cout << "Member File End: " << ArchiveFile.tellp() << "\n";
  }
  
  //Close archive file.
  ArchiveFile.close();
}


int main(int argc, char **argv) {

  //Parse Command line options
  cl::ParseCommandLineOptions(argc, argv, " llvm-ar\n");

  //Create archive!
  CreateArchive();

  return 0;
}
