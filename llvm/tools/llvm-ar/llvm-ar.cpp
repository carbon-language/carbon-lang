//===-- llvm-ar.cpp - LLVM archive librarian utility ----------------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// Builds up standard unix archive files (.a) containing LLVM bytecode.
//
//===----------------------------------------------------------------------===//

#include "llvm/Module.h"
#include "llvm/Bytecode/Reader.h"
#include "Support/CommandLine.h"
#include "Support/FileUtilities.h"
#include <string>
#include <fstream>
#include <cstdio>
#include <sys/stat.h>
#include <sys/types.h> 
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
using namespace llvm;

using std::string;


#define  ARFMAG    "\n"      /* header trailer string */ 
#define  ARMAG   "!<arch>\n"  /* magic string */ 
#define  SARMAG  8            /* length of magic string */ 
#define VERSION "llvm-ar is a part of the LLVM compiler infrastructure.\nPlease see http://llvm.cs.uiuc.edu for more information.\n";


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


//Option for X32_64, not used but must allow it to be present.
cl::opt<bool> X32Option ("X32_64", cl::desc("Ignored option spelt -X32_64, for compatibility with AIX"), cl::Optional);

//llvm-ar options
cl::opt<string> Options(cl::Positional, cl::desc("{dmpqrstx}[abcfilNoPsSuvV] "), cl::Required);

//llvm-ar options
cl::list<string> RestofArgs(cl::Positional, cl::desc("[relpos] [count]] <archive-file> [members..]"), cl::Optional);

//booleans to represent Operation, only one can be preformed at a time
bool Print, Delete, Move, QuickAppend, InsertWithReplacement, DisplayTable;
bool Extract;

//Modifiers to follow operation to vary behavior
bool AddAfter, AddBefore, Create, TruncateNames, InsertBefore, UseCount;
bool OriginalDates,  FullPath, SymTable, OnlyUpdate, Verbose;

//Realtive Pos Arg
string RelPos;

//Count, use for multiple entries in the archive with the same name
int Count;

//Archive
string Archive;

//Member Files
std::vector<string> Members;


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
  std::vector<unsigned> offsets; //Vector of offsets into archive file
  std::vector<char> names; //Vector of characters that are the symbol names. 

  //Loop over archive member files, parse bytecode, and generate symbol table.
  for(unsigned i=0; i<Members.size(); ++i) { 
    
    //Open Member file for reading and copy to buffer
    int FD = open(Members[i].c_str(),O_RDONLY);
    
    //Check for errors opening the file.
    if (FD == -1) {
      std::cerr << "Error opening file!\n";
      return false;
    }

    // Size of file
    unsigned Length = getFileSize(Members[i]);
    if (Length == (unsigned)-1) {
      std::cerr << "Error stating file\n";
      return false;
    }
    
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
  std::cout << "Symbol Table Size: " << symbolTableSize << "\n";

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
    
    std::cout << "Offset: " << offsets[i] << "\n";
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

  std::cout << "Member File Start: " << ArchiveFile.tellp() << "\n";

  ar_hdr Hdr; //Header for archive member file.

  //stat the file to get info
  struct stat StatBuf;
  if (stat(Member.c_str(), &StatBuf) == -1 || StatBuf.st_size == 0)
    return false;

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
  std::cout << "Size: " << Length << "\n";

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
  
  std::cout << "Member File End: " << ArchiveFile.tellp() << "\n";

  return true;
}


// CreateArchive - Generates archive with or without symbol table.
//
void CreateArchive() {
  
  std::cerr << "Archive File: " << Archive << "\n";

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
  if(SymTable) {
    std::cout << "Symbol Table Start: " << ArchiveFile.tellp() << "\n";
    if(!WriteSymbolTable(ArchiveFile)) {
      std::cerr << "Error creating symbol table. Exiting program.";
      exit(1);
    }
    std::cout << "Symbol Table End: " << ArchiveFile.tellp() << "\n";
  }
  //Loop over all member files, and add to the archive.
  for(unsigned i=0; i < Members.size(); ++i) {
    if(ArchiveFile.tellp() % 2 != 0)
      ArchiveFile << ARFMAG;
    if(AddMemberToArchive(Members[i],ArchiveFile) != true) {
      std::cerr << "Error adding " << Members[i] << "to archive. Exiting program.\n";
      exit(1);
    }
  }
  
  //Close archive file.
  ArchiveFile.close();
}

//Print out usage for errors in command line
void printUse() {
  std::cout << "USAGE: ar [-X32_64] [-]{dmpqrstx}[abcfilNoPsSuvV] [member-name] [count] archive-file [files..]\n\n";

  std::cout << "commands:\n" <<
    "d            - delete file(s) from the archive\n"
  << "m[ab]        - move file(s) in the archive\n"
  << "p            - print file(s) found in the archive\n"
  << "q[f]         - quick append file(s) to the archive\n"
  << "r[ab][f][u]  - replace existing or insert new file(s) into the archive\n"
  << "t            - display contents of archive\n"
  << "x[o]         - extract file(s) from the archive\n";

  std::cout << "\ncommand specific modifiers:\n"
	    << "[a]          - put file(s) after [member-name]\n"
	    << "[b]          - put file(s) before [member-name] (same as [i])\n"
	    << "[N]          - use instance [count] of name\n"
	    << "[f]          - truncate inserted file names\n"
	    << "[P]          - use full path names when matching\n"
	    << "[o]          - preserve original dates\n"
	    << "[u]          - only replace files that are newer than current archive contents\n";

  std::cout << "generic modifiers:\n"
	    << "[c]          - do not warn if the library had to be created\n"
	    << "[s]          - create an archive index (cf. ranlib)\n"
	    << "[S]          - do not build a symbol table\n"
	    << "[v]          - be verbose\n"
	    << "[V]          - display the version number\n";
  exit(1);
}


//Print version
void printVersion() {
  std::cout << VERSION;
  exit(0);
}

//Extract the memberfile name from the command line
void getRelPos() {
  if(RestofArgs.size() > 0) {
    RelPos = RestofArgs[0];
    RestofArgs.erase(RestofArgs.begin());
  }
  //Throw error if needed and not present
  else
    printUse();
}

//Extract count from the command line
void getCount() {
  if(RestofArgs.size() > 0) {
    Count = atoi(RestofArgs[0].c_str());
    RestofArgs.erase(RestofArgs.begin());
  }
  //Throw error if needed and not present
  else
    printUse();
}

//Get the Archive File Name from the command line
void getArchive() {
  std::cerr << RestofArgs.size() << "\n";
  if(RestofArgs.size() > 0) {
    Archive = RestofArgs[0];
    RestofArgs.erase(RestofArgs.begin());
  }
  //Throw error if needed and not present
  else
    printUse();
}


//Copy over remaining items in RestofArgs to our Member File vector.
//This is just for clarity.
void getMembers() {
  std::cerr << RestofArgs.size() << "\n";
  if(RestofArgs.size() > 0)
    Members = std::vector<string>(RestofArgs); 
}

// Parse the operations and operation modifiers
// FIXME: Not all of these options has been implemented, but we still
// do all the command line parsing for them.
void parseCL() {

  //Keep track of number of operations. We can only specify one
  //per execution
  unsigned NumOperations = 0;

  for(unsigned i=0; i<Options.size(); ++i) {
    switch(Options[i]) {
    case 'd':
      ++NumOperations;
      Delete = true;
      break;
    case 'm':
      ++NumOperations;
      Move = true;
      break;
    case 'p':
      ++NumOperations;
      Print = true;
      break;
    case 'r':
      ++NumOperations;
      InsertWithReplacement = true;
      break;
    case 't':
      ++NumOperations;
      DisplayTable = true;
      break;
    case 'x':
      ++NumOperations;
      Extract = true;
      break;
    case 'a':
      AddAfter = true;
      getRelPos();
      break;
    case 'b':
      AddBefore = true;
      getRelPos();
      break;
    case 'c':
      Create = true;
      break;
    case 'f':
      TruncateNames = true;
      break;
    case 'i':
      InsertBefore = true;
      getRelPos();
      break;
    case 'l':
      break;
    case 'N':
      UseCount = true;
      getCount();
      break;
    case 'o':
      OriginalDates = true;
      break;
    case 'P':
      FullPath = true;
      break;
    case 's':
      SymTable = true;
      break;
    case 'S':
      SymTable = false;
      break;
    case 'u':
      OnlyUpdate = true;
      break;
    case 'v':
      Verbose = true;
      break;
    case 'V':
      printVersion();
      break;
    default:
      printUse();
    }
  }

  //Check that only one operation has been specified
  if(NumOperations > 1)
    printUse();

  getArchive();
  getMembers();

}

int main(int argc, char **argv) {

  //Parse Command line options
  cl::ParseCommandLineOptions(argc, argv);
  parseCL();

  //Create archive!
  if(Create)
    CreateArchive();

  return 0;
}

