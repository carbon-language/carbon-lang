//===-- llvm-ar.cpp - LLVM archive librarian utility ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Builds up (relatively) standard unix archive files (.a) containing LLVM
// bitcode or other files.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Object/Archive.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstdlib>
#include <fcntl.h>
#include <memory>

#if !defined(_MSC_VER) && !defined(__MINGW32__)
#include <unistd.h>
#else
#include <io.h>
#endif

using namespace llvm;

// The name this program was invoked as.
static StringRef ToolName;

static const char *TemporaryOutput;

// fail - Show the error message and exit.
LLVM_ATTRIBUTE_NORETURN static void fail(Twine Error) {
  outs() << ToolName << ": " << Error << ".\n";
  if (TemporaryOutput)
    sys::fs::remove(TemporaryOutput);
  exit(1);
}

static void failIfError(error_code EC, Twine Context = "") {
  if (!EC)
    return;

  std::string ContextStr = Context.str();
  if (ContextStr == "")
    fail(EC.message());
  fail(Context + ": " + EC.message());
}

// Option for compatibility with AIX, not used but must allow it to be present.
static cl::opt<bool>
X32Option ("X32_64", cl::Hidden,
            cl::desc("Ignored option for compatibility with AIX"));

// llvm-ar operation code and modifier flags. This must come first.
static cl::opt<std::string>
Options(cl::Positional, cl::Required, cl::desc("{operation}[modifiers]..."));

// llvm-ar remaining positional arguments.
static cl::list<std::string>
RestOfArgs(cl::Positional, cl::OneOrMore,
    cl::desc("[relpos] [count] <archive-file> [members]..."));

// MoreHelp - Provide additional help output explaining the operations and
// modifiers of llvm-ar. This object instructs the CommandLine library
// to print the text of the constructor when the --help option is given.
static cl::extrahelp MoreHelp(
  "\nOPERATIONS:\n"
  "  d[NsS]       - delete file(s) from the archive\n"
  "  m[abiSs]     - move file(s) in the archive\n"
  "  p[kN]        - print file(s) found in the archive\n"
  "  q[ufsS]      - quick append file(s) to the archive\n"
  "  r[abfiuRsS]  - replace or insert file(s) into the archive\n"
  "  t            - display contents of archive\n"
  "  x[No]        - extract file(s) from the archive\n"
  "\nMODIFIERS (operation specific):\n"
  "  [a] - put file(s) after [relpos]\n"
  "  [b] - put file(s) before [relpos] (same as [i])\n"
  "  [i] - put file(s) before [relpos] (same as [b])\n"
  "  [N] - use instance [count] of name\n"
  "  [o] - preserve original dates\n"
  "  [s] - create an archive index (cf. ranlib)\n"
  "  [S] - do not build a symbol table\n"
  "  [u] - update only files newer than archive contents\n"
  "\nMODIFIERS (generic):\n"
  "  [c] - do not warn if the library had to be created\n"
  "  [v] - be verbose about actions taken\n"
);

// This enumeration delineates the kinds of operations on an archive
// that are permitted.
enum ArchiveOperation {
  Print,            ///< Print the contents of the archive
  Delete,           ///< Delete the specified members
  Move,             ///< Move members to end or as given by {a,b,i} modifiers
  QuickAppend,      ///< Quickly append to end of archive
  ReplaceOrInsert,  ///< Replace or Insert members
  DisplayTable,     ///< Display the table of contents
  Extract           ///< Extract files back to file system
};

// Modifiers to follow operation to vary behavior
static bool AddAfter = false;      ///< 'a' modifier
static bool AddBefore = false;     ///< 'b' modifier
static bool Create = false;        ///< 'c' modifier
static bool OriginalDates = false; ///< 'o' modifier
static bool OnlyUpdate = false;    ///< 'u' modifier
static bool Verbose = false;       ///< 'v' modifier

// Relative Positional Argument (for insert/move). This variable holds
// the name of the archive member to which the 'a', 'b' or 'i' modifier
// refers. Only one of 'a', 'b' or 'i' can be specified so we only need
// one variable.
static std::string RelPos;

// This variable holds the name of the archive file as given on the
// command line.
static std::string ArchiveName;

// This variable holds the list of member files to proecess, as given
// on the command line.
static std::vector<std::string> Members;

// show_help - Show the error message, the help message and exit.
LLVM_ATTRIBUTE_NORETURN static void
show_help(const std::string &msg) {
  errs() << ToolName << ": " << msg << "\n\n";
  cl::PrintHelpMessage();
  std::exit(1);
}

// getRelPos - Extract the member filename from the command line for
// the [relpos] argument associated with a, b, and i modifiers
static void getRelPos() {
  if(RestOfArgs.size() == 0)
    show_help("Expected [relpos] for a, b, or i modifier");
  RelPos = RestOfArgs[0];
  RestOfArgs.erase(RestOfArgs.begin());
}

// getArchive - Get the archive file name from the command line
static void getArchive() {
  if(RestOfArgs.size() == 0)
    show_help("An archive name must be specified");
  ArchiveName = RestOfArgs[0];
  RestOfArgs.erase(RestOfArgs.begin());
}

// getMembers - Copy over remaining items in RestOfArgs to our Members vector
// This is just for clarity.
static void getMembers() {
  if(RestOfArgs.size() > 0)
    Members = std::vector<std::string>(RestOfArgs);
}

// parseCommandLine - Parse the command line options as presented and return the
// operation specified. Process all modifiers and check to make sure that
// constraints on modifier/operation pairs have not been violated.
static ArchiveOperation parseCommandLine() {

  // Keep track of number of operations. We can only specify one
  // per execution.
  unsigned NumOperations = 0;

  // Keep track of the number of positional modifiers (a,b,i). Only
  // one can be specified.
  unsigned NumPositional = 0;

  // Keep track of which operation was requested
  ArchiveOperation Operation;

  for(unsigned i=0; i<Options.size(); ++i) {
    switch(Options[i]) {
    case 'd': ++NumOperations; Operation = Delete; break;
    case 'm': ++NumOperations; Operation = Move ; break;
    case 'p': ++NumOperations; Operation = Print; break;
    case 'q': ++NumOperations; Operation = QuickAppend; break;
    case 'r': ++NumOperations; Operation = ReplaceOrInsert; break;
    case 't': ++NumOperations; Operation = DisplayTable; break;
    case 'x': ++NumOperations; Operation = Extract; break;
    case 'c': Create = true; break;
    case 'l': /* accepted but unused */ break;
    case 'o': OriginalDates = true; break;
    case 's': break; // Ignore for now.
    case 'S': break; // Ignore for now.
    case 'u': OnlyUpdate = true; break;
    case 'v': Verbose = true; break;
    case 'a':
      getRelPos();
      AddAfter = true;
      NumPositional++;
      break;
    case 'b':
      getRelPos();
      AddBefore = true;
      NumPositional++;
      break;
    case 'i':
      getRelPos();
      AddBefore = true;
      NumPositional++;
      break;
    default:
      cl::PrintHelpMessage();
    }
  }

  // At this point, the next thing on the command line must be
  // the archive name.
  getArchive();

  // Everything on the command line at this point is a member.
  getMembers();

  // Perform various checks on the operation/modifier specification
  // to make sure we are dealing with a legal request.
  if (NumOperations == 0)
    show_help("You must specify at least one of the operations");
  if (NumOperations > 1)
    show_help("Only one operation may be specified");
  if (NumPositional > 1)
    show_help("You may only specify one of a, b, and i modifiers");
  if (AddAfter || AddBefore) {
    if (Operation != Move && Operation != ReplaceOrInsert)
      show_help("The 'a', 'b' and 'i' modifiers can only be specified with "
            "the 'm' or 'r' operations");
  }
  if (OriginalDates && Operation != Extract)
    show_help("The 'o' modifier is only applicable to the 'x' operation");
  if (OnlyUpdate && Operation != ReplaceOrInsert)
    show_help("The 'u' modifier is only applicable to the 'r' operation");

  // Return the parsed operation to the caller
  return Operation;
}

// Implements the 'p' operation. This function traverses the archive
// looking for members that match the path list.
static void doPrint(StringRef Name, object::Archive::child_iterator I) {
  if (Verbose)
    outs() << "Printing " << Name << "\n";

  StringRef Data = I->getBuffer();
  outs().write(Data.data(), Data.size());
}

// putMode - utility function for printing out the file mode when the 't'
// operation is in verbose mode.
static void printMode(unsigned mode) {
  if (mode & 004)
    outs() << "r";
  else
    outs() << "-";
  if (mode & 002)
    outs() << "w";
  else
    outs() << "-";
  if (mode & 001)
    outs() << "x";
  else
    outs() << "-";
}

// Implement the 't' operation. This function prints out just
// the file names of each of the members. However, if verbose mode is requested
// ('v' modifier) then the file type, permission mode, user, group, size, and
// modification time are also printed.
static void doDisplayTable(StringRef Name, object::Archive::child_iterator I) {
  if (Verbose) {
    sys::fs::perms Mode = I->getAccessMode();
    printMode((Mode >> 6) & 007);
    printMode((Mode >> 3) & 007);
    printMode(Mode & 007);
    outs() << ' ' << I->getUID();
    outs() << '/' << I->getGID();
    outs() << ' ' << format("%6llu", I->getSize());
    outs() << ' ' << I->getLastModified().str();
    outs() << ' ';
  }
  outs() << Name << "\n";
}

// Implement the 'x' operation. This function extracts files back to the file
// system.
static void doExtract(StringRef Name, object::Archive::child_iterator I) {
  // Open up a file stream for writing
  // FIXME: we should abstract this, O_BINARY in particular.
  int OpenFlags = O_TRUNC | O_WRONLY | O_CREAT;
#ifdef O_BINARY
  OpenFlags |= O_BINARY;
#endif

  // Retain the original mode.
  sys::fs::perms Mode = I->getAccessMode();

  int FD = open(Name.str().c_str(), OpenFlags, Mode);
  if (FD < 0)
    fail("Could not open output file");

  {
    raw_fd_ostream file(FD, false);

    // Get the data and its length
    StringRef Data = I->getBuffer();

    // Write the data.
    file.write(Data.data(), Data.size());
  }

  // If we're supposed to retain the original modification times, etc. do so
  // now.
  if (OriginalDates)
    failIfError(
        sys::fs::setLastModificationAndAccessTime(FD, I->getLastModified()));

  if (close(FD))
    fail("Could not close the file");
}

static bool shouldCreateArchive(ArchiveOperation Op) {
  switch (Op) {
  case Print:
  case Delete:
  case Move:
  case DisplayTable:
  case Extract:
    return false;

  case QuickAppend:
  case ReplaceOrInsert:
    return true;
  }

  llvm_unreachable("Missing entry in covered switch.");
}

static void performReadOperation(ArchiveOperation Operation,
                                 object::Archive *OldArchive) {
  for (object::Archive::child_iterator I = OldArchive->begin_children(),
                                       E = OldArchive->end_children();
       I != E; ++I) {
    StringRef Name;
    failIfError(I->getName(Name));

    if (!Members.empty() &&
        std::find(Members.begin(), Members.end(), Name) == Members.end())
      continue;

    switch (Operation) {
    default:
      llvm_unreachable("Not a read operation");
    case Print:
      doPrint(Name, I);
      break;
    case DisplayTable:
      doDisplayTable(Name, I);
      break;
    case Extract:
      doExtract(Name, I);
      break;
    }
  }
}

namespace {
class NewArchiveIterator {
  bool IsNewMember;
  SmallString<16> MemberName;
  object::Archive::child_iterator OldI;
  std::vector<std::string>::const_iterator NewI;

public:
  NewArchiveIterator(object::Archive::child_iterator I, Twine Name);
  NewArchiveIterator(std::vector<std::string>::const_iterator I, Twine Name);
  bool isNewMember() const;
  object::Archive::child_iterator getOld() const;
  const char *getNew() const;
  StringRef getMemberName() const { return MemberName; }
};
}

NewArchiveIterator::NewArchiveIterator(object::Archive::child_iterator I,
                                       Twine Name)
    : IsNewMember(false), OldI(I) {
  Name.toVector(MemberName);
}

NewArchiveIterator::NewArchiveIterator(
    std::vector<std::string>::const_iterator I, Twine Name)
    : IsNewMember(true), NewI(I) {
  Name.toVector(MemberName);
}

bool NewArchiveIterator::isNewMember() const { return IsNewMember; }

object::Archive::child_iterator NewArchiveIterator::getOld() const {
  assert(!IsNewMember);
  return OldI;
}

const char *NewArchiveIterator::getNew() const {
  assert(IsNewMember);
  return NewI->c_str();
}

template <typename T>
void addMember(std::vector<NewArchiveIterator> &Members,
               std::string &StringTable, T I, StringRef Name) {
  if (Name.size() < 16) {
    NewArchiveIterator NI(I, Twine(Name) + "/");
    Members.push_back(NI);
  } else {
    int MapIndex = StringTable.size();
    NewArchiveIterator NI(I, Twine("/") + Twine(MapIndex));
    Members.push_back(NI);
    StringTable += Name;
    StringTable += "/\n";
  }
}

namespace {
class HasName {
  StringRef Name;

public:
  HasName(StringRef Name) : Name(Name) {}
  bool operator()(StringRef Path) { return Name == sys::path::filename(Path); }
};
}

// We have to walk this twice and computing it is not trivial, so creating an
// explicit std::vector is actually fairly efficient.
static std::vector<NewArchiveIterator>
computeNewArchiveMembers(ArchiveOperation Operation,
                         object::Archive *OldArchive,
                         std::string &StringTable) {
  std::vector<NewArchiveIterator> Ret;
  std::vector<NewArchiveIterator> Moved;
  int InsertPos = -1;
  StringRef PosName = sys::path::filename(RelPos);
  if (OldArchive) {
    int Pos = 0;
    for (object::Archive::child_iterator I = OldArchive->begin_children(),
                                         E = OldArchive->end_children();
         I != E; ++I, ++Pos) {
      StringRef Name;
      failIfError(I->getName(Name));
      if (Name == PosName) {
        assert(AddAfter || AddBefore);
        if (AddBefore)
          InsertPos = Pos;
        else
          InsertPos = Pos + 1;
      }
      if (Operation != QuickAppend && !Members.empty()) {
        std::vector<std::string>::iterator MI =
            std::find_if(Members.begin(), Members.end(), HasName(Name));
        if (MI != Members.end()) {
          if (Operation == Move) {
            addMember(Moved, StringTable, I, Name);
            continue;
          }
          if (Operation != ReplaceOrInsert || !OnlyUpdate)
            continue;
          // Ignore if the file if it is older than the member.
          sys::fs::file_status Status;
          failIfError(sys::fs::status(*MI, Status));
          if (Status.getLastModificationTime() < I->getLastModified())
            Members.erase(MI);
          else
            continue;
        }
      }
      addMember(Ret, StringTable, I, Name);
    }
  }

  if (Operation == Delete)
    return Ret;

  if (Operation == Move) {
    if (RelPos.empty()) {
      Ret.insert(Ret.end(), Moved.begin(), Moved.end());
      return Ret;
    }
    if (InsertPos == -1)
      fail("Insertion point not found");
    assert(unsigned(InsertPos) <= Ret.size());
    Ret.insert(Ret.begin() + InsertPos, Moved.begin(), Moved.end());
    return Ret;
  }

  for (std::vector<std::string>::iterator I = Members.begin(),
                                          E = Members.end();
       I != E; ++I) {
    StringRef Name = sys::path::filename(*I);
    addMember(Ret, StringTable, I, Name);
  }

  return Ret;
}

template <typename T>
static void printWithSpacePadding(raw_ostream &OS, T Data, unsigned Size) {
  uint64_t OldPos = OS.tell();
  OS << Data;
  unsigned SizeSoFar = OS.tell() - OldPos;
  assert(Size >= SizeSoFar && "Data doesn't fit in Size");
  unsigned Remaining = Size - SizeSoFar;
  for (unsigned I = 0; I < Remaining; ++I)
    OS << ' ';
}

static void performWriteOperation(ArchiveOperation Operation,
                                  object::Archive *OldArchive) {
  int TmpArchiveFD;
  SmallString<128> TmpArchive;
  failIfError(sys::fs::createUniqueFile(ArchiveName + ".temp-archive-%%%%%%%.a",
                                        TmpArchiveFD, TmpArchive));

  TemporaryOutput = TmpArchive.c_str();
  tool_output_file Output(TemporaryOutput, TmpArchiveFD);
  raw_fd_ostream &Out = Output.os();
  Out << "!<arch>\n";

  std::string StringTable;
  std::vector<NewArchiveIterator> NewMembers =
      computeNewArchiveMembers(Operation, OldArchive, StringTable);
  if (!StringTable.empty()) {
    if (StringTable.size() % 2)
      StringTable += '\n';
    printWithSpacePadding(Out, "//", 48);
    printWithSpacePadding(Out, StringTable.size(), 10);
    Out << "`\n";
    Out << StringTable;
  }

  for (std::vector<NewArchiveIterator>::iterator I = NewMembers.begin(),
                                                 E = NewMembers.end();
       I != E; ++I) {
    StringRef Name = I->getMemberName();
    printWithSpacePadding(Out, Name, 16);

    if (I->isNewMember()) {
      // FIXME: we do a stat + open. We should do a open + fstat.
      const char *FileName = I->getNew();
      sys::fs::file_status Status;
      failIfError(sys::fs::status(FileName, Status), FileName);

      OwningPtr<MemoryBuffer> File;
      failIfError(MemoryBuffer::getFile(FileName, File), FileName);

      uint64_t secondsSinceEpoch =
          Status.getLastModificationTime().toEpochTime();
      printWithSpacePadding(Out, secondsSinceEpoch, 12);

      printWithSpacePadding(Out, Status.getUser(), 6);
      printWithSpacePadding(Out, Status.getGroup(), 6);
      printWithSpacePadding(Out, format("%o", Status.permissions()), 8);
      printWithSpacePadding(Out, Status.getSize(), 10);
      Out << "`\n";

      Out << File->getBuffer();
    } else {
      object::Archive::child_iterator OldMember = I->getOld();

      uint64_t secondsSinceEpoch = OldMember->getLastModified().toEpochTime();
      printWithSpacePadding(Out, secondsSinceEpoch, 12);

      printWithSpacePadding(Out, OldMember->getUID(), 6);
      printWithSpacePadding(Out, OldMember->getGID(), 6);
      printWithSpacePadding(Out, format("%o", OldMember->getAccessMode()), 8);
      printWithSpacePadding(Out, OldMember->getSize(), 10);
      Out << "`\n";

      Out << OldMember->getBuffer();
    }

    if (Out.tell() % 2)
      Out << '\n';
  }
  Output.keep();
  Out.close();
  sys::fs::rename(TemporaryOutput, ArchiveName);
  TemporaryOutput = NULL;
}

static void performOperation(ArchiveOperation Operation,
                             object::Archive *OldArchive) {
  switch (Operation) {
  case Print:
  case DisplayTable:
  case Extract:
    performReadOperation(Operation, OldArchive);
    return;

  case Delete:
  case Move:
  case QuickAppend:
  case ReplaceOrInsert:
    performWriteOperation(Operation, OldArchive);
    return;
  }
  llvm_unreachable("Unknown operation.");
}

// main - main program for llvm-ar .. see comments in the code
int main(int argc, char **argv) {
  ToolName = argv[0];
  // Print a stack trace if we signal out.
  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(argc, argv);
  llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.

  // Have the command line options parsed and handle things
  // like --help and --version.
  cl::ParseCommandLineOptions(argc, argv,
    "LLVM Archiver (llvm-ar)\n\n"
    "  This program archives bitcode files into single libraries\n"
  );

  // Do our own parsing of the command line because the CommandLine utility
  // can't handle the grouped positional parameters without a dash.
  ArchiveOperation Operation = parseCommandLine();

  // Create or open the archive object.
  OwningPtr<MemoryBuffer> Buf;
  error_code EC = MemoryBuffer::getFile(ArchiveName, Buf, -1, false);
  if (EC && EC != llvm::errc::no_such_file_or_directory) {
    errs() << argv[0] << ": error opening '" << ArchiveName
           << "': " << EC.message() << "!\n";
    return 1;
  }

  if (!EC) {
    object::Archive Archive(Buf.take(), EC);

    if (EC) {
      errs() << argv[0] << ": error loading '" << ArchiveName
             << "': " << EC.message() << "!\n";
      return 1;
    }
    performOperation(Operation, &Archive);
    return 0;
  }

  assert(EC == llvm::errc::no_such_file_or_directory);

  if (!shouldCreateArchive(Operation)) {
    failIfError(EC, Twine("error loading '") + ArchiveName + "'");
  } else {
    if (!Create) {
      // Produce a warning if we should and we're creating the archive
      errs() << argv[0] << ": creating " << ArchiveName << "\n";
    }
  }

  performOperation(Operation, NULL);
  return 0;
}
