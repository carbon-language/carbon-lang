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

#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstdlib>
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
static int TmpArchiveFD = -1;

// Show the error message and exit.
LLVM_ATTRIBUTE_NORETURN static void fail(Twine Error) {
  outs() << ToolName << ": " << Error << ".\n";
  if (TmpArchiveFD != -1)
    close(TmpArchiveFD);
  if (TemporaryOutput)
    sys::fs::remove(TemporaryOutput);
  exit(1);
}

static void failIfError(std::error_code EC, Twine Context = "") {
  if (!EC)
    return;

  std::string ContextStr = Context.str();
  if (ContextStr == "")
    fail(EC.message());
  fail(Context + ": " + EC.message());
}

// llvm-ar/llvm-ranlib remaining positional arguments.
static cl::list<std::string>
    RestOfArgs(cl::Positional, cl::ZeroOrMore,
               cl::desc("[relpos] [count] <archive-file> [members]..."));

static cl::opt<bool> MRI("M", cl::desc(""));

std::string Options;

// Provide additional help output explaining the operations and modifiers of
// llvm-ar. This object instructs the CommandLine library to print the text of
// the constructor when the --help option is given.
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
  Extract,          ///< Extract files back to file system
  CreateSymTab      ///< Create a symbol table in an existing archive
};

// Modifiers to follow operation to vary behavior
static bool AddAfter = false;      ///< 'a' modifier
static bool AddBefore = false;     ///< 'b' modifier
static bool Create = false;        ///< 'c' modifier
static bool OriginalDates = false; ///< 'o' modifier
static bool OnlyUpdate = false;    ///< 'u' modifier
static bool Verbose = false;       ///< 'v' modifier
static bool Symtab = true;         ///< 's' modifier

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
static std::vector<StringRef> Members;

// Show the error message, the help message and exit.
LLVM_ATTRIBUTE_NORETURN static void
show_help(const std::string &msg) {
  errs() << ToolName << ": " << msg << "\n\n";
  cl::PrintHelpMessage();
  std::exit(1);
}

// Extract the member filename from the command line for the [relpos] argument
// associated with a, b, and i modifiers
static void getRelPos() {
  if(RestOfArgs.size() == 0)
    show_help("Expected [relpos] for a, b, or i modifier");
  RelPos = RestOfArgs[0];
  RestOfArgs.erase(RestOfArgs.begin());
}

static void getOptions() {
  if(RestOfArgs.size() == 0)
    show_help("Expected options");
  Options = RestOfArgs[0];
  RestOfArgs.erase(RestOfArgs.begin());
}

// Get the archive file name from the command line
static void getArchive() {
  if(RestOfArgs.size() == 0)
    show_help("An archive name must be specified");
  ArchiveName = RestOfArgs[0];
  RestOfArgs.erase(RestOfArgs.begin());
}

// Copy over remaining items in RestOfArgs to our Members vector
static void getMembers() {
  for (auto &Arg : RestOfArgs)
    Members.push_back(Arg);
}

static void runMRIScript();

// Parse the command line options as presented and return the operation
// specified. Process all modifiers and check to make sure that constraints on
// modifier/operation pairs have not been violated.
static ArchiveOperation parseCommandLine() {
  if (MRI) {
    if (!RestOfArgs.empty())
      fail("Cannot mix -M and other options");
    runMRIScript();
  }

  getOptions();

  // Keep track of number of operations. We can only specify one
  // per execution.
  unsigned NumOperations = 0;

  // Keep track of the number of positional modifiers (a,b,i). Only
  // one can be specified.
  unsigned NumPositional = 0;

  // Keep track of which operation was requested
  ArchiveOperation Operation;

  bool MaybeJustCreateSymTab = false;

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
    case 's':
      Symtab = true;
      MaybeJustCreateSymTab = true;
      break;
    case 'S':
      Symtab = false;
      break;
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

 if (NumOperations == 0 && MaybeJustCreateSymTab) {
    NumOperations = 1;
    Operation = CreateSymTab;
    if (!Members.empty())
      show_help("The s operation takes only an archive as argument");
  }

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

// Utility function for printing out the file mode when the 't' operation is in
// verbose mode.
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
  // Retain the original mode.
  sys::fs::perms Mode = I->getAccessMode();
  SmallString<128> Storage = Name;

  int FD;
  failIfError(
      sys::fs::openFileForWrite(Storage.c_str(), FD, sys::fs::F_None, Mode),
      Storage.c_str());

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
  case CreateSymTab:
    return false;

  case QuickAppend:
  case ReplaceOrInsert:
    return true;
  }

  llvm_unreachable("Missing entry in covered switch.");
}

static void performReadOperation(ArchiveOperation Operation,
                                 object::Archive *OldArchive) {
  for (object::Archive::child_iterator I = OldArchive->child_begin(),
                                       E = OldArchive->child_end();
       I != E; ++I) {
    ErrorOr<StringRef> NameOrErr = I->getName();
    failIfError(NameOrErr.getError());
    StringRef Name = NameOrErr.get();

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
  StringRef Name;

  object::Archive::child_iterator OldI;

  StringRef NewFilename;

public:
  NewArchiveIterator(object::Archive::child_iterator I, StringRef Name);
  NewArchiveIterator(StringRef I, StringRef Name);
  NewArchiveIterator();
  bool isNewMember() const;
  StringRef getName() const;

  object::Archive::child_iterator getOld() const;

  StringRef getNew() const;
  int getFD(sys::fs::file_status &NewStatus) const;
  const sys::fs::file_status &getStatus() const;
};
}

NewArchiveIterator::NewArchiveIterator() {}

NewArchiveIterator::NewArchiveIterator(object::Archive::child_iterator I,
                                       StringRef Name)
    : IsNewMember(false), Name(Name), OldI(I) {}

NewArchiveIterator::NewArchiveIterator(StringRef NewFilename, StringRef Name)
    : IsNewMember(true), Name(Name), NewFilename(NewFilename) {}

StringRef NewArchiveIterator::getName() const { return Name; }

bool NewArchiveIterator::isNewMember() const { return IsNewMember; }

object::Archive::child_iterator NewArchiveIterator::getOld() const {
  assert(!IsNewMember);
  return OldI;
}

StringRef NewArchiveIterator::getNew() const {
  assert(IsNewMember);
  return NewFilename;
}

int NewArchiveIterator::getFD(sys::fs::file_status &NewStatus) const {
  assert(IsNewMember);
  int NewFD;
  failIfError(sys::fs::openFileForRead(NewFilename, NewFD), NewFilename);
  assert(NewFD != -1);

  failIfError(sys::fs::status(NewFD, NewStatus), NewFilename);

  // Opening a directory doesn't make sense. Let it fail.
  // Linux cannot open directories with open(2), although
  // cygwin and *bsd can.
  if (NewStatus.type() == sys::fs::file_type::directory_file)
    failIfError(make_error_code(errc::is_a_directory), NewFilename);

  return NewFD;
}

template <typename T>
void addMember(std::vector<NewArchiveIterator> &Members, T I, StringRef Name,
               int Pos = -1) {
  NewArchiveIterator NI(I, Name);
  if (Pos == -1)
    Members.push_back(NI);
  else
    Members[Pos] = NI;
}

enum InsertAction {
  IA_AddOldMember,
  IA_AddNewMeber,
  IA_Delete,
  IA_MoveOldMember,
  IA_MoveNewMember
};

static InsertAction computeInsertAction(ArchiveOperation Operation,
                                        object::Archive::child_iterator I,
                                        StringRef Name,
                                        std::vector<StringRef>::iterator &Pos) {
  if (Operation == QuickAppend || Members.empty())
    return IA_AddOldMember;

  auto MI =
      std::find_if(Members.begin(), Members.end(), [Name](StringRef Path) {
        return Name == sys::path::filename(Path);
      });

  if (MI == Members.end())
    return IA_AddOldMember;

  Pos = MI;

  if (Operation == Delete)
    return IA_Delete;

  if (Operation == Move)
    return IA_MoveOldMember;

  if (Operation == ReplaceOrInsert) {
    StringRef PosName = sys::path::filename(RelPos);
    if (!OnlyUpdate) {
      if (PosName.empty())
        return IA_AddNewMeber;
      return IA_MoveNewMember;
    }

    // We could try to optimize this to a fstat, but it is not a common
    // operation.
    sys::fs::file_status Status;
    failIfError(sys::fs::status(*MI, Status), *MI);
    if (Status.getLastModificationTime() < I->getLastModified()) {
      if (PosName.empty())
        return IA_AddOldMember;
      return IA_MoveOldMember;
    }

    if (PosName.empty())
      return IA_AddNewMeber;
    return IA_MoveNewMember;
  }
  llvm_unreachable("No such operation");
}

// We have to walk this twice and computing it is not trivial, so creating an
// explicit std::vector is actually fairly efficient.
static std::vector<NewArchiveIterator>
computeNewArchiveMembers(ArchiveOperation Operation,
                         object::Archive *OldArchive) {
  std::vector<NewArchiveIterator> Ret;
  std::vector<NewArchiveIterator> Moved;
  int InsertPos = -1;
  StringRef PosName = sys::path::filename(RelPos);
  if (OldArchive) {
    for (auto &Child : OldArchive->children()) {
      int Pos = Ret.size();
      ErrorOr<StringRef> NameOrErr = Child.getName();
      failIfError(NameOrErr.getError());
      StringRef Name = NameOrErr.get();
      if (Name == PosName) {
        assert(AddAfter || AddBefore);
        if (AddBefore)
          InsertPos = Pos;
        else
          InsertPos = Pos + 1;
      }

      std::vector<StringRef>::iterator MemberI = Members.end();
      InsertAction Action =
          computeInsertAction(Operation, Child, Name, MemberI);
      switch (Action) {
      case IA_AddOldMember:
        addMember(Ret, Child, Name);
        break;
      case IA_AddNewMeber:
        addMember(Ret, *MemberI, Name);
        break;
      case IA_Delete:
        break;
      case IA_MoveOldMember:
        addMember(Moved, Child, Name);
        break;
      case IA_MoveNewMember:
        addMember(Moved, *MemberI, Name);
        break;
      }
      if (MemberI != Members.end())
        Members.erase(MemberI);
    }
  }

  if (Operation == Delete)
    return Ret;

  if (!RelPos.empty() && InsertPos == -1)
    fail("Insertion point not found");

  if (RelPos.empty())
    InsertPos = Ret.size();

  assert(unsigned(InsertPos) <= Ret.size());
  Ret.insert(Ret.begin() + InsertPos, Moved.begin(), Moved.end());

  Ret.insert(Ret.begin() + InsertPos, Members.size(), NewArchiveIterator());
  int Pos = InsertPos;
  for (auto &Member : Members) {
    StringRef Name = sys::path::filename(Member);
    addMember(Ret, Member, Name, Pos);
    ++Pos;
  }

  return Ret;
}

template <typename T>
static void printWithSpacePadding(raw_fd_ostream &OS, T Data, unsigned Size,
				  bool MayTruncate = false) {
  uint64_t OldPos = OS.tell();
  OS << Data;
  unsigned SizeSoFar = OS.tell() - OldPos;
  if (Size > SizeSoFar) {
    unsigned Remaining = Size - SizeSoFar;
    for (unsigned I = 0; I < Remaining; ++I)
      OS << ' ';
  } else if (Size < SizeSoFar) {
    assert(MayTruncate && "Data doesn't fit in Size");
    // Some of the data this is used for (like UID) can be larger than the
    // space available in the archive format. Truncate in that case.
    OS.seek(OldPos + Size);
  }
}

static void print32BE(raw_fd_ostream &Out, unsigned Val) {
  for (int I = 3; I >= 0; --I) {
    char V = (Val >> (8 * I)) & 0xff;
    Out << V;
  }
}

static void printRestOfMemberHeader(raw_fd_ostream &Out,
                                    const sys::TimeValue &ModTime, unsigned UID,
                                    unsigned GID, unsigned Perms,
                                    unsigned Size) {
  printWithSpacePadding(Out, ModTime.toEpochTime(), 12);
  printWithSpacePadding(Out, UID, 6, true);
  printWithSpacePadding(Out, GID, 6, true);
  printWithSpacePadding(Out, format("%o", Perms), 8);
  printWithSpacePadding(Out, Size, 10);
  Out << "`\n";
}

static void printMemberHeader(raw_fd_ostream &Out, StringRef Name,
                              const sys::TimeValue &ModTime, unsigned UID,
                              unsigned GID, unsigned Perms, unsigned Size) {
  printWithSpacePadding(Out, Twine(Name) + "/", 16);
  printRestOfMemberHeader(Out, ModTime, UID, GID, Perms, Size);
}

static void printMemberHeader(raw_fd_ostream &Out, unsigned NameOffset,
                              const sys::TimeValue &ModTime, unsigned UID,
                              unsigned GID, unsigned Perms, unsigned Size) {
  Out << '/';
  printWithSpacePadding(Out, NameOffset, 15);
  printRestOfMemberHeader(Out, ModTime, UID, GID, Perms, Size);
}

static void writeStringTable(raw_fd_ostream &Out,
                             ArrayRef<NewArchiveIterator> Members,
                             std::vector<unsigned> &StringMapIndexes) {
  unsigned StartOffset = 0;
  for (ArrayRef<NewArchiveIterator>::iterator I = Members.begin(),
                                              E = Members.end();
       I != E; ++I) {
    StringRef Name = I->getName();
    if (Name.size() < 16)
      continue;
    if (StartOffset == 0) {
      printWithSpacePadding(Out, "//", 58);
      Out << "`\n";
      StartOffset = Out.tell();
    }
    StringMapIndexes.push_back(Out.tell() - StartOffset);
    Out << Name << "/\n";
  }
  if (StartOffset == 0)
    return;
  if (Out.tell() % 2)
    Out << '\n';
  int Pos = Out.tell();
  Out.seek(StartOffset - 12);
  printWithSpacePadding(Out, Pos - StartOffset, 10);
  Out.seek(Pos);
}

// Returns the offset of the first reference to a member offset.
static unsigned writeSymbolTable(raw_fd_ostream &Out,
                                 ArrayRef<NewArchiveIterator> Members,
                                 ArrayRef<MemoryBufferRef> Buffers,
                                 std::vector<unsigned> &MemberOffsetRefs) {
  unsigned StartOffset = 0;
  unsigned MemberNum = 0;
  std::string NameBuf;
  raw_string_ostream NameOS(NameBuf);
  unsigned NumSyms = 0;
  LLVMContext &Context = getGlobalContext();
  for (ArrayRef<NewArchiveIterator>::iterator I = Members.begin(),
                                              E = Members.end();
       I != E; ++I, ++MemberNum) {
    MemoryBufferRef MemberBuffer = Buffers[MemberNum];
    ErrorOr<std::unique_ptr<object::SymbolicFile>> ObjOrErr =
        object::SymbolicFile::createSymbolicFile(
            MemberBuffer, sys::fs::file_magic::unknown, &Context);
    if (!ObjOrErr)
      continue;  // FIXME: check only for "not an object file" errors.
    object::SymbolicFile &Obj = *ObjOrErr.get();

    if (!StartOffset) {
      printMemberHeader(Out, "", sys::TimeValue::now(), 0, 0, 0, 0);
      StartOffset = Out.tell();
      print32BE(Out, 0);
    }

    for (const object::BasicSymbolRef &S : Obj.symbols()) {
      uint32_t Symflags = S.getFlags();
      if (Symflags & object::SymbolRef::SF_FormatSpecific)
        continue;
      if (!(Symflags & object::SymbolRef::SF_Global))
        continue;
      if (Symflags & object::SymbolRef::SF_Undefined)
        continue;
      failIfError(S.printName(NameOS));
      NameOS << '\0';
      ++NumSyms;
      MemberOffsetRefs.push_back(MemberNum);
      print32BE(Out, 0);
    }
  }
  Out << NameOS.str();

  if (StartOffset == 0)
    return 0;

  if (Out.tell() % 2)
    Out << '\0';

  unsigned Pos = Out.tell();
  Out.seek(StartOffset - 12);
  printWithSpacePadding(Out, Pos - StartOffset, 10);
  Out.seek(StartOffset);
  print32BE(Out, NumSyms);
  Out.seek(Pos);
  return StartOffset + 4;
}

static void
performWriteOperation(ArchiveOperation Operation, object::Archive *OldArchive,
                      std::vector<NewArchiveIterator> &NewMembers) {
  SmallString<128> TmpArchive;
  failIfError(sys::fs::createUniqueFile(ArchiveName + ".temp-archive-%%%%%%%.a",
                                        TmpArchiveFD, TmpArchive));

  TemporaryOutput = TmpArchive.c_str();
  tool_output_file Output(TemporaryOutput, TmpArchiveFD);
  raw_fd_ostream &Out = Output.os();
  Out << "!<arch>\n";

  std::vector<unsigned> MemberOffsetRefs;

  std::vector<std::unique_ptr<MemoryBuffer>> Buffers;
  std::vector<MemoryBufferRef> Members;
  std::vector<sys::fs::file_status> NewMemberStatus;

  for (unsigned I = 0, N = NewMembers.size(); I < N; ++I) {
    NewArchiveIterator &Member = NewMembers[I];
    MemoryBufferRef MemberRef;

    if (Member.isNewMember()) {
      StringRef Filename = Member.getNew();
      NewMemberStatus.resize(NewMemberStatus.size() + 1);
      sys::fs::file_status &Status = NewMemberStatus.back();
      int FD = Member.getFD(Status);
      ErrorOr<std::unique_ptr<MemoryBuffer>> MemberBufferOrErr =
          MemoryBuffer::getOpenFile(FD, Filename, Status.getSize(), false);
      failIfError(MemberBufferOrErr.getError(), Filename);
      if (close(FD) != 0)
        fail("Could not close file");
      Buffers.push_back(std::move(MemberBufferOrErr.get()));
      MemberRef = Buffers.back()->getMemBufferRef();
    } else {
      object::Archive::child_iterator OldMember = Member.getOld();
      ErrorOr<MemoryBufferRef> MemberBufferOrErr =
          OldMember->getMemoryBufferRef();
      failIfError(MemberBufferOrErr.getError());
      MemberRef = MemberBufferOrErr.get();
    }
    Members.push_back(MemberRef);
  }

  unsigned MemberReferenceOffset = 0;
  if (Symtab) {
    MemberReferenceOffset =
        writeSymbolTable(Out, NewMembers, Members, MemberOffsetRefs);
  }

  std::vector<unsigned> StringMapIndexes;
  writeStringTable(Out, NewMembers, StringMapIndexes);

  unsigned MemberNum = 0;
  unsigned LongNameMemberNum = 0;
  unsigned NewMemberNum = 0;
  std::vector<unsigned> MemberOffset;
  for (std::vector<NewArchiveIterator>::iterator I = NewMembers.begin(),
                                                 E = NewMembers.end();
       I != E; ++I, ++MemberNum) {

    unsigned Pos = Out.tell();
    MemberOffset.push_back(Pos);

    MemoryBufferRef File = Members[MemberNum];
    if (I->isNewMember()) {
      StringRef FileName = I->getNew();
      const sys::fs::file_status &Status = NewMemberStatus[NewMemberNum];
      NewMemberNum++;

      StringRef Name = sys::path::filename(FileName);
      if (Name.size() < 16)
        printMemberHeader(Out, Name, Status.getLastModificationTime(),
                          Status.getUser(), Status.getGroup(),
                          Status.permissions(), Status.getSize());
      else
        printMemberHeader(Out, StringMapIndexes[LongNameMemberNum++],
                          Status.getLastModificationTime(), Status.getUser(),
                          Status.getGroup(), Status.permissions(),
                          Status.getSize());
    } else {
      object::Archive::child_iterator OldMember = I->getOld();
      StringRef Name = I->getName();

      if (Name.size() < 16)
        printMemberHeader(Out, Name, OldMember->getLastModified(),
                          OldMember->getUID(), OldMember->getGID(),
                          OldMember->getAccessMode(), OldMember->getSize());
      else
        printMemberHeader(Out, StringMapIndexes[LongNameMemberNum++],
                          OldMember->getLastModified(), OldMember->getUID(),
                          OldMember->getGID(), OldMember->getAccessMode(),
                          OldMember->getSize());
    }

    Out << File.getBuffer();

    if (Out.tell() % 2)
      Out << '\n';
  }

  if (MemberReferenceOffset) {
    Out.seek(MemberReferenceOffset);
    for (unsigned MemberNum : MemberOffsetRefs)
      print32BE(Out, MemberOffset[MemberNum]);
  }

  Output.keep();
  Out.close();
  sys::fs::rename(TemporaryOutput, ArchiveName);
  TemporaryOutput = nullptr;
}

static void
performWriteOperation(ArchiveOperation Operation, object::Archive *OldArchive,
                      std::vector<NewArchiveIterator> *NewMembersP) {
  if (NewMembersP) {
    performWriteOperation(Operation, OldArchive, *NewMembersP);
    return;
  }
  std::vector<NewArchiveIterator> NewMembers =
      computeNewArchiveMembers(Operation, OldArchive);
  performWriteOperation(Operation, OldArchive, NewMembers);
}

static void createSymbolTable(object::Archive *OldArchive) {
  // When an archive is created or modified, if the s option is given, the
  // resulting archive will have a current symbol table. If the S option
  // is given, it will have no symbol table.
  // In summary, we only need to update the symbol table if we have none.
  // This is actually very common because of broken build systems that think
  // they have to run ranlib.
  if (OldArchive->hasSymbolTable())
    return;

  performWriteOperation(CreateSymTab, OldArchive, nullptr);
}

static void performOperation(ArchiveOperation Operation,
                             object::Archive *OldArchive,
                             std::vector<NewArchiveIterator> *NewMembers) {
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
    performWriteOperation(Operation, OldArchive, NewMembers);
    return;
  case CreateSymTab:
    createSymbolTable(OldArchive);
    return;
  }
  llvm_unreachable("Unknown operation.");
}

static int performOperation(ArchiveOperation Operation,
                            std::vector<NewArchiveIterator> *NewMembers) {
  // Create or open the archive object.
  ErrorOr<std::unique_ptr<MemoryBuffer>> Buf =
      MemoryBuffer::getFile(ArchiveName, -1, false);
  std::error_code EC = Buf.getError();
  if (EC && EC != errc::no_such_file_or_directory) {
    errs() << ToolName << ": error opening '" << ArchiveName
           << "': " << EC.message() << "!\n";
    return 1;
  }

  if (!EC) {
    object::Archive Archive(Buf.get()->getMemBufferRef(), EC);

    if (EC) {
      errs() << ToolName << ": error loading '" << ArchiveName
             << "': " << EC.message() << "!\n";
      return 1;
    }
    performOperation(Operation, &Archive, NewMembers);
    return 0;
  }

  assert(EC == errc::no_such_file_or_directory);

  if (!shouldCreateArchive(Operation)) {
    failIfError(EC, Twine("error loading '") + ArchiveName + "'");
  } else {
    if (!Create) {
      // Produce a warning if we should and we're creating the archive
      errs() << ToolName << ": creating " << ArchiveName << "\n";
    }
  }

  performOperation(Operation, nullptr, NewMembers);
  return 0;
}

static void runMRIScript() {
  enum class MRICommand { AddLib, AddMod, Create, Save, End, Invalid };

  ErrorOr<std::unique_ptr<MemoryBuffer>> Buf = MemoryBuffer::getSTDIN();
  failIfError(Buf.getError());
  const MemoryBuffer &Ref = *Buf.get();
  bool Saved = false;
  std::vector<NewArchiveIterator> NewMembers;
  std::vector<std::unique_ptr<MemoryBuffer>> ArchiveBuffers;
  std::vector<std::unique_ptr<object::Archive>> Archives;

  for (line_iterator I(Ref, /*SkipBlanks*/ true, ';'), E; I != E; ++I) {
    StringRef Line = *I;
    StringRef CommandStr, Rest;
    std::tie(CommandStr, Rest) = Line.split(' ');
    Rest = Rest.trim();
    if (!Rest.empty() && Rest.front() == '"' && Rest.back() == '"')
      Rest = Rest.drop_front().drop_back();
    auto Command = StringSwitch<MRICommand>(CommandStr.lower())
                       .Case("addlib", MRICommand::AddLib)
                       .Case("addmod", MRICommand::AddMod)
                       .Case("create", MRICommand::Create)
                       .Case("save", MRICommand::Save)
                       .Case("end", MRICommand::End)
                       .Default(MRICommand::Invalid);

    switch (Command) {
    case MRICommand::AddLib: {
      auto BufOrErr = MemoryBuffer::getFile(Rest, -1, false);
      failIfError(BufOrErr.getError(), "Could not open library");
      ArchiveBuffers.push_back(std::move(*BufOrErr));
      auto LibOrErr =
          object::Archive::create(ArchiveBuffers.back()->getMemBufferRef());
      failIfError(LibOrErr.getError(), "Could not parse library");
      Archives.push_back(std::move(*LibOrErr));
      object::Archive &Lib = *Archives.back();
      for (auto &Member : Lib.children()) {
        ErrorOr<StringRef> NameOrErr = Member.getName();
        failIfError(NameOrErr.getError());
        addMember(NewMembers, Member, *NameOrErr);
      }
      break;
    }
    case MRICommand::AddMod:
      addMember(NewMembers, Rest, sys::path::filename(Rest));
      break;
    case MRICommand::Create:
      Create = true;
      if (!ArchiveName.empty())
        fail("Editing multiple archives not supported");
      if (Saved)
        fail("File already saved");
      ArchiveName = Rest;
      break;
    case MRICommand::Save:
      Saved = true;
      break;
    case MRICommand::End:
      break;
    case MRICommand::Invalid:
      fail("Unknown command: " + CommandStr);
    }
  }

  // Nothing to do if not saved.
  if (Saved)
    performOperation(ReplaceOrInsert, &NewMembers);
  exit(0);
}

static int ar_main() {
  // Do our own parsing of the command line because the CommandLine utility
  // can't handle the grouped positional parameters without a dash.
  ArchiveOperation Operation = parseCommandLine();
  return performOperation(Operation, nullptr);
}

static int ranlib_main() {
  if (RestOfArgs.size() != 1)
    fail(ToolName + "takes just one archive as argument");
  ArchiveName = RestOfArgs[0];
  return performOperation(CreateSymTab, nullptr);
}

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

  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();

  StringRef Stem = sys::path::stem(ToolName);
  if (Stem.find("ar") != StringRef::npos)
    return ar_main();
  if (Stem.find("ranlib") != StringRef::npos)
    return ranlib_main();
  fail("Not ranlib or ar!");
}
