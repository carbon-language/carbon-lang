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

#include "Archive.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
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
bool AddAfter = false;           ///< 'a' modifier
bool AddBefore = false;          ///< 'b' modifier
bool Create = false;             ///< 'c' modifier
bool OriginalDates = false;      ///< 'o' modifier
bool SymTable = true;            ///< 's' & 'S' modifiers
bool OnlyUpdate = false;         ///< 'u' modifier
bool Verbose = false;            ///< 'v' modifier

// Relative Positional Argument (for insert/move). This variable holds
// the name of the archive member to which the 'a', 'b' or 'i' modifier
// refers. Only one of 'a', 'b' or 'i' can be specified so we only need
// one variable.
std::string RelPos;

// This variable holds the name of the archive file as given on the
// command line.
std::string ArchiveName;

// This variable holds the list of member files to proecess, as given
// on the command line.
std::vector<std::string> Members;

// This variable holds the (possibly expanded) list of path objects that
// correspond to files we will
std::set<std::string> Paths;

// The Archive object to which all the editing operations will be sent.
Archive* TheArchive = 0;

// The name this program was invoked as.
static const char *program_name;

// show_help - Show the error message, the help message and exit.
LLVM_ATTRIBUTE_NORETURN static void
show_help(const std::string &msg) {
  errs() << program_name << ": " << msg << "\n\n";
  cl::PrintHelpMessage();
  if (TheArchive)
    delete TheArchive;
  std::exit(1);
}

// fail - Show the error message and exit.
LLVM_ATTRIBUTE_NORETURN static void
fail(const std::string &msg) {
  errs() << program_name << ": " << msg << "\n\n";
  if (TheArchive)
    delete TheArchive;
  std::exit(1);
}

// getRelPos - Extract the member filename from the command line for
// the [relpos] argument associated with a, b, and i modifiers
void getRelPos() {
  if(RestOfArgs.size() == 0)
    show_help("Expected [relpos] for a, b, or i modifier");
  RelPos = RestOfArgs[0];
  RestOfArgs.erase(RestOfArgs.begin());
}

// getArchive - Get the archive file name from the command line
void getArchive() {
  if(RestOfArgs.size() == 0)
    show_help("An archive name must be specified");
  ArchiveName = RestOfArgs[0];
  RestOfArgs.erase(RestOfArgs.begin());
}

// getMembers - Copy over remaining items in RestOfArgs to our Members vector
// This is just for clarity.
void getMembers() {
  if(RestOfArgs.size() > 0)
    Members = std::vector<std::string>(RestOfArgs);
}

// parseCommandLine - Parse the command line options as presented and return the
// operation specified. Process all modifiers and check to make sure that
// constraints on modifier/operation pairs have not been violated.
ArchiveOperation parseCommandLine() {

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

// buildPaths - Convert the strings in the Members vector to sys::Path objects
// and make sure they are valid and exist exist. This check is only needed for
// the operations that add/replace files to the archive ('q' and 'r')
bool buildPaths(bool checkExistence, std::string* ErrMsg) {
  for (unsigned i = 0; i < Members.size(); i++) {
    std::string aPath = Members[i];
    if (checkExistence) {
      bool IsDirectory;
      error_code EC = sys::fs::is_directory(aPath, IsDirectory);
      if (EC)
        fail(aPath + ": " + EC.message());
      if (IsDirectory)
        fail(aPath + " Is a directory");

      Paths.insert(aPath);
    } else {
      Paths.insert(aPath);
    }
  }
  return false;
}

// doPrint - Implements the 'p' operation. This function traverses the archive
// looking for members that match the path list. It is careful to uncompress
// things that should be and to skip bitcode files unless the 'k' modifier was
// given.
bool doPrint(std::string* ErrMsg) {
  if (buildPaths(false, ErrMsg))
    return true;
  for (Archive::iterator I = TheArchive->begin(), E = TheArchive->end();
       I != E; ++I ) {
    if (Paths.empty() ||
        (std::find(Paths.begin(), Paths.end(), I->getPath()) != Paths.end())) {
      const char *data = reinterpret_cast<const char *>(I->getData());

      // Skip things that don't make sense to print
      if (I->isSVR4SymbolTable() || I->isBSD4SymbolTable())
        continue;

      if (Verbose)
        outs() << "Printing " << I->getPath().str() << "\n";

      unsigned len = I->getSize();
      outs().write(data, len);
    }
  }
  return false;
}

// putMode - utility function for printing out the file mode when the 't'
// operation is in verbose mode.
void
printMode(unsigned mode) {
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

// doDisplayTable - Implement the 't' operation. This function prints out just
// the file names of each of the members. However, if verbose mode is requested
// ('v' modifier) then the file type, permission mode, user, group, size, and
// modification time are also printed.
bool
doDisplayTable(std::string* ErrMsg) {
  if (buildPaths(false, ErrMsg))
    return true;
  for (Archive::iterator I = TheArchive->begin(), E = TheArchive->end();
       I != E; ++I ) {
    if (Paths.empty() ||
        (std::find(Paths.begin(), Paths.end(), I->getPath()) != Paths.end())) {
      if (Verbose) {
        unsigned mode = I->getMode();
        printMode((mode >> 6) & 007);
        printMode((mode >> 3) & 007);
        printMode(mode & 007);
        outs() << ' ' << I->getUser();
        outs() << "/" << I->getGroup();
        outs() << ' ' << format("%6llu", I->getSize());
        sys::TimeValue ModTime = I->getModTime();
        outs() << " " << ModTime.str();
        outs() << " " << I->getPath().str() << "\n";
      } else {
        outs() << I->getPath().str() << "\n";
      }
    }
  }
  return false;
}

// doExtract - Implement the 'x' operation. This function extracts files back to
// the file system.
bool
doExtract(std::string* ErrMsg) {
  if (buildPaths(false, ErrMsg))
    return true;
  for (Archive::iterator I = TheArchive->begin(), E = TheArchive->end();
       I != E; ++I ) {
    if (Paths.empty() ||
        (std::find(Paths.begin(), Paths.end(), I->getPath()) != Paths.end())) {

      // Open up a file stream for writing
      int OpenFlags = O_TRUNC | O_WRONLY | O_CREAT;
#ifdef O_BINARY
      OpenFlags |= O_BINARY;
#endif

      // Retain the original mode.
      sys::fs::perms Mode = sys::fs::perms(I->getMode());

      int FD = open(I->getPath().str().c_str(), OpenFlags, Mode);
      if (FD < 0)
        return true;

      {
        raw_fd_ostream file(FD, false);

        // Get the data and its length
        const char* data = reinterpret_cast<const char*>(I->getData());
        unsigned len = I->getSize();

        // Write the data.
        file.write(data, len);
      }

      // If we're supposed to retain the original modification times, etc. do so
      // now.
      if (OriginalDates) {
        error_code EC =
            sys::fs::setLastModificationAndAccessTime(FD, I->getModTime());
        if (EC)
          fail(EC.message());
      }
      if (close(FD))
        return true;
    }
  }
  return false;
}

// doDelete - Implement the delete operation. This function deletes zero or more
// members from the archive. Note that if the count is specified, there should
// be no more than one path in the Paths list or else this algorithm breaks.
// That check is enforced in parseCommandLine (above).
bool
doDelete(std::string* ErrMsg) {
  if (buildPaths(false, ErrMsg))
    return true;
  if (Paths.empty())
    return false;
  for (Archive::iterator I = TheArchive->begin(), E = TheArchive->end();
       I != E; ) {
    if (std::find(Paths.begin(), Paths.end(), I->getPath()) != Paths.end()) {
      Archive::iterator J = I;
      ++I;
      TheArchive->erase(J);
    } else {
      ++I;
    }
  }

  // We're done editting, reconstruct the archive.
  if (TheArchive->writeToDisk(ErrMsg))
    return true;
  return false;
}

// doMore - Implement the move operation. This function re-arranges just the
// order of the archive members so that when the archive is written the move
// of the members is accomplished. Note the use of the RelPos variable to
// determine where the items should be moved to.
bool
doMove(std::string* ErrMsg) {
  if (buildPaths(false, ErrMsg))
    return true;

  // By default and convention the place to move members to is the end of the
  // archive.
  Archive::iterator moveto_spot = TheArchive->end();

  // However, if the relative positioning modifiers were used, we need to scan
  // the archive to find the member in question. If we don't find it, its no
  // crime, we just move to the end.
  if (AddBefore || AddAfter) {
    for (Archive::iterator I = TheArchive->begin(), E= TheArchive->end();
         I != E; ++I ) {
      if (RelPos == I->getPath().str()) {
        if (AddAfter) {
          moveto_spot = I;
          moveto_spot++;
        } else {
          moveto_spot = I;
        }
        break;
      }
    }
  }

  // Keep a list of the paths remaining to be moved
  std::set<std::string> remaining(Paths);

  // Scan the archive again, this time looking for the members to move to the
  // moveto_spot.
  for (Archive::iterator I = TheArchive->begin(), E= TheArchive->end();
       I != E && !remaining.empty(); ++I ) {
    std::set<std::string>::iterator found =
      std::find(remaining.begin(),remaining.end(), I->getPath());
    if (found != remaining.end()) {
      if (I != moveto_spot)
        TheArchive->splice(moveto_spot,*TheArchive,I);
      remaining.erase(found);
    }
  }

  // We're done editting, reconstruct the archive.
  if (TheArchive->writeToDisk(ErrMsg))
    return true;
  return false;
}

// doQuickAppend - Implements the 'q' operation. This function just
// indiscriminantly adds the members to the archive and rebuilds it.
bool
doQuickAppend(std::string* ErrMsg) {
  // Get the list of paths to append.
  if (buildPaths(true, ErrMsg))
    return true;
  if (Paths.empty())
    return false;

  // Append them quickly.
  for (std::set<std::string>::iterator PI = Paths.begin(), PE = Paths.end();
       PI != PE; ++PI) {
    if (TheArchive->addFileBefore(*PI, TheArchive->end(), ErrMsg))
      return true;
  }

  // We're done editting, reconstruct the archive.
  if (TheArchive->writeToDisk(ErrMsg))
    return true;
  return false;
}

// doReplaceOrInsert - Implements the 'r' operation. This function will replace
// any existing files or insert new ones into the archive.
bool
doReplaceOrInsert(std::string* ErrMsg) {

  // Build the list of files to be added/replaced.
  if (buildPaths(true, ErrMsg))
    return true;
  if (Paths.empty())
    return false;

  // Keep track of the paths that remain to be inserted.
  std::set<std::string> remaining(Paths);

  // Default the insertion spot to the end of the archive
  Archive::iterator insert_spot = TheArchive->end();

  // Iterate over the archive contents
  for (Archive::iterator I = TheArchive->begin(), E = TheArchive->end();
       I != E && !remaining.empty(); ++I ) {

    // Determine if this archive member matches one of the paths we're trying
    // to replace.

    std::set<std::string>::iterator found = remaining.end();
    for (std::set<std::string>::iterator RI = remaining.begin(),
         RE = remaining.end(); RI != RE; ++RI ) {
      std::string compare(sys::path::filename(*RI));
      if (compare == I->getPath().str()) {
        found = RI;
        break;
      }
    }

    if (found != remaining.end()) {
      sys::fs::file_status Status;
      error_code EC = sys::fs::status(*found, Status);
      if (EC)
        return true;
      if (!sys::fs::is_directory(Status)) {
        if (OnlyUpdate) {
          // Replace the item only if it is newer.
          if (Status.getLastModificationTime() > I->getModTime())
            if (I->replaceWith(*found, ErrMsg))
              return true;
        } else {
          // Replace the item regardless of time stamp
          if (I->replaceWith(*found, ErrMsg))
            return true;
        }
      } else {
        // We purposefully ignore directories.
      }

      // Remove it from our "to do" list
      remaining.erase(found);
    }

    // Determine if this is the place where we should insert
    if (AddBefore && RelPos == I->getPath().str())
      insert_spot = I;
    else if (AddAfter && RelPos == I->getPath().str()) {
      insert_spot = I;
      insert_spot++;
    }
  }

  // If we didn't replace all the members, some will remain and need to be
  // inserted at the previously computed insert-spot.
  if (!remaining.empty()) {
    for (std::set<std::string>::iterator PI = remaining.begin(),
         PE = remaining.end(); PI != PE; ++PI) {
      if (TheArchive->addFileBefore(*PI, insert_spot, ErrMsg))
        return true;
    }
  }

  // We're done editting, reconstruct the archive.
  if (TheArchive->writeToDisk(ErrMsg))
    return true;
  return false;
}

bool shouldCreateArchive(ArchiveOperation Op) {
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

// main - main program for llvm-ar .. see comments in the code
int main(int argc, char **argv) {
  program_name = argv[0];
  // Print a stack trace if we signal out.
  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(argc, argv);
  LLVMContext &Context = getGlobalContext();
  llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.

  // Have the command line options parsed and handle things
  // like --help and --version.
  cl::ParseCommandLineOptions(argc, argv,
    "LLVM Archiver (llvm-ar)\n\n"
    "  This program archives bitcode files into single libraries\n"
  );

  int exitCode = 0;

  // Do our own parsing of the command line because the CommandLine utility
  // can't handle the grouped positional parameters without a dash.
  ArchiveOperation Operation = parseCommandLine();

  // Create or open the archive object.
  if (shouldCreateArchive(Operation) && !llvm::sys::fs::exists(ArchiveName)) {
    // Produce a warning if we should and we're creating the archive
    if (!Create)
      errs() << argv[0] << ": creating " << ArchiveName << "\n";
    TheArchive = Archive::CreateEmpty(ArchiveName, Context);
    TheArchive->writeToDisk();
  }

  if (!TheArchive) {
    std::string Error;
    TheArchive = Archive::OpenAndLoad(ArchiveName, Context, &Error);
    if (TheArchive == 0) {
      errs() << argv[0] << ": error loading '" << ArchiveName << "': "
             << Error << "!\n";
      return 1;
    }
  }

  // Make sure we're not fooling ourselves.
  assert(TheArchive && "Unable to instantiate the archive");

  // Perform the operation
  std::string ErrMsg;
  bool haveError = false;
  switch (Operation) {
    case Print:           haveError = doPrint(&ErrMsg); break;
    case Delete:          haveError = doDelete(&ErrMsg); break;
    case Move:            haveError = doMove(&ErrMsg); break;
    case QuickAppend:     haveError = doQuickAppend(&ErrMsg); break;
    case ReplaceOrInsert: haveError = doReplaceOrInsert(&ErrMsg); break;
    case DisplayTable:    haveError = doDisplayTable(&ErrMsg); break;
    case Extract:         haveError = doExtract(&ErrMsg); break;
  }
  if (haveError) {
    errs() << argv[0] << ": " << ErrMsg << "\n";
    return 1;
  }

  delete TheArchive;
  TheArchive = 0;

  // Return result code back to operating system.
  return exitCode;
}
