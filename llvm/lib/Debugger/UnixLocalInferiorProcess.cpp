//===-- UnixLocalInferiorProcess.cpp - A Local process on a Unixy system --===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
// 
// This file provides one implementation of the InferiorProcess class, which is
// designed to be used on unixy systems (those that support pipe, fork, exec,
// and signals).
//
// When the process is started, the debugger creates a pair of pipes, forks, and
// makes the child start executing the program.  The child executes the process
// with an IntrinsicLowering instance that turns debugger intrinsics into actual
// callbacks.
//
// This target takes advantage of the fact that the Module* addresses in the
// parent and the Module* addresses in the child will be the same, due to the
// use of fork().  As such, global addresses looked up in the child can be sent
// over the pipe to the debugger.
//
//===----------------------------------------------------------------------===//

#include "llvm/Debugger/InferiorProcess.h"
#include "llvm/Constant.h"
#include "llvm/Module.h"
#include "llvm/ModuleProvider.h"
#include "llvm/Type.h"
#include "llvm/iOther.h"
#include "llvm/CodeGen/IntrinsicLowering.h"
#include "llvm/ExecutionEngine/GenericValue.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "Support/FileUtilities.h"
#include "Support/StringExtras.h"
#include <cerrno>
#include <csignal>
#include <unistd.h>        // Unix-specific debugger support
#include <sys/types.h>
#include <sys/wait.h>
using namespace llvm;

// runChild - Entry point for the child process.
static void runChild(Module *M, const std::vector<std::string> &Arguments,
                     const char * const *envp,
                     FDHandle ReadFD, FDHandle WriteFD);

//===----------------------------------------------------------------------===//
//                        Parent/Child Pipe Protocol
//===----------------------------------------------------------------------===//
//
// The parent/child communication protocol is designed to have the child process
// responding to requests that the debugger makes.  Whenever the child process
// has stopped (due to a break point, single stepping, etc), the child process
// enters a message processing loop, where it reads and responds to commands
// until the parent decides that it wants to continue execution in some way.
//
// Whenever the child process stops, it notifies the debugger by sending a
// character over the wire.
//

namespace {
  /// LocationToken - Objects of this type are sent across the pipe from the
  /// child to the parent to indicate where various stack frames are located.
  struct LocationToken {
    unsigned Line, Col;
    const GlobalVariable *File;
    LocationToken(unsigned L = 0, unsigned C = 0, const GlobalVariable *F = 0)
      : Line(L), Col(C), File(F) {}
  };
}

// Once the debugger process has received the LocationToken, it can make
// requests of the child by sending one of the following enum values followed by
// any data required by that command.  The child responds with data appropriate
// to the command.
//
namespace {
  /// CommandID - This enum defines all of the commands that the child process
  /// can respond to.  The actual expected data and responses are defined as the
  /// enum values are defined.
  ///
  enum CommandID {
    //===------------------------------------------------------------------===//
    // Execution commands - These are sent to the child to from the debugger to
    // get it to do certain things.
    //

    // StepProgram: void->char - This command causes the program to continue
    // execution, but stop as soon as it reaches another stoppoint.
    StepProgram,

    // FinishProgram: FrameDesc*->char - This command causes the program to
    // continue execution until the specified function frame returns.
    FinishProgram, 

    // ContProgram: void->char - This command causes the program to continue
    // execution, stopping at some point in the future.
    ContProgram,

    // GetSubprogramDescriptor: FrameDesc*->GlobalValue* - This command returns
    // the GlobalValue* descriptor object for the specified stack frame.
    GetSubprogramDescriptor,

    // GetParentFrame: FrameDesc*->FrameDesc* - This command returns the frame
    // descriptor for the parent stack frame to the specified one, or null if
    // there is none.
    GetParentFrame,

    // GetFrameLocation - FrameDesc*->LocationToken - This command returns the
    // location that a particular stack frame is stopped at.
    GetFrameLocation,

    // AddBreakpoint - LocationToken->unsigned - This command instructs the
    // target to install a breakpoint at the specified location.
    AddBreakpoint,

    // RemoveBreakpoint - unsigned->void - This command instructs the target to
    // remove a breakpoint.
    RemoveBreakpoint,
  };
}




//===----------------------------------------------------------------------===//
//                            Parent Process Code
//===----------------------------------------------------------------------===//

namespace {
  class IP : public InferiorProcess {
    // ReadFD, WriteFD - The file descriptors to read/write to the inferior
    // process.
    FDHandle ReadFD, WriteFD;

    // ChildPID - The unix PID of the child process we forked.
    mutable pid_t ChildPID;
  public:
    IP(Module *M, const std::vector<std::string> &Arguments,
       const char * const *envp);
    ~IP();

    std::string getStatus() const;

    /// Execution method implementations...
    virtual void stepProgram();
    virtual void finishProgram(void *Frame);
    virtual void contProgram();


    // Stack frame method implementations...
    virtual void *getPreviousFrame(void *Frame) const;
    virtual const GlobalVariable *getSubprogramDesc(void *Frame) const;
    virtual void getFrameLocation(void *Frame, unsigned &LineNo,
                                  unsigned &ColNo,
                                  const GlobalVariable *&SourceDesc) const;

    // Breakpoint implementation methods
    virtual unsigned addBreakpoint(unsigned LineNo, unsigned ColNo,
                                   const GlobalVariable *SourceDesc);
    virtual void removeBreakpoint(unsigned ID);


  private:
    /// startChild - This starts up the child process, and initializes the
    /// ChildPID member.
    ///
    void startChild(Module *M, const std::vector<std::string> &Arguments,
                    const char * const *envp);

    /// killChild - Kill or reap the child process.  This throws the
    /// InferiorProcessDead exception an exit code if the process had already
    /// died, otherwise it just kills it and returns.
    void killChild() const;

  private:
    // Methods for communicating with the child process.  If the child exits or
    // dies while attempting to communicate with it, ChildPID is set to zero and
    // an exception is thrown.

    /// readFromChild - Low-level primitive to read some data from the child,
    /// throwing an exception if it dies.
    void readFromChild(void *Buffer, unsigned Size) const;

    /// writeToChild - Low-level primitive to send some data to the child
    /// process, throwing an exception if the child died.
    void writeToChild(void *Buffer, unsigned Size) const;

    /// sendCommand - Send a command token and the request data to the child.
    ///
    void sendCommand(CommandID Command, void *Data, unsigned Size) const;

    /// waitForStop - This method waits for the child process to reach a stop
    /// point.
    void waitForStop();
  };
}

// create - This is the factory method for the InferiorProcess class.  Since
// there is currently only one subclass of InferiorProcess, we just define it
// here.
InferiorProcess *
InferiorProcess::create(Module *M, const std::vector<std::string> &Arguments,
                        const char * const *envp) {
  return new IP(M, Arguments, envp);
}

/// IP constructor - Create some pipes, them fork a child process.  The child
/// process should start execution of the debugged program, but stop at the
/// first available opportunity.
IP::IP(Module *M, const std::vector<std::string> &Arguments,
       const char * const *envp)
  : InferiorProcess(M) {

  // Start the child running...
  startChild(M, Arguments, envp);
  
  // Okay, we created the program and it is off and running.  Wait for it to
  // stop now.
  try {
    waitForStop();
  } catch (InferiorProcessDead &IPD) {
    throw "Error waiting for the child process to stop.  "
          "It exited with status " + itostr(IPD.getExitCode());
  }
}

IP::~IP() {
  // If the child is still running, kill it.
  if (!ChildPID) return;

  killChild();
}

/// getStatus - Return information about the unix process being debugged.
///
std::string IP::getStatus() const {
  if (ChildPID == 0)
    return "Unix target.  ERROR: child process appears to be dead!\n";

  return "Unix target: PID #" + utostr((unsigned)ChildPID) + "\n";
}


/// startChild - This starts up the child process, and initializes the
/// ChildPID member.
///
void IP::startChild(Module *M, const std::vector<std::string> &Arguments,
                    const char * const *envp) {
  // Create the pipes.  Make sure to immediately assign the returned file
  // descriptors to FDHandle's so they get destroyed if an exception is thrown.
  int FDs[2];
  if (pipe(FDs)) throw "Error creating a pipe!";
  FDHandle ChildReadFD(FDs[0]);
  WriteFD = FDs[1];

  if (pipe(FDs)) throw "Error creating a pipe!";
  ReadFD = FDs[0];
  FDHandle ChildWriteFD(FDs[1]);

  // Fork off the child process.
  switch (ChildPID = fork()) {
  case -1: throw "Error forking child process!";
  case 0:  // child
    delete this;       // Free parent pipe file descriptors
    runChild(M, Arguments, envp, ChildReadFD, ChildWriteFD);
    exit(1);
  default: break;
  }
}

/// sendCommand - Send a command token and the request data to the child.
///
void IP::sendCommand(CommandID Command, void *Data, unsigned Size) const {
  writeToChild(&Command, sizeof(Command));
  writeToChild(Data, Size);
}

/// stepProgram - Implement the 'step' command, continuing execution until
/// the next possible stop point.
void IP::stepProgram() {
  sendCommand(StepProgram, 0, 0);
  waitForStop();
}

/// finishProgram - Implement the 'finish' command, executing the program until
/// the current function returns to its caller.
void IP::finishProgram(void *Frame) {
  sendCommand(FinishProgram, &Frame, sizeof(Frame));
  waitForStop();
}

/// contProgram - Implement the 'cont' command, continuing execution until
/// a breakpoint is encountered.
void IP::contProgram() {
  sendCommand(ContProgram, 0, 0);
  waitForStop();
}


//===----------------------------------------------------------------------===//
// Stack manipulation methods
//

/// getPreviousFrame - Given the descriptor for the current stack frame,
/// return the descriptor for the caller frame.  This returns null when it
/// runs out of frames.
void *IP::getPreviousFrame(void *Frame) const {
  sendCommand(GetParentFrame, &Frame, sizeof(Frame));
  readFromChild(&Frame, sizeof(Frame));
  return Frame;
}

/// getSubprogramDesc - Return the subprogram descriptor for the current
/// stack frame.
const GlobalVariable *IP::getSubprogramDesc(void *Frame) const {
  sendCommand(GetSubprogramDescriptor, &Frame, sizeof(Frame));
  const GlobalVariable *Desc;
  readFromChild(&Desc, sizeof(Desc));
  return Desc;
}

/// getFrameLocation - This method returns the source location where each stack
/// frame is stopped.
void IP::getFrameLocation(void *Frame, unsigned &LineNo, unsigned &ColNo, 
                          const GlobalVariable *&SourceDesc) const {
  sendCommand(GetFrameLocation, &Frame, sizeof(Frame));
  LocationToken Loc;
  readFromChild(&Loc, sizeof(Loc));
  LineNo     = Loc.Line;
  ColNo      = Loc.Col;
  SourceDesc = Loc.File;
}


//===----------------------------------------------------------------------===//
// Breakpoint manipulation methods
//
unsigned IP::addBreakpoint(unsigned LineNo, unsigned ColNo,
                           const GlobalVariable *SourceDesc) {
  LocationToken Loc;
  Loc.Line = LineNo;
  Loc.Col = ColNo;
  Loc.File = SourceDesc;
  sendCommand(AddBreakpoint, &Loc, sizeof(Loc));
  unsigned ID;
  readFromChild(&ID, sizeof(ID));
  return ID;
}

void IP::removeBreakpoint(unsigned ID) {
  sendCommand(RemoveBreakpoint, &ID, sizeof(ID));
}


//===----------------------------------------------------------------------===//
//             Methods for communication with the child process
//
// Methods for communicating with the child process.  If the child exits or dies
// while attempting to communicate with it, ChildPID is set to zero and an
// exception is thrown.
//

/// readFromChild - Low-level primitive to read some data from the child,
/// throwing an exception if it dies.
void IP::readFromChild(void *Buffer, unsigned Size) const {
  assert(ChildPID &&
         "Child process died and still attempting to communicate with it!");
  while (Size) {
    ssize_t Amount = read(ReadFD, Buffer, Size);
    if (Amount == 0) {
      // If we cannot communicate with the process, kill it.
      killChild();
      // If killChild succeeded, then the process must have closed the pipe FD
      // or something, because the child existed, but we cannot communicate with
      // it.
      throw InferiorProcessDead(-1);
    } else if (Amount == -1) {
      if (errno != EINTR) {
        ChildPID = 0;
        killChild();
        throw "Error reading from child process!";
      }
    } else {
      // We read a chunk.
      Buffer = (char*)Buffer + Amount;
      Size -= Amount;
    }
  }
}

/// writeToChild - Low-level primitive to send some data to the child
/// process, throwing an exception if the child died.
void IP::writeToChild(void *Buffer, unsigned Size) const {
  while (Size) {
    ssize_t Amount = write(WriteFD, Buffer, Size);
    if (Amount < 0 && errno == EINTR) continue;
    if (Amount <= 0) {
      // If we cannot communicate with the process, kill it.
      killChild();

      // If killChild succeeded, then the process must have closed the pipe FD
      // or something, because the child existed, but we cannot communicate with
      // it.
      throw InferiorProcessDead(-1);
    } else {
      // We wrote a chunk.
      Buffer = (char*)Buffer + Amount;
      Size -= Amount;
    }
  }
}

/// killChild - Kill or reap the child process.  This throws the
/// InferiorProcessDead exception an exit code if the process had already
/// died, otherwise it just returns the exit code if it had to be killed.
void IP::killChild() const {
  assert(ChildPID != 0 && "Child has already been reaped!");
  
  // If the process terminated on its own accord, closing the pipe file
  // descriptors, we will get here.  Check to see if the process has already
  // died in this manner, gracefully.
  int Status = 0;
  int PID;
  do {
    PID = waitpid(ChildPID, &Status, WNOHANG);
  } while (PID < 0 && errno == EINTR);
  if (PID < 0) throw "Error waiting for child to exit!";

  // Ok, there is a slight race condition here.  It's possible that we will find
  // out that the file descriptor closed before waitpid will indicate that the
  // process gracefully died.  If we don't know that the process gracefully
  // died, wait a bit and try again.  This is pretty nasty.
  if (PID == 0) {
    usleep(10000);   // Wait a bit.

    // Try again.
    Status = 0;
    do {
      PID = waitpid(ChildPID, &Status, WNOHANG);
    } while (PID < 0 && errno == EINTR);
    if (PID < 0) throw "Error waiting for child to exit!";
  }

  // If the child process was already dead, then indicate that the process
  // terminated on its own.
  if (PID) {
    assert(PID == ChildPID && "Didn't reap child?");
    ChildPID = 0;            // Child has been reaped
    if (WIFEXITED(Status))
      throw InferiorProcessDead(WEXITSTATUS(Status));
    else if (WIFSIGNALED(Status))
      throw InferiorProcessDead(WTERMSIG(Status));
    throw InferiorProcessDead(-1);
  }
  
  // Otherwise, the child exists and has not yet been killed.
  if (kill(ChildPID, SIGKILL) < 0)
    throw "Error killing child process!";

  do {
    PID = waitpid(ChildPID, 0, 0);
  } while (PID < 0 && errno == EINTR);
  if (PID <= 0) throw "Error waiting for child to exit!";

  assert(PID == ChildPID && "Didn't reap child?");
}


/// waitForStop - This method waits for the child process to reach a stop
/// point.  When it does, it fills in the CurLocation member and returns.
void IP::waitForStop() {
  char Dummy;
  readFromChild(&Dummy, sizeof(char));
}


//===----------------------------------------------------------------------===//
//                             Child Process Code
//===----------------------------------------------------------------------===//

namespace {
  class SourceSubprogram;

  /// SourceRegion - Instances of this class represent the regions that are
  /// active in the program.
  class SourceRegion {
    /// Parent - A pointer to the region that encloses the current one.
    SourceRegion *Parent;

    /// CurSubprogram - The subprogram that contains this region.  This allows
    /// efficient stack traversals.
    SourceSubprogram *CurSubprogram;

    /// CurLine, CurCol, CurFile - The last location visited by this region.
    /// This is used for getting the source location of callers in stack frames.
    unsigned CurLine, CurCol;
    void *CurFileDesc;

    //std::vector<void*> ActiveObjects;
  public:
    SourceRegion(SourceRegion *p, SourceSubprogram *Subprogram = 0)
     : Parent(p), CurSubprogram(Subprogram ? Subprogram : p->getSubprogram()) {
      CurLine = 0; CurCol = 0;
      CurFileDesc = 0;
    }

    virtual ~SourceRegion() {}

    SourceRegion *getParent() const { return Parent; }
    SourceSubprogram *getSubprogram() const { return CurSubprogram; }

    void updateLocation(unsigned Line, unsigned Col, void *File) {
      CurLine = Line;
      CurCol = Col;
      CurFileDesc = File;
    }

    /// Return a LocationToken for the place that this stack frame stopped or
    /// called a sub-function.
    LocationToken getLocation(ExecutionEngine *EE) {
      LocationToken LT;
      LT.Line = CurLine;
      LT.Col = CurCol;
      const GlobalValue *GV = EE->getGlobalValueAtAddress(CurFileDesc);
      LT.File = dyn_cast_or_null<GlobalVariable>(GV);
      return LT;
    }
  };

  /// SourceSubprogram - This is a stack-frame that represents a source program.
  ///
  class SourceSubprogram : public SourceRegion {
    /// Desc - A pointer to the descriptor for the subprogram that this frame
    /// represents.
    void *Desc;
  public:
    SourceSubprogram(SourceRegion *P, void *desc)
      : SourceRegion(P, this), Desc(desc) {}
    void *getDescriptor() const { return Desc; }
  };


  /// Child class - This class contains all of the information and methods used
  /// by the child side of the debugger.  The single instance of this object is
  /// pointed to by the "TheChild" global variable.
  class Child {
    /// M - The module for the program currently being debugged.
    ///
    Module *M;

    /// EE - The execution engine that we are using to run the program.
    ///
    ExecutionEngine *EE;

    /// ReadFD, WriteFD - The file descriptor handles for this side of the
    /// debugger pipe.
    FDHandle ReadFD, WriteFD;

    /// RegionStack - A linked list of all of the regions dynamically active.
    ///
    SourceRegion *RegionStack;

    /// StopAtNextOpportunity - If this flag is set, the child process will stop
    /// and report to the debugger at the next possible chance it gets.
    volatile bool StopAtNextOpportunity;

    /// StopWhenSubprogramReturns - If this is non-null, the debugger requests
    /// that the program stops when the specified function frame is destroyed.
    SourceSubprogram *StopWhenSubprogramReturns;

    /// Breakpoints - This contains a list of active breakpoints and their IDs.
    ///
    std::vector<std::pair<unsigned, LocationToken> > Breakpoints;

    /// CurBreakpoint - The last assigned breakpoint.
    ///
    unsigned CurBreakpoint;

  public:
    Child(Module *m, ExecutionEngine *ee, FDHandle &Read, FDHandle &Write)
      : M(m), EE(ee), ReadFD(Read), WriteFD(Write),
        RegionStack(0), CurBreakpoint(0) {
      StopAtNextOpportunity = true;
      StopWhenSubprogramReturns = 0;
    }

    /// writeToParent - Send the specified buffer of data to the debugger
    /// process.
    ///
    void writeToParent(const void *Buffer, unsigned Size);

    /// readFromParent - Read the specified number of bytes from the parent.
    ///
    void readFromParent(void *Buffer, unsigned Size);

    /// childStopped - This method is called whenever the child has stopped
    /// execution due to a breakpoint, step command, interruption, or whatever.
    /// This stops the process, responds to any requests from the debugger, and
    /// when commanded to, can continue execution by returning.
    ///
    void childStopped();

    /// startSubprogram - This method creates a new region for the subroutine
    /// with the specified descriptor.
    ///
    void startSubprogram(void *FuncDesc);

    /// startRegion - This method initiates the creation of an anonymous region.
    ///
    void startRegion();

    /// endRegion - This method terminates the last active region.
    ///
    void endRegion();

    /// reachedLine - This method is automatically called by the program every
    /// time it executes an llvm.dbg.stoppoint intrinsic.  If the debugger wants
    /// us to stop here, we do so, otherwise we continue execution.
    ///
    void reachedLine(unsigned Line, unsigned Col, void *SourceDesc);
  };

  /// TheChild - The single instance of the Child class, which only gets created
  /// in the child process.
  Child *TheChild = 0;
} // end anonymous namespace


// writeToParent - Send the specified buffer of data to the debugger process.
void Child::writeToParent(const void *Buffer, unsigned Size) {
  while (Size) {
    ssize_t Amount = write(WriteFD, Buffer, Size);
    if (Amount < 0 && errno == EINTR) continue;
    if (Amount <= 0) {
      write(2, "ERROR: Connection to debugger lost!\n", 36);
      abort();
    } else {
      // We wrote a chunk.
      Buffer = (const char*)Buffer + Amount;
      Size -= Amount;
    }
  }
}

// readFromParent - Read the specified number of bytes from the parent.
void Child::readFromParent(void *Buffer, unsigned Size) {
  while (Size) {
    ssize_t Amount = read(ReadFD, Buffer, Size);
    if (Amount < 0 && errno == EINTR) continue;
    if (Amount <= 0) {
      write(2, "ERROR: Connection to debugger lost!\n", 36);
      abort();
    } else {
      // We read a chunk.
      Buffer = (char*)Buffer + Amount;
      Size -= Amount;
    }
  }
}

/// childStopped - This method is called whenever the child has stopped
/// execution due to a breakpoint, step command, interruption, or whatever.
/// This stops the process, responds to any requests from the debugger, and when
/// commanded to, can continue execution by returning.
///
void Child::childStopped() {
  // Since we stopped, notify the parent that we did so.
  char Token = 0;
  writeToParent(&Token, sizeof(char));

  StopAtNextOpportunity = false;
  StopWhenSubprogramReturns = 0;

  // Now that the debugger knows that we stopped, read commands from it and
  // respond to them appropriately.
  CommandID Command;
  while (1) {
    SourceRegion *Frame;
    const void *Result;
    readFromParent(&Command, sizeof(CommandID));

    switch (Command) {
    case StepProgram:
      // To step the program, just return.
      StopAtNextOpportunity = true;
      return;

    case FinishProgram:         // Run until exit from the specified function...
      readFromParent(&Frame, sizeof(Frame));
      // The user wants us to stop when the specified FUNCTION exits, not when
      // the specified REGION exits.
      StopWhenSubprogramReturns = Frame->getSubprogram();
      return;

    case ContProgram:
      // To continue, just return back to execution.
      return;

    case GetSubprogramDescriptor:
      readFromParent(&Frame, sizeof(Frame));
      Result =
        EE->getGlobalValueAtAddress(Frame->getSubprogram()->getDescriptor());
      writeToParent(&Result, sizeof(Result));
      break;

    case GetParentFrame:
      readFromParent(&Frame, sizeof(Frame));
      Result = Frame ? Frame->getSubprogram()->getParent() : RegionStack;
      writeToParent(&Result, sizeof(Result));
      break;

    case GetFrameLocation: {
      readFromParent(&Frame, sizeof(Frame));
      LocationToken LT = Frame->getLocation(EE);
      writeToParent(&LT, sizeof(LT));
      break;
    }
    case AddBreakpoint: {
      LocationToken Loc;
      readFromParent(&Loc, sizeof(Loc));
      // Convert the GlobalVariable pointer to the address it was emitted to.
      Loc.File = (GlobalVariable*)EE->getPointerToGlobal(Loc.File);
      unsigned ID = CurBreakpoint++;
      Breakpoints.push_back(std::make_pair(ID, Loc));
      writeToParent(&ID, sizeof(ID));
      break;
    }
    case RemoveBreakpoint: {
      unsigned ID = 0;
      readFromParent(&ID, sizeof(ID));
      for (unsigned i = 0, e = Breakpoints.size(); i != e; ++i)
        if (Breakpoints[i].first == ID) {
          Breakpoints.erase(Breakpoints.begin()+i);
          break;
        }
      break;
    }
    default:
      assert(0 && "Unknown command!");
    }
  }
}



/// startSubprogram - This method creates a new region for the subroutine
/// with the specified descriptor.
void Child::startSubprogram(void *SPDesc) {
  RegionStack = new SourceSubprogram(RegionStack, SPDesc);
}

/// startRegion - This method initiates the creation of an anonymous region.
///
void Child::startRegion() {
  RegionStack = new SourceRegion(RegionStack);
}

/// endRegion - This method terminates the last active region.
///
void Child::endRegion() {
  SourceRegion *R = RegionStack->getParent();

  // If the debugger wants us to stop when this frame is destroyed, do so.
  if (RegionStack == StopWhenSubprogramReturns) {
    StopAtNextOpportunity = true;
    StopWhenSubprogramReturns = 0;
  }

  delete RegionStack;
  RegionStack = R;
}




/// reachedLine - This method is automatically called by the program every time
/// it executes an llvm.dbg.stoppoint intrinsic.  If the debugger wants us to
/// stop here, we do so, otherwise we continue execution.  Note that the Data
/// pointer coming in is a pointer to the LLVM global variable that represents
/// the source file we are in.  We do not use the contents of the global
/// directly in the child, but we do use its address.
///
void Child::reachedLine(unsigned Line, unsigned Col, void *SourceDesc) {
  if (RegionStack)
    RegionStack->updateLocation(Line, Col, SourceDesc);

  // If we hit a breakpoint, stop the program.
  for (unsigned i = 0, e = Breakpoints.size(); i != e; ++i)
    if (Line       == Breakpoints[i].second.Line &&
        SourceDesc == (void*)Breakpoints[i].second.File &&
        Col        == Breakpoints[i].second.Col) {
      childStopped();
      return;
    }

  // If we are single stepping the program, make sure to stop it.
  if (StopAtNextOpportunity)
    childStopped();
}




//===----------------------------------------------------------------------===//
//                        Child class wrapper functions
//
// These functions are invoked directly by the program as it executes, in place
// of the debugging intrinsic functions that it contains.
//


/// llvm_debugger_stop - Every time the program reaches a new source line, it
/// will call back to this function.  If the debugger has a breakpoint or
/// otherwise wants us to stop on this line, we do so, and notify the debugger
/// over the pipe.
///
extern "C"
void *llvm_debugger_stop(void *Dummy, unsigned Line, unsigned Col,
                         void *SourceDescriptor) {
  TheChild->reachedLine(Line, Col, SourceDescriptor);
  return Dummy;
}


/// llvm_dbg_region_start - This function is invoked every time an anonymous
/// region of the source program is entered.
///
extern "C"
void *llvm_dbg_region_start(void *Dummy) {
  TheChild->startRegion();
  return Dummy;
}

/// llvm_dbg_subprogram - This function is invoked every time a source-language
/// subprogram has been entered.
///
extern "C"
void *llvm_dbg_subprogram(void *FuncDesc) {
  TheChild->startSubprogram(FuncDesc);
  return 0;
}

/// llvm_dbg_region_end - This function is invoked every time a source-language
/// region (started with llvm.dbg.region.start or llvm.dbg.func.start) is
/// terminated.
///
extern "C"
void llvm_dbg_region_end(void *Dummy) {
  TheChild->endRegion();
}




namespace {
  /// DebuggerIntrinsicLowering - This class implements a simple intrinsic
  /// lowering class that revectors debugging intrinsics to call actual
  /// functions (defined above), instead of being turned into noops.
  struct DebuggerIntrinsicLowering : public DefaultIntrinsicLowering {
    virtual void LowerIntrinsicCall(CallInst *CI) {
      Module *M = CI->getParent()->getParent()->getParent();
      switch (CI->getCalledFunction()->getIntrinsicID()) {
      case Intrinsic::dbg_stoppoint:
        // Turn call into a call to llvm_debugger_stop
        CI->setOperand(0, M->getOrInsertFunction("llvm_debugger_stop",
                                  CI->getCalledFunction()->getFunctionType()));
        break;
      case Intrinsic::dbg_region_start:
        // Turn call into a call to llvm_dbg_region_start
        CI->setOperand(0, M->getOrInsertFunction("llvm_dbg_region_start",
                                  CI->getCalledFunction()->getFunctionType()));
        break;

      case Intrinsic::dbg_region_end:
        // Turn call into a call to llvm_dbg_region_end
        CI->setOperand(0, M->getOrInsertFunction("llvm_dbg_region_end",
                                  CI->getCalledFunction()->getFunctionType()));
        break;
      case Intrinsic::dbg_func_start:
        // Turn call into a call to llvm_dbg_subprogram
        CI->setOperand(0, M->getOrInsertFunction("llvm_dbg_subprogram",
                                  CI->getCalledFunction()->getFunctionType()));
        break;
      default:
        DefaultIntrinsicLowering::LowerIntrinsicCall(CI);
        break;
      }
    }
  };
} // end anonymous namespace


static void runChild(Module *M, const std::vector<std::string> &Arguments,
                     const char * const *envp,
                     FDHandle ReadFD, FDHandle WriteFD) {

  // Create an execution engine that uses our custom intrinsic lowering object
  // to revector debugging intrinsic functions into actual functions defined
  // above.
  ExecutionEngine *EE =
    ExecutionEngine::create(new ExistingModuleProvider(M), false,
                            new DebuggerIntrinsicLowering());
  assert(EE && "Couldn't create an ExecutionEngine, not even an interpreter?");
  
  // Call the main function from M as if its signature were:
  //   int main (int argc, char **argv, const char **envp)
  // using the contents of Args to determine argc & argv, and the contents of
  // EnvVars to determine envp.
  //
  Function *Fn = M->getMainFunction();
  if (!Fn) exit(1);

  // Create the child class instance which will be used by the debugger
  // callbacks to keep track of the current state of the process.
  assert(TheChild == 0 && "A child process has already been created??");
  TheChild = new Child(M, EE, ReadFD, WriteFD);

  // Run main...
  int Result = EE->runFunctionAsMain(Fn, Arguments, envp);

  // If the program didn't explicitly call exit, call exit now, for the program.
  // This ensures that any atexit handlers get called correctly.
  Function *Exit = M->getOrInsertFunction("exit", Type::VoidTy, Type::IntTy, 0);

  std::vector<GenericValue> Args;
  GenericValue ResultGV;
  ResultGV.IntVal = Result;
  Args.push_back(ResultGV);
  EE->runFunction(Exit, Args);
  abort();
}
