//===-- UserInput.cpp - Interpreter Input Loop support --------------------===//
// 
//  This file implements the interpreter Input I/O loop.
//
//===----------------------------------------------------------------------===//

#include "Interpreter.h"
#include "llvm/Bytecode/Reader.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Transforms/Utils/Linker.h"
#include <algorithm>
using std::string;
using std::cout;
using std::cin;

enum CommandID {
  Quit, Help,                                 // Basics
  Print, Info, List, StackTrace, Up, Down,    // Inspection
  Next, Step, Run, Finish, Call,              // Control flow changes
  Break, Watch,                               // Debugging
  Load, Flush,
  TraceOpt, ProfileOpt                              // Toggle features
};

// CommandTable - Build a lookup table for the commands available to the user...
static struct CommandTableElement {
  const char *Name;
  enum CommandID CID;

  inline bool operator<(const CommandTableElement &E) const {
    return string(Name) < string(E.Name);
  }
  inline bool operator==(const string &S) const { 
    return string(Name) == S;
  }
} CommandTable[] = {
  { "quit"     , Quit       }, { "q", Quit }, { "", Quit }, // Empty str = eof
  { "help"     , Help       }, { "h", Help },

  { "print"    , Print      }, { "p", Print },
  { "list"     , List       },
  { "info"     , Info       },
  { "backtrace", StackTrace }, { "bt", StackTrace }, { "where", StackTrace },
  { "up"       , Up         },
  { "down"     , Down       },

  { "next"     , Next       }, { "n", Next },
  { "step"     , Step       }, { "s", Step },
  { "run"      , Run        },
  { "finish"   , Finish     },
  { "call"     , Call       },

  { "break"    , Break      }, { "b", Break },
  { "watch"    , Watch      },

  { "load"     , Load       },
  { "flush"    , Flush      },

  { "trace"    , TraceOpt   },
  { "profile"  , ProfileOpt },
};
static CommandTableElement *CommandTableEnd = 
   CommandTable+sizeof(CommandTable)/sizeof(CommandTable[0]);


//===----------------------------------------------------------------------===//
// handleUserInput - Enter the input loop for the interpreter.  This function
// returns when the user quits the interpreter.
//
void Interpreter::handleUserInput() {
  bool UserQuit = false;

  // Sort the table...
  std::sort(CommandTable, CommandTableEnd);

  // Print the instruction that we are stopped at...
  printCurrentInstruction();

  do {
    string Command;
    cout << "lli> " << std::flush;
    cin >> Command;

    CommandTableElement *E = find(CommandTable, CommandTableEnd, Command);

    if (E == CommandTableEnd) {
      cout << "Error: '" << Command << "' not recognized!\n";
      continue;
    }

    switch (E->CID) {
    case Quit:       UserQuit = true;   break;
    case Load:
      cin >> Command;
      loadModule(Command);
      break;
    case Flush: flushModule(); break;
    case Print:
      cin >> Command;
      print(Command);
      break;
    case Info:
      cin >> Command;
      infoValue(Command);
      break;
     
    case List:       list();            break;
    case StackTrace: printStackTrace(); break;
    case Up: 
      if (CurFrame > 0) { --CurFrame; printStackFrame(); }
      else cout << "Error: Already at root of stack!\n";
      break;
    case Down:
      if ((unsigned)CurFrame < ECStack.size()-1) {
        ++CurFrame;
        printStackFrame();
      } else
        cout << "Error: Already at bottom of stack!\n";
      break;
    case Next:       nextInstruction(); break;
    case Step:       stepInstruction(); break;
    case Run:        run();             break;
    case Finish:     finish();          break;
    case Call:
      cin >> Command;
      callMethod(Command);    // Enter the specified function
      finish();               // Run until it's complete
      break;

    case TraceOpt:
      Trace = !Trace;
      cout << "Tracing " << (Trace ? "enabled\n" : "disabled\n");
      break;

    case ProfileOpt:
      Profile = !Profile;
      cout << "Profiling " << (Trace ? "enabled\n" : "disabled\n");
      break;

    default:
      cout << "Command '" << Command << "' unimplemented!\n";
      break;
    }

  } while (!UserQuit);
}

//===----------------------------------------------------------------------===//
// loadModule - Load a new module to execute...
//
void Interpreter::loadModule(const string &Filename) {
  string ErrorMsg;
  if (CurMod && !flushModule()) return;  // Kill current execution

  CurMod = ParseBytecodeFile(Filename, &ErrorMsg);
  if (CurMod == 0) {
    cout << "Error parsing '" << Filename << "': No module loaded: "
         << ErrorMsg << "\n";
    return;
  }
  CW.setModule(CurMod);  // Update Writer

#if 0
  string RuntimeLib = getCurrentExecutablePath();
  if (!RuntimeLib.empty()) RuntimeLib += "/";
  RuntimeLib += "RuntimeLib.bc";

  if (Module *SupportLib = ParseBytecodeFile(RuntimeLib, &ErrorMsg)) {
    if (LinkModules(CurMod, SupportLib, &ErrorMsg))
      std::cerr << "Error Linking runtime library into current module: "
                << ErrorMsg << "\n";
  } else {
    std::cerr << "Error loading runtime library '"+RuntimeLib+"': "
              << ErrorMsg << "\n";
  }
#endif
}


//===----------------------------------------------------------------------===//
// flushModule - Return true if the current program has been unloaded.
//
bool Interpreter::flushModule() {
  if (CurMod == 0) {
    cout << "Error flushing: No module loaded!\n";
    return false;
  }

  if (!ECStack.empty()) {
    // TODO: if use is not sure, return false
    cout << "Killing current execution!\n";
    ECStack.clear();
    CurFrame = -1;
  }

  CW.setModule(0);
  delete CurMod;
  CurMod = 0;
  ExitCode = 0;
  return true;
}

//===----------------------------------------------------------------------===//
// setBreakpoint - Enable a breakpoint at the specified location
//
void Interpreter::setBreakpoint(const string &Name) {
  Value *PickedVal = ChooseOneOption(Name, LookupMatchingNames(Name));
  // TODO: Set a breakpoint on PickedVal
}

//===----------------------------------------------------------------------===//
// callMethod - Enter the specified method...
//
bool Interpreter::callMethod(const string &Name) {
  std::vector<Value*> Options = LookupMatchingNames(Name);

  for (unsigned i = 0; i < Options.size(); ++i) { // Remove non-fn matches...
    if (!isa<Function>(Options[i])) {
      Options.erase(Options.begin()+i);
      --i;
    }
  }

  Value *PickedMeth = ChooseOneOption(Name, Options);
  if (PickedMeth == 0)
    return true;

  Function *F = cast<Function>(PickedMeth);

  std::vector<GenericValue> Args;
  // TODO, get args from user...

  callMethod(F, Args);  // Start executing it...

  // Reset the current frame location to the top of stack
  CurFrame = ECStack.size()-1;

  return false;
}

// callMainMethod - This is a nasty gross hack that will dissapear when
// callMethod can parse command line options and stuff for us.
//
bool Interpreter::callMainMethod(const string &Name,
                                 const std::vector<string> &InputArgv) {
  std::vector<Value*> Options = LookupMatchingNames(Name);

  for (unsigned i = 0; i < Options.size(); ++i) { // Remove non-fn matches...
    if (!isa<Function>(Options[i])) {
      Options.erase(Options.begin()+i);
      --i;
    }
  }

  Value *PickedMeth = ChooseOneOption(Name, Options);
  if (PickedMeth == 0)
    return true;

  Function *M = cast<Function>(PickedMeth);
  const FunctionType *MT = M->getFunctionType();

  std::vector<GenericValue> Args;
  switch (MT->getParamTypes().size()) {
  default:
    cout << "Unknown number of arguments to synthesize for '" << Name << "'!\n";
    return true;
  case 2: {
    PointerType *SPP = PointerType::get(PointerType::get(Type::SByteTy));
    if (MT->getParamTypes()[1] != SPP) {
      CW << "Second argument of '" << Name << "' should have type: '"
         << SPP << "'!\n";
      return true;
    }

    Args.push_back(CreateArgv(InputArgv));
  }
    // fallthrough
  case 1:
    if (!MT->getParamTypes()[0]->isInteger()) {
      cout << "First argument of '" << Name << "' should be an integer!\n";
      return true;
    } else {
      GenericValue GV; GV.UIntVal = InputArgv.size();
      Args.insert(Args.begin(), GV);
    }
    // fallthrough
  case 0:
    break;
  }

  callMethod(M, Args);  // Start executing it...

  // Reset the current frame location to the top of stack
  CurFrame = ECStack.size()-1;

  return false;
}



void Interpreter::list() {
  if (ECStack.empty())
    cout << "Error: No program executing!\n";
  else
    CW << ECStack[CurFrame].CurMethod;   // Just print the function out...
}

void Interpreter::printStackTrace() {
  if (ECStack.empty()) cout << "No program executing!\n";

  for (unsigned i = 0; i < ECStack.size(); ++i) {
    printStackFrame((int)i);
  }
}
