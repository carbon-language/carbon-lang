//===-- UserInput.cpp - Interpreter Input Loop support --------------------===//
// 
//  This file implements the interpreter Input I/O loop.
//
//===----------------------------------------------------------------------===//

#include "Interpreter.h"
#include "llvm/Bytecode/Reader.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Transforms/Utils/Linker.h"
#include <algorithm>

enum CommandID {
  Quit, Help,                                 // Basics
  Print, Info, List, StackTrace, Up, Down,    // Inspection
  Next, Step, Run, Finish, Call,              // Control flow changes
  Break, Watch,                               // Debugging
  Flush,
  TraceOpt,                                   // Toggle features
};

// CommandTable - Build a lookup table for the commands available to the user...
static struct CommandTableElement {
  const char *Name;
  enum CommandID CID;

  inline bool operator<(const CommandTableElement &E) const {
    return std::string(Name) < std::string(E.Name);
  }
  inline bool operator==(const std::string &S) const { 
    return std::string(Name) == S;
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

  { "flush"    , Flush      },

  { "trace"    , TraceOpt   },
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
    std::string Command;
    std::cout << "lli> " << std::flush;
    std::cin >> Command;

    CommandTableElement *E = find(CommandTable, CommandTableEnd, Command);

    if (E == CommandTableEnd) {
      std::cout << "Error: '" << Command << "' not recognized!\n";
      continue;
    }

    switch (E->CID) {
    case Quit:       UserQuit = true;   break;
    case Print:
      std::cin >> Command;
      print(Command);
      break;
    case Info:
      std::cin >> Command;
      infoValue(Command);
      break;
     
    case List:       list();            break;
    case StackTrace: printStackTrace(); break;
    case Up: 
      if (CurFrame > 0) { --CurFrame; printStackFrame(); }
      else std::cout << "Error: Already at root of stack!\n";
      break;
    case Down:
      if ((unsigned)CurFrame < ECStack.size()-1) {
        ++CurFrame;
        printStackFrame();
      } else
        std::cout << "Error: Already at bottom of stack!\n";
      break;
    case Next:       nextInstruction(); break;
    case Step:       stepInstruction(); break;
    case Run:        run();             break;
    case Finish:     finish();          break;
    case Call:
      std::cin >> Command;
      callFunction(Command);    // Enter the specified function
      finish();               // Run until it's complete
      break;

    case TraceOpt:
      Trace = !Trace;
      std::cout << "Tracing " << (Trace ? "enabled\n" : "disabled\n");
      break;

    default:
      std::cout << "Command '" << Command << "' unimplemented!\n";
      break;
    }

  } while (!UserQuit);
}

//===----------------------------------------------------------------------===//
// setBreakpoint - Enable a breakpoint at the specified location
//
void Interpreter::setBreakpoint(const std::string &Name) {
  Value *PickedVal = ChooseOneOption(Name, LookupMatchingNames(Name));
  // TODO: Set a breakpoint on PickedVal
}

//===----------------------------------------------------------------------===//
// callFunction - Enter the specified function...
//
bool Interpreter::callFunction(const std::string &Name) {
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

  callFunction(F, Args);  // Start executing it...

  // Reset the current frame location to the top of stack
  CurFrame = ECStack.size()-1;

  return false;
}

// callMainFunction - This is a nasty gross hack that will dissapear when
// callFunction can parse command line options and stuff for us.
//
bool Interpreter::callMainFunction(const std::string &Name,
                                   const std::vector<std::string> &InputArgv) {
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
    std::cout << "Unknown number of arguments to synthesize for '" << Name
              << "'!\n";
    return true;
  case 2: {
    PointerType *SPP = PointerType::get(PointerType::get(Type::SByteTy));
    if (MT->getParamTypes()[1] != SPP) {
      CW << "Second argument of '" << Name << "' should have type: '"
         << SPP << "'!\n";
      return true;
    }

    Args.push_back(PTOGV(CreateArgv(InputArgv)));
  }
    // fallthrough
  case 1:
    if (!MT->getParamTypes()[0]->isInteger()) {
      std::cout << "First argument of '" << Name << "' should be an integer!\n";
      return true;
    } else {
      GenericValue GV; GV.UIntVal = InputArgv.size();
      Args.insert(Args.begin(), GV);
    }
    // fallthrough
  case 0:
    break;
  }

  callFunction(M, Args);  // Start executing it...

  // Reset the current frame location to the top of stack
  CurFrame = ECStack.size()-1;

  return false;
}



void Interpreter::list() {
  if (ECStack.empty())
    std::cout << "Error: No program executing!\n";
  else
    CW << ECStack[CurFrame].CurFunction;   // Just print the function out...
}

void Interpreter::printStackTrace() {
  if (ECStack.empty()) std::cout << "No program executing!\n";

  for (unsigned i = 0; i < ECStack.size(); ++i) {
    printStackFrame((int)i);
  }
}
