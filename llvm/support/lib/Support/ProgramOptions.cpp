// $Id$
//***************************************************************************
//
// File:
//    ProgramOptions.C
//
// Purpose:
//    General options processing for any program.
//
// History:
//    08/08/95 - adve  - created in the dHPF compiler
//    10/10/96 - mpal, dbaker - converted to const member functions.
//    11/26/96 - adve  - fixed to handle options that consume 0+ arguments
//    07/15/01 - vadve - Copied to LLVM system and modified
//
//**************************************************************************/

//************************** System Include Files **************************/

#include <iostream.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <string>
#ifndef MAXINT
#define MAXINT	((1 << sizeof(int)-1) - 1)
#endif

//*************************** User Include Files ***************************/

#include "llvm/Support/ProgramOptions.h"
#include "llvm/Support/ProgramOption.h"


//************************** Method Definitions ****************************/

ProgramOptions::ProgramOptions(int _argc,
			       const char* _argv[],
			       const char* _envp[])
  : optionRegistry(),
    argc(_argc),
    argv(_argv),
    envp(_envp),
    argsConsumed(0)
{}

const char*
ProgramOptions::StringOptionValue(const char* optString) const
{
    const StringOption* handler = (const StringOption*) OptionHandler(optString);
    return (handler == NULL) ? NULL : handler->Value();
}

bool
ProgramOptions::FlagOptionValue(const char* optString) const
{
    const FlagOption* handler = (const FlagOption*) OptionHandler(optString);
    return (handler == NULL) ? false : handler->Value();
}

double
ProgramOptions::RealOptionValue(const char* optString) const
{
    const RealValuedOption* handler =
	(const RealValuedOption*) OptionHandler(optString);
    return (handler == NULL) ? MAXFLOAT : handler->Value();
}

int
ProgramOptions::IntOptionValue(const char* optString) const
{
    const IntegerValuedOption* handler =
	(const IntegerValuedOption*) OptionHandler(optString);
    return (handler == NULL) ? MAXINT : handler->Value();
}

bool
ProgramOptions::OptionSpecified(const char* optString) const
{
    const ProgramOption* handler = OptionHandler(optString);
    return handler->OptionSpecified();
}

const char* 
ProgramOptions::ProgramName() const
{
    return argv[0];
}

int
ProgramOptions::NumberOfOtherOptions() const
{
   return argc - argsConsumed;
} 

const char*
ProgramOptions::OtherOption(int i) const
{
   i += argsConsumed;
   assert(i >= 0 && i < argc);
   return argv[i];
}

const char**
ProgramOptions::GetOriginalArgs() const
{
   return argv;
}

vector<string>
ProgramOptions::GetDescription() const
{  
  vector<string> optDesc;
  
  if (optDesc.size() < (unsigned) argc)
    {
      for (hash_map<string,ProgramOption*>::const_iterator iter=optionRegistry.begin();
	   ! (iter == optionRegistry.end());
	   ++iter)
	{
	  const ProgramOption* handler = iter->second;
	  optDesc.push_back(handler->ArgString());	// 1st
	  optDesc.push_back(handler->HelpMesg());	// 2nd
	  optDesc.push_back(handler->GetTextValue());		// 3rd
	}
    }
  
  return optDesc;
}

void
ProgramOptions::Register(ProgramOption* option)
{
  optionRegistry[option->ArgString()] = option;
}

//----------------------------------------------------------------------
// function ProgramOptions::ParseArgs
// 
// Parse command-line options until you run out of options or see
// an unrecognized option.  `getopt' is a standard package to do this,
// but incredibly, it limited to single-letter options.
//
// -- Each option can consume zero or one additional arguments of argv
// -- "--" can be used to mark the end of options
//	  so that a program argument can also start with "-".
//---------------------------------------------------------------------/

void
ProgramOptions::ParseArgs(int argc,
			  const char* argv[],
			  const char* envp[])
{
  if (argc == 0) {
    Usage();
  }
  // consume the program name
  argsConsumed = 1;
  
  while (argsConsumed < argc)
    {
      const char* arg = argv[argsConsumed];
      if (arg[0] == '-')
	{
	  if (strcmp(arg, "--") == 0) {	// "--" marks end of options
	    argsConsumed++;		// consume and
	    break;			// discontinue the for loop
	  }
	  ProgramOption* handler = OptionHandler(arg+1);
	  if (handler == NULL) {
	    cerr << "Unrecognized option: " << arg+1 << endl;
	    Usage();
	  }

	  if (argc - argsConsumed < handler->MinExpectedArgs()) {
	    cerr << "Option " << (char*) arg+1 << " needs "
		 << handler->MinExpectedArgs() << " arguments" << endl;
	    Usage();
	  }

	  argsConsumed++;  // consumed the option

	  const char* nextArg = (argsConsumed < argc)? argv[argsConsumed]
						     : "";
	  int numAdditionalArgsConsumed = handler->EvalOpt(nextArg);
	  if (numAdditionalArgsConsumed < 0)
	    Usage();
	  argsConsumed += numAdditionalArgsConsumed;
	}
      else
	{
	  break; // quit the while loop 
	}
    }
  
  ParseExtraArgs(); 
}

void
ProgramOptions::PrintArgs(ostream& stream) const
{
  for (int i = 0; i < argc; i++) {
    stream << argv[i] << " ";
  }
  stream << endl; 
}


void
ProgramOptions::PrintOptions(ostream& stream) const
{
  stream << "OPTIONS:" << endl;
  stream << "\tUse argument 0 to turn OFF a flag option: "
	 << "-<flag_opt> 0" << endl << endl;
    
  for (hash_map<string,ProgramOption*>::const_iterator iter = optionRegistry.begin();
       iter != optionRegistry.end(); ++iter) {
      const ProgramOption* handler = (*iter).second;
      
      stream << "\t-" << handler->ArgString();
      
      const char* const showarg = " <arg>";
      int i = 1;
      for (i=1; i <= handler->MinExpectedArgs(); i++)
	stream << showarg; 
      
      int numCharsPrinted = 1 + strlen(handler->ArgString())
	+ strlen(showarg) * handler->MinExpectedArgs();
      for (i=1; i > numCharsPrinted / 8; i--)
	stream << "\t";
      
      stream << "\t" << handler->HelpMesg()
	     << endl;
    }
}

void
ProgramOptions::Usage() const
{
  PrintUsage(cerr); 
  exit(1); 
}


//**************************************************************************/
