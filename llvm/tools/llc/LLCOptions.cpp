// $Id$
//**************************************************************************/
// File:
//	LLCOptions.cpp
// 
// Purpose:
//	Options for the llc compiler.
// 
// History:
//	7/15/01	 -  Vikram Adve  -  Created
// 
//**************************************************************************/

//************************** System Include Files **************************/

#include <iostream.h>
#include <unistd.h>


//*************************** User Include Files ***************************/

#include "llvm/Support/ProgramOptions.h"
#include "llvm/Support/ProgramOption.h"
#include "llvm/LLC/LLCOptions.h"


//---------------------------------------------------------------------------
// class LLCOptions
//---------------------------------------------------------------------------

/*ctor*/
LLCOptions::LLCOptions (int _argc,
			const char* _argv[],
			const char* _envp[]) 
  : ProgramOptions(_argc, _argv, _envp)
{
  InitializeOptions();
  ParseArgs(argc, argv, envp);
  CheckParse();
}

/*dtor*/
LLCOptions::~LLCOptions()
{}

//--------------------------------------------------------------------
// Initialize all our compiler options
//--------------------------------------------------------------------

void
LLCOptions::InitializeOptions()
{
  Register(new FlagOption(HELP_OPT,
			  "print usage message",
			  false /*initValue*/));
  
  Register(new FlagOption(DEBUG_OPT,
			  "turn on default debugging options",
			  false /*initValue*/));
  
  Register(new FlagOption(DEBUG_OPT,
			  "turn off all diagnostic messages",
			  false /*initValue*/));
  
  Register(new StringOption(OUTFILENAME_OPT,
			    "output file name",
			    "" /*initValue*/));
  
  Register(new IntegerValuedOption(DEBUG_INSTR_SELECT_OPT,
       "control amount of debugging information for instruction selection",
				   0 /*initValue*/));
}


void
LLCOptions::ParseExtraArgs()
{
  if (argsConsumed != argc-1)
    Usage();
  
  // input file name should be the last argument
  inputFileName = argv[argc-1];
  
  // output file name may be specified with -o option;
  // otherwise create it from the input file name by replace ".ll" with ".o"
  const string &outfilenameOpt = StringOptionValue(OUTFILENAME_OPT);
  if (outfilenameOpt.length())
    {// "-o" option was used
      outputFileName = outfilenameOpt; 
    }
  else
    {
      outputFileName = inputFileName;
      unsigned int suffixPos = outputFileName.rfind(".bc");
      
      if (suffixPos >= outputFileName.length())
	suffixPos = outputFileName.rfind(".ll");
      
      if (suffixPos >= outputFileName.length())
	{
	  cerr << "Unrecognized suffix in file name " << inputFileName << endl;
	  Usage();
	}
      
      outputFileName.replace(suffixPos, 3, ".o"); 
    }
}

//--------------------------------------------------------------------
// Functions that must be overridden in subclass of ProgramOptions
//--------------------------------------------------------------------

void
LLCOptions::CheckParse()
{}

void
LLCOptions::PrintUsage(ostream& stream) const
{
  stream << "\nUSAGE:\n\t" << ProgramName() << " [options] " 
	 << "llvm-file" << endl << endl;
  PrintOptions(stream); 
}


