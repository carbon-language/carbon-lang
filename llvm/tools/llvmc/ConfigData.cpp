//===- ConfigData.cpp - Configuration Data Mgmt -----------------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements the parsing of configuration files for the LLVM Compiler
// Driver (llvmc).
//
//===------------------------------------------------------------------------===

#include "ConfigData.h"
#include "ConfigLexer.h"
#include "CompilerDriver.h"
#include "Support/StringExtras.h"
#include <iostream>
#include <fstream>

using namespace llvm;

extern int ::Configlex();

namespace llvm {
  ConfigLexerInfo ConfigLexerData;
  InputProvider* ConfigLexerInput = 0;
  unsigned ConfigLexerLine = 1;

  InputProvider::~InputProvider() {}
  void InputProvider::error(const std::string& msg) {
    std::cerr << name << ":" << ConfigLexerLine << ": Error: " << msg << "\n";
    errCount++;
  }

  void InputProvider::checkErrors() {
    if (errCount > 0) {
      std::cerr << name << " had " << errCount << " errors. Terminating.\n";
      exit(errCount);
    }
  }

}

namespace {

  class FileInputProvider : public InputProvider {
    public:
      FileInputProvider(const std::string & fname)
        : InputProvider(fname) 
        , F(fname.c_str()) {
        ConfigLexerInput = this;
      }
      virtual ~FileInputProvider() { F.close(); ConfigLexerInput = 0; }
      virtual unsigned read(char *buffer, unsigned max_size) {
        if (F.good()) {
          F.read(buffer,max_size);
          if ( F.gcount() ) return F.gcount() - 1;
        }
        return 0;
      }

      bool okay() { return F.good(); }
    private:
      std::ifstream F;
  };

  struct ParseContext
  {
    int token;
    InputProvider* provider;
    CompilerDriver::ConfigData* confDat;
    CompilerDriver::Action* action;

    int next() { return token = Configlex(); }

    bool next_is_real() { 
      token = Configlex(); 
      return (token != EOLTOK) && (token != ERRORTOK) && (token != 0);
    }

    void eatLineRemnant() {
      while (next_is_real()) ;
    }

    void error(const std::string& msg, bool skip = true) {
      provider->error(msg);
      if (skip)
        eatLineRemnant();
    }

    std::string parseName() {
      std::string result;
      if (next() == EQUALS) {
        while (next_is_real()) {
          switch (token ) {
            case STRING :
            case OPTION : 
              result += ConfigLexerData.StringVal + " ";
              break;
            default:
              error("Invalid name");
              break;
          }
        }
        if (result.empty())
          error("Name exepected");
        else
          result.erase(result.size()-1,1);
      } else
        error("= expected");
      return result;
    }

    bool parseBoolean() {
      bool result = true;
      if (next() == EQUALS) {
        if (next() == FALSETOK) {
          result = false;
        } else if (token != TRUETOK) {
          error("Expecting boolean value");
          return false;
        }
        if (next() != EOLTOK && token != 0) {
          error("Extraneous tokens after boolean");
        }
      }
      else
        error("Expecting '='");
      return result;
    }

    void parseLang() {
      if ( next() == NAME ) {
        confDat->langName = parseName();
      } else if (token == TRANSLATOR) {
        switch (next()) {
          case PREPROCESSES:
            confDat->TranslatorPreprocesses = parseBoolean();
            break;
          case OPTIMIZES:
            confDat->TranslatorOptimizes = parseBoolean();
            break;
          case GROKS_DASH_O:
            confDat->TranslatorGroksDashO = parseBoolean();
            break;
          default:
            error("Invalid lang.translator identifier");
            break;
        }
      }
      else if (token == PREPROCESSOR) {
        if (next() == NEEDED)
          confDat->PreprocessorNeeded = parseBoolean();
      }
      else {
        error("Expecting valid identifier after 'lang.'");
      }
    }

    void parseCommand(CompilerDriver::Action& action) {
      if (next() == EQUALS) {
        next();
        if (token == EOLTOK) {
          // no value (valid)
          action.program.clear();
          action.args.clear();
          action.inputAt = 0;
          action.outputAt = 0;
        } else {
          if (token == STRING || token == OPTION) {
            action.program = ConfigLexerData.StringVal;
          } else {
            error("Expecting a program name");
          }
          while (next_is_real()) {
            if (token == STRING || token == OPTION)
              action.args.push_back(ConfigLexerData.StringVal);
            else if (token == IN_SUBST) {
              action.inputAt = action.args.size();
              action.args.push_back("in");
            } else if (token == OUT_SUBST) {
              action.outputAt = action.args.size();
              action.args.push_back("out");
            } else
              error("Expecting a program argument", false);
          }
        }
      }
    }

    void parsePreProcessor() {
      if (next() != COMMAND) {
        error("Expecting 'command'");
        return;
      }
      parseCommand(confDat->PreProcessor);
    }

    void parseTranslator() {
      if (next() != COMMAND) {
        error("Expecting 'command'");
        return;
      }
      parseCommand(confDat->Translator);
    }

    void parseOptimizer() {
      if (next() != COMMAND) {
        error("Expecting 'command'");
        return;
      }
      parseCommand(confDat->Optimizer);
    }

    void parseAssembler() {
      if (next() != COMMAND) {
        error("Expecting 'command'");
        return;
      }
      parseCommand(confDat->Assembler);
    }

    void parseLinker() {
      if (next() != COMMAND) {
        error("Expecting 'command'");
        return;
      }
      parseCommand(confDat->Linker);
    }

    void parseAssignment() {
      switch (token) {
        case LANG:          return parseLang();
        case PREPROCESSOR:  return parsePreProcessor();
        case TRANSLATOR:    return parseTranslator();
        case OPTIMIZER:     return parseOptimizer();
        case ASSEMBLER:     return parseAssembler();
        case LINKER:        return parseLinker();
        case EOLTOK:        break; // just ignore
        case ERRORTOK:
        default:          
          error("Invalid top level configuration item identifier");
      }
    }

    void parseFile() {
      while ( next() != 0 ) {
        parseAssignment();
      }
      provider->checkErrors();
    }
  };

  void
  ParseConfigData(InputProvider& provider, CompilerDriver::ConfigData& confDat) {
    ParseContext ctxt;
    ctxt.token = 0;
    ctxt.provider = &provider;
    ctxt.confDat = &confDat;
    ctxt.action = 0;
    ctxt.parseFile();
  }
}

CompilerDriver::ConfigData*
LLVMC_ConfigDataProvider::ReadConfigData(const std::string& ftype) {
  CompilerDriver::ConfigData* result = 0;
  if (configDir.empty()) {
    FileInputProvider fip( std::string("/etc/llvm/") + ftype );
    if (!fip.okay()) {
      fip.error("Configuration for '" + ftype + "' is not available.");
      fip.checkErrors();
    }
    else {
      result = new CompilerDriver::ConfigData();
      ParseConfigData(fip,*result);
    }
  } else {
    FileInputProvider fip( configDir + "/" + ftype );
    if (!fip.okay()) {
      fip.error("Configuration for '" + ftype + "' is not available.");
      fip.checkErrors();
    }
    else {
      result = new CompilerDriver::ConfigData();
      ParseConfigData(fip,*result);
    }
  }
  return result;
}

LLVMC_ConfigDataProvider::LLVMC_ConfigDataProvider() 
  : Configurations() 
  , configDir() 
{
  Configurations.clear();
}

LLVMC_ConfigDataProvider::~LLVMC_ConfigDataProvider()
{
  ConfigDataMap::iterator cIt = Configurations.begin();
  while (cIt != Configurations.end()) {
    CompilerDriver::ConfigData* cd = cIt->second;
    ++cIt;
    delete cd;
  }
  Configurations.clear();
}

CompilerDriver::ConfigData* 
LLVMC_ConfigDataProvider::ProvideConfigData(const std::string& filetype) {
  CompilerDriver::ConfigData* result = 0;
  if (!Configurations.empty()) {
    ConfigDataMap::iterator cIt = Configurations.find(filetype);
    if ( cIt != Configurations.end() ) {
      // We found one in the case, return it.
      result = cIt->second;
    }
  }
  if (result == 0) {
    // The configuration data doesn't exist, we have to go read it.
    result = ReadConfigData(filetype);
    // If we got one, cache it
    if ( result != 0 )
      Configurations.insert(std::make_pair(filetype,result));
  }
  return result; // Might return 0
}

// vim: sw=2 smartindent smarttab tw=80 autoindent expandtab
