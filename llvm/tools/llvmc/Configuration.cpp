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

extern int ::Configlineno;

namespace llvm {
  ConfigLexerInfo ConfigLexerData;
  InputProvider* ConfigLexerInput = 0;

  InputProvider::~InputProvider() {}
  void InputProvider::error(const std::string& msg) {
    std::cerr << name << ":" << Configlineno << ": Error: " << msg << "\n";
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

  struct Parser
  {
    ConfigLexerTokens token;
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

    void parseOptionList(CompilerDriver::StringVector& optList ) {
      while (next_is_real()) {
        if (token == STRING || token == OPTION)
          optList.push_back(ConfigLexerData.StringVal);
        else {
          error("Expecting a program option", false);
          break;
        }
      }
    }

    void parseLang() {
      switch (next() ) {
        case NAME: 
          confDat->langName = parseName(); 
          break;
        case OPT1: 
          parseOptionList(confDat->opts[CompilerDriver::OPT_FAST_COMPILE]); 
          break;
        case OPT2: 
          parseOptionList(confDat->opts[CompilerDriver::OPT_SIMPLE]); 
          break;
        case OPT3: 
          parseOptionList(confDat->opts[CompilerDriver::OPT_AGGRESSIVE]); 
          break;
        case OPT4: 
          parseOptionList(confDat->opts[CompilerDriver::OPT_LINK_TIME]); 
          break;
        case OPT5: 
          parseOptionList(
            confDat->opts[CompilerDriver::OPT_AGGRESSIVE_LINK_TIME]);
          break;
        default:   
          error("Expecting 'name' or 'optN' after 'lang.'"); 
          break;
      }
    }

    void parseCommand(CompilerDriver::Action& action) {
      if (next() == EQUALS) {
        if (next() == EOLTOK) {
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
            if (token == STRING || token == OPTION) {
              action.args.push_back(ConfigLexerData.StringVal);
            } else if (token == IN_SUBST) {
              action.inputAt = action.args.size();
              action.args.push_back("@in@");
            } else if (token == OUT_SUBST) {
              action.outputAt = action.args.size();
              action.args.push_back("@out@");
            } else {
              error("Expecting a program argument", false);
              break;
            }
          }
        }
      }
    }

    void parsePreprocessor() {
      switch (next()) {
        case COMMAND:
          parseCommand(confDat->PreProcessor);
          break;
        case REQUIRED:
          if (parseBoolean())
            confDat->PreProcessor.set(CompilerDriver::REQUIRED_FLAG);
          else
            confDat->PreProcessor.clear(CompilerDriver::REQUIRED_FLAG);
          break;
        default:
          error("Expecting 'command' or 'required'");
          break;
      }
    }

    void parseTranslator() {
      switch (next()) {
        case COMMAND: 
          parseCommand(confDat->Translator);
          break;
        case REQUIRED:
          if (parseBoolean())
            confDat->Translator.set(CompilerDriver::REQUIRED_FLAG);
          else
            confDat->Translator.clear(CompilerDriver::REQUIRED_FLAG);
          break;
        case PREPROCESSES:
          if (parseBoolean())
            confDat->Translator.set(CompilerDriver::PREPROCESSES_FLAG);
          else 
            confDat->Translator.clear(CompilerDriver::PREPROCESSES_FLAG);
          break;
        case OPTIMIZES:
          if (parseBoolean())
            confDat->Translator.set(CompilerDriver::OPTIMIZES_FLAG);
          else
            confDat->Translator.clear(CompilerDriver::OPTIMIZES_FLAG);
          break;
        case GROKS_DASH_O:
          if (parseBoolean())
            confDat->Translator.set(CompilerDriver::GROKS_DASH_O_FLAG);
          else
            confDat->Translator.clear(CompilerDriver::GROKS_DASH_O_FLAG);
          break;
        case GROKS_O10N:
          if (parseBoolean())
            confDat->Translator.set(CompilerDriver::GROKS_O10N_FLAG);
          else
            confDat->Translator.clear(CompilerDriver::GROKS_O10N_FLAG);
          break;
        default:
          error("Expecting 'command', 'required', 'preprocesses', "
                "'groks_dash_O' or 'optimizes'");
          break;
      }
    }

    void parseOptimizer() {
      switch (next()) {
        case COMMAND:
          parseCommand(confDat->Optimizer);
          break;
        case GROKS_DASH_O:
          if (parseBoolean())
            confDat->Optimizer.set(CompilerDriver::GROKS_DASH_O_FLAG);
          else
            confDat->Optimizer.clear(CompilerDriver::GROKS_DASH_O_FLAG);
          break;
        case GROKS_O10N:
          if (parseBoolean())
            confDat->Optimizer.set(CompilerDriver::GROKS_O10N_FLAG);
          else
            confDat->Optimizer.clear(CompilerDriver::GROKS_O10N_FLAG);
          break;
        default:
          error("Expecting 'command' or 'groks_dash_O'");
          break;
      }
    }

    void parseAssembler() {
      switch(next()) {
        case COMMAND:
          parseCommand(confDat->Assembler);
          break;
        default:
          error("Expecting 'command'");
          break;
      }
    }

    void parseLinker() {
      switch(next()) {
        case COMMAND:
          parseCommand(confDat->Linker);
          break;
        case GROKS_DASH_O:
          if (parseBoolean())
            confDat->Linker.set(CompilerDriver::GROKS_DASH_O_FLAG);
          else
            confDat->Linker.clear(CompilerDriver::GROKS_DASH_O_FLAG);
          break;
        case GROKS_O10N:
          if (parseBoolean())
            confDat->Linker.set(CompilerDriver::GROKS_O10N_FLAG);
          else
            confDat->Linker.clear(CompilerDriver::GROKS_O10N_FLAG);
          break;
        default:
          error("Expecting 'command'");
          break;
      }
    }

    void parseAssignment() {
      switch (token) {
        case LANG:          parseLang(); break;
        case PREPROCESSOR:  parsePreprocessor(); break;
        case TRANSLATOR:    parseTranslator(); break;
        case OPTIMIZER:     parseOptimizer(); break;
        case ASSEMBLER:     parseAssembler(); break;
        case LINKER:        parseLinker(); break;
        case EOLTOK:        break; // just ignore
        case ERRORTOK:
        default:          
          error("Invalid top level configuration item");
          break;
      }
    }

    void parseFile() {
      while ( next() != EOFTOK ) {
        if (token == ERRORTOK)
          error("Invalid token");
        else if (token != EOLTOK)
          parseAssignment();
      }
      provider->checkErrors();
    }
  };

  void
  ParseConfigData(InputProvider& provider, CompilerDriver::ConfigData& confDat) {
    Parser p;
    p.token = EOFTOK;
    p.provider = &provider;
    p.confDat = &confDat;
    p.action = 0;
    p.parseFile();
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
