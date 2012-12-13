=============
Clang Plugins
=============

Clang Plugins make it possible to run extra user defined actions during
a compilation. This document will provide a basic walkthrough of how to
write and run a Clang Plugin.

Introduction
============

Clang Plugins run FrontendActions over code. See the :doc:`FrontendAction
tutorial <RAVFrontendAction>` on how to write a FrontendAction
using the RecursiveASTVisitor. In this tutorial, we'll demonstrate how
to write a simple clang plugin.

Writing a PluginASTAction
=========================

The main difference from writing normal FrontendActions is that you can
handle plugin command line options. The PluginASTAction base class
declares a ParseArgs method which you have to implement in your plugin.

::

      bool ParseArgs(const CompilerInstance &CI,
                     const std::vector<std::string>& args) {
        for (unsigned i = 0, e = args.size(); i != e; ++i) {
          if (args[i] == "-some-arg") {
            // Handle the command line argument.
          }
        }
        return true;
      }

Registering a plugin
====================

A plugin is loaded from a dynamic library at runtime by the compiler. To
register a plugin in a library, use FrontendPluginRegistry::Add:

::

      static FrontendPluginRegistry::Add<MyPlugin> X("my-plugin-name", "my plugin description");

Putting it all together
=======================

Let's look at an example plugin that prints top-level function names.
This example is also checked into the clang repository; please also take
a look at the latest `checked in version of
PrintFunctionNames.cpp <http://llvm.org/viewvc/llvm-project/cfe/trunk/examples/PrintFunctionNames/PrintFunctionNames.cpp?view=markup>`_.

::

    #include "clang/Frontend/FrontendPluginRegistry.h"
    #include "clang/AST/ASTConsumer.h"
    #include "clang/AST/AST.h"
    #include "clang/Frontend/CompilerInstance.h"
    #include "llvm/Support/raw_ostream.h"
    using namespace clang;

    namespace {

    class PrintFunctionsConsumer : public ASTConsumer {
    public:
      virtual bool HandleTopLevelDecl(DeclGroupRef DG) {
        for (DeclGroupRef::iterator i = DG.begin(), e = DG.end(); i != e; ++i) {
          const Decl *D = *i;
          if (const NamedDecl *ND = dyn_cast<NamedDecl>(D))
            llvm::errs() << "top-level-decl: \"" << ND->getNameAsString() << "\"\n";
        }

        return true;
      }
    };

    class PrintFunctionNamesAction : public PluginASTAction {
    protected:
      ASTConsumer *CreateASTConsumer(CompilerInstance &CI, llvm::StringRef) {
        return new PrintFunctionsConsumer();
      }

      bool ParseArgs(const CompilerInstance &CI,
                     const std::vector<std::string>& args) {
        for (unsigned i = 0, e = args.size(); i != e; ++i) {
          llvm::errs() << "PrintFunctionNames arg = " << args[i] << "\n";

          // Example error handling.
          if (args[i] == "-an-error") {
            DiagnosticsEngine &D = CI.getDiagnostics();
            unsigned DiagID = D.getCustomDiagID(
              DiagnosticsEngine::Error, "invalid argument '" + args[i] + "'");
            D.Report(DiagID);
            return false;
          }
        }
        if (args.size() && args[0] == "help")
          PrintHelp(llvm::errs());

        return true;
      }
      void PrintHelp(llvm::raw_ostream& ros) {
        ros << "Help for PrintFunctionNames plugin goes here\n";
      }

    };

    }

    static FrontendPluginRegistry::Add<PrintFunctionNamesAction>
    X("print-fns", "print function names");

Running the plugin
==================

To run a plugin, the dynamic library containing the plugin registry must
be loaded via the -load command line option. This will load all plugins
that are registered, and you can select the plugins to run by specifying
the -plugin option. Additional parameters for the plugins can be passed
with -plugin-arg-<plugin-name>.

Note that those options must reach clang's cc1 process. There are two
ways to do so:

-  Directly call the parsing process by using the -cc1 option; this has
   the downside of not configuring the default header search paths, so
   you'll need to specify the full system path configuration on the
   command line.
-  Use clang as usual, but prefix all arguments to the cc1 process with
   -Xclang.

For example, to run the print-function-names plugin over a source file
in clang, first build the plugin, and then call clang with the plugin
from the source tree:

::

      $ export BD=/path/to/build/directory
      $ (cd $BD && make PrintFunctionNames )
      $ clang++ -D_GNU_SOURCE -D_DEBUG -D__STDC_CONSTANT_MACROS \
            -D__STDC_FORMAT_MACROS -D__STDC_LIMIT_MACROS -D_GNU_SOURCE \
            -I$BD/tools/clang/include -Itools/clang/include -I$BD/include -Iinclude \
            tools/clang/tools/clang-check/ClangCheck.cpp -fsyntax-only \
            -Xclang -load -Xclang $BD/lib/PrintFunctionNames.so -Xclang \
            -plugin -Xclang print-fns

Also see the print-function-name plugin example's
`README <http://llvm.org/viewvc/llvm-project/cfe/trunk/examples/PrintFunctionNames/README.txt?view=markup>`_
