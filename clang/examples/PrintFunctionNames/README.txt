This is a simple example demonstrating how to use clang's facility for
providing AST consumers using a plugin.

You will probably need to build clang so that it exports all symbols (disable
TOOL_NO_EXPORT in the tools/clang Makefile).

Once the plugin is built, you can run it using:
--
$ clang -cc1 -load path/to/PrintFunctionNames.so -plugin=print-fns some-input-file.c
--
