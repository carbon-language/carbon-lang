This is a simple example demonstrating how to use clang's facility for
providing AST consumers using a plugin.

Build the plugin by running `make` in this directory.

Once the plugin is built, you can run it using:
--
Linux:
$ clang -cc1 -load ../../Debug+Asserts/lib/libPrintFunctionNames.so -plugin print-fns some-input-file.c

Mac:
$ clang -cc1 -load ../../Debug+Asserts/lib/libPrintFunctionNames.dylib -plugin print-fns some-input-file.c
