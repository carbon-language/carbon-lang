===========
ClangFormat
===========

`ClangFormat` describes a set of tools that are built on top of
:doc:`LibFormat`. It can support your workflow in a variety of ways including a
standalone tool and editor integrations.


Standalone Tool
===============

:program:`clang-format` is located in `clang/tools/clang-format` and can be used
to format C/C++/Obj-C code.

.. code-block:: console

  $ clang-format --help
  OVERVIEW: A tool to format C/C++/Obj-C code.

  Currently supports LLVM and Google style guides.
  If no arguments are specified, it formats the code from standard input
  and writes the result to the standard output.
  If <file> is given, it reformats the file. If -i is specified together
  with <file>, the file is edited in-place. Otherwise, the result is
  written to the standard output.

  USAGE: clang-format [options] [<file>]

  OPTIONS:
    -fatal-assembler-warnings - Consider warnings as error
    -help                     - Display available options (-help-hidden for more)
    -i                        - Inplace edit <file>, if specified.
    -length=<int>             - Format a range of this length, -1 for end of file.
    -offset=<int>             - Format a range starting at this file offset.
    -stats                    - Enable statistics output from program
    -style=<string>           - Coding style, currently supports: LLVM, Google, Chromium.
    -version                  - Display the version of this program


Vim Integration
===============

There is an integration for :program:`vim` which lets you run the
:program:`clang-format` standalone tool on your current buffer, optionally
selecting regions to reformat. The integration has the form of a `python`-file
which can be found under `clang/tools/clang-format/clang-format.py`.

This can be integrated by adding the following to your `.vimrc`:

.. code-block:: vim

  map <C-K> :pyf <path-to-this-file>/clang-format.py<CR>
  imap <C-K> <ESC>:pyf <path-to-this-file>/clang-format.py<CR>i

The first line enables :program:`clang-format` for NORMAL and VISUAL mode, the
second line adds support for INSERT mode. Change "C-K" to another binding if
you need :program:`clang-format` on a different key (C-K stands for Ctrl+k).

With this integration you can press the bound key and clang-format will
format the current line in NORMAL and INSERT mode or the selected region in
VISUAL mode. The line or region is extended to the next bigger syntactic
entity.

It operates on the current, potentially unsaved buffer and does not create
or save any files. To revert a formatting, just undo.


Script for patch reformatting
=============================

The python script `clang/tools/clang-format-diff.py` parses the output of
a unified diff and reformats all contained lines with :program:`clang-format`.

.. code-block:: console

  usage: clang-format-diff.py [-h] [-p P] [-style STYLE]

  Reformat changed lines in diff

  optional arguments:
    -h, --help    show this help message and exit
    -p P          strip the smallest prefix containing P slashes
    -style STYLE  formatting style to apply (LLVM, Google)

So to reformat all the lines in the latest :program:`git` commit, just do:

.. code-block:: console

  git diff -U0 HEAD^ | clang-format-diff.py

The :option:`-U0` will create a diff without context lines (the script would format
those as well).
