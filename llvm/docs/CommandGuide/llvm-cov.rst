llvm-cov - emit coverage information
====================================

SYNOPSIS
--------

:program:`llvm-cov` [options] SOURCEFILE

DESCRIPTION
-----------

The :program:`llvm-cov` tool reads code coverage data files and displays the
coverage information for a specified source file. It is compatible with the
``gcov`` tool from version 4.2 of ``GCC`` and may also be compatible with
some later versions of ``gcov``.

To use llvm-cov, you must first build an instrumented version of your
application that collects coverage data as it runs. Compile with the
``-fprofile-arcs`` and ``-ftest-coverage`` options to add the
instrumentation. (Alternatively, you can use the ``--coverage`` option, which
includes both of those other options.) You should compile with debugging
information (``-g``) and without optimization (``-O0``); otherwise, the
coverage data cannot be accurately mapped back to the source code.

At the time you compile the instrumented code, a ``.gcno`` data file will be
generated for each object file. These ``.gcno`` files contain half of the
coverage data. The other half of the data comes from ``.gcda`` files that are
generated when you run the instrumented program, with a separate ``.gcda``
file for each object file. Each time you run the program, the execution counts
are summed into any existing ``.gcda`` files, so be sure to remove any old
files if you do not want their contents to be included.

By default, the ``.gcda`` files are written into the same directory as the
object files, but you can override that by setting the ``GCOV_PREFIX`` and
``GCOV_PREFIX_STRIP`` environment variables. The ``GCOV_PREFIX_STRIP``
variable specifies a number of directory components to be removed from the
start of the absolute path to the object file directory. After stripping those
directories, the prefix from the ``GCOV_PREFIX`` variable is added. These
environment variables allow you to run the instrumented program on a machine
where the original object file directories are not accessible, but you will
then need to copy the ``.gcda`` files back to the object file directories
where llvm-cov expects to find them.

Once you have generated the coverage data files, run llvm-cov for each main
source file where you want to examine the coverage results. This should be run
from the same directory where you previously ran the compiler. The results for
the specified source file are written to a file named by appending a ``.gcov``
suffix. A separate output file is also created for each file included by the
main source file, also with a ``.gcov`` suffix added.

The basic content of an llvm-cov output file is a copy of the source file with
an execution count and line number prepended to every line. The execution
count is shown as ``-`` if a line does not contain any executable code. If
a line contains code but that code was never executed, the count is displayed
as ``#####``.


OPTIONS
-------

.. option:: -a, --all-blocks

 Display all basic blocks. If there are multiple blocks for a single line of
 source code, this option causes llvm-cov to show the count for each block
 instead of just one count for the entire line.

.. option:: -b, --branch-probabilities

 Display conditional branch probabilities and a summary of branch information. 

.. option:: -c, --branch-counts

 Display branch counts instead of probabilities (requires -b).

.. option:: -f, --function-summaries

 Show a summary of coverage for each function instead of just one summary for
 an entire source file.

.. option:: --help

 Display available options (--help-hidden for more).

.. option:: -l, --long-file-names

 For coverage output of files included from the main source file, add the
 main file name followed by ``##`` as a prefix to the output file names. This
 can be combined with the --preserve-paths option to use complete paths for
 both the main file and the included file.

.. option:: -n, --no-output

 Do not output any ``.gcov`` files. Summary information is still
 displayed.

.. option:: -o=<DIR|FILE>, --object-directory=<DIR>, --object-file=<FILE>

 Find objects in DIR or based on FILE's path. If you specify a particular
 object file, the coverage data files are expected to have the same base name
 with ``.gcno`` and ``.gcda`` extensions. If you specify a directory, the
 files are expected in that directory with the same base name as the source
 file.

.. option:: -p, --preserve-paths

 Preserve path components when naming the coverage output files. In addition
 to the source file name, include the directories from the path to that
 file. The directories are separate by ``#`` characters, with ``.`` directories
 removed and ``..`` directories replaced by ``^`` characters. When used with
 the --long-file-names option, this applies to both the main file name and the
 included file name.

.. option:: -u, --unconditional-branches

 Include unconditional branches in the output for the --branch-probabilities
 option.

.. option:: -version

 Display the version of llvm-cov.

EXIT STATUS
-----------

:program:`llvm-cov` returns 1 if it cannot read input files.  Otherwise, it
exits with zero.

